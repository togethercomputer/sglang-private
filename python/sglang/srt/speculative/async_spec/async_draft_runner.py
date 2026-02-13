"""Dedicated-GPU draft model runner for async speculative decoding.

This process runs on a separate GPU and receives commands from the target
scheduler via NCCL. It maintains a tree cache of speculative
continuations that can be served instantly on cache hits.

The target manages draft KV-cache block tables and sends them with every
request. The draft uses them for attention during forward passes via
the _compute_slot_map / _update_kv_mapping / _populate_kv_from_block_table
helpers.

Adapted from SSD's DraftRunner.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.speculative.async_spec.handshake import (
    CMD_EXIT,
    CMD_PREFILL,
    CMD_SPEC_REQUEST,
)

logger = logging.getLogger(__name__)


class AsyncDraftRunner:
    """Runs on a dedicated GPU, receives spec requests, manages tree cache.

    The draft runner operates in a loop:
    1. Receive command from target scheduler via NCCL
    2. For spec requests: cache lookup -> respond -> background tree decode
    3. For prefill: run draft model prefill to warm up KV cache
    """

    def __init__(
        self,
        server_args,
        draft_model_path: str,
        draft_gpu_id: int,
        channel,  # AsyncSpecNcclChannel
        vocab_size: int,
        model_runner,  # ModelRunner instance
    ):
        self.server_args = server_args
        self.draft_model_path = draft_model_path
        self.draft_gpu_id = draft_gpu_id
        self.channel = channel
        self.device = torch.device(f"cuda:{draft_gpu_id}")

        self.spec_k = server_args.speculative_num_steps
        self.fan_out = server_args.speculative_async_fan_out
        self.jit_speculate = server_args.speculative_async_jit_speculate

        # Model runner and memory pools
        self.model_runner = model_runner
        self.page_size = server_args.page_size
        self.max_blocks = channel.max_blocks
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.attn_backend = model_runner.attn_backend

        # Tree cache (matching SSD _reset_tree_cache_tensors)
        self.tree_cache_keys = torch.zeros(
            (0, 3), dtype=torch.int64, device=self.device
        )
        self.tree_cache_tokens: Optional[torch.Tensor] = None
        self.tree_cache_logits: Optional[torch.Tensor] = None

        self.vocab_size = vocab_size

    def draft_loop(self):
        """Main event loop."""
        logger.info(f"AsyncDraftRunner starting draft loop on GPU {self.draft_gpu_id}")

        while True:
            try:
                cmd = self.channel.recv_command()

                if cmd == CMD_EXIT:
                    logger.info("AsyncDraftRunner received exit command")
                    break
                elif cmd == CMD_SPEC_REQUEST:
                    self._handle_spec_request()
                elif cmd == CMD_PREFILL:
                    self._handle_prefill()
                else:
                    logger.error(
                        f"AsyncDraftRunner received unknown command: {cmd}"
                    )
                    break
            except Exception as e:
                logger.error(
                    f"AsyncDraftRunner error in draft loop: {e}", exc_info=True
                )
                break

        logger.info("AsyncDraftRunner exiting draft loop")

    def _handle_prefill(self):
        """Receive prefill tensors from target via NCCL, run draft model forward."""
        (
            num_reqs,
            total_tokens,
            input_ids,
            req_pool_indices,
            seq_lens,
            positions,
            block_tables,
        ) = self.channel.unpack_prefill()
        logger.debug(
            f"Draft prefill: {num_reqs} reqs, {total_tokens} tokens"
        )

        # Populate req_to_token_pool from block tables
        self._populate_kv_from_block_table(req_pool_indices, seq_lens, block_tables)

        # Compute out_cache_loc for all prefill tokens using block table mapping
        out_cache_loc = self._compute_out_cache_loc_for_prefill(
            req_pool_indices, seq_lens, block_tables
        )

        # Build extend_* fields
        extend_seq_lens = seq_lens.to(torch.int32)
        extend_prefix_lens = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        extend_start_loc = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        if num_reqs > 1:
            extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)

        # Construct ForwardBatch
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=num_reqs,
            input_ids=input_ids.to(torch.int32),
            req_pool_indices=req_pool_indices.to(torch.int32),
            seq_lens=seq_lens.to(torch.int32),
            out_cache_loc=out_cache_loc.to(torch.int32),
            seq_lens_sum=total_tokens,
            positions=positions.to(torch.int64),
            extend_num_tokens=total_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend,
        )

        # Run the draft model forward to populate KV cache
        self.model_runner.forward(forward_batch)

    def _handle_spec_request(self):
        """Core async spec logic: cache lookup -> respond -> background tree decode."""
        (
            B,
            K,
            fan_out,
            vocab_size,
            cache_keys,
            temperatures,
            num_tokens,
            draft_block_tables,
        ) = self.channel.unpack_spec_request()

        # Step 1: Cache lookup and build response
        out_tokens, out_logits, glue_decode_input_ids, cache_hits = (
            self._hit_cache_and_respond(
                cache_keys, B, K, vocab_size, num_tokens, temperatures, draft_block_tables
            )
        )

        # Build speculations [B, K+1] = recovery_token + K draft tokens
        recovery_tokens = cache_keys[:, 2].unsqueeze(1)  # [B, 1]
        speculations = torch.cat(
            [recovery_tokens, out_tokens.to(torch.int64)], dim=1
        )

        # Send speculations back to target via NCCL
        self.channel.send_speculations(speculations)

        # --- Target proceeds to verify while we continue ---
        # Build partial_tree_decode_args (matching SSD structure, for future tree decode)
        partial_tree_decode_args = {
            "num_tokens": num_tokens,
            "seq_ids": cache_keys[:, 0],
            "temperatures": temperatures,
            "dbt": draft_block_tables,
            "cache_hits": cache_hits,
            "returned_tokens": out_tokens,
        }

        # Reset tree cache for fresh population
        self._reset_tree_cache()

        # TODO: Tree decode (glue decode + tree decode steps)
        # tree_decode_args = self._build_tree_batch(partial_tree_decode_args, glue_decode_input_ids)
        # tokens, logits, activations = self._decode_tree(tree_decode_args)
        # self._populate_tree_cache(tree_decode_args, tokens, logits, cache_hits)

    def _hit_cache_and_respond(
        self,
        cache_keys: torch.Tensor,
        B: int,
        K: int,
        V: int,
        num_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        draft_block_tables: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Check tree cache, return cached/jit-speculated/random tokens (matching SSD).

        Returns: (out_tokens, out_logits, glue_decode_input_ids, cache_hits)
        """
        # Init with random logits so token IDs are in-vocab (matching SSD)
        out_logits = torch.empty(
            (B, K, V), dtype=torch.float32, device=self.device
        ).uniform_()
        out_tokens = out_logits.argmax(dim=-1)  # [B, K]
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)

        if self.tree_cache_keys.numel() > 0:
            # Vectorized membership against tensor cache
            eq = cache_keys.unsqueeze(1) == self.tree_cache_keys.unsqueeze(0)  # [B,T,3]
            match = torch.all(eq, dim=2)  # [B,T]
            cache_hits_bool = match.any(dim=1)  # [B]
            cache_hits = cache_hits_bool.to(torch.int64)

            if cache_hits_bool.all() or (cache_hits_bool.any() and not self.jit_speculate):
                # Fill hit slots from cache
                idx = match.float().argmax(dim=1).to(torch.int64)
                sel = cache_hits_bool
                cached_k = min(K, self.tree_cache_tokens.shape[1]) if self.tree_cache_tokens is not None else 0
                if cached_k > 0:
                    out_tokens[sel, :cached_k] = self.tree_cache_tokens[idx[sel], :cached_k]
                    if self.tree_cache_logits is not None:
                        out_logits[sel, :cached_k] = self.tree_cache_logits[idx[sel], :cached_k]
            else:
                # Cache entries exist but NOT all hits — run draft model for all
                self.jit_speculate_decode(
                    cache_keys, num_tokens, out_logits, out_tokens,
                    temperatures, draft_block_tables,
                )
        else:
            # TODO: throw error here when we properly build a cache.
            # Cache is empty — always run draft model (no cache to serve from)
            self.jit_speculate_decode(
                cache_keys, num_tokens, out_logits, out_tokens,
                temperatures, draft_block_tables,
            )

        # Build glue_decode_input_ids: [B*(K+1)] = [rec_tok, draft_tok_0, ..., draft_tok_{K-1}] per request
        rec_toks = cache_keys[:, 2]  # [B]
        glue_decode_input_ids = torch.cat(
            [rec_toks.unsqueeze(1), out_tokens], dim=1
        ).reshape(-1)  # [B*(K+1)]

        return out_tokens, out_logits, glue_decode_input_ids, cache_hits

    def jit_speculate_decode(
        self,
        request_keys: torch.Tensor,  # [B, 3]
        num_tokens: torch.Tensor,  # [B]
        out_logits: torch.Tensor,  # [B, K, V] - written in-place
        out_tokens: torch.Tensor,  # [B, K] - written in-place
        temperatures: torch.Tensor,  # [B]
        draft_block_tables: torch.Tensor,  # [B, max_blocks]
    ):
        """Run K decode steps to generate draft tokens (matching SSD jit_speculate)."""
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )

        B = request_keys.shape[0]
        input_ids = request_keys[:, -1]  # recovery tokens [B]
        positions = num_tokens - 1  # want to write rec token at pos N-1
        req_pool_indices = request_keys[:, 0]  # seq_ids as req_pool_indices

        for step in range(self.spec_k):
            # Compute slot map from block tables
            slot_map = self._compute_slot_map(positions, draft_block_tables)

            # Update KV mapping so attention can see new positions
            self._update_kv_mapping(req_pool_indices, positions, slot_map)

            # Build ForwardBatch for decode
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=B,
                input_ids=input_ids.to(torch.int32),
                req_pool_indices=req_pool_indices.to(torch.int32),
                seq_lens=(positions + 1).to(torch.int32),
                out_cache_loc=slot_map.to(torch.int32),
                seq_lens_sum=B,
                positions=positions.to(torch.int64),
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                attn_backend=self.attn_backend,
            )

            # Forward pass
            output = self.model_runner.forward(forward_batch)
            logits = output.logits_output.next_token_logits  # [B, V]

            out_logits[:, step, :] = logits
            next_tokens = self._sample(logits, temperatures)
            out_tokens[:, step] = next_tokens

            # Update for next iteration
            input_ids = next_tokens
            positions = positions + 1

    def _sample(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """Simple temperature-scaled sampling."""
        scaled = logits / temperatures.unsqueeze(-1).clamp(min=1e-5)
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # ── Shared helpers for block-table-to-KV mapping ──

    def _compute_slot_map(
        self,
        positions: torch.Tensor,  # [B] or [N]
        block_tables: torch.Tensor,  # [B, max_blocks] or [N, max_blocks]
    ) -> torch.Tensor:
        """Vectorized: block_tables[i, pos // page_size] * page_size + pos % page_size."""
        block_idx = (positions // self.page_size).to(torch.int64)
        pos_in_block = (positions % self.page_size).to(torch.int32)
        batch_indices = torch.arange(positions.shape[0], device=self.device)
        blk_ids = block_tables[batch_indices, block_idx]
        return (blk_ids * self.page_size + pos_in_block).to(torch.int32)

    def _update_kv_mapping(
        self,
        req_pool_indices: torch.Tensor,  # [B]
        positions: torch.Tensor,  # [B]
        slot_map: torch.Tensor,  # [B]
    ):
        """Write req_to_token_pool[req_pool_idx, pos] = slot_map for each entry."""
        for i in range(req_pool_indices.shape[0]):
            rpi = req_pool_indices[i].item()
            pos = positions[i].item()
            self.req_to_token_pool.req_to_token[rpi, pos] = slot_map[i].item()

    def _populate_kv_from_block_table(
        self,
        req_pool_indices: torch.Tensor,  # [R]
        seq_lens: torch.Tensor,  # [R]
        block_tables: torch.Tensor,  # [R, max_blocks]
    ):
        """Bulk-write all token positions into req_to_token_pool from block tables."""
        for r in range(req_pool_indices.shape[0]):
            rpi = req_pool_indices[r].item()
            sl = seq_lens[r].item()
            if sl == 0:
                continue
            # Vectorized computation of all KV indices for this request
            pos_range = torch.arange(sl, device=self.device, dtype=torch.int64)
            blk_idx = pos_range // self.page_size
            pos_in_blk = pos_range % self.page_size
            blk_ids = block_tables[r][blk_idx]
            kv_indices = (blk_ids * self.page_size + pos_in_blk).to(torch.int32)
            self.req_to_token_pool.req_to_token[rpi, :sl] = kv_indices

    def _compute_out_cache_loc_for_prefill(
        self,
        req_pool_indices: torch.Tensor,  # [R]
        seq_lens: torch.Tensor,  # [R]
        block_tables: torch.Tensor,  # [R, max_blocks]
    ) -> torch.Tensor:
        """Compute out_cache_loc for all prefill tokens from block tables."""
        parts = []
        for r in range(req_pool_indices.shape[0]):
            sl = seq_lens[r].item()
            if sl == 0:
                continue
            pos_range = torch.arange(sl, device=self.device, dtype=torch.int64)
            blk_idx = pos_range // self.page_size
            pos_in_blk = pos_range % self.page_size
            blk_ids = block_tables[r][blk_idx]
            kv_indices = blk_ids * self.page_size + pos_in_blk
            parts.append(kv_indices)
        return torch.cat(parts).to(torch.int32) if parts else torch.empty(0, dtype=torch.int32, device=self.device)

    def _reset_tree_cache(self):
        """Reset tensor-backed tree cache (matching SSD _reset_tree_cache_tensors)."""
        self.tree_cache_keys = torch.zeros(
            (0, 3), dtype=torch.int64, device=self.device
        )
        self.tree_cache_tokens = None
        self.tree_cache_logits = None


def run_async_draft_runner_process(
    server_args,
    draft_gpu_id: int,
    nccl_port: int,
    result_pipe,  # multiprocessing.Connection (for init status)
):
    """Entry point for the draft runner subprocess."""
    try:
        import setproctitle

        setproctitle.setproctitle(f"sglang::async_draft_runner(gpu={draft_gpu_id})")
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Draft GPU {draft_gpu_id}] %(levelname)s %(message)s",
    )

    logger.info(f"Starting async draft runner on GPU {draft_gpu_id}")

    try:
        torch.cuda.set_device(draft_gpu_id)

        # Load draft model config
        from sglang.srt.configs.model_config import ModelConfig

        draft_model_path = server_args.speculative_draft_model_path
        draft_revision = server_args.speculative_draft_model_revision or "main"

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=draft_model_path,
            model_revision=draft_revision,
            is_draft_model=True,
        )
        vocab_size = model_config.vocab_size

        # Compute max_blocks from context_length and page_size
        page_size = server_args.page_size
        context_length = model_config.context_len
        max_blocks = (context_length + page_size - 1) // page_size

        # Create ModelRunner for the draft model (single-GPU, tp_size=1)
        from sglang.srt.model_executor.model_runner import ModelRunner

        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=draft_gpu_id,
            tp_rank=0,
            tp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=nccl_port + 100,  # Use different port to avoid conflict
            server_args=server_args,
            is_draft_worker=True,
        )

        # Compute num_kv_pages from the model runner's memory pool
        num_kv_pages = model_runner.max_total_num_tokens // page_size

        # Signal readiness to parent process (before NCCL init so parent can
        # proceed to create its own NCCL channel concurrently)
        result_pipe.send(
            {
                "status": "ready",
                "draft_gpu_id": draft_gpu_id,
                "vocab_size": vocab_size,
                "num_kv_pages": num_kv_pages,
                "max_blocks": max_blocks,
            }
        )
        result_pipe.close()

        # Create NCCL channel (rank=1, draft side)
        from sglang.srt.speculative.async_spec.nccl_comm import create_nccl_channel

        device = torch.device(f"cuda:{draft_gpu_id}")
        channel = create_nccl_channel(
            rank=1,
            device=device,
            nccl_port=nccl_port,
            max_batch_size=server_args.max_running_requests or 64,
            max_spec_k=server_args.speculative_num_steps,
            max_prefill_tokens=server_args.max_prefill_tokens or 16384,
            max_blocks=max_blocks,
        )

        # Create the draft runner
        runner = AsyncDraftRunner(
            server_args=server_args,
            draft_model_path=draft_model_path,
            draft_gpu_id=draft_gpu_id,
            channel=channel,
            vocab_size=vocab_size,
            model_runner=model_runner,
        )

        # Enter the main loop
        runner.draft_loop()

    except Exception as e:
        logger.error(f"AsyncDraftRunner process failed: {e}", exc_info=True)
        try:
            result_pipe.send({"status": "error", "error": str(e)})
            result_pipe.close()
        except Exception:
            pass
        raise
