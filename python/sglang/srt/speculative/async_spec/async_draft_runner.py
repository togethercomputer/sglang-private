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
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.speculative.async_spec.handshake import (
    CMD_EXIT,
    CMD_PREFILL,
    CMD_SPEC_REQUEST,
)
from sglang.srt.speculative.async_spec.tree_utils import (
    get_forked_recovery_tokens_from_logits,
)

logger = logging.getLogger(__name__)


class _TreeDecodeKVSpec:
    """Lightweight spec_info-like object for FlashInfer kv_indptr/kv_indices bypass.

    When passed as forward_batch.spec_info, the FlashInfer backend's
    call_begin_forward skips the Triton kernel and GPU cumsum, using
    these precomputed values directly (see flashinfer_backend.py line 1103).
    """

    __slots__ = ("kv_indptr", "kv_indices")

    def __init__(self, kv_indptr: torch.Tensor, kv_indices: torch.Tensor):
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices


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

        # Fan-out lists: [K+1] entries for K+1 glue decode positions
        self.fan_out_list = server_args.speculative_async_fan_out_list
        self.fan_out_list_miss = server_args.speculative_async_fan_out_list_miss
        self.mq_len = sum(self.fan_out_list)  # Total tree width per request

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

        # Pre-allocate constant tensors used every draft step
        self._init_prealloc_buffers()

        # CUDA graph runners
        self.tree_cuda_graph_runner = None
        self.glue_decode_cuda_graph_runner = None
        if not getattr(server_args, "disable_cuda_graph", False):
            max_bs = getattr(server_args, "max_running_requests", None) or 64

            # Initialize attention backend CUDA graph state once with the max
            # of both tree decode (max_bs * mq_len) and glue decode (max_bs)
            max_tree_N = max_bs * self.mq_len
            max_glue_tokens = max_bs * (self.spec_k + 1)
            cuda_graph_max_bs = max(max_tree_N, max_bs)
            cuda_graph_max_tokens = max(max_tree_N, max_glue_tokens)
            self.attn_backend.init_cuda_graph_state(
                cuda_graph_max_bs, cuda_graph_max_tokens
            )

            # Tree decode CUDA graph runner
            try:
                from sglang.srt.speculative.async_spec.tree_cuda_graph_runner import (
                    TreeDecodeCudaGraphRunner,
                )

                self.tree_cuda_graph_runner = TreeDecodeCudaGraphRunner(
                    model_runner=self.model_runner,
                    max_batch_size=max_bs,
                    mq_len=self.mq_len,
                    device=self.device,
                )
                self.tree_cuda_graph_runner.capture()
                if not self.tree_cuda_graph_runner.graphs:
                    logger.warning("Tree CUDA graph capture failed, using eager mode")
                    self.tree_cuda_graph_runner = None
            except Exception as e:
                logger.warning(f"Tree CUDA graph runner init failed: {e}, using eager mode")
                self.tree_cuda_graph_runner = None

            # Glue decode CUDA graph runner (multi-query decode, matches SSD)
            try:
                from sglang.srt.speculative.async_spec.glue_decode_cuda_graph_runner import (
                    GlueDecodeCudaGraphRunner,
                )

                self.glue_decode_cuda_graph_runner = GlueDecodeCudaGraphRunner(
                    model_runner=self.model_runner,
                    spec_k=self.spec_k,
                    max_batch_size=max_bs,
                    device=self.device,
                )
                self.glue_decode_cuda_graph_runner.capture()
                if not self.glue_decode_cuda_graph_runner.graphs:
                    logger.warning("Glue decode CUDA graph capture failed, using eager mode")
                    self.glue_decode_cuda_graph_runner = None
            except Exception as e:
                logger.warning(f"Glue decode CUDA graph runner init failed: {e}, using eager mode")
                self.glue_decode_cuda_graph_runner = None

        # Profiling
        self._draft_step_times = []

    def _init_prealloc_buffers(self):
        """Pre-allocate constant tensors to avoid repeated CUDA mallocs."""
        K = self.spec_k
        MQ_LEN = self.mq_len
        d = self.device

        # Step position offsets: at step i, KV write positions shift by i * MQ_LEN
        self._step_pos_offsets = (
            torch.arange(K, device=d, dtype=torch.int64)[:, None] * MQ_LEN
        )
        # Step rope offsets: at step i, rope positions shift by i
        self._step_rope_offsets = torch.arange(K, device=d, dtype=torch.int64)[
            :, None
        ]

        # Fan-out depth index tensors:
        # _fan_idx_hit[m] = depth of branch m for cache hits
        # e.g. for fan_out_list=[2,2,2], K=2: [0,0,1,1,2,2]
        fan_out_t = torch.as_tensor(self.fan_out_list, device=d, dtype=torch.int64)
        fan_out_t_miss = torch.as_tensor(
            self.fan_out_list_miss, device=d, dtype=torch.int64
        )
        self._fan_idx_hit = torch.arange(K + 1, device=d, dtype=torch.int64).repeat_interleave(fan_out_t)
        self._fan_idx_miss = torch.arange(K + 1, device=d, dtype=torch.int64).repeat_interleave(fan_out_t_miss)

        # Arange for MQ_LEN positions and K+1 glue positions
        self._arange_mq = torch.arange(MQ_LEN, device=d, dtype=torch.int64)
        self._arange_kp1 = torch.arange(K + 1, device=d, dtype=torch.int64)

        # Pre-stack hit/miss fan indices for vectorized selection [2, MQ_LEN]
        # Row 0 = miss, row 1 = hit
        self._fan_idx_stack = torch.stack(
            [self._fan_idx_miss, self._fan_idx_hit], dim=0
        )  # [2, MQ_LEN]

    def _vectorized_fan_idx(self, cache_hits: torch.Tensor, B: int) -> torch.Tensor:
        """Select fan_idx_hit or fan_idx_miss per request without GPU→CPU syncs.

        Uses the pre-stacked [2, MQ_LEN] tensor and advanced indexing to
        select the correct fan indices based on cache_hits[b], avoiding the
        Python loop with B .item() calls.
        """
        # cache_hits: [B] int64 (0 or 1)
        selector = cache_hits.long().clamp(0, 1)  # [B] -> indices into dim 0
        return self._fan_idx_stack[selector].reshape(-1)  # [B, MQ_LEN] -> [B*MQ_LEN]

    def draft_loop(self):
        """Main event loop."""
        logger.info(f"AsyncDraftRunner starting draft loop on GPU {self.draft_gpu_id}")

        while True:
            try:
                cmd = self.channel.recv_command()

                if cmd == CMD_EXIT:
                    logger.info("AsyncDraftRunner received exit command")
                    if self._draft_step_times:
                        avg_ms = (
                            sum(self._draft_step_times)
                            * 1000
                            / len(self._draft_step_times)
                        )
                        logger.info(f"Avg draft step time: {avg_ms:.2f}ms")
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

    @torch.inference_mode()
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
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        seq_lens_i32 = seq_lens.to(torch.int32)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=num_reqs,
            input_ids=input_ids.to(torch.int32),
            req_pool_indices=req_pool_indices.to(torch.int32),
            seq_lens=seq_lens_i32,
            seq_lens_cpu=seq_lens_i32.cpu(),
            out_cache_loc=out_cache_loc.to(torch.int32),
            seq_lens_sum=total_tokens,
            positions=positions.to(torch.int64),
            extend_num_tokens=total_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens.tolist(),
            extend_seq_lens_cpu=extend_seq_lens.tolist(),
            extend_start_loc=extend_start_loc,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        # Run the draft model forward to populate KV cache
        self.model_runner.forward(forward_batch)

    @torch.inference_mode()
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

        _ds0 = time.perf_counter()
        _prof = os.environ.get("SGLANG_ASYNC_SPEC_PROFILE", "0") == "1"
        if _prof:
            torch.cuda.synchronize()
            _d0 = time.perf_counter()

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

        if _prof:
            torch.cuda.synchronize()
            _d1 = time.perf_counter()

        # --- Target proceeds to verify while we continue ---
        # Build partial_tree_decode_args (matching SSD structure)
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

        # Tree decode: glue decode + tree decode steps + cache populate
        tree_decode_args = self._build_tree_batch(
            partial_tree_decode_args, glue_decode_input_ids
        )

        if _prof:
            torch.cuda.synchronize()
            _d2 = time.perf_counter()

        tokens, logits = self._decode_tree(tree_decode_args)

        if _prof:
            torch.cuda.synchronize()
            _d3 = time.perf_counter()

        self._populate_tree_cache(
            tree_decode_args, tokens, logits, tree_decode_args["cache_hits"]
        )

        self._draft_step_times.append(time.perf_counter() - _ds0)

        if _prof:
            torch.cuda.synchronize()
            _d4 = time.perf_counter()
            logger.info(
                f"[PROFILE draft] service={(_d1-_d0)*1000:.2f}ms "
                f"build_tree={(_d2-_d1)*1000:.2f}ms "
                f"decode_tree={(_d3-_d2)*1000:.2f}ms "
                f"populate={(_d4-_d3)*1000:.2f}ms "
                f"total={(_d4-_d0)*1000:.2f}ms"
            )

    @torch.inference_mode()
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

    @torch.inference_mode()
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
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        B = request_keys.shape[0]
        input_ids = request_keys[:, -1]  # recovery tokens [B]
        positions = num_tokens - 1  # want to write rec token at pos N-1
        req_pool_indices = request_keys[:, 0]  # seq_ids as req_pool_indices

        # Pre-convert dtypes outside the loop
        req_pool_indices_i32 = req_pool_indices.to(torch.int32)

        # Pre-allocate a single ForwardBatch and reuse across steps
        slot_map = self._compute_slot_map(positions, draft_block_tables)
        self._update_kv_mapping(req_pool_indices, positions, slot_map)

        seq_lens_i32 = (positions + 1).to(torch.int32)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=B,
            input_ids=input_ids.to(torch.int32),
            req_pool_indices=req_pool_indices_i32,
            seq_lens=seq_lens_i32,
            seq_lens_cpu=seq_lens_i32.cpu(),
            out_cache_loc=slot_map.to(torch.int32),
            seq_lens_sum=B,
            positions=positions.to(torch.int64),
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        # Forward pass — call forward_decode directly
        logits_output = self.model_runner.forward_decode(forward_batch)
        logits = logits_output.next_token_logits  # [B, V]

        out_logits[:, 0, :] = logits
        next_tokens = self._sample(logits, temperatures)
        out_tokens[:, 0] = next_tokens

        input_ids = next_tokens
        positions = positions + 1

        for step in range(1, self.spec_k):
            # Compute slot map from block tables
            slot_map = self._compute_slot_map(positions, draft_block_tables)

            # Update KV mapping so attention can see new positions
            self._update_kv_mapping(req_pool_indices, positions, slot_map)

            # Update ForwardBatch in-place
            seq_lens_i32 = (positions + 1).to(torch.int32)
            forward_batch.input_ids = input_ids.to(torch.int32)
            forward_batch.seq_lens = seq_lens_i32
            forward_batch.seq_lens_cpu = seq_lens_i32.cpu()
            forward_batch.out_cache_loc = slot_map.to(torch.int32)
            forward_batch.positions = positions.to(torch.int64)

            # Forward pass — call forward_decode directly
            logits_output = self.model_runner.forward_decode(forward_batch)
            logits = logits_output.next_token_logits  # [B, V]

            out_logits[:, step, :] = logits
            next_tokens = self._sample(logits, temperatures)
            out_tokens[:, step] = next_tokens

            # Update for next iteration
            input_ids = next_tokens
            positions = positions + 1

    # ── Tree decode methods (adapted from SSD) ──

    @torch.inference_mode()
    def _build_tree_batch(self, partial_tree_decode_args, glue_decode_input_ids):
        """Run glue decode and fork tokens to construct tree decode arguments.

        Adapted from SSD's _build_tree_batch. Flow:
        1. Run EXTEND forward for glue decode (K+1 tokens per request)
        2. Fork alternative tokens from glue decode logits
        3. Construct tree_decode_args dictionary
        """
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        K = self.spec_k
        B = glue_decode_input_ids.shape[0] // (K + 1)
        num_tokens = partial_tree_decode_args["num_tokens"]
        dbt = partial_tree_decode_args["dbt"]
        cache_hits = partial_tree_decode_args["cache_hits"]
        temperatures = partial_tree_decode_args["temperatures"]
        seq_ids = partial_tree_decode_args["seq_ids"]  # req_pool_indices [B]

        assert B == num_tokens.shape[0], (
            f"_build_tree_batch: B={B} != num_tokens.shape[0]={num_tokens.shape[0]}"
        )

        # ── 1. Prepare and run glue decode (EXTEND forward) ──

        # Positions for glue decode: [num_tokens-1, num_tokens, ..., num_tokens+K-1] per request
        positions_start = (num_tokens - 1).unsqueeze(-1)  # [B, 1]
        positions_grid = positions_start + self._arange_kp1  # [B, K+1]
        positions_flat = positions_grid.reshape(-1).to(torch.int64)  # [B*(K+1)]

        # Compute slot map for glue decode from block tables
        b_expanded = torch.arange(B, device=self.device).unsqueeze(-1).expand(-1, K + 1)
        block_indices = (positions_grid // self.page_size).to(torch.int64)
        offsets = (positions_grid % self.page_size).to(torch.int32)
        blk_ids = dbt[b_expanded, block_indices]
        slot_map_grid = blk_ids * self.page_size + offsets
        slot_map_flat = slot_map_grid.reshape(-1).to(torch.int32)

        # Update req_to_token_pool for glue decode positions
        req_pool_indices_glue = seq_ids.unsqueeze(-1).expand(-1, K + 1).reshape(-1)
        self._update_kv_mapping(req_pool_indices_glue, positions_flat, slot_map_flat)

        # Build common tensors for glue decode
        seq_lens = (num_tokens + K).to(torch.int32)  # [B] total after extension
        extend_prefix_lens = (num_tokens - 1).to(torch.int32)  # [B]
        seq_ids_i32 = seq_ids.to(torch.int32)
        input_ids_i32 = glue_decode_input_ids.to(torch.int32)
        seq_lens_sum = int(seq_lens.sum().item())

        # ── CUDA graph path for glue decode (matches SSD's approach) ──
        use_glue_graph = (
            self.glue_decode_cuda_graph_runner is not None
            and self.glue_decode_cuda_graph_runner.can_run(B)
        )

        if use_glue_graph:
            seq_lens_cpu = seq_lens.cpu()
            output = self.glue_decode_cuda_graph_runner.replay(
                B=B,
                input_ids=input_ids_i32,
                positions=positions_flat,
                out_cache_loc=slot_map_flat,
                req_pool_indices=seq_ids_i32,
                seq_lens=seq_lens,
                seq_lens_sum=seq_lens_sum,
                seq_lens_cpu=seq_lens_cpu,
                extend_prefix_lens=extend_prefix_lens,
            )
            # Extract hidden_states from CUDA graph output
            hidden_states = output.hidden_states  # [bucket_B*(K+1), H]
            if hidden_states is not None:
                hidden_states = hidden_states[: B * (K + 1)]
            else:
                # Fallback: output might store logits differently
                hidden_states = output.next_token_logits  # unlikely but safe
        else:
            # ── Eager fallback ──
            extend_seq_lens = torch.full((B,), K + 1, dtype=torch.int32, device=self.device)
            extend_start_loc = torch.zeros(B, dtype=torch.int32, device=self.device)
            if B > 1:
                extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)

            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=B,
                input_ids=input_ids_i32,
                req_pool_indices=seq_ids_i32,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens.cpu(),
                out_cache_loc=slot_map_flat,
                seq_lens_sum=seq_lens_sum,
                positions=positions_flat,
                extend_num_tokens=B * (K + 1),
                extend_seq_lens=extend_seq_lens,
                extend_prefix_lens=extend_prefix_lens,
                extend_prefix_lens_cpu=extend_prefix_lens.tolist(),
                extend_seq_lens_cpu=extend_seq_lens.tolist(),
                extend_start_loc=extend_start_loc,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                attn_backend=self.attn_backend,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )

            output = self.model_runner.forward(forward_batch)
            hidden_states = output.logits_output.hidden_states  # [B*(K+1), H]

        # Compute all-position logits from hidden_states via lm_head
        lm_head_weight = self.model_runner.model.lm_head.weight  # [V, H]
        glue_decode_logits_flat = F.linear(
            hidden_states.to(lm_head_weight.dtype), lm_head_weight
        )  # [B*(K+1), V]
        glue_decode_logits = glue_decode_logits_flat.view(B, K + 1, -1)  # [B, K+1, V]

        # ── 2. Fork alternative tokens from glue decode logits ──

        forked_rec_tokens = get_forked_recovery_tokens_from_logits(
            glue_decode_logits,
            self.fan_out_list,
            cache_hits,
            glue_decode_input_ids.reshape(B, K + 1),
            self.fan_out_list_miss,
        ).view(-1)  # [B*MQ_LEN] = [N]

        # ── 3. Construct tree_decode_args ──

        N = B * self.mq_len
        assert forked_rec_tokens.shape[0] == N, (
            f"forked_rec_tokens.shape[0]={forked_rec_tokens.shape[0]} != N={N}"
        )

        # b_flat: maps each of the N branches to its parent request index (0..B-1)
        b_flat = (
            torch.arange(B, device=self.device, dtype=torch.int64)[:, None]
            .expand(B, self.mq_len)
            .flatten()
        )  # [N]

        # fkp1_flat: branch index within each request (0..MQ_LEN-1)
        fkp1_flat = self._arange_mq.repeat(B)  # [N]

        # j_idx_flat: tree depth for each branch (accounts for fan_out per depth)
        # Vectorized: select hit/miss fan_idx per request without B GPU→CPU syncs
        j_idx_flat = self._vectorized_fan_idx(cache_hits, B)

        # KV write positions: after glue decode, starting at num_tokens + K
        initial_positions = (num_tokens[b_flat] - 1) + (K + 1) + fkp1_flat  # [N]
        # = num_tokens[b] + K + branch_offset

        # Rope positions: based on tree depth
        initial_rope_positions = (num_tokens[b_flat] - 1) + j_idx_flat + 1  # [N]
        # = num_tokens[b] + depth

        seq_ids_expanded = seq_ids[b_flat]  # [N]
        temperatures_expanded = temperatures[b_flat]  # [N]

        tree_decode_args = {
            "B": B,
            "K": K,
            "fan_out": self.fan_out,
            "N": N,
            "input_ids": forked_rec_tokens,  # [N]
            "positions": initial_positions,  # [N] KV write positions
            "rope_positions": initial_rope_positions,  # [N] for positional encoding
            "block_tables": dbt,  # [B, max_blocks]
            "temps": temperatures_expanded,  # [N]
            "rec_flat": forked_rec_tokens,  # [N]
            "seq_ids_expanded": seq_ids_expanded,  # [N] req_pool_indices per branch
            "b_flat": b_flat,  # [N] parent request index per branch
            "cache_hits": cache_hits,  # [B]
        }

        return tree_decode_args

    @torch.inference_mode()
    def _compute_step_positions_and_slot_maps(
        self,
        initial_positions: torch.Tensor,
        initial_rope_positions: torch.Tensor,
        dbt: torch.Tensor,
        B: int,
        K: int,
        N: int,
    ):
        """Precompute positions, rope positions, context lens, and slot maps for all K tree steps.

        Returns:
            step_positions: [K, N] KV write positions per step
            step_rope_positions: [K, N] rope positions per step
            step_context_lens: [K, B] context lens per step per request
            step_slot_maps: [K, N] KV cache slot indices per step
        """
        MQ_LEN = self.mq_len

        # Position arrays for all K steps via broadcasting
        step_positions = initial_positions[None, :] + self._step_pos_offsets  # [K, N]
        step_rope_positions = initial_rope_positions[None, :] + self._step_rope_offsets  # [K, N]

        # Context lens: last branch position per request per step + 1
        step_context_lens = step_positions.view(K, B, MQ_LEN)[:, :, -1] + 1  # [K, B]

        # Slot maps for all steps from block tables
        b_flat = (
            torch.arange(B, device=self.device, dtype=torch.int64)[:, None]
            .expand(B, MQ_LEN)
            .flatten()
        )  # [N]
        batch_indices = torch.arange(N, device=self.device)
        dbt_expanded = dbt[b_flat]  # [N, max_blocks]

        step_offsets = (step_positions % self.page_size).to(torch.int32)  # [K, N]
        step_blk_idx = (step_positions // self.page_size).to(torch.int64)  # [K, N]
        step_blk_ids = dbt_expanded[batch_indices[None, :], step_blk_idx]  # [K, N]
        step_slot_maps = step_blk_ids * self.page_size + step_offsets  # [K, N]

        return step_positions, step_rope_positions, step_context_lens, step_slot_maps

    @torch.inference_mode()
    def _decode_tree(self, tree_decode_args):
        """Run K autoregressive decode steps for all N tree branches.

        Adapted from SSD's _decode_tree. Each step:
        1. Update KV mapping for current step's positions
        2. Build DECODE ForwardBatch with N entries
        3. Forward pass, sample tokens
        4. Store logits and tokens

        Returns:
            spec_tokens: [N, K] sampled tokens per branch per step
            spec_logits: [N, K, V] logits per branch per step
        """
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        B = tree_decode_args["B"]
        K = tree_decode_args["K"]
        N = tree_decode_args["N"]

        V = self.vocab_size
        spec_tokens = torch.zeros((N, K), dtype=torch.int64, device=self.device)
        spec_logits = torch.zeros((N, K, V), dtype=torch.float32, device=self.device)

        initial_positions = tree_decode_args["positions"]  # [N]
        initial_rope_positions = tree_decode_args["rope_positions"]  # [N]
        current_input_ids = tree_decode_args["input_ids"]  # [N]
        dbt = tree_decode_args["block_tables"]  # [B, max_blocks]
        temps = tree_decode_args["temps"]  # [N]
        seq_ids_expanded = tree_decode_args["seq_ids_expanded"]  # [N]
        b_flat = tree_decode_args["b_flat"]  # [N]

        # Precompute all step positions, rope positions, context lens, slot maps
        step_positions, step_rope_positions, step_context_lens, step_slot_maps = (
            self._compute_step_positions_and_slot_maps(
                initial_positions, initial_rope_positions, dbt, B, K, N
            )
        )

        _prof = os.environ.get("SGLANG_ASYNC_SPEC_PROFILE", "0") == "1"

        if _prof:
            torch.cuda.synchronize()
            _t_kv0 = time.perf_counter()

        # Bulk-populate ALL steps' KV mappings upfront.
        # Safe because seq_lens at step i limits attention to positions <= step_context_lens[i].
        all_positions = step_positions.reshape(-1)  # [K*N]
        all_slot_maps = step_slot_maps.reshape(-1)  # [K*N]
        all_rpis = seq_ids_expanded.repeat(K)  # [K*N]
        self._update_kv_mapping(all_rpis, all_positions, all_slot_maps)

        if _prof:
            torch.cuda.synchronize()
            _t_kv1 = time.perf_counter()

        # Pre-convert dtypes outside the loop
        step_slot_maps_i32 = step_slot_maps.to(torch.int32)  # [K, N]
        seq_ids_expanded_i32 = seq_ids_expanded.to(torch.int32)  # [N]
        step_rope_positions_i64 = step_rope_positions.to(torch.int64)  # [K, N]

        # Pre-expand step_context_lens [K, B] -> [K, N] and convert dtype
        step_ctx_lens_expanded = step_context_lens[:, b_flat].to(torch.int32)  # [K, N]

        # Pre-compute seq_lens_sum per step
        step_seq_lens_sums = step_ctx_lens_expanded.sum(dim=1)  # [K] on GPU

        # ── CUDA graph path (if available) ──
        use_graph = (
            self.tree_cuda_graph_runner is not None
            and self.tree_cuda_graph_runner.can_run(N)
        )

        if use_graph:
            # Precompute all K steps' CPU-side plan args (1 GPU→CPU sync)
            self.tree_cuda_graph_runner.precompute_plans(
                K, N,
                step_ctx_lens_expanded,
                step_seq_lens_sums,
                seq_ids_expanded_i32,
                step_slot_maps_i32,
                step_rope_positions_i64,
            )

            for depth in range(K):
                if _prof:
                    torch.cuda.synchronize()
                    _st = time.perf_counter()

                logits = self.tree_cuda_graph_runner.replay(
                    step=depth, input_ids=current_input_ids
                )

                spec_logits[:, depth, :] = logits
                next_tokens = self._sample(logits, temps)
                spec_tokens[:, depth] = next_tokens
                current_input_ids = next_tokens

                if _prof:
                    torch.cuda.synchronize()
                    _et = time.perf_counter()
                    logger.info(
                        f"[PROFILE draft] tree_step[{depth}] (graph)={(_et-_st)*1000:.2f}ms"
                    )
        else:
            # ── Eager fallback path ──
            # Batch GPU→CPU transfers (1 sync instead of K separate per-step syncs).
            # This provides seq_lens_cpu to the FlashInfer backend so it avoids
            # internal .cpu() transfers in call_begin_forward.
            step_seq_lens_sums_cpu = step_seq_lens_sums.cpu()  # [K]
            step_ctx_lens_cpu = step_ctx_lens_expanded.cpu()  # [K, N]

            # ── Pre-compute FlashInfer KV metadata for all K steps (SSD-style) ──
            # This lets call_begin_forward skip the per-step Triton kernel +
            # cumsum by passing kv_indptr & kv_indices via spec_info.
            step_kv_specs = self._precompute_kv_specs(
                K, N, seq_ids_expanded_i32, step_ctx_lens_expanded,
                step_seq_lens_sums_cpu,
            )

            # Reuse a single ForwardBatch, mutating fields each step
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=N,
                input_ids=current_input_ids.to(torch.int32),
                req_pool_indices=seq_ids_expanded_i32,
                seq_lens=step_ctx_lens_expanded[0],
                seq_lens_cpu=step_ctx_lens_cpu[0],
                out_cache_loc=step_slot_maps_i32[0],
                seq_lens_sum=int(step_seq_lens_sums_cpu[0]),
                positions=step_rope_positions_i64[0],
                spec_info=step_kv_specs[0] if step_kv_specs else None,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                attn_backend=self.attn_backend,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            for depth in range(K):
                if _prof:
                    torch.cuda.synchronize()
                    _st = time.perf_counter()

                # Update ForwardBatch fields in-place for this step
                forward_batch.input_ids = current_input_ids.to(torch.int32)
                forward_batch.seq_lens = step_ctx_lens_expanded[depth]
                forward_batch.seq_lens_cpu = step_ctx_lens_cpu[depth]
                forward_batch.out_cache_loc = step_slot_maps_i32[depth]
                forward_batch.seq_lens_sum = int(step_seq_lens_sums_cpu[depth])
                forward_batch.positions = step_rope_positions_i64[depth]
                if step_kv_specs:
                    forward_batch.spec_info = step_kv_specs[depth]

                # Forward pass — call forward_decode directly
                logits_output = self.model_runner.forward_decode(forward_batch)
                logits = logits_output.next_token_logits  # [N, V]

                spec_logits[:, depth, :] = logits
                next_tokens = self._sample(logits, temps)
                spec_tokens[:, depth] = next_tokens
                current_input_ids = next_tokens

                if _prof:
                    torch.cuda.synchronize()
                    _et = time.perf_counter()
                    logger.info(
                        f"[PROFILE draft] tree_step[{depth}] (eager)={(_et-_st)*1000:.2f}ms"
                    )

        if _prof:
            torch.cuda.synchronize()
            _t_end = time.perf_counter()
            mode = "graph" if use_graph else "eager"
            logger.info(
                f"[PROFILE draft] _decode_tree ({mode}): "
                f"kv_mapping={(_t_kv1-_t_kv0)*1000:.2f}ms "
                f"total_tree={(_t_end-_t_kv0)*1000:.2f}ms "
                f"B={B} K={K} N={N}"
            )

        return spec_tokens, spec_logits

    @torch.inference_mode()
    def _populate_tree_cache(
        self,
        tree_decode_args: dict,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        cache_hits: torch.Tensor,
    ):
        """Store tree decode results in the tree cache for future cache hits.

        Adapted from SSD's _populate_tree_cache.

        Keys are (seq_id, depth_index, recovery_token) tuples.
        Values are the K speculated tokens and K logits from that starting point.
        """
        seq_ids_expanded = tree_decode_args["seq_ids_expanded"].to(torch.int64)  # [N]
        rec_flat = tree_decode_args["rec_flat"].to(torch.int64)  # [N]

        # j_idx: tree depth per branch, same construction as in _build_tree_batch
        j_idx_flat = self._vectorized_fan_idx(cache_hits, cache_hits.shape[0])

        # Keys: (seq_id, depth, recovery_token) per branch
        keys = torch.stack([seq_ids_expanded, j_idx_flat, rec_flat], dim=1).contiguous()  # [N, 3]

        self.tree_cache_keys = keys
        self.tree_cache_tokens = tokens  # [N, K]
        self.tree_cache_logits = logits  # [N, K, V]

        logger.debug(
            f"Tree cache populated: {keys.shape[0]} entries, "
            f"tokens={tokens.shape}, logits={logits.shape}"
        )

    # ── Pre-computed KV spec info for FlashInfer bypass ──

    def _precompute_kv_specs(
        self,
        K: int,
        N: int,
        req_pool_indices: torch.Tensor,  # [N] int32
        step_ctx_lens: torch.Tensor,  # [K, N] int32 GPU
        step_sums_cpu: torch.Tensor,  # [K] CPU
    ):
        """Pre-compute kv_indptr + kv_indices for all K tree decode steps.

        This matches SSD's step-0 precomputation: by providing kv_indptr and
        kv_indices directly, the FlashInfer backend skips the per-step Triton
        kernel (create_flashinfer_kv_indices_triton) and GPU cumsum.

        Returns a list of K _TreeDecodeKVSpec objects, or [] if the backend
        doesn't support the spec_info bypass (non-FlashInfer backends).
        """
        try:
            req_to_token = self.req_to_token_pool.req_to_token
        except AttributeError:
            return []

        specs = []
        for s in range(K):
            seq_lens_s = step_ctx_lens[s]  # [N] int32
            # kv_indptr: cumulative sum of context lens per entry, [N+1]
            kv_indptr = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
            kv_indptr[1:] = torch.cumsum(seq_lens_s, dim=0)
            total_kv = int(step_sums_cpu[s])
            # kv_indices: flatten req_to_token[rpi, 0:seq_len] for all N entries
            kv_indices = torch.empty(total_kv, dtype=torch.int32, device=self.device)
            # Use the same Triton kernel for extraction (runs once per step at
            # precompute time, not during the hot decode loop)
            from sglang.srt.layers.attention.utils import (
                create_flashinfer_kv_indices_triton,
            )
            create_flashinfer_kv_indices_triton[(N,)](
                req_to_token,
                req_pool_indices,
                seq_lens_s,
                kv_indptr,
                None,  # kv_start_idx
                kv_indices,
                req_to_token.shape[1],
            )
            specs.append(_TreeDecodeKVSpec(kv_indptr=kv_indptr, kv_indices=kv_indices))
        return specs

    # ── Sampling ──

    def _sample(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """Gumbel-Max sampling (matches SSD's sampler).

        Uses the Gumbel-Max trick: sample = argmax(probs / Exp(1)) which is
        equivalent to multinomial sampling but uses argmax instead of the
        more expensive torch.multinomial (which does internal sorting/scanning).
        Greedy (temp=0) requests use direct argmax.
        """
        logits_f = logits.float()
        greedy_tokens = logits_f.argmax(dim=-1)
        zero_mask = temperatures == 0

        logits_f.div_(temperatures.unsqueeze(-1).clamp(min=1e-5))
        probs = torch.softmax(logits_f, dim=-1)
        # Gumbel-Max: divide by Exp(1) noise, take argmax
        scores = probs.div_(torch.empty_like(probs).exponential_(1).clamp_(min=1e-10))
        sample_tokens = scores.argmax(dim=-1)
        return torch.where(zero_mask, greedy_tokens, sample_tokens)

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
        self.req_to_token_pool.req_to_token[
            req_pool_indices.long(), positions.long()
        ] = slot_map.to(self.req_to_token_pool.req_to_token.dtype)

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
