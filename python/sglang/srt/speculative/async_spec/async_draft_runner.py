"""Dedicated-GPU draft model runner for async speculative decoding.

This process runs on a separate GPU and receives commands from the target
scheduler via NCCL. It maintains a tree cache of speculative
continuations that can be served instantly on cache hits.

The draft runner loads its own model via sglang's ModelRunner infrastructure
(with world_size=1) and performs real forward passes for:
  - Prefill: populate draft KV cache for new sequences
  - JIT speculate: K decode steps on cache misses for immediate response
  - Glue decode: process returned speculations to fork tree branches
  - Tree decode: K steps of branched speculation to populate tree cache

KV cache management: Unlike SSD where the target manages draft block tables,
here the draft manages its own req_to_token_pool and token_to_kv_pool_allocator.
The target's seq_lens (sent via NCCL) is the authoritative post-verification
state; the draft rolls back its KV cache to match at the start of each spec
request. All slot assignments use vectorized tensor indexing to avoid
per-element .item() calls and CUDA syncs.

Adapted from SSD's DraftRunner.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.speculative.async_spec.handshake import (
    CMD_EXIT,
    CMD_PREFILL,
    CMD_SPEC_REQUEST,
)
from sglang.srt.speculative.async_spec.tree_utils import (
    compute_mq_len,
    get_forked_recovery_tokens_from_logits,
    make_glue_decode_input_ids,
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
    ):
        self.server_args = server_args
        self.draft_model_path = draft_model_path
        self.draft_gpu_id = draft_gpu_id
        self.channel = channel
        self.device = torch.device(f"cuda:{draft_gpu_id}")

        self.spec_k = server_args.speculative_num_steps
        self.fan_out = server_args.speculative_async_fan_out
        self.fan_out_list = server_args.speculative_async_fan_out_list
        self.fan_out_list_miss = server_args.speculative_async_fan_out_list_miss
        self.jit_speculate = server_args.speculative_async_jit_speculate
        self.draft_temperature = server_args.speculative_async_draft_temperature

        # Compute MQ_LEN (total tree query tokens per seq)
        self.mq_len = compute_mq_len(self.fan_out_list)
        self.mq_len_miss = compute_mq_len(self.fan_out_list_miss)

        # Tree cache (matching SSD _reset_tree_cache_tensors)
        self.tree_cache_keys = torch.zeros(
            (0, 3), dtype=torch.int64, device=self.device
        )
        self.tree_cache_tokens: Optional[torch.Tensor] = None
        self.tree_cache_logits: Optional[torch.Tensor] = None

        self.vocab_size = vocab_size

        # Model loading (must happen before tensor-based state init)
        self._init_model()

        max_reqs = self.model_runner.req_to_token_pool.size
        max_context_len = self.model_runner.req_to_token_pool.req_to_token.shape[1]

        # Tensor-based state tracking (avoids Python dicts and .item() syncs)
        # Maps target req_pool_idx -> draft req_pool_idx (-1 = unmapped)
        self.target_to_draft = torch.full(
            (max_context_len,), -1, dtype=torch.int64, device=self.device
        )
        # Current KV cache length per draft slot
        self.draft_seq_lens = torch.zeros(
            max_reqs, dtype=torch.int64, device=self.device
        )
        # Free slot tracking for draft req pool (CPU — only touched during prefill)
        self.draft_pool_free_slots: List[int] = list(range(max_reqs - 1, -1, -1))

        # Pre-allocate fan_out expansion tensors for tree decode
        # These map each (batch, depth) pair to its fan-out count, avoiding
        # per-element Python loops during the hot path.
        self._precompute_fan_out_tensors()

        # Timing stats
        self._draft_step_times: List[float] = []

    def _precompute_fan_out_tensors(self):
        """Pre-compute repeat_interleave patterns for tree expansion.

        fan_out_list has K+1 entries (one per glue decode position: recovery + K draft).
        Matches SSD's _init_prealloc_buffers.
        """
        K = self.spec_k
        d = self.device

        # fan_out tensors: K+1 entries each
        self._fo_hit_t = torch.tensor(
            self.fan_out_list, dtype=torch.int64, device=d
        )
        self._fo_miss_t = torch.tensor(
            self.fan_out_list_miss, dtype=torch.int64, device=d
        )

        # Fan index: maps each MQ position to its K+1 depth index.
        # E.g. for fan_out_list=[2,2,1], _fan_idx_hit = [0,0,1,1,2]
        self._fan_idx_hit = torch.arange(
            K + 1, device=d, dtype=torch.int64
        ).repeat_interleave(self._fo_hit_t)
        self._fan_idx_miss = torch.arange(
            K + 1, device=d, dtype=torch.int64
        ).repeat_interleave(self._fo_miss_t)

        # Pre-allocate step offset tensors for tree decode precomputation.
        # _step_pos_offsets[d] = d * MQ_LEN: position advance per tree step
        # _step_rope_offsets[d] = d: rope position advance per tree step
        self._step_pos_offsets = (
            torch.arange(K, device=d, dtype=torch.int64)[:, None] * self.mq_len
        )
        self._step_rope_offsets = torch.arange(
            K, device=d, dtype=torch.int64
        )[:, None]
        self._arange_mq = torch.arange(
            self.mq_len, device=d, dtype=torch.int64
        )

    def _init_model(self):
        """Initialize distributed env and create ModelRunner for draft model."""
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.srt.layers.dp_attention import initialize_dp_attention
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.server_args import set_global_server_args_for_scheduler
        from sglang.srt.utils.common import get_free_port

        # Create a modified copy of server_args for the draft worker
        draft_server_args = copy.deepcopy(self.server_args)
        draft_server_args.tp_size = 1
        draft_server_args.dp_size = 1
        draft_server_args.pp_size = 1
        # Enable CUDA graphs for decode forward passes (JIT + tree decode).
        # This matches SSD's approach of capturing decode graphs for the draft model.
        draft_server_args.disable_cuda_graph = False
        draft_server_args.speculative_algorithm = None
        # Use a reasonable memory fraction for the draft model
        draft_server_args.mem_fraction_static = 0.80
        # Ensure the draft model path is used
        draft_server_args.model_path = self.draft_model_path

        # Set global server args (required by initialize_dp_attention -> is_nsa_enable_prefill_cp)
        set_global_server_args_for_scheduler(draft_server_args)

        # Load draft model config
        draft_model_config = ModelConfig.from_server_args(
            draft_server_args,
            model_path=self.draft_model_path,
            model_revision=self.server_args.speculative_draft_model_revision or "main",
            is_draft_model=True,
        )

        # Initialize distributed environment (world_size=1 for draft)
        free_port = get_free_port()
        dist_init_method = f"tcp://127.0.0.1:{free_port}"

        init_distributed_environment(
            backend="nccl",
            world_size=1,
            rank=0,
            local_rank=self.draft_gpu_id,
            distributed_init_method=dist_init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
        initialize_dp_attention(
            server_args=draft_server_args,
            model_config=draft_model_config,
        )

        # Create ModelRunner (is_draft_worker=True, no shared pools)
        self.model_runner = ModelRunner(
            model_config=draft_model_config,
            mem_fraction_static=draft_server_args.mem_fraction_static,
            gpu_id=self.draft_gpu_id,
            tp_rank=0,
            tp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=free_port,
            server_args=draft_server_args,
            dp_rank=0,
            is_draft_worker=True,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
        )

        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.token_to_kv_pool_allocator = self.model_runner.token_to_kv_pool_allocator

        logger.info(
            f"Draft model loaded: {self.draft_model_path}, "
            f"max_total_num_tokens={self.model_runner.max_total_num_tokens}"
        )

    # ── Forward pass helpers ──

    def _make_decode_forward_batch(
        self,
        input_ids: torch.Tensor,  # [B]
        req_pool_indices: torch.Tensor,  # [B]
        seq_lens: torch.Tensor,  # [B]
        out_cache_loc: torch.Tensor,  # [B]
    ):
        """Construct a ForwardBatch for decode (one token per sequence).

        Minimizes CPU-GPU sync: uses non_blocking .cpu() transfer and
        defers .item() to a single call.
        """
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        B = input_ids.shape[0]
        positions = (seq_lens - 1).clamp(min=0)
        # Single non-blocking CPU transfer for seq_lens
        seq_lens_cpu = seq_lens.cpu()

        return ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=B,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens_cpu.sum()),
            positions=positions,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
        )

    def _make_extend_forward_batch(
        self,
        input_ids: torch.Tensor,  # [N] (total tokens)
        req_pool_indices: torch.Tensor,  # [B]
        seq_lens: torch.Tensor,  # [B] (final seq len after extend)
        out_cache_loc: torch.Tensor,  # [N]
        extend_prefix_lens: torch.Tensor,  # [B]
        extend_seq_lens: torch.Tensor,  # [B] (number of new tokens per req)
    ):
        """Construct a ForwardBatch for extend (prefill/glue decode).

        Minimizes CPU-GPU sync by batching transfers and computing
        max on CPU after a single .cpu() call.
        """
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        B = req_pool_indices.shape[0]
        N = input_ids.shape[0]

        # Single CPU transfer for extend_seq_lens, then compute max on CPU
        extend_seq_lens_cpu = extend_seq_lens.cpu()
        max_el = int(extend_seq_lens_cpu.max())

        # Compute positions vectorized: arange per req, offset by prefix_len
        offsets = torch.arange(
            max_el, device=self.device, dtype=torch.long
        ).unsqueeze(0)  # [1, max_el]
        # mask[b, j] = True if j < extend_seq_lens[b]
        mask = offsets < extend_seq_lens.unsqueeze(1)  # [B, max_el]
        positions = (extend_prefix_lens.unsqueeze(1) + offsets).masked_select(mask)

        # Compute extend_start_loc via cumsum
        extend_start_loc = torch.zeros(B, device=self.device, dtype=torch.int32)
        if B > 0:
            extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0).to(
                torch.int32
            )

        # Single CPU transfer for seq_lens, compute sum on CPU
        seq_lens_cpu = seq_lens.cpu()
        extend_prefix_lens_cpu = extend_prefix_lens.cpu()

        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=B,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens_cpu.sum()),
            positions=positions,
            extend_num_tokens=N,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu.tolist(),
            extend_seq_lens_cpu=extend_seq_lens_cpu.tolist(),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
        )

    @torch.inference_mode()
    def _run_forward(self, fb) -> torch.Tensor:
        """Run model forward and return next_token_logits [num_tokens, V].

        Uses CUDA graph replay for decode mode when available,
        falling back to eager forward otherwise.
        """
        graph_runner = self.model_runner.graph_runner
        if (
            fb.forward_mode.is_decode()
            and graph_runner is not None
            and graph_runner.can_run(fb)
        ):
            # CUDA graph path: replay captured decode graph
            logits_output = graph_runner.replay(fb)
        elif fb.forward_mode.is_decode():
            # Eager decode fallback
            self.model_runner.attn_backend.init_forward_metadata(fb)
            logits_output = self.model_runner.forward_decode(
                fb, skip_attn_backend_init=True
            )
        else:
            # Extend path (glue decode) — always eager
            self.model_runner.attn_backend.init_forward_metadata(fb)
            logits_output, _ = self.model_runner.forward_extend(
                fb, skip_attn_backend_init=True
            )
        return logits_output.next_token_logits

    def _sample_tokens(
        self, logits: torch.Tensor, temperatures: torch.Tensor
    ) -> torch.Tensor:
        """Sample tokens from logits using Gumbel-max trick. Returns [B] token ids.

        Matches SSD's sampler: for stochastic sampling, uses
        probs / exponential_noise → argmax, which is ~2.4x faster than
        torch.multinomial while producing equivalent samples.
        """
        if self.draft_temperature is not None and self.draft_temperature > 0:
            # Fixed draft temperature — Gumbel-max trick
            logits_f = logits.float() / self.draft_temperature
            probs = F.softmax(logits_f, dim=-1)
            scores = probs / (torch.empty_like(probs).exponential_(1) + 1e-10)
            return scores.argmax(dim=-1)
        elif temperatures is not None and (temperatures > 0).any():
            # Per-request temperatures with greedy/stochastic mix
            logits_f = logits.float()
            greedy_tokens = logits_f.argmax(dim=-1)
            zero_mask = temperatures <= 0
            if zero_mask.all():
                return greedy_tokens
            # Gumbel-max for stochastic requests
            temps = temperatures.unsqueeze(-1).clamp(min=1e-6)
            probs = F.softmax(logits_f / temps, dim=-1)
            scores = probs / (torch.empty_like(probs).exponential_(1) + 1e-10)
            sample_tokens = scores.argmax(dim=-1)
            return torch.where(zero_mask, greedy_tokens, sample_tokens)
        else:
            # Pure greedy
            return logits.argmax(dim=-1)

    # ── Vectorized req_to_token_pool helpers ──

    def _assign_slots_to_pool(
        self,
        draft_indices: torch.Tensor,  # [B] draft pool row indices
        base_positions: torch.Tensor,  # [B] starting column position per req
        tokens_per_req: int,  # number of contiguous slots per req
        alloc_locs: torch.Tensor,  # [B * tokens_per_req] allocated KV slots
    ):
        """Vectorized assignment: req_to_token[draft_idx, base+j] = loc for j in [0, tokens_per_req).

        Replaces the pattern:
            for i in range(B):
                for j in range(tokens_per_req):
                    req_to_token[draft_idx[i], base[i]+j] = locs[i*tpr+j]
        """
        pool = self.req_to_token_pool.req_to_token
        # Build row indices: each draft_idx repeated tokens_per_req times
        rows = draft_indices.repeat_interleave(tokens_per_req)  # [B * tpr]
        # Build col indices: base_positions[i] + 0, 1, ..., tpr-1
        offsets = torch.arange(tokens_per_req, device=self.device, dtype=torch.int64)
        cols = (base_positions.unsqueeze(1) + offsets).reshape(-1)  # [B * tpr]
        pool[rows, cols] = alloc_locs.to(pool.dtype)

    def _assign_slots_per_element(
        self,
        draft_indices: torch.Tensor,  # [N] draft pool row indices (may repeat)
        positions: torch.Tensor,  # [N] column positions
        alloc_locs: torch.Tensor,  # [N] allocated KV slots
    ):
        """Vectorized assignment: req_to_token[draft_idx[i], pos[i]] = loc[i]."""
        pool = self.req_to_token_pool.req_to_token
        pool[draft_indices, positions] = alloc_locs.to(pool.dtype)

    # ── Req pool management ──

    def _alloc_draft_slot(self) -> int:
        """Allocate a draft req pool index."""
        if not self.draft_pool_free_slots:
            raise RuntimeError("Draft req pool exhausted")
        return self.draft_pool_free_slots.pop()

    def _free_draft_slot(self, idx: int):
        """Return a draft req pool index to the free list."""
        self.draft_pool_free_slots.append(idx)

    def _resolve_draft_indices(
        self, target_req_indices: torch.Tensor
    ) -> torch.Tensor:
        """Look up draft pool indices for a batch of target req indices.

        Uses the tensor-based target_to_draft mapping. Returns -1 for
        unmapped target indices.
        """
        return self.target_to_draft[target_req_indices]

    def _rollback_kv_cache(
        self,
        draft_indices: torch.Tensor,  # [B] draft pool indices
        target_seq_lens: torch.Tensor,  # [B] authoritative lengths from target
    ):
        """Roll back draft KV cache to match verified target state.

        After the target verifies speculated tokens, only A of K may be accepted.
        The draft's KV cache may contain stale entries from previous JIT speculate
        and tree decode rounds. This method frees KV slots beyond the target's
        verified length and updates draft_seq_lens to match.

        Analogous to SSD's target-side block deallocation after verification.
        """
        cur_lens = self.draft_seq_lens[draft_indices]  # [B]
        needs_rollback = (cur_lens > target_seq_lens) & (target_seq_lens > 0)

        if not needs_rollback.any():
            return

        # Collect all excess KV slot indices to free in one batch
        excess_locs_list = []
        rb_indices = needs_rollback.nonzero(as_tuple=True)[0]
        for idx in rb_indices:
            di = draft_indices[idx]
            tl = target_seq_lens[idx]
            dl = cur_lens[idx]
            excess_locs_list.append(
                self.req_to_token_pool.req_to_token[di, tl:dl]
            )

        if excess_locs_list:
            self.token_to_kv_pool_allocator.free(torch.cat(excess_locs_list))

        # Update draft_seq_lens for rolled-back requests
        self.draft_seq_lens[draft_indices[needs_rollback]] = target_seq_lens[
            needs_rollback
        ]

    # ── Main event loop ──

    def draft_loop(self):
        """Main event loop."""
        logger.info(
            f"AsyncDraftRunner starting draft loop on GPU {self.draft_gpu_id}"
        )

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
                        logger.info(f"Avg draft step time: {avg_ms:.2f} ms")
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

    # ── Handle prefill ──

    @torch.inference_mode()
    def _handle_prefill(self):
        """Receive prefill data from target, run draft model prefill to populate KV cache."""
        num_reqs, total_tokens, input_ids, target_req_pool_indices, seq_lens = (
            self.channel.unpack_prefill()
        )
        logger.info(f"Draft prefill: {num_reqs} reqs, {total_tokens} tokens")

        if num_reqs == 0:
            return

        # Allocate draft req pool slots (CPU-side — only during prefill)
        draft_req_indices = []
        per_req_lens = []
        for i in range(num_reqs):
            target_idx = target_req_pool_indices[i].item()
            sl = seq_lens[i].item()

            # Free old slot if this target req was already tracked
            old_draft_idx = self.target_to_draft[target_idx].item()
            if old_draft_idx >= 0:
                old_sl = self.draft_seq_lens[old_draft_idx].item()
                if old_sl > 0:
                    old_locs = self.req_to_token_pool.req_to_token[
                        old_draft_idx, :old_sl
                    ]
                    self.token_to_kv_pool_allocator.free(old_locs)
                self._free_draft_slot(old_draft_idx)
                self.draft_seq_lens[old_draft_idx] = 0
                self.target_to_draft[target_idx] = -1

            draft_idx = self._alloc_draft_slot()
            self.target_to_draft[target_idx] = draft_idx
            draft_req_indices.append(draft_idx)
            per_req_lens.append(sl)

        # Allocate KV cache slots for all tokens
        alloc_locs = self.token_to_kv_pool_allocator.alloc(total_tokens)
        if alloc_locs is None:
            logger.warning(
                f"Draft prefill: failed to allocate {total_tokens} KV cache slots, skipping"
            )
            for i in range(num_reqs):
                target_idx = target_req_pool_indices[i].item()
                draft_idx = self.target_to_draft[target_idx].item()
                if draft_idx >= 0:
                    self._free_draft_slot(draft_idx)
                    self.target_to_draft[target_idx] = -1
            return

        # Assign allocated slots to req_to_token_pool (vectorized per req)
        draft_req_indices_t = torch.tensor(
            draft_req_indices, dtype=torch.int64, device=self.device
        )
        per_req_lens_t = torch.tensor(
            per_req_lens, dtype=torch.int64, device=self.device
        )

        # Build (row, col) pairs for all tokens across all reqs
        # For variable-length reqs, we need per-req offsets into alloc_locs
        pool = self.req_to_token_pool.req_to_token
        pool_dtype = pool.dtype
        slot_offset = 0
        for i in range(num_reqs):
            sl = per_req_lens[i]
            di = draft_req_indices[i]
            pool[di, :sl] = alloc_locs[slot_offset : slot_offset + sl].to(pool_dtype)
            slot_offset += sl

        self.draft_seq_lens[draft_req_indices_t] = per_req_lens_t

        # Build and run forward batch (EXTEND)
        extend_prefix_lens = torch.zeros(
            num_reqs, dtype=torch.int64, device=self.device
        )

        fb = self._make_extend_forward_batch(
            input_ids=input_ids[:total_tokens].to(torch.int32),
            req_pool_indices=draft_req_indices_t,
            seq_lens=per_req_lens_t,
            out_cache_loc=alloc_locs,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=per_req_lens_t,
        )

        self._run_forward(fb)
        logger.info("Draft prefill complete")

    # ── Handle spec request ──

    @torch.inference_mode()
    def _handle_spec_request(self):
        """Core async spec logic: cache lookup -> respond -> background tree decode."""
        t0 = time.perf_counter()

        B, K, fan_out, vocab_size, cache_keys, temperatures, seq_lens = (
            self.channel.unpack_spec_request()
        )

        # Resolve draft indices from target req pool indices (vectorized lookup)
        target_req_indices = cache_keys[:, 0]
        draft_indices = self._resolve_draft_indices(target_req_indices)

        # Roll back draft KV cache to match verified target state.
        self._rollback_kv_cache(draft_indices, seq_lens)

        # Step 1: Cache lookup + optional JIT speculate -> respond to target
        speculations, out_tokens, cache_hits = self._hit_cache_and_respond_with_model(
            cache_keys, B, K, vocab_size, temperatures, draft_indices
        )

        # Send speculations back to target via NCCL
        self.channel.send_speculations(speculations)

        # --- Target proceeds to verify while we continue ---
        # Step 2: Reset tree cache and run background tree decode
        self._reset_tree_cache()
        self._build_and_decode_tree(
            cache_keys, B, K, out_tokens, temperatures, draft_indices, cache_hits
        )

        self._draft_step_times.append(time.perf_counter() - t0)

    def _hit_cache_and_respond_with_model(
        self,
        cache_keys: torch.Tensor,
        B: int,
        K: int,
        V: int,
        temperatures: torch.Tensor,
        draft_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Check tree cache; on miss with jit_speculate, run K decode steps.

        Returns: (speculations [B, K+1], out_tokens [B, K], cache_hits [B])
        """
        # Init with random logits so token IDs are in-vocab
        out_logits = torch.empty(
            (B, K, V), dtype=torch.float32, device=self.device
        ).uniform_()
        out_tokens = out_logits.argmax(dim=-1)  # [B, K]
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)

        recovery_tokens = cache_keys[:, 2]  # [B]

        # Vectorized cache lookup
        if self.tree_cache_keys.numel() > 0:
            eq = cache_keys.unsqueeze(1) == self.tree_cache_keys.unsqueeze(0)
            match = torch.all(eq, dim=2)
            cache_hits_bool = match.any(dim=1)
            cache_hits = cache_hits_bool.to(torch.int64)

            if cache_hits_bool.any() and self.tree_cache_tokens is not None:
                idx = match.float().argmax(dim=1).to(torch.int64)
                sel = cache_hits_bool
                cached_k = min(K, self.tree_cache_tokens.shape[1])
                out_tokens[sel, :cached_k] = self.tree_cache_tokens[
                    idx[sel], :cached_k
                ]
                if self.tree_cache_logits is not None:
                    out_logits[sel, :cached_k] = self.tree_cache_logits[
                        idx[sel], :cached_k
                    ]

        # JIT speculate for cache misses
        all_hit = cache_hits.all().item()
        if self.jit_speculate and not all_hit:
            self._jit_speculate(
                cache_keys, draft_indices, out_logits, out_tokens, temperatures
            )

        # Build speculations [B, K+1] = recovery_token + K draft tokens
        speculations = torch.cat(
            [recovery_tokens.unsqueeze(1), out_tokens.to(torch.int64)], dim=1
        )

        return speculations, out_tokens, cache_hits

    @torch.inference_mode()
    def _jit_speculate(
        self,
        cache_keys: torch.Tensor,  # [B, 3]
        draft_indices: torch.Tensor,  # [B]
        out_logits: torch.Tensor,  # [B, K, V] - written into
        out_tokens: torch.Tensor,  # [B, K] - written into
        temperatures: torch.Tensor,  # [B]
    ):
        """Run K sequential decode steps to produce JIT speculations.

        All req_to_token assignments are vectorized (one indexed write per step).
        """
        B = cache_keys.shape[0]
        K = self.spec_k

        input_ids = cache_keys[:, 2].to(torch.int32)  # recovery tokens
        draft_cur_lens = self.draft_seq_lens[draft_indices].clone()  # [B]

        for step in range(K):
            # Allocate 1 KV slot per request
            new_locs = self.token_to_kv_pool_allocator.alloc(B)
            if new_locs is None:
                logger.warning(f"JIT speculate: KV alloc failed at step {step}")
                break

            # Vectorized slot assignment: req_to_token[draft_idx[i], cur_len[i]] = loc[i]
            self._assign_slots_per_element(draft_indices, draft_cur_lens, new_locs)

            cur_seq_lens = draft_cur_lens + 1

            # Build decode forward batch
            fb = self._make_decode_forward_batch(
                input_ids=input_ids,
                req_pool_indices=draft_indices,
                seq_lens=cur_seq_lens,
                out_cache_loc=new_locs,
            )

            logits = self._run_forward(fb)  # [B, V]
            out_logits[:, step, :] = logits
            next_tokens = self._sample_tokens(logits, temperatures)
            out_tokens[:, step] = next_tokens

            input_ids = next_tokens.to(torch.int32)
            draft_cur_lens = cur_seq_lens

        # Update tracked seq lens (vectorized)
        self.draft_seq_lens[draft_indices] = draft_cur_lens

    # ── Glue decode + Tree decode ──

    @torch.inference_mode()
    def _build_and_decode_tree(
        self,
        cache_keys: torch.Tensor,  # [B, 3]
        B: int,
        K: int,
        returned_tokens: torch.Tensor,  # [B, K]
        temperatures: torch.Tensor,  # [B]
        draft_indices: torch.Tensor,  # [B]
        cache_hits: torch.Tensor,  # [B]
    ):
        """Run glue decode + tree decode + populate cache.

        This runs in the background while the target model verifies.
        Matches SSD's _build_tree_batch + _decode_tree flow.
        """
        if B == 0:
            return

        recovery_tokens = cache_keys[:, 2]
        draft_cur_lens = self.draft_seq_lens[draft_indices]  # [B]

        # ── Phase 1: Glue decode ──
        # Build glue input: [recovery_token, tok_0, ..., tok_{K-1}] per seq
        glue_flat = make_glue_decode_input_ids(
            returned_tokens.to(torch.int64), recovery_tokens
        ).to(torch.int32)  # [B*(K+1)]
        glue_2d = glue_flat.to(torch.int64).view(B, K + 1)
        num_glue_tokens = B * (K + 1)

        # Allocate KV cache for glue tokens
        glue_locs = self.token_to_kv_pool_allocator.alloc(num_glue_tokens)
        if glue_locs is None:
            logger.warning("Tree decode: glue KV alloc failed")
            return

        # Vectorized slot assignment for glue tokens
        self._assign_slots_to_pool(draft_indices, draft_cur_lens, K + 1, glue_locs)

        glue_seq_lens = draft_cur_lens + (K + 1)
        glue_extend_prefix_lens = draft_cur_lens.clone()
        glue_extend_seq_lens = torch.full(
            (B,), K + 1, dtype=torch.int64, device=self.device
        )

        # Run glue decode forward (EXTEND)
        fb = self._make_extend_forward_batch(
            input_ids=glue_flat,
            req_pool_indices=draft_indices,
            seq_lens=glue_seq_lens,
            out_cache_loc=glue_locs,
            extend_prefix_lens=glue_extend_prefix_lens,
            extend_seq_lens=glue_extend_seq_lens,
        )

        glue_logits_flat = self._run_forward(fb)  # [B*(K+1), V]
        glue_logits = glue_logits_flat.view(B, K + 1, -1)  # [B, K+1, V]

        # Fork: sample fan_out alternative tokens at each K+1 position.
        # Pass returned_tokens (glue_2d) to mask out chain tokens via -inf.
        forked_tokens = get_forked_recovery_tokens_from_logits(
            logits=glue_logits,
            fan_out_list=self.fan_out_list,
            temperatures=temperatures,
            cache_hits=cache_hits,
            fan_out_list_miss=self.fan_out_list_miss,
            returned_tokens=glue_2d,
        )  # [B, MQ_LEN]

        # Update draft seq lens after glue (vectorized)
        self.draft_seq_lens[draft_indices] = glue_seq_lens

        # ── Phase 2: Tree decode (K steps) with precomputed positions ──
        MQ_LEN = self.mq_len
        N_tree = B * MQ_LEN
        tree_input_flat = forked_tokens.reshape(-1)  # [N_tree]

        if N_tree == 0:
            return

        # Compute batch IDs and fan depth IDs for tree tokens
        tree_batch_ids = torch.arange(
            B, device=self.device
        ).repeat_interleave(MQ_LEN)  # [N_tree]
        all_hit = cache_hits.all().item()
        all_miss = (cache_hits == 0).all().item()
        if all_hit:
            tree_fan_idx = self._fan_idx_hit.repeat(B)
        elif all_miss:
            tree_fan_idx = self._fan_idx_miss.repeat(B)
        else:
            tree_fan_idx = torch.cat(
                [
                    self._fan_idx_hit if cache_hits[b].item() else self._fan_idx_miss
                    for b in range(B)
                ]
            )

        tree_draft_indices = draft_indices[tree_batch_ids]
        tree_base_lens = glue_seq_lens[tree_batch_ids]
        tree_temps = temperatures[tree_batch_ids]

        # Precompute positions and seq_lens for all K steps.
        # Each tree token gets a unique KV position to avoid slot conflicts.
        # Token (b, j) at step d: kv_pos = base + d*MQ_LEN + j
        fkp1_flat = self._arange_mq.repeat(B)  # [N_tree]: 0..MQ_LEN-1 per batch
        initial_positions = tree_base_lens + fkp1_flat  # [N_tree]

        # Precompute [K, N_tree] position/seq_lens tensors (matching SSD pattern)
        all_step_positions = (
            initial_positions.unsqueeze(0) + self._step_pos_offsets
        )  # [K, N_tree]
        all_step_seq_lens = all_step_positions + 1  # [K, N_tree]

        # Tree decode: K steps with precomputed values
        spec_tokens = torch.zeros(
            (N_tree, K), dtype=torch.int64, device=self.device
        )
        spec_logits = torch.zeros(
            (N_tree, K, self.vocab_size),
            dtype=torch.float32,
            device=self.device,
        )

        current_input_ids = tree_input_flat.to(torch.int32)

        for depth in range(K):
            step_locs = self.token_to_kv_pool_allocator.alloc(N_tree)
            if step_locs is None:
                logger.warning(f"Tree decode: KV alloc failed at depth {depth}")
                break

            # Use precomputed positions for this step (no recomputation)
            step_positions = all_step_positions[depth]  # [N_tree]
            step_seq_lens = all_step_seq_lens[depth]  # [N_tree]

            # Vectorized slot assignment with unique positions
            self._assign_slots_per_element(
                tree_draft_indices, step_positions, step_locs
            )

            fb = self._make_decode_forward_batch(
                input_ids=current_input_ids,
                req_pool_indices=tree_draft_indices,
                seq_lens=step_seq_lens,
                out_cache_loc=step_locs,
            )

            logits = self._run_forward(fb)  # [N_tree, V]
            spec_logits[:, depth, :] = logits

            next_tokens = self._sample_tokens(logits, tree_temps)
            spec_tokens[:, depth] = next_tokens
            current_input_ids = next_tokens.to(torch.int32)

        # ── Phase 3: Populate tree cache ──
        self._populate_tree_cache(
            tree_batch_ids,
            tree_fan_idx,
            tree_input_flat,
            spec_tokens,
            spec_logits,
            cache_keys,
        )

    def _populate_tree_cache(
        self,
        tree_batch_ids: torch.Tensor,  # [N]
        tree_depth_ids: torch.Tensor,  # [N]
        tree_recovery_tokens: torch.Tensor,  # [N] the forked input tokens
        spec_tokens: torch.Tensor,  # [N, K]
        spec_logits: torch.Tensor,  # [N, K, V]
        cache_keys_orig: torch.Tensor,  # [B, 3]
    ):
        """Populate tensor-backed tree cache from tree decode results."""
        N = tree_batch_ids.shape[0]
        if N == 0:
            return

        # Build cache keys [N, 3] = (target_req_pool_idx, depth_index, recovery_token)
        seq_ids = cache_keys_orig[tree_batch_ids, 0]
        keys = torch.stack(
            [seq_ids, tree_depth_ids, tree_recovery_tokens.to(torch.int64)], dim=1
        )

        self.tree_cache_keys = keys
        self.tree_cache_tokens = spec_tokens
        self.tree_cache_logits = spec_logits

    def _reset_tree_cache(self):
        """Reset tensor-backed tree cache (matching SSD _reset_tree_cache_tensors)."""
        self.tree_cache_keys = torch.zeros(
            (0, 3), dtype=torch.int64, device=self.device
        )
        self.tree_cache_tokens = None
        self.tree_cache_logits = None


def _cleanup_draft_runner(draft_gpu_id: int):
    """Clean up GPU and distributed resources in the draft runner process."""
    import gc

    from sglang.srt.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    logger.info(f"Draft runner GPU {draft_gpu_id}: cleaning up resources")
    try:
        destroy_model_parallel()
    except Exception as e:
        logger.debug(f"destroy_model_parallel: {e}")
    try:
        destroy_distributed_environment()
    except Exception as e:
        logger.debug(f"destroy_distributed_environment: {e}")

    gc.collect()
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.debug(f"CUDA cleanup: {e}")
    logger.info(f"Draft runner GPU {draft_gpu_id}: cleanup complete")


def run_async_draft_runner_process(
    server_args,
    draft_gpu_id: int,
    nccl_port: int,
    result_pipe,  # multiprocessing.Connection (for init status)
):
    """Entry point for the draft runner subprocess."""
    import signal

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

    # Install SIGTERM handler so the process can clean up before exiting.
    _shutdown_requested = [False]

    def _sigterm_handler(signum, frame):
        logger.info("Draft runner received SIGTERM, will exit after cleanup")
        _shutdown_requested[0] = True

    signal.signal(signal.SIGTERM, _sigterm_handler)

    runner = None
    try:
        torch.cuda.set_device(draft_gpu_id)

        # Load draft model config to get vocab size
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

        # Signal readiness to parent process (before NCCL init so parent can
        # proceed to create its own NCCL channel concurrently)
        result_pipe.send(
            {
                "status": "ready",
                "draft_gpu_id": draft_gpu_id,
                "vocab_size": vocab_size,
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
        )

        # Create the draft runner (loads model inside __init__)
        runner = AsyncDraftRunner(
            server_args=server_args,
            draft_model_path=draft_model_path,
            draft_gpu_id=draft_gpu_id,
            channel=channel,
            vocab_size=vocab_size,
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
    finally:
        # Always clean up GPU/distributed resources before exiting.
        del runner
        _cleanup_draft_runner(draft_gpu_id)
