"""CUDA graph runner for async speculative glue decode.

Converts glue decode from eager EXTEND to CUDA-graph-captured multi-query
forward, matching SSD's approach of using CUDA graphs for K+1 tokens per
request.

The glue decode processes B requests × (K+1) tokens each:
  [recovery_token, spec_tok_0, ..., spec_tok_{K-1}]
with causal attention within each request's tokens.

This uses ForwardMode.DRAFT_EXTEND_V2 for capture/replay, which routes through
the FlashInfer prefill wrapper with CUDA graph support.
"""

from __future__ import annotations

import bisect
import logging
from typing import Dict, List

import torch

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)

logger = logging.getLogger(__name__)


class GlueDecodeCudaGraphRunner:
    """Captures and replays CUDA graphs for draft model glue decode.

    The glue decode always processes B × (K+1) tokens (fixed per-request
    token count), making it ideal for CUDA graph capture at power-of-2
    batch size buckets.
    """

    def __init__(
        self,
        model_runner,
        spec_k: int,
        max_batch_size: int,
        device: torch.device,
    ):
        self.model_runner = model_runner
        self.model = model_runner.model
        self.attn_backend = model_runner.attn_backend
        self.device = device
        self.spec_k = spec_k
        self.tokens_per_seq = spec_k + 1  # K+1 tokens per request

        self.max_bs = max_batch_size
        max_tokens = max_batch_size * self.tokens_per_seq

        # Padding fill value from attention backend
        self.seq_len_fill_value = (
            self.attn_backend.get_cuda_graph_seq_len_fill_value()
        )

        # Bucket sizes by B (number of requests)
        self.capture_bs = self._compute_bucket_sizes(max_batch_size)
        self.max_capture_bs = max(self.capture_bs) if self.capture_bs else 0

        # Pre-allocate persistent GPU buffers
        with torch.device(device):
            self.input_ids_buf = torch.zeros(max_tokens, dtype=torch.int32)
            self.positions_buf = torch.zeros(max_tokens, dtype=torch.int64)
            self.out_cache_loc_buf = torch.zeros(max_tokens, dtype=torch.int32)
            self.req_pool_indices_buf = torch.zeros(
                max_batch_size, dtype=torch.int32
            )
            self.seq_lens_buf = torch.full(
                (max_batch_size,), self.seq_len_fill_value, dtype=torch.int32
            )
            # EXTEND-specific fields
            self.extend_seq_lens_buf = torch.full(
                (max_batch_size,), self.tokens_per_seq, dtype=torch.int32
            )
            self.extend_prefix_lens_buf = torch.zeros(
                max_batch_size, dtype=torch.int32
            )
            self.extend_start_loc_buf = torch.zeros(
                max_batch_size, dtype=torch.int32
            )

        self.seq_lens_cpu_buf = torch.full(
            (max_batch_size,), self.seq_len_fill_value, dtype=torch.int32
        )

        # Graph storage
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.output_buffers: Dict[int, object] = {}

        # NOTE: init_cuda_graph_state must be called by the caller before
        # constructing this runner to avoid reinitializing shared state.

        self.stream = torch.cuda.Stream(device=device)
        self.graph_pool = None

    @staticmethod
    def _compute_bucket_sizes(max_batch_size: int) -> List[int]:
        """Bucket sizes: B = 1, 2, 4, ..., max_batch_size."""
        buckets = set()
        b = 1
        while b <= max_batch_size:
            buckets.add(b)
            b *= 2
        if max_batch_size > 0:
            buckets.add(max_batch_size)
        return sorted(buckets)

    def can_run(self, B: int) -> bool:
        return bool(self.graphs) and B <= self.max_capture_bs

    # ── Capture ────────────────────────────────────────────────────────

    def capture(self):
        """Capture CUDA graphs for all bucket sizes."""
        from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode

        if not self.capture_bs:
            return

        self.graph_pool = torch.cuda.graph_pool_handle()
        forward = self.model.forward

        with model_capture_mode():
            for B in reversed(self.capture_bs):
                try:
                    graph, output = self._capture_one_batch_size(B, forward)
                    self.graphs[B] = graph
                    self.output_buffers[B] = output
                    self.graph_pool = graph.pool()
                except RuntimeError as e:
                    logger.warning(
                        f"GlueDecodeCudaGraphRunner: capture failed for B={B}: {e}"
                    )

        if self.graphs:
            logger.info(
                f"GlueDecodeCudaGraphRunner: captured {len(self.graphs)} graphs, "
                f"buckets={sorted(self.graphs.keys())}"
            )
        else:
            logger.warning("GlueDecodeCudaGraphRunner: no graphs captured")

    def _capture_one_batch_size(self, B: int, forward):
        """Capture CUDA graph for batch size B (B requests × (K+1) tokens)."""
        num_tokens = B * self.tokens_per_seq

        # Slice persistent buffers
        input_ids = self.input_ids_buf[:num_tokens]
        positions = self.positions_buf[:num_tokens]
        out_cache_loc = self.out_cache_loc_buf[:num_tokens]
        req_pool_indices = self.req_pool_indices_buf[:B]
        seq_lens = self.seq_lens_buf[:B]
        seq_lens_cpu = self.seq_lens_cpu_buf[:B]
        extend_seq_lens = self.extend_seq_lens_buf[:B]
        extend_prefix_lens = self.extend_prefix_lens_buf[:B]
        extend_start_loc = self.extend_start_loc_buf[:B]

        # Set up extend_start_loc: [0, K+1, 2*(K+1), ...]
        extend_start_loc[:] = (
            torch.arange(B, device=self.device, dtype=torch.int32) * self.tokens_per_seq
        )

        # ForwardBatch for capture — use DRAFT_EXTEND which has prefill
        # CUDA graph capture/replay support in the FlashInfer backend
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=B,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens.sum().item()),
            positions=positions,
            extend_num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.attn_backend,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        # Initialize attention metadata for capture
        self.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=B,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            spec_info=None,
        )

        def run_once():
            return forward(input_ids, positions, forward_batch)

        # Warmup
        torch.cuda.synchronize(self.device)
        for _ in range(2):
            run_once()
        torch.cuda.synchronize(self.device)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool, stream=self.stream):
            output = run_once()

        torch.cuda.synchronize(self.device)
        return graph, output

    # ── Replay ─────────────────────────────────────────────────────────

    def replay(
        self,
        B: int,
        input_ids: torch.Tensor,  # [B*(K+1)] int32
        positions: torch.Tensor,  # [B*(K+1)] int64
        out_cache_loc: torch.Tensor,  # [B*(K+1)] int32
        req_pool_indices: torch.Tensor,  # [B] int32
        seq_lens: torch.Tensor,  # [B] int32 (total seq len per request)
        seq_lens_sum: int,
        seq_lens_cpu: torch.Tensor,  # [B] int32 CPU
        extend_prefix_lens: torch.Tensor,  # [B] int32
    ):
        """Replay glue decode CUDA graph.

        Returns the model output (with hidden_states for all positions).
        """
        # Find bucket
        idx = bisect.bisect_left(self.capture_bs, B)
        bucket_B = self.capture_bs[idx]
        num_tokens = B * self.tokens_per_seq
        bucket_tokens = bucket_B * self.tokens_per_seq

        # Populate persistent buffers
        self.input_ids_buf[:num_tokens].copy_(input_ids)
        self.positions_buf[:num_tokens].copy_(positions)
        self.out_cache_loc_buf[:num_tokens].copy_(out_cache_loc)
        self.req_pool_indices_buf[:B].copy_(req_pool_indices)
        self.seq_lens_buf[:B].copy_(seq_lens)
        self.seq_lens_cpu_buf[:B].copy_(seq_lens_cpu)
        self.extend_prefix_lens_buf[:B].copy_(extend_prefix_lens)

        # Pad if needed
        if B < bucket_B:
            self.seq_lens_buf[B:bucket_B].fill_(self.seq_len_fill_value)
            self.seq_lens_cpu_buf[B:bucket_B].fill_(self.seq_len_fill_value)
            self.req_pool_indices_buf[B:bucket_B].fill_(0)
            self.extend_prefix_lens_buf[B:bucket_B].fill_(0)

        # Update extend_start_loc for actual B
        self.extend_start_loc_buf[:bucket_B] = (
            torch.arange(bucket_B, device=self.device, dtype=torch.int32)
            * self.tokens_per_seq
        )

        # Update attention metadata for replay
        padded_sum = seq_lens_sum + (bucket_B - B) * self.seq_len_fill_value
        self.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bucket_B,
            req_pool_indices=self.req_pool_indices_buf[:bucket_B],
            seq_lens=self.seq_lens_buf[:bucket_B],
            seq_lens_sum=padded_sum,
            encoder_lens=None,
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            spec_info=None,
            seq_lens_cpu=self.seq_lens_cpu_buf[:bucket_B],
        )

        # Replay
        self.graphs[bucket_B].replay()

        # Return output sliced to actual tokens
        output = self.output_buffers[bucket_B]
        return output
