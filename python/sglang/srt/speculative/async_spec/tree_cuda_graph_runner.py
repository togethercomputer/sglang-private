"""CUDA graph runner for async speculative tree decode steps.

Captures CUDA graphs for the draft model's DECODE forward pass at
various batch sizes (N = B * mq_len buckets). At replay time, populates
pre-allocated persistent buffers, updates attention metadata in-place,
and replays the captured graph.

This follows the same capture/replay pattern as SGLang's CudaGraphRunner
but is specialized for the tree decode hot path where:
- Forward mode is always DECODE
- spec_info is None (this IS the draft model)
- No encoder (encoder_lens=None)
- Batch size = N = B * mq_len where B varies per spec request
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


class TreeDecodeCudaGraphRunner:
    """Captures and replays CUDA graphs for draft model tree decode.

    Architecture:
    - Pre-allocates persistent GPU buffers for max batch size
    - Captures CUDA graphs at power-of-2 bucket sizes (N = B * mq_len)
    - precompute_plans() does a single GPU→CPU sync for all K steps
    - replay() populates buffers, updates attention metadata, replays graph
    - Returns sliced output for actual (non-padded) batch size
    """

    def __init__(
        self,
        model_runner,
        max_batch_size: int,
        mq_len: int,
        device: torch.device,
    ):
        self.model_runner = model_runner
        self.model = model_runner.model
        self.attn_backend = model_runner.attn_backend
        self.device = device
        self.mq_len = mq_len

        max_N = max_batch_size * mq_len
        self.max_N = max_N

        # Padding fill value from attention backend
        self.seq_len_fill_value = (
            self.attn_backend.get_cuda_graph_seq_len_fill_value()
        )

        # Bucket sizes: N = B * mq_len for B = 1, 2, 4, ..., max_batch_size
        self.capture_bs = self._compute_bucket_sizes(max_batch_size, mq_len)
        self.max_capture_N = max(self.capture_bs) if self.capture_bs else 0

        # Pre-allocate persistent GPU input buffers.
        # These are the SAME tensors used during capture and replay — the CUDA
        # graph captures kernel launches referencing these pointers, and we
        # update the data in-place before each replay.
        with torch.device(device):
            self.input_ids_buf = torch.zeros(max_N, dtype=torch.int32)
            self.positions_buf = torch.zeros(max_N, dtype=torch.int64)
            self.out_cache_loc_buf = torch.zeros(max_N, dtype=torch.int32)
            self.req_pool_indices_buf = torch.zeros(max_N, dtype=torch.int32)
            self.seq_lens_buf = torch.full(
                (max_N,), self.seq_len_fill_value, dtype=torch.int32
            )

        # CPU buffer for seq_lens (avoids GPU→CPU sync during replay)
        self.seq_lens_cpu_buf = torch.full(
            (max_N,), self.seq_len_fill_value, dtype=torch.int32
        )

        # Graph + output storage keyed by bucket size N
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.output_buffers: Dict[int, object] = {}

        # Precomputed per-step plan data (set by precompute_plans)
        self._plans_ready = False

        # NOTE: init_cuda_graph_state must be called by the caller before
        # constructing this runner to avoid reinitializing shared state.

        # Capture infrastructure
        self.stream = torch.cuda.Stream(device=device)
        self.graph_pool = None

    @staticmethod
    def _compute_bucket_sizes(max_batch_size: int, mq_len: int) -> List[int]:
        """Bucket sizes as B * mq_len for B = 1, 2, 4, ..., max_batch_size."""
        buckets = set()
        b = 1
        while b <= max_batch_size:
            buckets.add(b * mq_len)
            b *= 2
        if max_batch_size > 0:
            buckets.add(max_batch_size * mq_len)
        return sorted(buckets)

    def can_run(self, N: int) -> bool:
        """Check if a captured graph exists for batch size >= N."""
        return bool(self.graphs) and N <= self.max_capture_N

    # ── Capture ────────────────────────────────────────────────────────

    def capture(self):
        """Capture CUDA graphs for all bucket sizes."""
        from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode

        if not self.capture_bs:
            logger.warning("TreeDecodeCudaGraphRunner: no bucket sizes to capture")
            return

        self.graph_pool = torch.cuda.graph_pool_handle()
        forward = self.model.forward

        with model_capture_mode():
            for N in reversed(self.capture_bs):
                try:
                    graph, output = self._capture_one_batch_size(N, forward)
                    self.graphs[N] = graph
                    self.output_buffers[N] = output
                    self.graph_pool = graph.pool()
                except RuntimeError as e:
                    logger.warning(
                        f"TreeDecodeCudaGraphRunner: capture failed for N={N}: {e}"
                    )

        if self.graphs:
            logger.info(
                f"TreeDecodeCudaGraphRunner: captured {len(self.graphs)} graphs, "
                f"buckets={sorted(self.graphs.keys())}"
            )
        else:
            logger.warning("TreeDecodeCudaGraphRunner: no graphs captured")

    def _capture_one_batch_size(self, N: int, forward):
        """Capture a CUDA graph for batch size N (a bucket size)."""
        # Slice persistent buffers to this bucket size
        input_ids = self.input_ids_buf[:N]
        positions = self.positions_buf[:N]
        out_cache_loc = self.out_cache_loc_buf[:N]
        req_pool_indices = self.req_pool_indices_buf[:N]
        seq_lens = self.seq_lens_buf[:N]
        seq_lens_cpu = self.seq_lens_cpu_buf[:N]

        # ForwardBatch references the persistent buffer slices
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=N,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens.sum().item()),
            positions=positions,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.attn_backend,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        # Initialize attention metadata for capture (allocates persistent
        # internal tensors keyed by N)
        self.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=N,
            num_tokens=N,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )

        def run_once():
            return forward(input_ids, positions, forward_batch)

        # Warmup (2 runs to flush stale state)
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

    # ── Precompute + Replay ────────────────────────────────────────────

    def precompute_plans(
        self,
        K: int,
        N: int,
        step_ctx_lens_expanded: torch.Tensor,  # [K, N] int32 GPU
        step_seq_lens_sums: torch.Tensor,  # [K] GPU
        req_pool_indices_i32: torch.Tensor,  # [N] int32 GPU
        step_slot_maps_i32: torch.Tensor,  # [K, N] int32 GPU
        step_rope_positions_i64: torch.Tensor,  # [K, N] int64 GPU
    ):
        """Pre-compute CPU-side plan args for all K steps.

        Performs ONE batched GPU→CPU transfer instead of K separate syncs.
        """
        self._K = K
        self._N = N
        # GPU tensors (referenced during replay)
        self._step_ctx_lens_gpu = step_ctx_lens_expanded
        self._req_pool_indices_i32 = req_pool_indices_i32
        self._step_slot_maps_i32 = step_slot_maps_i32
        self._step_rope_positions_i64 = step_rope_positions_i64
        # Batched GPU→CPU transfer (1 sync for all K steps)
        self._step_ctx_lens_cpu = step_ctx_lens_expanded.cpu()  # [K, N]
        self._step_sums_cpu = step_seq_lens_sums.cpu()  # [K]
        self._plans_ready = True

    def replay(self, step: int, input_ids: torch.Tensor) -> torch.Tensor:
        """Replay graph for one tree decode step.

        Args:
            step: Tree decode step index (0..K-1)
            input_ids: [N] token ids for this step

        Returns:
            next_token_logits: [N, V] tensor (valid until next replay call)
        """
        assert self._plans_ready, "Call precompute_plans() before replay()"

        N = self._N
        seq_lens = self._step_ctx_lens_gpu[step]
        out_cache_loc = self._step_slot_maps_i32[step]
        req_pool_indices = self._req_pool_indices_i32
        positions = self._step_rope_positions_i64[step]
        seq_lens_cpu = self._step_ctx_lens_cpu[step]
        seq_lens_sum = int(self._step_sums_cpu[step])

        # Find smallest bucket >= N
        idx = bisect.bisect_left(self.capture_bs, N)
        bucket_N = self.capture_bs[idx]

        # Populate persistent buffers with this step's data
        self.input_ids_buf[:N].copy_(input_ids.to(torch.int32))
        self.positions_buf[:N].copy_(positions)
        self.out_cache_loc_buf[:N].copy_(out_cache_loc)
        self.req_pool_indices_buf[:N].copy_(req_pool_indices)
        self.seq_lens_buf[:N].copy_(seq_lens)
        self.seq_lens_cpu_buf[:N].copy_(seq_lens_cpu)

        # Pad remaining entries if bucket > actual batch
        if N < bucket_N:
            self.seq_lens_buf[N:bucket_N].fill_(self.seq_len_fill_value)
            self.seq_lens_cpu_buf[N:bucket_N].fill_(self.seq_len_fill_value)
            self.req_pool_indices_buf[N:bucket_N].fill_(0)

        # Update attention metadata in-place for replay
        padded_sum = seq_lens_sum + (bucket_N - N) * self.seq_len_fill_value
        self.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bucket_N,
            req_pool_indices=self.req_pool_indices_buf[:bucket_N],
            seq_lens=self.seq_lens_buf[:bucket_N],
            seq_lens_sum=padded_sum,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=self.seq_lens_cpu_buf[:bucket_N],
        )

        # Replay the captured graph
        self.graphs[bucket_N].replay()

        # Return logits sliced to actual (non-padded) batch size.
        # The caller MUST copy/consume this tensor before the next replay().
        output = self.output_buffers[bucket_N]
        return output.next_token_logits[:N]
