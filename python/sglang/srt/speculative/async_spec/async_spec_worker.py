"""Scheduler-side orchestrator for async speculative decoding.

This class lives in the scheduler process and orchestrates the
speculate -> verify -> postprocess cycle. It wraps the target TpModelWorker
and communicates with a remote AsyncDraftRunner running on a dedicated GPU
via NCCL.

Reuses SGLang's existing EagleVerifyInput/verify infrastructure for the
verification step — async spec only changes how speculation is derived
(dedicated GPU), not how tokens are verified (standard chain verify).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.eagle_info import EagleVerifyInput

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.async_spec.nccl_comm import AsyncSpecNcclChannel

logger = logging.getLogger(__name__)


class AsyncSpecWorker:
    """Scheduler-side orchestrator for async speculation.

    Wraps the target TpModelWorker and communicates with a remote
    AsyncDraftRunner running on a dedicated GPU via NCCL.

    The verify step reuses the existing EagleVerifyInput infrastructure
    since async spec uses topk=1 (a chain, not a tree), and the verification
    algorithm is identical to standard speculative decoding verification.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.device = target_worker.device

        # Async spec config
        self.spec_k = server_args.speculative_num_steps
        self.fan_out = server_args.speculative_async_fan_out
        self.jit_speculate = server_args.speculative_async_jit_speculate
        self.sampler_x = server_args.speculative_async_sampler_x
        self.draft_temperature = server_args.speculative_async_draft_temperature
        self.topk = 1  # Async spec uses chain (topk=1), not tree
        self.num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size

        # NCCL channel (set by scheduler after spawning draft process)
        self.nccl_channel: Optional[AsyncSpecNcclChannel] = None

        # Model info
        self.vocab_size = target_worker.model_runner.model_config.vocab_size
        self.draft_dtype = torch.float32

        # Allocator reference (shared with target worker)
        _, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def model_runner(self):
        return self._target_worker.model_runner

    @property
    def model_config(self):
        return self._target_worker.model_runner.model_config

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Main entry point called by Scheduler.run_batch().

        Implements the full speculate -> verify cycle for async spec.
        """
        if batch.forward_mode.is_extend() or batch.forward_mode == ForwardMode.EXTEND:
            return self._handle_prefill(batch)
        else:
            return self._handle_decode(batch)

    def _handle_prefill(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Prefill: run target prefill, set recovery tokens, notify draft."""
        model_worker_batch = batch.get_model_worker_batch()
        target_result = self._target_worker.forward_batch_generation(model_worker_batch)

        next_token_ids = target_result.next_token_ids

        # Set recovery tokens on each request for first decode
        if next_token_ids is not None:
            next_token_ids_cpu = next_token_ids.cpu()
            for i, req in enumerate(batch.reqs):
                if i < len(next_token_ids_cpu):
                    req.recovery_token_id = next_token_ids_cpu[i].item()
                    req.last_spec_step_accepted_len = -1

        # Notify draft runner about prefill via NCCL
        if self.nccl_channel is not None:
            try:
                input_ids = model_worker_batch.input_ids
                req_pool_indices = torch.tensor(
                    [req.req_pool_idx for req in batch.reqs],
                    dtype=torch.int64,
                    device=self.device,
                )
                seq_lens = torch.tensor(
                    [
                        len(req.origin_input_ids) + len(req.output_ids)
                        for req in batch.reqs
                    ],
                    dtype=torch.int64,
                    device=self.device,
                )
                self.nccl_channel.send_prefill(
                    input_ids, req_pool_indices, seq_lens
                )
            except Exception as e:
                logger.warning(f"Failed to send prefill to draft runner: {e}")

        return target_result

    def _handle_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Decode with async speculation using the standard verify pipeline.

        1. Get speculations from draft runner
        2. Build EagleVerifyInput (chain structure with topk=1)
        3. Run verify() using the existing eagle verify infrastructure
        """
        reqs = batch.reqs
        B = len(reqs)

        if B == 0:
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=torch.empty(0, dtype=torch.int64, device=self.device),
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )

        # STEP 1: SPECULATE — get draft predictions from draft runner
        speculations = self._speculate(reqs)
        # STEP 2: BUILD VERIFY INPUT — construct EagleVerifyInput for the chain
        # For topk=1 (chain), the tree structures are trivial:
        # - draft_token: [recovery, tok0, tok1, ..., tokK-1] flattened across batch
        # - retrive_index: linear indices [0, 1, 2, ..., K] per request
        # - retrive_next_token: [1, 2, ..., K, -1] per request (chain)
        # - retrive_next_sibling: all -1 (no siblings in chain)
        # - positions: [seq_len, seq_len+1, ..., seq_len+K] per request
        K = self.spec_k
        num_verify_tokens = self.num_draft_tokens  # K + 1

        # Flatten speculations [B, K+1] into draft_token [B*(K+1)]
        all_draft_tokens = speculations.to(torch.long).flatten()

        # Build positions: [seq_len+0, seq_len+1, ..., seq_len+K] per request
        seq_lens_cpu = batch.seq_lens.cpu().to(torch.int32)
        positions = torch.empty(
            B * num_verify_tokens, dtype=torch.long, device=self.device
        )
        for b in range(B):
            sl = batch.seq_lens[b].item()
            for k in range(num_verify_tokens):
                positions[b * num_verify_tokens + k] = sl + k

        # Build retrive_index: [0, 1, ..., K] per request (per-request indices)
        retrive_index = (
            torch.arange(num_verify_tokens, device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )

        # Build retrive_next_token: chain links [1, 2, ..., K, -1] per request
        # These are per-request indices (0-based within each request's tokens)
        retrive_next_token = torch.full(
            (B, num_verify_tokens), -1, device=self.device, dtype=torch.long
        )
        for k in range(K):
            retrive_next_token[:, k] = k + 1

        # Build retrive_next_sibling: all -1 (no branches in chain)
        retrive_next_sibling = torch.full(
            (B, num_verify_tokens), -1, device=self.device, dtype=torch.long
        )

        # Build tree_mask: for chain, all True (causal mask handled by attention backend)
        seq_lens_sum = batch.seq_lens.sum().item()
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * B,
            ),
            True,
            device=self.device,
        )

        spec_info = EagleVerifyInput(
            draft_token=all_draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=K,
            topk=self.topk,
            draft_token_num=num_verify_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
        )

        # STEP 3: VERIFY — use standard eagle verify pipeline
        logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
            self._verify(batch, spec_info)
        )

        # STEP 4: UPDATE recovery tokens for next round
        next_token_ids = verify_output.verified_id
        if next_token_ids is not None and len(next_token_ids) > 0:
            # The last token in verified_id per request is the new recovery token
            accept_lens = verify_output.accept_length_per_req_cpu
            idx = 0
            for i, req in enumerate(reqs):
                if not req.finished():
                    al = accept_lens[i] if i < len(accept_lens) else 0
                    # Recovery token is the last verified token for this request
                    recovery_idx = idx + al
                    if recovery_idx < len(next_token_ids):
                        req.recovery_token_id = next_token_ids[recovery_idx].item()
                    req.last_spec_step_accepted_len = al
                    idx += al + 1
                else:
                    req.last_spec_step_accepted_len = 0

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _speculate(self, reqs: List[Req]) -> torch.Tensor:
        """Send spec request to draft via NCCL, receive speculations."""
        B = len(reqs)

        # Build cache_keys [B, 3] and temperatures [B] on GPU
        cache_keys = torch.zeros(B, 3, dtype=torch.int64, device=self.device)
        temperatures = torch.zeros(B, dtype=torch.float32, device=self.device)
        for i, req in enumerate(reqs):
            req_pool_idx = req.req_pool_idx if req.req_pool_idx is not None else 0
            accepted_len = req.last_spec_step_accepted_len
            recovery_token = (
                req.recovery_token_id if req.recovery_token_id is not None else 0
            )
            cache_keys[i, 0] = req_pool_idx
            cache_keys[i, 1] = accepted_len
            cache_keys[i, 2] = recovery_token
            temperatures[i] = req.sampling_params.temperature

        # Compute current sequence lengths for the draft runner
        seq_lens = torch.tensor(
            [
                len(req.origin_input_ids) + len(req.output_ids)
                for req in reqs
            ],
            dtype=torch.int64,
            device=self.device,
        )

        # Send request via NCCL
        self.nccl_channel.send_spec_request(
            batch_size=B,
            lookahead=self.spec_k,
            fan_out=self.fan_out,
            vocab_size=self.vocab_size,
            cache_keys=cache_keys,
            temperatures=temperatures,
            seq_lens=seq_lens,
        )

        # Receive speculations [B, K+1]
        speculations = self.nccl_channel.recv_speculations(B, self.spec_k)
        return speculations

    def _verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        """Run verification using existing eagle verify infrastructure.

        This is essentially the same as EAGLEWorker.verify(), adapted for
        async spec (no hidden states, no draft extend after decode).
        """
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_req = self.spec_k + 1
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )

        # Forward with target model (is_verify=True skips sampling)
        batch_result = self._target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output = batch_result.logits_output
        can_run_cuda_graph = batch_result.can_run_cuda_graph

        # Set hidden_states to None (async spec doesn't need them for draft)
        spec_info.hidden_states = logits_output.hidden_states

        # Run the actual verification (acceptance/rejection)
        verify_output = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
        )

        # Reset forward mode for next iteration
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        # We don't need draft_input for next iteration (draft is on separate GPU)
        batch.spec_info = None

        return logits_output, verify_output, model_worker_batch, can_run_cuda_graph

    def clear_cache_pool(self):
        """Clean up draft KV cache allocations."""
        pass

    def send_exit(self):
        """Send exit command to draft runner."""
        if self.nccl_channel is not None:
            try:
                self.nccl_channel.send_exit()
            except Exception as e:
                logger.warning(f"Failed to send exit to draft runner: {e}")

    # Delegate properties to target worker
    def get_worker_info(self):
        return self._target_worker.get_worker_info()

    def get_pad_input_ids_func(self):
        return self._target_worker.get_pad_input_ids_func()

    def get_memory_pool(self):
        return self._target_worker.get_memory_pool()

    @property
    def sliding_window_size(self):
        return self._target_worker.sliding_window_size

    @property
    def is_hybrid_swa(self):
        return self._target_worker.is_hybrid_swa
