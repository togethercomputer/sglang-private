"""Scheduler-side orchestrator for async speculative decoding.

This class lives in the scheduler process and orchestrates the
speculate -> verify -> postprocess cycle. It wraps the target TpModelWorker
and communicates with a remote AsyncDraftRunner running on a dedicated GPU
via NCCL.

Reuses SGLang's existing EagleVerifyInput/verify infrastructure for the
verification step — async spec only changes how speculation is derived
(dedicated GPU), not how tokens are verified (standard chain verify).

The target manages draft KV-cache block tables via DraftBlockAllocator
and sends them with every request so the draft can use them for attention.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional

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


class DraftBlockAllocator:
    """Simple free-list block allocator for draft KV cache (no prefix caching).

    Mirrors SSD's BlockManager. The target manages these blocks and sends
    block tables to the draft runner with every request.
    """

    def __init__(self, num_blocks: int, page_size: int):
        self.num_blocks = num_blocks
        self.page_size = page_size
        self.free_blocks: deque = deque(range(num_blocks))

    def can_allocate(self, num_blocks_needed: int) -> bool:
        return len(self.free_blocks) >= num_blocks_needed

    def allocate(self, num_blocks_needed: int) -> List[int]:
        if not self.can_allocate(num_blocks_needed):
            raise RuntimeError(
                f"DraftBlockAllocator: cannot allocate {num_blocks_needed} blocks, "
                f"only {len(self.free_blocks)} free"
            )
        return [self.free_blocks.popleft() for _ in range(num_blocks_needed)]

    def free(self, block_ids: List[int]):
        for bid in block_ids:
            self.free_blocks.append(bid)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)


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
        num_draft_kv_pages: int = 0,
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
        self.mq_len = sum(server_args.speculative_async_fan_out_list)  # Tree width

        # NCCL channel (set by scheduler after spawning draft process)
        self.nccl_channel: Optional[AsyncSpecNcclChannel] = None

        # Model info
        self.vocab_size = target_worker.model_runner.model_config.vocab_size
        self.draft_dtype = torch.float32

        # Allocator reference (shared with target worker)
        _, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()

        # Draft block allocator and per-request tracking
        self.draft_block_allocator: Optional[DraftBlockAllocator] = None
        self.draft_block_tables: Dict[int, List[int]] = {}
        self.draft_num_tokens: Dict[int, int] = {}

        if num_draft_kv_pages > 0:
            self.draft_block_allocator = DraftBlockAllocator(
                num_blocks=num_draft_kv_pages, page_size=self.page_size
            )
            logger.info(
                f"DraftBlockAllocator initialized: {num_draft_kv_pages} blocks, "
                f"page_size={self.page_size}"
            )

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
        """Prefill: run target prefill, allocate draft blocks, notify draft."""
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
                self._send_prefill_to_draft(batch, model_worker_batch)
            except Exception as e:
                logger.warning(f"Failed to send prefill to draft runner: {e}")

        return target_result

    def _send_prefill_to_draft(self, batch: ScheduleBatch, model_worker_batch):
        """Allocate draft blocks and send prefill data to draft runner."""
        max_blocks = self.nccl_channel.max_blocks
        input_ids = model_worker_batch.input_ids
        num_reqs = len(batch.reqs)

        req_pool_indices_list = []
        seq_lens_list = []
        positions_parts = []
        block_tables_list = []

        for req in batch.reqs:
            rpi = req.req_pool_idx
            seq_len = len(req.origin_input_ids) + len(req.output_ids)
            req_pool_indices_list.append(rpi)
            seq_lens_list.append(seq_len)

            # Build positions for this request
            positions_parts.append(
                torch.arange(seq_len, device=self.device, dtype=torch.int64)
            )

            # Allocate draft blocks for this request
            if self.draft_block_allocator is not None:
                blocks_needed = (seq_len + self.page_size - 1) // self.page_size
                block_ids = self.draft_block_allocator.allocate(blocks_needed)
                self.draft_block_tables[rpi] = block_ids
                self.draft_num_tokens[rpi] = seq_len

                # Pad block table to max_blocks
                padded = block_ids + [0] * (max_blocks - len(block_ids))
                block_tables_list.append(padded[:max_blocks])
            else:
                block_tables_list.append([0] * max_blocks)

        req_pool_indices = torch.tensor(
            req_pool_indices_list, dtype=torch.int64, device=self.device
        )
        seq_lens = torch.tensor(
            seq_lens_list, dtype=torch.int64, device=self.device
        )
        positions = torch.cat(positions_parts)
        block_tables = torch.tensor(
            block_tables_list, dtype=torch.int64, device=self.device
        )

        self.nccl_channel.send_prefill(
            input_ids, req_pool_indices, seq_lens, positions, block_tables
        )

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
        # Vectorized: avoid B×(K+1) Python loop with .item() GPU→CPU syncs
        seq_lens_cpu = batch.seq_lens.cpu().to(torch.int32)
        offsets = torch.arange(
            num_verify_tokens, dtype=torch.long, device=self.device
        )  # [K+1]
        positions = (
            batch.seq_lens.unsqueeze(1).to(torch.long) + offsets.unsqueeze(0)
        ).reshape(-1)  # [B*(K+1)]

        # Build retrive_index: [0, 1, ..., K] per request (per-request indices)
        retrive_index = (
            torch.arange(num_verify_tokens, device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )

        # Build retrive_next_token: chain links [1, 2, ..., K, -1] per request
        # Vectorized: [1, 2, ..., K, -1] broadcast to all B rows
        chain_links = torch.arange(
            1, num_verify_tokens + 1, device=self.device, dtype=torch.long
        )
        chain_links[-1] = -1  # Last token has no next
        retrive_next_token = chain_links.unsqueeze(0).expand(B, -1).contiguous()

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

        # STEP 4: UPDATE recovery tokens and draft block tracking for next round
        next_token_ids = verify_output.verified_id
        if next_token_ids is not None and len(next_token_ids) > 0:
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

                    # Update draft_num_tokens based on accepted tokens
                    rpi = req.req_pool_idx
                    if rpi in self.draft_num_tokens:
                        self.draft_num_tokens[rpi] += al + 1

                    idx += al + 1
                else:
                    req.last_spec_step_accepted_len = 0
                    # Free draft blocks for finished requests
                    self._free_draft_blocks_for_req(req.req_pool_idx)

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
        K = self.spec_k
        max_blocks = self.nccl_channel.max_blocks

        # Build cache_keys [B, 3], temperatures [B], num_tokens [B], block_tables [B, max_blocks]
        # Collect as CPU lists first, convert to GPU tensors once (avoids per-request torch.tensor calls)
        cache_keys_list = []
        temps_list = []
        num_tokens_list = []
        block_tables_list = []

        for i, req in enumerate(reqs):
            req_pool_idx = req.req_pool_idx if req.req_pool_idx is not None else 0
            accepted_len = req.last_spec_step_accepted_len
            recovery_token = (
                req.recovery_token_id if req.recovery_token_id is not None else 0
            )
            cache_keys_list.append([req_pool_idx, accepted_len, recovery_token])
            temps_list.append(req.sampling_params.temperature)

            # Populate num_tokens and block_tables from tracking
            rpi = req_pool_idx
            bt_row = [0] * max_blocks
            nt = 0
            if rpi in self.draft_num_tokens:
                nt = self.draft_num_tokens[rpi]

                # Ensure enough blocks for current tokens + tree decode
                # Tree decode needs: K+1 (glue) + K*MQ_LEN (tree steps) extra positions
                if self.draft_block_allocator is not None and rpi in self.draft_block_tables:
                    current_blocks = self.draft_block_tables[rpi]
                    tree_lookahead = K + 1 + K * self.mq_len
                    needed_blocks = (nt + tree_lookahead + self.page_size - 1) // self.page_size
                    additional = needed_blocks - len(current_blocks)
                    if additional > 0 and self.draft_block_allocator.can_allocate(additional):
                        new_blocks = self.draft_block_allocator.allocate(additional)
                        current_blocks.extend(new_blocks)

                    bt = current_blocks[:max_blocks]
                    bt_row[:len(bt)] = bt

            num_tokens_list.append(nt)
            block_tables_list.append(bt_row)

        # Single CPU→GPU transfer per tensor (instead of B per-element writes)
        cache_keys = torch.tensor(cache_keys_list, dtype=torch.int64, device=self.device)
        temperatures = torch.tensor(temps_list, dtype=torch.float32, device=self.device)
        num_tokens = torch.tensor(num_tokens_list, dtype=torch.int64, device=self.device)
        block_tables = torch.tensor(block_tables_list, dtype=torch.int64, device=self.device)

        # Send request via NCCL
        self.nccl_channel.send_spec_request(
            batch_size=B,
            lookahead=self.spec_k,
            fan_out=self.fan_out,
            vocab_size=self.vocab_size,
            cache_keys=cache_keys,
            temperatures=temperatures,
            num_tokens=num_tokens,
            draft_block_tables=block_tables,
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

    def _free_draft_blocks_for_req(self, req_pool_idx: int):
        """Free draft blocks for a finished request."""
        if self.draft_block_allocator is not None and req_pool_idx in self.draft_block_tables:
            blocks = self.draft_block_tables.pop(req_pool_idx)
            self.draft_block_allocator.free(blocks)
        self.draft_num_tokens.pop(req_pool_idx, None)

    def clear_cache_pool(self):
        """Clean up all draft KV cache allocations."""
        if self.draft_block_allocator is not None:
            for rpi in list(self.draft_block_tables.keys()):
                self._free_draft_blocks_for_req(rpi)

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
