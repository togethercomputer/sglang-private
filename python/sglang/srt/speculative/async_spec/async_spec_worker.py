import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

from ssd.engine.helpers.runner_helpers import (
    prepare_prefill_metadata,
    send_prefill_request,
    send_speculation_request,
    receive_speculation_response,
)

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.spec_worker import SpecWorker
from sglang.srt.speculative.spec_utils import _ts, _decode_ids, _decode_id_list
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import LogitsProcessorOutput
from sglang.srt.utils import empty_context, set_random_seed
from sglang.srt.distributed import get_tp_group

logger = logging.getLogger(__name__)
NCCL_LOG = os.environ.get("SSD_NCCL_LOG", "0") == "1"
_SGLANG_PROF = os.environ.get('SSD_PROFILE', '0') == '1'
import time as _time



class ModelConfigStub:
    def __init__(self, context_len: int, vocab_size: int, dtype: torch.dtype):
        self.is_encoder_decoder = False
        self.context_len = context_len
        self.attention_arch = AttentionArch.MHA
        self.is_local_attention_model = False
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.hf_config = None


class ModelRunnerStub:

    def __init__(
        self,
        server_args: ServerArgs,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.device = torch.device(server_args.device)
        self.max_total_num_tokens = target_worker.max_total_num_tokens
        self.max_running_requests = target_worker.max_running_requests
        self.max_token_pool_size = target_worker.model_runner.max_token_pool_size
        self.req_to_token_pool = target_worker.model_runner.req_to_token_pool  # draft block table
        self.token_to_kv_pool = None  # The KV cache is allocated on the draft process, not here.
        self.attn_backend = None
        self.model_is_mrope = False
        self.sliding_window_size = None
        self.model_config = ModelConfigStub(
            context_len=self.max_total_num_tokens,
            vocab_size=target_worker.model_runner.model_config.vocab_size,
            dtype=target_worker.model_runner.model_config.dtype,
        )
        self.use_ngram_embedding = False
        self.is_hybrid_swa = False


class AsyncSpecWorker(SpecWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
        async_process_group: dist.ProcessGroup,
        async_rank: int,
    ):
        super().__init__(server_args, gpu_id, tp_rank, dp_rank, moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port, target_worker)
        self.async_process_group = async_process_group
        self.async_rank = async_rank
        self._is_async_leader = (async_process_group is not None)
        self.draft_attn_backend = None
        K = self.speculative_num_steps
        MQ_LEN = sum(self.server_args.speculative_async_fan_out_list)
        # K from the glue decode, MQ_LEN * K from the tree decode.
        self.num_tokens_for_async_draft_tree = K * (MQ_LEN + 1) + 1
        self._alloc_handshake_bufs(1)

    def _alloc_handshake_bufs(self, B, max_blocks: int = -1):
        self._hs_B = B
        K = self.speculative_num_steps
        d = self.device
        self._cmd = torch.zeros(1, dtype=torch.int64, device=d)
        self._meta = torch.tensor([
            B,
            K,
            self.server_args.speculative_async_fan_out,
            max_blocks,
        ], dtype=torch.int64, device=d)
        self._cache_keys = torch.empty(B, 3, dtype=torch.int64, device=d)
        self._num_tokens_buf = torch.empty(B, dtype=torch.int64, device=d)
        self._temps_buf = torch.zeros(B, dtype=torch.int64, device=d)
        # self._block_tables_buf = torch.full((B, self.max_blocks), -1, dtype=torch.int32, device=d)
        self._fused_response = torch.empty(B + B * K, dtype=torch.int64, device=d)
        # For now we don't need logits because speculation is always greedy
        self._logits_q = None
        self._extend_counts = torch.zeros(B, dtype=torch.int64, device=d)
        self._parent_list = torch.arange(
            -1, K - 1, dtype=torch.int64, device=d,
        ).unsqueeze(0).repeat(B, 1)
        self._top_scores_index = torch.arange(
            0, K, dtype=torch.int64, device=d,
        ).unsqueeze(0).repeat(B, 1)

    def _get_draft_block_table(
        self,
        forward_batch: ForwardBatch,
    ) -> torch.tensor:
        r2t = forward_batch.req_to_token_pool.req_to_token
        max_blocks = min(
            forward_batch.seq_lens.max() + self.num_tokens_for_async_draft_tree,
            forward_batch.req_to_token_pool.req_to_token.shape[1],
        )
        draft_block_table = torch.stack(
            [r2t[idx, :max_blocks] for idx in forward_batch.req_pool_indices],
            dim=0,
        ).to(device=self.device, dtype=torch.int64)
        return draft_block_table, max_blocks

    def _init_model_runner(self):
        self._model_runner = ModelRunnerStub(
            self.server_args,
            self.target_worker,
        )

    def init_attention_backend(self):
        pass

    def init_cuda_graphs(self):
        pass

    def _set_random_seed(self, seed: int):
        self.random_seed = seed
        set_random_seed(self.random_seed)

    def _can_cuda_graph(self, forward_batch: ForwardBatch) -> bool:
        return False

    def _get_alloc_len_for_speculation(self) -> int:
        return self.num_tokens_for_async_draft_tree

    def _get_context_managers_for_draft(self, return_empty_contexts: bool = False):
        # Always return empty context managers for async spec worker.
        return (empty_context(), empty_context(), empty_context())

    def _capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        if self.speculative_algorithm.is_eagle():
            draft_input.hidden_states = logits_output.hidden_states

    def _prepare_for_extend(self, batch: ScheduleBatch):
        if self.speculative_algorithm.is_eagle():
            batch.spec_info.prepare_for_extend(batch)

    # Prefill forward pass for async spec worker.
    def _draft_extend_forward_pass(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        draft_block_table, max_blocks = self._get_draft_block_table(forward_batch)
        cmd = torch.tensor([1], dtype=torch.int64, device=self.device)
        eagle_acts = forward_batch.spec_info.hidden_states if self.speculative_algorithm.is_eagle() else None
        if NCCL_LOG:
            print(f'[{_ts()}] [draft_extend_forward_pass] max_blocks={max_blocks}', flush=True)
            print(f'[{_ts()}] [draft_extend_forward_pass] input_ids.shape={forward_batch.input_ids.shape}', flush=True)
        metadata = prepare_prefill_metadata(
            forward_batch.input_ids.shape[0],
            forward_batch.batch_size,
            max_blocks,
            eagle_acts is not None,
            eagle_acts.shape[1] if eagle_acts is not None else 0,
            self.device,
        )
        if NCCL_LOG:
            sep = '=' * 80
            print(f"[{_ts()}] \n{sep}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] batch_size={forward_batch.batch_size}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] input_ids shape={forward_batch.input_ids.shape}, values={forward_batch.input_ids.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] input_ids decoded='{_decode_ids(forward_batch.input_ids)}'", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] extend_seq_lens={forward_batch.extend_seq_lens.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] draft_block_table shape={draft_block_table.shape}, values={draft_block_table.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] metadata={metadata.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] req_pool_indices={forward_batch.req_pool_indices.tolist()}", flush=True)
            print(f"[{_ts()}] {sep}\n", flush=True)
        if self._is_async_leader:
            send_prefill_request(
                cmd,
                metadata,
                forward_batch.input_ids,
                forward_batch.extend_seq_lens,
                draft_block_table,
                eagle_acts,
                self.async_process_group,
                self.async_rank,
            )
        return LogitsProcessorOutput(None, None)

    # Decode forward pass for async spec worker (runs on separate process).
    def _draft_forward(self, forward_batch: ForwardBatch, request_ids: torch.tensor = None):
        assert request_ids is not None
        if NCCL_LOG:
            print(f'[{_ts()}] [draft_forward] SENDING SPECULATION REQUEST', flush=True)
        B = forward_batch.batch_size
        draft_block_table, max_blocks = self._get_draft_block_table(forward_batch)
        if B != self._hs_B:
            self._alloc_handshake_bufs(B, max_blocks)
        else:
            self._meta[3] = max_blocks

        accept_length = forward_batch.spec_info.accept_length
        if accept_length is None:
            accept_length = -2  # first decode, no prior acceptance
        else:
            # SGLang's accept_length includes the verified token (+1 added in
            # prepare_extend_after_decode), but SSD expects it without the
            # verified token.  Subtract 1 to match SSD convention.
            accept_length = accept_length - 1
        self._cache_keys[:, 0] = request_ids
        self._cache_keys[:, 1] = accept_length
        self._cache_keys[:, 2] = forward_batch.spec_info.verified_id
        self._num_tokens_buf = forward_batch.seq_lens + 1
        # self._temps_buf = forward_batch.spec_info.temperature
        # self._block_tables_buf = forward_batch.spec_info.draft_block_table
        if NCCL_LOG:
            sep = '=' * 80
            print(f"[{_ts()}] \n{sep}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] B={B}, K={self.speculative_num_steps}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] request_ids={request_ids.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] accept_length={accept_length if isinstance(accept_length, int) else accept_length.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] verified_id={forward_batch.spec_info.verified_id.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] verified_id decoded={_decode_id_list(forward_batch.spec_info.verified_id)}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] cache_keys shape={self._cache_keys.shape}, values={self._cache_keys.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] num_tokens (seq_lens+1)={self._num_tokens_buf.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] draft_block_table shape={draft_block_table.shape}, values={draft_block_table.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] temps={self._temps_buf.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] meta={self._meta.tolist()}", flush=True)
            print(f"[{_ts()}] {sep}\n", flush=True)
        if self._is_async_leader:
            if _SGLANG_PROF:
                torch.cuda.synchronize()
                _df_t0 = _time.perf_counter()
            send_speculation_request(
                self._cmd,
                self._meta,
                self._cache_keys,
                self._num_tokens_buf,
                draft_block_table,
                self._temps_buf,
                self.async_process_group,
                self.async_rank,
            )
            if _SGLANG_PROF:
                torch.cuda.synchronize()
                _df_t1 = _time.perf_counter()
            if NCCL_LOG:
                print(f'[{_ts()}] [draft_forward] SPECULATION REQUEST SENT', flush=True)
                print(f'[{_ts()}] [draft_forward] RECEIVING SPECULATION RESPONSE', flush=True)
            speculations, _, _ = receive_speculation_response(
                B,
                self.speculative_num_steps,
                self._fused_response,
                self._logits_q,
                self.async_process_group,
                self.async_rank,
                skip_logits=True
            )
            if _SGLANG_PROF:
                torch.cuda.synchronize()
                _df_t2 = _time.perf_counter()
                print(f"[PROFILE sglang_draft] nccl_send={(_df_t1-_df_t0)*1000:.2f}ms nccl_recv={(_df_t2-_df_t1)*1000:.2f}ms total={(_df_t2-_df_t0)*1000:.2f}ms", flush=True)
        else:
            speculations = torch.empty(B, self.speculative_num_steps, dtype=torch.int64, device=self.device)

        # Broadcast speculations from TP rank 0 to all other TP ranks
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            dist.broadcast(speculations, src=tp_group.ranks[0], group=tp_group.device_group)
        if NCCL_LOG:
            print(f'[{_ts()}] [draft_forward] SPECULATION RESPONE RECEIVED', flush=True)
            sep = '=' * 80
            print(f"[{_ts()}] \n{sep}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC_RESP] speculations shape={speculations.shape}", flush=True)
            for i in range(B):
                spec_ids = speculations[i].tolist()
                spec_text = _decode_id_list(speculations[i])
                print(f"[{_ts()}]   req[{i}]: speculations={spec_ids}", flush=True)
                print(f"[{_ts()}]            decoded={spec_text}", flush=True)
            print(f"[{_ts()}] {sep}\n", flush=True)
        return self._parent_list, self._top_scores_index, speculations

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        batch.spec_info.prepare_extend_after_decode(
            batch,
            self.speculative_num_steps,
        )
