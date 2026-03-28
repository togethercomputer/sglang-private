import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

from ssd.engine.helpers.runner_helpers import (
    PrefillRequest,
    SpeculationRequest,
    SpeculationResponse,
)
from ssd.utils.misc import compress_neg_ones_and_zeros

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
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
        target_hidden_size: int,
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
        B = 1
        eagle = self.speculative_algorithm == SpeculativeAlgorithm.ASYNC_EAGLE
        eagle3 = self.speculative_algorithm == SpeculativeAlgorithm.ASYNC_EAGLE3
        phoenix_v1 = self.speculative_algorithm == SpeculativeAlgorithm.ASYNC_PHOENIX
        phoenix_v2 = self.speculative_algorithm == SpeculativeAlgorithm.ASYNC_PHOENIX2
        standalone = self.speculative_algorithm == SpeculativeAlgorithm.ASYNC_STANDALONE
        if eagle3 or phoenix_v2:
            eagle_act_dim = 3 * target_hidden_size
        elif eagle or phoenix_v1:
            eagle_act_dim = target_hidden_size
        elif standalone:
            eagle_act_dim = 0
        else:
            raise ValueError(f"Unsupported speculative algorithm: {self.speculative_algorithm}")

        self._speculation_request = SpeculationRequest.prepare(
            batch_size=B,
            lookahead=K,
            max_blocks=-1,
            vocab_size=-1,
            draft_dtype=self.model_runner.model_config.dtype,
            device=self.device,
            eagle=self.speculative_algorithm.is_eagle(),
            eagle_act_dim=eagle_act_dim,
        )
        self._speculation_response = SpeculationResponse.prepare(
            lookahead=K,
            device=self.device,
            draft_dtype=self.model_runner.model_config.dtype,
            batch_size=B,
        )
        self._alloc_tree_verification_bufs(B)

    def _alloc_tree_verification_bufs(self, B):
        K = self.speculative_num_steps
        self._parent_list = torch.arange(
            -1, K - 1, dtype=torch.int64, device=self.device,
        ).unsqueeze(0).repeat(B, 1)
        self._top_scores_index = torch.arange(
            0, K, dtype=torch.int64, device=self.device,
        ).unsqueeze(0).repeat(B, 1)

    def _get_draft_block_table(
        self,
        forward_batch: ForwardBatch,
    ) -> torch.tensor:
        r2t = forward_batch.req_to_token_pool.req_to_token
        page_size = self.server_args.page_size
        max_tokens = min(
            forward_batch.seq_lens.max() + self.num_tokens_for_async_draft_tree,
            forward_batch.req_to_token_pool.req_to_token.shape[1],
        )
        if page_size == 1:
            draft_block_table = torch.stack(
                [r2t[idx, :max_tokens] for idx in forward_batch.req_pool_indices],
                dim=0,
            ).to(device=self.device, dtype=torch.int64)
            return draft_block_table, max_tokens
        else:
            # Convert per-token indices to page-level block table.
            # Sample req_to_token at stride intervals and divide by page_size.
            pool_len = r2t.shape[1]
            max_pages = (max_tokens + page_size - 1) // page_size
            # Clamp strided indices to stay within req_to_token bounds.
            strided_indices = torch.arange(
                0, max_pages * page_size, page_size,
                device=self.device,
            ).clamp_(max=pool_len - 1)
            per_token_indices = torch.stack(
                [r2t[idx, strided_indices] for idx in forward_batch.req_pool_indices],
                dim=0,
            ).to(device=self.device, dtype=torch.int64)
            draft_block_table = per_token_indices // page_size
            return draft_block_table, max_pages

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
        if batch.forward_mode.is_idle():
            return

        if self.speculative_algorithm.is_eagle():
            B = len(batch.seq_lens)
            eagle_act_dim = batch.spec_info.hidden_states.shape[1]
            if batch.spec_info.last_hidden_states is None or batch.spec_info.last_hidden_states.shape[0] != B:
                batch.spec_info.last_hidden_states = torch.empty(
                    B, eagle_act_dim, device=self.device, dtype=batch.spec_info.hidden_states.dtype,
                )

            assert len(batch.spec_info.verified_id) == B
            assert batch.spec_info.hidden_states.shape[0] == sum(batch.extend_lens), (
                f"hidden_states.shape[0]={batch.spec_info.hidden_states.shape[0]} != sum(extend_lens)={sum(batch.extend_lens)}"
            )
            # Duplicate the first hidden state (h0), and remove the last hidden state, so that the sequence
            # length is the same as the input tokens (t0, t1, ...). So the inputs to the Eagle prefill are:
            # [t0, h0], [t1, h0], [t2, h1], [t3, h2], [t4, h3], ...
            pt = 0
            for i, extend_len in enumerate(batch.extend_lens):
                hidden_states = batch.spec_info.hidden_states[pt : pt + extend_len, :]
                batch.spec_info.last_hidden_states[i, :] = hidden_states[-1, :]
                batch.spec_info.hidden_states[pt : pt + extend_len, :] = torch.cat([
                    hidden_states[:1, :], hidden_states[:-1, :],
                ], dim=0)
                pt += extend_len

    def _draft_extend_forward_pass(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        draft_block_table, max_blocks = self._get_draft_block_table(forward_batch)
        target_hidden_states = forward_batch.spec_info.hidden_states if self.speculative_algorithm.is_eagle() else None
        if NCCL_LOG:
            print(f'[{_ts()}] [draft_extend_forward_pass] max_blocks={max_blocks}', flush=True)
            print(f'[{_ts()}] [draft_extend_forward_pass] input_ids.shape={forward_batch.input_ids.shape}', flush=True)
            if target_hidden_states is not None:
                print(f'[{_ts()}] [draft_extend_forward_pass] target_hidden_states.shape={target_hidden_states.shape}', flush=True)
            else:
                print(f'[{_ts()}] [draft_extend_forward_pass] target_hidden_states is None', flush=True)

        prefill_request = PrefillRequest.prepare(
            input_ids=forward_batch.input_ids,
            num_tokens=forward_batch.extend_seq_lens,
            draft_block_table=draft_block_table,
            eagle_acts=target_hidden_states,
            max_blocks=max_blocks,
            device=self.device,
        )
        if NCCL_LOG:
            sep = '=' * 80
            print(f"[{_ts()}] \n{sep}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] batch_size={forward_batch.batch_size}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] input_ids shape={forward_batch.input_ids.shape}, values={forward_batch.input_ids.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] input_ids decoded='{_decode_ids(forward_batch.input_ids)}'", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] extend_seq_lens={forward_batch.extend_seq_lens.tolist()}", flush=True)
            draft_block_table_values_str = compress_neg_ones_and_zeros(f"{draft_block_table.tolist()}")  # Replace 3 or more -1's with "-1, ..., -1" to avoid long lists
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] draft_block_table shape={draft_block_table.shape}, values={draft_block_table_values_str}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] metadata={prefill_request.metadata.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_PREFILL] req_pool_indices={forward_batch.req_pool_indices.tolist()}", flush=True)
            print(f"[{_ts()}] {sep}\n", flush=True)
        if self._is_async_leader:
            prefill_request.send(
                async_pg=self.async_process_group,
                draft_rank=self.async_rank,
            )
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=target_hidden_states)

    def _draft_forward(self, forward_batch: ForwardBatch, request_ids: torch.tensor = None):
        assert request_ids is not None
        if NCCL_LOG:
            print(f'[{_ts()}] [draft_forward] SENDING SPECULATION REQUEST', flush=True)
        B = forward_batch.batch_size
        draft_block_table, max_blocks = self._get_draft_block_table(forward_batch)
        if B != self._speculation_request.batch_size:
            self._speculation_request.maybe_update_buffers(B, max_blocks=-1)
            self._alloc_tree_verification_bufs(B)

        self._speculation_request.metadata[2] = max_blocks
        self._speculation_request.block_tables = draft_block_table

        accept_length = forward_batch.spec_info.accept_length
        if accept_length is None:
            accept_length = -2  # first decode, no prior acceptance
        else:
            # SGLang's accept_length includes the verified token (+1 added in
            # prepare_extend_after_decode), but SSD expects it without the
            # verified token.  Subtract 1 to match SSD convention.
            accept_length = accept_length - 1

        self._speculation_request.cache_keys[:, 0] = request_ids
        self._speculation_request.cache_keys[:, 1] = accept_length
        self._speculation_request.cache_keys[:, 2] = forward_batch.spec_info.verified_id
        self._speculation_request.num_tokens[:] = forward_batch.seq_lens + 1
        # TODO: Set temperatures

        if self.speculative_algorithm.is_eagle():
            eagle_acts = forward_batch.spec_info.hidden_states
            accepted_token_ids = forward_batch.input_ids
            last_hidden_states = forward_batch.spec_info.last_hidden_states
            seq_lens = forward_batch.seq_lens
            if isinstance(accept_length, int):
                assert accept_length == -2
            else:
                assert isinstance(accept_length, torch.Tensor)
                assert accept_length.shape == (B,)

            is_first_decode = accept_length == -2
            a = 0
            for i in range(B):
                acc_len = seq_lens[i] if is_first_decode else accept_length[i]
                b = max(min(a + acc_len, eagle_acts.shape[0]), 0)
                self._speculation_request.extend_counts[i] = 0 if is_first_decode else acc_len
                if eagle_acts is not None:
                    self._speculation_request.recovery_activations[i, :] = last_hidden_states[i, :]
                    if not is_first_decode and acc_len > 0:
                        self._speculation_request.extend_activations[i, :acc_len] = eagle_acts[a: b, :]
                        self._speculation_request.extend_token_ids[i, :acc_len] = accepted_token_ids[a: b]
                        a += acc_len

        if NCCL_LOG:
            cache_keys = self._speculation_request.cache_keys
            num_tokens = self._speculation_request.num_tokens
            temps = self._speculation_request.temps
            metadata = self._speculation_request.metadata
            sep = '=' * 80
            print(f"[{_ts()}] \n{sep}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] B={B}, K={self.speculative_num_steps}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] request_ids={request_ids.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] accept_length={accept_length if isinstance(accept_length, int) else accept_length.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] verified_id={forward_batch.spec_info.verified_id.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] verified_id decoded={_decode_id_list(forward_batch.spec_info.verified_id)}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] cache_keys shape={cache_keys.shape}, values={cache_keys.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] num_tokens (seq_lens+1)={num_tokens.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] temps={temps.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] metadata={metadata.tolist()}", flush=True)
            draft_block_table_values_str = compress_neg_ones_and_zeros(f"{draft_block_table.tolist()}")
            print(f"[{_ts()}] [NCCL_LOG SGLANG_SPEC] draft_block_table shape={draft_block_table.shape}, values={draft_block_table_values_str}", flush=True)
            print(f"[{_ts()}] {sep}\n", flush=True)

        if self._is_async_leader:
            if _SGLANG_PROF:
                torch.cuda.synchronize()
                _df_t0 = _time.perf_counter()
            self._speculation_request.send(
                async_pg=self.async_process_group,
                draft_rank=self.async_rank,
            )
            if _SGLANG_PROF:
                torch.cuda.synchronize()
                _df_t1 = _time.perf_counter()
            if NCCL_LOG:
                print(f'[{_ts()}] [draft_forward] SPECULATION REQUEST SENT', flush=True)
                print(f'[{_ts()}] [draft_forward] RECEIVING SPECULATION RESPONSE', flush=True)


            self._speculation_response.receive(
                async_pg=self.async_process_group,
                draft_rank=self.async_rank,
                batch_size=B,
            )
            speculations = self._speculation_response.speculations
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
        if self.speculative_algorithm.is_eagle() and not batch.forward_mode.is_idle():
            # Extract the last target hidden state per request for recovery_activations.
            # After prepare_extend_after_decode, batch.extend_lens[i] gives the number of
            # hidden states for request i (accepted tokens + verified token).
            # The last hidden state per request is the recovery activation.
            B = len(batch.extend_lens)
            hidden_states = batch.spec_info.hidden_states
            assert hidden_states.shape[0] == sum(batch.extend_lens), (
                f"hidden_states.shape[0]={hidden_states.shape[0]} != sum(extend_lens)={sum(batch.extend_lens)}"
            )
            eagle_act_dim = hidden_states.shape[1]
            if batch.spec_info.last_hidden_states is None or batch.spec_info.last_hidden_states.shape[0] != B:
                batch.spec_info.last_hidden_states = torch.empty(
                    B, eagle_act_dim, device=self.device, dtype=hidden_states.dtype,
                )
            pt = 0
            for i in range(B):
                extend_len = batch.extend_lens[i]
                batch.spec_info.last_hidden_states[i, :] = hidden_states[pt + extend_len - 1, :]
                pt += extend_len
