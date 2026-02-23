import logging
from typing import Optional

import torch
from ssd.engine.helpers.runner_helpers import prepare_prefill_payload

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.async_spec.nccl_comm import NcclDraftChannel
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.spec_worker import SpecWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import LogitsProcessorOutput
from sglang.srt.utils import empty_context, set_random_seed

logger = logging.getLogger(__name__)


class ModelRunnerStub:
    def __init__(
        self,
        server_args: ServerArgs,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.device = server_args.device
        self.max_total_num_tokens = target_worker.max_total_num_tokens
        self.max_running_requests = target_worker.max_running_requests
        self.max_token_pool_size = target_worker.max_token_pool_size
        self.req_to_token_pool = target_worker.req_to_token_pool  # draft block table
        self.token_to_kv_pool = None  # KV cache is allocated on the draft process, not here.
        self.attn_backend = None
        self.model_is_mrope = False


class AsyncSpecWorker(SpecWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
        nccl_channel: NcclDraftChannel,
    ):
        super().__init__(server_args, gpu_id, tp_rank, dp_rank, moe_ep_rank, nccl_port, target_worker)
        self.nccl_channel = nccl_channel

    # def __init__(  # This init is written from scratch, but for now we are reusing the SpecWorker init.
    #     self,
    #     server_args: ServerArgs,
    #     gpu_id: int,
    #     tp_rank: int,
    #     dp_rank: Optional[int],
    #     moe_ep_rank: int,
    #     nccl_port: int,
    #     target_worker: TpModelWorker,
    #     nccl_channel: NcclDraftChannel,
    # ):
    #     # TODO: Can we update this to use the same as SpecWorker?
    #     self.server_args = server_args
    #     self.tp_size = server_args.tp_size
    #     self.ep_size = server_args.ep_size
    #     self.pp_size = server_args.pp_size
    #     self.tp_rank = tp_rank
    #     self.moe_ep_rank = moe_ep_rank
    #     self.pp_rank = 0
    #     self.dp_rank = dp_rank
    #     self.gpu_id = gpu_id
    #     self.nccl_port = nccl_port
    #     self.is_draft_worker = True
    #     self.is_multi_layer_eagle = False

    #     self.target_worker = target_worker
    #     self.nccl_channel = nccl_channel
    #     self.num_tokens_for_async_draft_tree = (
    #         sum(server_args.speculative_async_fan_out_list) * server_args.speculative_num_steps
    #     )
    #     self.model_runner = ModelRunnerStub(
    #         server_args,
    #         self.target_worker,
    #     )

    #     self.topk = server_args.speculative_eagle_topk
    #     self.speculative_num_steps = server_args.speculative_num_steps
    #     self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
    #     self.enable_nan_detection = server_args.enable_nan_detection
    #     self.device = server_args.device
    #     self.page_size = server_args.page_size
    #     self.speculative_algorithm = SpeculativeAlgorithm.from_string(
    #         server_args.speculative_algorithm
    #     )

    #     #############################################################
    #     # INITIALIZATION CODE BELOW COPIED FROM TpWorker.__init__()
    #     #############################################################
    #     self._init_model_config()

    #     self.max_total_num_tokens = target_worker.max_total_num_tokens
    #     self.max_prefill_tokens = server_args.max_prefill_tokens
    #     self.max_running_requests = target_worker.max_running_requests
    #     assert self.max_running_requests > 0, "max_running_request is zero"
    #     self.max_queued_requests = server_args.max_queued_requests
    #     assert (
    #         self.max_queued_requests is None or self.max_queued_requests >= 1
    #     ), "If configured, max_queued_requests must be at least 1 for any work to be scheduled."
    #     self.max_req_len = min(
    #         self.model_config.context_len - 1,
    #         target_worker.max_token_pool_size - 1,
    #     )
    #     self.max_req_input_len = self.max_req_len - 5
    #     assert (
    #         self.max_req_len > 0 and self.max_req_input_len > 0
    #     ), "Memory pool size is too small"

    #     self.enable_overlap = not server_args.disable_overlap_schedule
    #     self.enable_spec = server_args.speculative_algorithm is not None
    #     self.hicache_layer_transfer_counter = None


    #     #############################################################
    #     # INITIALIZATION CODE BELOW COPIED FROM SpecWorker.__init__()
    #     #############################################################

    #     # Override the context length of the draft model to be the same as the target model.
    #     server_args.context_length = target_worker.model_runner.model_config.context_len

    #     # Do not capture cuda graph in `super().__init__()`
    #     # It will be captured later.
    #     backup_disable_cuda_graph = server_args.disable_cuda_graph
    #     server_args.disable_cuda_graph = True
    #     # Share the allocator with a target worker.
    #     # Draft and target worker own their own KV cache pools.
    #     self.req_to_token_pool, self.token_to_kv_pool_allocator = (
    #         target_worker.get_memory_pool()
    #     )

    #     # Load hot token ids
    #     if self.speculative_algorithm.is_eagle3():
    #         if server_args.speculative_token_map is not None:
    #             logger.warning(
    #                 "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
    #             )
    #         self.hot_token_id = None
    #     elif server_args.speculative_token_map is not None:
    #         self.hot_token_id = load_token_map(server_args.speculative_token_map)
    #         server_args.json_model_override_args = (
    #             f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
    #         )
    #     else:
    #         self.hot_token_id = None

    #     # Init attention backend and cuda graphs
    #     self.draft_model_runner.server_args.disable_cuda_graph = (
    #         backup_disable_cuda_graph
    #     )
    #     self.eagle_use_aux_hidden_state = False
    #     if self.speculative_algorithm.is_eagle3():
    #         self.eagle_use_aux_hidden_state = True
    #         eagle_config = getattr(
    #             self.draft_model_runner.model_config.hf_config, "eagle_config", {}
    #         )
    #         self.eagle_use_aux_hidden_state = eagle_config.get(
    #             "use_aux_hidden_state", True
    #         )

    def _init_model_runner(self):
        self.model_runner = ModelRunnerStub(
            self.server_args,
            self.target_worker,
        )

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
        return None

    def _prepare_for_extend(self, batch: ScheduleBatch):
        pass

    def _send_prefill_request(
        self,
        cmd: torch.Tensor,  # TODO: Add shapes here
        metadata: torch.Tensor,
        input_ids: torch.Tensor,
        num_tokens: torch.Tensor,
        draft_block_table: torch.Tensor,
        eagle_acts: torch.Tensor,
    ):
        # TODO: Send to draft runner
        # dist.send(cmd, dst=self.draft_runner_rank, group=self.async_pg)
        # dist.send(metadata, dst=self.draft_runner_rank, group=self.async_pg)
        # for t in (input_ids, num_tokens, draft_block_table, eagle_acts):
        #     if t is not None:
        #         dist.send(t, dst=self.draft_runner_rank, group=self.async_pg)
        pass

    # Prefill forward pass for async spec worker.
    def _draft_extend_forward_pass(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        r2t = forward_batch.req_to_token_pool.req_to_token
        draft_block_tables = torch.stack([r2t[idx, :] for idx in forward_batch.req_pool_indices], dim=0)
        cmd, metadata, input_ids, num_tokens, draft_block_table, eagle_acts = prepare_prefill_payload(
            forward_batch.input_ids,
            forward_batch.hidden_states,
            self.device,
            draft_block_tables.shape[1],
            draft_block_tables,
        )
        self._send_prefill_request(cmd, metadata, input_ids, num_tokens, draft_block_table, eagle_acts)
        return LogitsProcessorOutput(None, None)

    # Decode forward pass for async spec worker (runs on separate process).
    def _draft_forward(self, forward_batch: ForwardBatch):
        # TODO: Implement remote draft forward pass
        pass

    # # batch, hidden_states, next_token_ids, seq_lens_cpu:
    # def _remote_draft_extend(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
    #     """Draft extend after target prefill (once per request, separate RPC)."""
    #     # 1. Prepare batch locally (set spec_info, prepare_for_extend, allocate out_cache_loc)
    #     # 2. Extract req_to_token_rows
    #     # 3. Send inputs via NCCL
    #     # 4. Receive topk_p, topk_index, hidden_states
    #     # 5. Store into batch.spec_info (consumed by next decode's _remote_extend_and_draft)

    # def _remote_extend_and_draft(self, batch):
    #     """Fused: send accepted tokens + draft alloc info → remote extends KV + runs K draft steps → returns tree."""
    #     # 1. Prepare extend_after_decode inputs locally
    #     # 2. Run _draft_preprocess_decode locally (allocates out_cache_loc for draft,
    #     #    writes to req_to_token_pool, restores allocator)
    #     # 3. Extract req_to_token_rows for current batch
    #     # 4. Send via NCCL: hidden_states, verified_id, accept_length,
    #     #    out_cache_loc (extend + draft combined), seq_lens, positions,
    #     #    req_pool_indices, req_to_token_rows
    #     # 5. Receive via NCCL: tree_mask, positions, retrive_*, draft_tokens
    #     # 6. Construct and return EagleVerifyInput

