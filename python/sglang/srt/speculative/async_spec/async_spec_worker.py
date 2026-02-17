import logging
from typing import Optional

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import load_token_map
from sglang.srt.speculative.spec_worker import SpecWorker
from sglang.srt.speculative.async_spec.nccl_comm import NcclDraftChannel

logger = logging.getLogger(__name__)


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
        self.server_args = server_args
        self.tp_size = server_args.tp_size
        self.ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = 0
        self.dp_rank = dp_rank
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self.is_draft_worker = True
        self.is_multi_layer_eagle = False

        self.target_worker = target_worker
        self.nccl_channel = nccl_channel

        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.device = server_args.device
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        #############################################################
        # INITIALIZATION CODE BELOW COPIED FROM TpWorker.__init__()
        #############################################################
        self._init_model_config()

        self.max_total_num_tokens = target_worker.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = target_worker.max_running_requests
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_queued_requests = server_args.max_queued_requests
        assert (
            self.max_queued_requests is None or self.max_queued_requests >= 1
        ), "If configured, max_queued_requests must be at least 1 for any work to be scheduled."
        self.max_req_len = min(
            self.model_config.context_len - 1,
            target_worker.max_token_pool_size - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_spec = server_args.speculative_algorithm is not None
        self.hicache_layer_transfer_counter = None


        #############################################################
        # INITIALIZATION CODE BELOW COPIED FROM SpecWorker.__init__()
        #############################################################

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            self.eagle_use_aux_hidden_state = True
            eagle_config = getattr(
                self.draft_model_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend():
            # Prefill path: target extend locally, then draft extend via RPC
            logits_output, next_token_ids, seq_lens_cpu = self.forward_target_extend(batch)
            self._remote_draft_extend(batch, logits_output.hidden_states, next_token_ids, seq_lens_cpu)
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )
        else:
            # Decode path: fused extend+draft via single RPC, then verify locally
            spec_info = self._remote_extend_and_draft(batch)  # ← single NCCL round-trip
            logits_output, verify_output, _, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def _remote_draft_extend(self, batch, hidden_states, next_token_ids, seq_lens_cpu):
        """Draft extend after target prefill (once per request, separate RPC)."""
        # 1. Prepare batch locally (set spec_info, prepare_for_extend, allocate out_cache_loc)
        # 2. Extract req_to_token_rows
        # 3. Send inputs via NCCL
        # 4. Receive topk_p, topk_index, hidden_states
        # 5. Store into batch.spec_info (consumed by next decode's _remote_extend_and_draft)

    def _remote_extend_and_draft(self, batch):
        """Fused: send accepted tokens + draft alloc info → remote extends KV + runs K draft steps → returns tree."""
        # 1. Prepare extend_after_decode inputs locally
        # 2. Run _draft_preprocess_decode locally (allocates out_cache_loc for draft,
        #    writes to req_to_token_pool, restores allocator)
        # 3. Extract req_to_token_rows for current batch
        # 4. Send via NCCL: hidden_states, verified_id, accept_length,
        #    out_cache_loc (extend + draft combined), seq_lens, positions,
        #    req_pool_indices, req_to_token_rows
        # 5. Receive via NCCL: tree_mask, positions, retrive_*, draft_tokens
        # 6. Construct and return EagleVerifyInput


