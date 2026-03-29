"""Dedicated-GPU draft model runner for async speculative decoding.

This process runs on a separate GPU and receives commands from the target
scheduler via NCCL. It maintains a tree cache of speculative
continuations that can be served instantly on cache hits.

Adapted from SSD's DraftRunner.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.speculative.async_spec.handshake import (
    CMD_EXIT,
    CMD_PREFILL,
    CMD_SPEC_REQUEST,
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
        self.jit_speculate = server_args.speculative_async_jit_speculate

        # Tree cache (matching SSD _reset_tree_cache_tensors)
        self.tree_cache_keys = torch.zeros(
            (0, 3), dtype=torch.int64, device=self.device
        )
        self.tree_cache_tokens: Optional[torch.Tensor] = None
        self.tree_cache_logits: Optional[torch.Tensor] = None

        self.vocab_size = vocab_size

    def draft_loop(self):
        """Main event loop."""
        logger.info(f"AsyncDraftRunner starting draft loop on GPU {self.draft_gpu_id}")

        while True:
            try:
                cmd = self.channel.recv_command()

                if cmd == CMD_EXIT:
                    logger.info("AsyncDraftRunner received exit command")
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

    def _handle_prefill(self):
        """Receive prefill tensors from target via NCCL."""
        num_reqs, total_tokens, input_ids, req_pool_indices, seq_lens = (
            self.channel.unpack_prefill()
        )
        logger.debug(
            f"Draft prefill: {num_reqs} reqs, {total_tokens} tokens"
        )
        # TODO: Run draft model forward to populate KV cache

    def _handle_spec_request(self):
        """Core async spec logic: cache lookup -> respond -> background tree decode."""
        B, K, fan_out, vocab_size, cache_keys, temperatures = (
            self.channel.unpack_spec_request()
        )

        # Step 1: Cache lookup and build response
        speculations = self._hit_cache_and_respond(cache_keys, B, K, vocab_size)

        # Send speculations back to target via NCCL
        self.channel.send_speculations(speculations)

        # --- Target proceeds to verify while we continue ---
        # Background tree decode to populate cache for next iteration
        self._reset_tree_cache()
        # TODO: Implement full tree decode (glue decode + tree decode)

    def _hit_cache_and_respond(
        self,
        cache_keys: torch.Tensor,
        B: int,
        K: int,
        V: int,
    ) -> torch.Tensor:
        """Check tree cache, return cached/random tokens (matching SSD)."""
        # Init with random logits so token IDs are in-vocab (matching SSD)
        out_logits = torch.empty(
            (B, K, V), dtype=torch.float32, device=self.device
        ).uniform_()
        out_tokens = out_logits.argmax(dim=-1)  # [B, K]
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)

        # Set recovery tokens as position 0
        recovery_tokens = cache_keys[:, 2].unsqueeze(1)  # [B, 1]

        # Vectorized cache lookup (matching SSD)
        if self.tree_cache_keys.numel() > 0:
            eq = cache_keys.unsqueeze(1) == self.tree_cache_keys.unsqueeze(0)
            match = torch.all(eq, dim=2)
            cache_hits_bool = match.any(dim=1)
            cache_hits = cache_hits_bool.to(torch.int64)

            if cache_hits_bool.any() and self.tree_cache_tokens is not None:
                idx = match.float().argmax(dim=1).to(torch.int64)
                sel = cache_hits_bool
                cached_k = min(K, self.tree_cache_tokens.shape[1])
                out_tokens[sel, :cached_k] = self.tree_cache_tokens[idx[sel], :cached_k]
                if self.tree_cache_logits is not None:
                    out_logits[sel, :cached_k] = self.tree_cache_logits[
                        idx[sel], :cached_k
                    ]

        # Build speculations [B, K+1] = recovery_token + K draft tokens
        speculations = torch.cat([recovery_tokens, out_tokens.to(torch.int64)], dim=1)

        return speculations

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

        # Create the draft runner
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
        raise
