"""IPC communication protocol between target scheduler and draft runner.

Uses multiprocessing Pipe for communication. The protocol is synchronous
from the target side: send a request, wait for response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)

# Command codes for target -> draft communication
CMD_SPEC_REQUEST = 0
CMD_PREFILL = 1
CMD_EXIT = 2


@dataclass
class SpecRequest:
    """Spec request sent from target to draft."""

    cmd: int
    batch_size: int
    lookahead: int
    fan_out: int
    vocab_size: int
    # Per-request data as lists for pickling
    cache_keys: List[List[int]]  # [B, 3]
    temperatures: List[float]  # [B]


@dataclass
class SpecResponse:
    """Spec response sent from draft to target."""

    # Tensors are sent as CPU tensors for pickling
    speculations: torch.Tensor  # [B, K+1] int64 CPU
    logits_q: torch.Tensor  # [B, K, V] float32 CPU
    cache_hits: torch.Tensor  # [B] int64 CPU


@dataclass
class PrefillRequest:
    """Prefill request sent from target to draft."""

    cmd: int
    num_reqs: int
    total_tokens: int
    input_ids: torch.Tensor  # CPU
    req_pool_indices: torch.Tensor  # CPU
    seq_lens: torch.Tensor  # CPU


# For unit testing
def concat_int64(*tensors: torch.Tensor) -> torch.Tensor:
    """Concatenate tensors into a single flat int64 payload."""
    parts = []
    for t in tensors:
        if t is None:
            continue
        if t.dtype != torch.int64:
            t = t.to(torch.int64)
        parts.append(t.reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.int64)
    return torch.cat(parts, dim=0)


class TargetDraftHandshake:
    """Handles the target -> draft request/response protocol.

    Uses multiprocessing Pipe for communication.
    """

    def __init__(
        self,
        reqs: List,
        lookahead: int,
        async_fan_out: int,
        vocab_size: int,
        draft_dtype: torch.dtype,
        device: torch.device,
        comm_pipe,  # multiprocessing.Connection
    ):
        self.reqs = reqs
        self.lookahead = lookahead
        self.async_fan_out = async_fan_out
        self.vocab_size = vocab_size
        self.draft_dtype = draft_dtype
        self.device = device
        self.comm_pipe = comm_pipe
        self.batch_size = len(reqs)

        # Build cache_keys from reqs: [B, 3] = (req_pool_idx, accepted_len - 1, recovery_token)
        cache_keys = []
        temperatures = []
        for req in reqs:
            req_pool_idx = req.req_pool_idx if req.req_pool_idx is not None else 0
            accepted_len = req.last_spec_step_accepted_len
            recovery_token = (
                req.recovery_token_id if req.recovery_token_id is not None else 0
            )
            cache_keys.append([req_pool_idx, accepted_len, recovery_token])
            temperatures.append(req.sampling_params.temperature)

        self.cache_keys = cache_keys
        self.temperatures = temperatures

    def execute_full_handshake(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Send request and receive response."""
        request = SpecRequest(
            cmd=CMD_SPEC_REQUEST,
            batch_size=self.batch_size,
            lookahead=self.lookahead,
            fan_out=self.async_fan_out,
            vocab_size=self.vocab_size,
            cache_keys=self.cache_keys,
            temperatures=self.temperatures,
        )

        self.comm_pipe.send(request)
        response: SpecResponse = self.comm_pipe.recv()

        # Move tensors to device
        speculations = response.speculations.to(self.device)
        logits_q = response.logits_q.to(self.device)
        cache_hits = response.cache_hits.to(self.device)

        return speculations, logits_q, cache_hits


def send_prefill_command(
    comm_pipe,
    device: torch.device,
    input_ids: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Send prefill command and tensors to draft runner."""
    request = PrefillRequest(
        cmd=CMD_PREFILL,
        num_reqs=len(seq_lens),
        total_tokens=input_ids.shape[0],
        input_ids=input_ids.cpu(),
        req_pool_indices=req_pool_indices.cpu(),
        seq_lens=seq_lens.cpu(),
    )
    comm_pipe.send(request)


def send_exit_command(comm_pipe) -> None:
    """Send exit command to draft runner."""
    comm_pipe.send(CMD_EXIT)
