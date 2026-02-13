"""NCCL-based communication channel for async speculative decoding.

Replaces multiprocessing.Pipe + pickle with GPU-direct NCCL send/recv
using pre-allocated buffers. Eliminates CPU serialization overhead and
avoids sending unused data (logits_q, cache_hits).

Buffer layout for decode request (request_int_buf):
  [cmd | B | K | fan_out | vocab_size | reserved*3 | cache_keys(B*3)...]

Protocol (all commands start with a grouped send/recv of 3 fixed-size
tensors: request_int_buf + request_temp_buf + seq_lens_buf so that NCCL
element counts always match):

  Decode:  target group{send int, send temp, send seq_lens} -> draft recvs,
           processes, draft send response_buf -> target recvs
  Prefill: target group{send int, send temp, send seq_lens} -> draft recvs,
           target send prefill_buf -> draft recvs
  Exit:    target group{send int, send temp, send seq_lens} -> draft recvs,
           exits
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

HEADER_SIZE = 8  # cmd, B, K, fan_out, vocab_size, reserved*3


class AsyncSpecNcclChannel:
    """Pre-allocated NCCL buffer channel for target <-> draft communication."""

    def __init__(
        self,
        comm,  # PyNcclCommunicator
        rank: int,
        device: torch.device,
        max_batch_size: int = 64,
        max_spec_k: int = 16,
        max_prefill_tokens: int = 16384,
    ):
        self.comm = comm
        self.rank = rank
        self.device = device
        self.peer = 1 - rank  # 2-process group: peer is the other rank
        self.max_batch_size = max_batch_size
        self.max_spec_k = max_spec_k

        # Pre-allocate fixed-size buffers (always send/recv full to match NCCL counts)
        # Request: header + cache_keys [max_B, 3] flattened
        self.request_int_buf = torch.zeros(
            HEADER_SIZE + max_batch_size * 3,
            dtype=torch.int64,
            device=device,
        )
        # Temperatures [max_B]
        self.request_temp_buf = torch.zeros(
            max_batch_size,
            dtype=torch.float32,
            device=device,
        )
        # Sequence lengths [max_B] (current total length per request)
        self.seq_lens_buf = torch.zeros(
            max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        # Response: speculations [max_B, max_K+1] flattened
        self.response_buf = torch.zeros(
            max_batch_size * (max_spec_k + 1),
            dtype=torch.int64,
            device=device,
        )
        # Prefill: input_ids + req_pool_indices + seq_lens (variable, secondary transfer)
        self.prefill_buf = torch.zeros(
            max_prefill_tokens + 2 * max_batch_size,
            dtype=torch.int64,
            device=device,
        )

    def _send(self, tensor: torch.Tensor):
        self.comm.send(tensor, dst=self.peer, stream=self.comm.stream)

    def _recv(self, tensor: torch.Tensor):
        self.comm.recv(tensor, src=self.peer, stream=self.comm.stream)

    def _sync(self):
        self.comm.stream.synchronize()

    # ── Target-side methods (rank=0) ──

    def send_spec_request(
        self,
        batch_size: int,
        lookahead: int,
        fan_out: int,
        vocab_size: int,
        cache_keys: torch.Tensor,  # [B, 3] int64 on device
        temperatures: torch.Tensor,  # [B] float32 on device
        seq_lens: torch.Tensor,  # [B] int64 on device
    ):
        """Target sends decode request to draft via NCCL."""
        from sglang.srt.speculative.async_spec.handshake import CMD_SPEC_REQUEST

        B = batch_size
        # Pack header
        self.request_int_buf[0] = CMD_SPEC_REQUEST
        self.request_int_buf[1] = B
        self.request_int_buf[2] = lookahead
        self.request_int_buf[3] = fan_out
        self.request_int_buf[4] = vocab_size
        # Pack cache_keys [B, 3] -> flat
        self.request_int_buf[HEADER_SIZE : HEADER_SIZE + B * 3] = (
            cache_keys.reshape(-1)
        )
        # Pack temperatures
        self.request_temp_buf[:B] = temperatures
        # Pack seq_lens
        self.seq_lens_buf[:B] = seq_lens

        # Always send full fixed-size buffers so NCCL counts match
        self.comm.group_start()
        self._send(self.request_int_buf)
        self._send(self.request_temp_buf)
        self._send(self.seq_lens_buf)
        self.comm.group_end()
        self._sync()

    def recv_speculations(
        self, batch_size: int, lookahead: int
    ) -> torch.Tensor:
        """Target receives speculation response from draft."""
        B = batch_size
        K = lookahead
        n = B * (K + 1)
        self._recv(self.response_buf[:n])
        self._sync()
        return self.response_buf[:n].reshape(B, K + 1).clone()

    def send_prefill(
        self,
        input_ids: torch.Tensor,  # [total_tokens] int64 on device
        req_pool_indices: torch.Tensor,  # [num_reqs] int64 on device
        seq_lens: torch.Tensor,  # [num_reqs] int64 on device
    ):
        """Target sends prefill data to draft via NCCL."""
        from sglang.srt.speculative.async_spec.handshake import CMD_PREFILL

        num_reqs = len(seq_lens)
        total_tokens = len(input_ids)

        # Pack header
        self.request_int_buf[0] = CMD_PREFILL
        self.request_int_buf[1] = num_reqs
        self.request_int_buf[2] = total_tokens

        # Phase 1: send full fixed-size command buffers (same as decode/exit)
        self.comm.group_start()
        self._send(self.request_int_buf)
        self._send(self.request_temp_buf)
        self._send(self.seq_lens_buf)
        self.comm.group_end()
        self._sync()

        # Phase 2: send variable-size prefill payload
        # (draft knows the size from header received in phase 1)
        payload_size = total_tokens + 2 * num_reqs
        self.prefill_buf[:total_tokens] = input_ids.to(torch.int64)
        self.prefill_buf[
            total_tokens : total_tokens + num_reqs
        ] = req_pool_indices.to(torch.int64)
        self.prefill_buf[
            total_tokens + num_reqs : total_tokens + 2 * num_reqs
        ] = seq_lens.to(torch.int64)

        self._send(self.prefill_buf[:payload_size])
        self._sync()

    def send_exit(self):
        """Target sends exit command to draft."""
        from sglang.srt.speculative.async_spec.handshake import CMD_EXIT

        self.request_int_buf[0] = CMD_EXIT
        # Send full fixed-size buffers (same pattern as decode/prefill)
        self.comm.group_start()
        self._send(self.request_int_buf)
        self._send(self.request_temp_buf)
        self._send(self.seq_lens_buf)
        self.comm.group_end()
        self._sync()

    # ── Draft-side methods (rank=1) ──

    def recv_command(self) -> int:
        """Draft receives command header, returns command code.

        Always receives the full fixed-size request_int_buf and
        request_temp_buf (grouped) so NCCL element counts match for
        all command types. For prefill, also receives the variable-size
        prefill payload in a second transfer.
        """
        from sglang.srt.speculative.async_spec.handshake import CMD_PREFILL

        # Phase 1: receive full fixed-size command buffers
        self.comm.group_start()
        self._recv(self.request_int_buf)
        self._recv(self.request_temp_buf)
        self._recv(self.seq_lens_buf)
        self.comm.group_end()
        self._sync()

        cmd = self.request_int_buf[0].item()

        if cmd == CMD_PREFILL:
            # Phase 2: receive variable-size prefill payload
            num_reqs = self.request_int_buf[1].item()
            total_tokens = self.request_int_buf[2].item()
            payload_size = total_tokens + 2 * num_reqs
            self._recv(self.prefill_buf[:payload_size])
            self._sync()

        return cmd

    def unpack_spec_request(
        self,
    ) -> Tuple[int, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draft unpacks a spec request from pre-filled buffers.

        Returns: (batch_size, lookahead, fan_out, vocab_size, cache_keys, temperatures, seq_lens)
        """
        B = self.request_int_buf[1].item()
        K = self.request_int_buf[2].item()
        fan_out = self.request_int_buf[3].item()
        vocab_size = self.request_int_buf[4].item()

        cache_keys = self.request_int_buf[HEADER_SIZE : HEADER_SIZE + B * 3].reshape(
            B, 3
        )
        temperatures = self.request_temp_buf[:B]
        seq_lens = self.seq_lens_buf[:B]

        return B, K, fan_out, vocab_size, cache_keys, temperatures, seq_lens

    def unpack_prefill(
        self,
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draft unpacks prefill data from pre-filled buffers.

        Returns: (num_reqs, total_tokens, input_ids, req_pool_indices, seq_lens)
        """
        num_reqs = self.request_int_buf[1].item()
        total_tokens = self.request_int_buf[2].item()

        input_ids = self.prefill_buf[:total_tokens]
        req_pool_indices = self.prefill_buf[
            total_tokens : total_tokens + num_reqs
        ]
        seq_lens = self.prefill_buf[
            total_tokens + num_reqs : total_tokens + 2 * num_reqs
        ]

        return num_reqs, total_tokens, input_ids, req_pool_indices, seq_lens

    def send_speculations(
        self,
        speculations: torch.Tensor,  # [B, K+1] int64 on device
    ):
        """Draft sends speculation results back to target."""
        n = speculations.numel()
        self.response_buf[:n] = speculations.reshape(-1)
        self._send(self.response_buf[:n])
        self._sync()


def create_nccl_channel(
    rank: int,
    device: torch.device,
    nccl_port: int,
    max_batch_size: int = 64,
    max_spec_k: int = 16,
    max_prefill_tokens: int = 16384,
) -> AsyncSpecNcclChannel:
    """Create an AsyncSpecNcclChannel with a new NCCL communicator.

    Sets up a StatelessProcessGroup (world_size=2) and PyNcclCommunicator,
    then wraps them in an AsyncSpecNcclChannel with pre-allocated buffers.
    """
    from sglang.srt.distributed.device_communicators.pynccl import (
        PyNcclCommunicator,
    )
    from sglang.srt.distributed.utils import StatelessProcessGroup

    logger.info(
        f"Creating NCCL channel: rank={rank}, device={device}, port={nccl_port}"
    )

    group = StatelessProcessGroup.create(
        host="localhost",
        port=nccl_port,
        rank=rank,
        world_size=2,
    )

    comm = PyNcclCommunicator(
        group=group,
        device=device,
    )
    # Enable the communicator (default is disabled)
    comm.disabled = False

    channel = AsyncSpecNcclChannel(
        comm=comm,
        rank=rank,
        device=device,
        max_batch_size=max_batch_size,
        max_spec_k=max_spec_k,
        max_prefill_tokens=max_prefill_tokens,
    )

    logger.info(f"NCCL channel created: rank={rank}")
    return channel
