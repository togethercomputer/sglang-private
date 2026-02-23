import logging
from typing import Dict

import torch
import torch.distributed as dist
from torch.distributed import TCPStore

logger = logging.getLogger(__name__)

# Dtype encoding for control tensor metadata
_DTYPE_TO_ID = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
    torch.int8: 4,
    torch.int16: 5,
    torch.int32: 6,
    torch.int64: 7,
    torch.bool: 8,
    torch.uint8: 9,
}
_ID_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ID.items()}

# Layout constants for control tensors
_MAX_TENSORS = 32
_MAX_NAME_LEN = 48
_MAX_NDIM = 8
_MAX_SCALARS = 16

# Per-tensor entry: [ndim, dtype_id, name_len, shape[0..MAX_NDIM-1], name_chars[0..MAX_NAME_LEN-1]]
_TENSOR_ENTRY_SIZE = 3 + _MAX_NDIM + _MAX_NAME_LEN
_CTRL_SIZE = 1 + _MAX_TENSORS * _TENSOR_ENTRY_SIZE

# Per-scalar entry: [value, name_chars[0..MAX_NAME_LEN-1]]
_SCALAR_ENTRY_SIZE = 1 + _MAX_NAME_LEN
_SCALAR_CTRL_SIZE = 1 + _MAX_SCALARS * _SCALAR_ENTRY_SIZE


class NcclDraftChannel:
    """Bidirectional NCCL P2P channel between local and remote draft process.

    Rank 0 = target (scheduler) side, Rank 1 = draft runner side.
    Uses pre-allocated GPU control tensors and pinned CPU staging buffers
    to avoid dynamic allocation on the critical path.
    """

    def __init__(self, group: dist.ProcessGroup, rank: int, device: torch.device):
        self.group = group
        self.rank = rank
        self.peer_rank = 1 - rank
        self.device = device

        # Pre-allocated GPU buffers for NCCL transfer
        self._send_ctrl = torch.zeros(_CTRL_SIZE, dtype=torch.int64, device=device)
        self._recv_ctrl = torch.zeros(_CTRL_SIZE, dtype=torch.int64, device=device)
        self._send_scalar_ctrl = torch.zeros(
            _SCALAR_CTRL_SIZE, dtype=torch.int64, device=device
        )
        self._recv_scalar_ctrl = torch.zeros(
            _SCALAR_CTRL_SIZE, dtype=torch.int64, device=device
        )

        # Pinned CPU staging buffers for fast metadata encoding/decoding
        self._ctrl_cpu = torch.zeros(_CTRL_SIZE, dtype=torch.int64).pin_memory()
        self._scalar_ctrl_cpu = torch.zeros(
            _SCALAR_CTRL_SIZE, dtype=torch.int64
        ).pin_memory()

    def send_tensors(self, tensor_dict: Dict[str, torch.Tensor]):
        """Send named tensors via NCCL.

        Sends a fixed-size control tensor with metadata (count, shapes,
        dtypes, names) followed by each tensor individually.
        """
        names = list(tensor_dict.keys())
        tensors = [tensor_dict[n] for n in names]
        n = len(tensors)
        assert n <= _MAX_TENSORS, f"Too many tensors: {n} > {_MAX_TENSORS}"

        # Build control metadata on CPU (fast scalar indexing)
        ctrl = self._ctrl_cpu
        ctrl.zero_()
        ctrl[0] = n

        for i, (name, t) in enumerate(zip(names, tensors)):
            base = 1 + i * _TENSOR_ENTRY_SIZE
            ctrl[base] = t.ndim
            ctrl[base + 1] = _DTYPE_TO_ID[t.dtype]
            name_len = min(len(name), _MAX_NAME_LEN)
            ctrl[base + 2] = name_len
            for d in range(t.ndim):
                ctrl[base + 3 + d] = t.shape[d]
            for j in range(name_len):
                ctrl[base + 3 + _MAX_NDIM + j] = ord(name[j])

        # Copy to GPU and send control tensor
        self._send_ctrl.copy_(ctrl, non_blocking=True)
        dist.send(self._send_ctrl, dst=self.peer_rank, group=self.group)

        # Send each tensor
        for t in tensors:
            dist.send(t.contiguous(), dst=self.peer_rank, group=self.group)

    def recv_tensors(self) -> Dict[str, torch.Tensor]:
        """Receive named tensors via NCCL.

        Receives a control tensor with metadata, then receives each tensor
        individually into freshly allocated buffers.
        """
        # Receive control tensor
        dist.recv(self._recv_ctrl, src=self.peer_rank, group=self.group)
        ctrl = self._recv_ctrl.cpu()

        n = ctrl[0].item()
        if n == 0:
            return {}

        # Parse metadata and receive each tensor
        result = {}
        for i in range(n):
            base = 1 + i * _TENSOR_ENTRY_SIZE
            ndim = ctrl[base].item()
            dtype = _ID_TO_DTYPE[ctrl[base + 1].item()]
            name_len = ctrl[base + 2].item()
            shape = tuple(ctrl[base + 3 + d].item() for d in range(ndim))
            name = "".join(
                chr(ctrl[base + 3 + _MAX_NDIM + j].item()) for j in range(name_len)
            )

            t = torch.empty(shape, dtype=dtype, device=self.device)
            dist.recv(t, src=self.peer_rank, group=self.group)
            result[name] = t

        return result

    def send_scalars(self, scalars: Dict[str, int]):
        """Send small scalar metadata via a control tensor."""
        n = len(scalars)
        assert n <= _MAX_SCALARS, f"Too many scalars: {n} > {_MAX_SCALARS}"

        ctrl = self._scalar_ctrl_cpu
        ctrl.zero_()
        ctrl[0] = n

        for i, (name, value) in enumerate(scalars.items()):
            base = 1 + i * _SCALAR_ENTRY_SIZE
            ctrl[base] = value
            name_len = min(len(name), _MAX_NAME_LEN)
            for j in range(name_len):
                ctrl[base + 1 + j] = ord(name[j])

        self._send_scalar_ctrl.copy_(ctrl, non_blocking=True)
        dist.send(self._send_scalar_ctrl, dst=self.peer_rank, group=self.group)

    def recv_scalars(self) -> Dict[str, int]:
        """Receive scalar metadata."""
        dist.recv(self._recv_scalar_ctrl, src=self.peer_rank, group=self.group)
        ctrl = self._recv_scalar_ctrl.cpu()

        n = ctrl[0].item()
        result = {}

        for i in range(n):
            base = 1 + i * _SCALAR_ENTRY_SIZE
            value = ctrl[base].item()
            chars = []
            for j in range(_MAX_NAME_LEN):
                c = ctrl[base + 1 + j].item()
                if c == 0:
                    break
                chars.append(chr(c))
            result["".join(chars)] = value

        return result


def create_nccl_channel(
    rank: int,
    device: torch.device,
    nccl_port: int,
    max_batch_size: int,
    max_spec_k: int,
    max_prefill_tokens: int = 16384,
) -> NcclDraftChannel:
    """Create a 2-rank NCCL process group and wrap it in an NcclDraftChannel.

    Both the target side (rank=0) and draft side (rank=1) must call this
    function with the same ``nccl_port`` for the rendezvous to complete.

    Args:
        rank: 0 for target (scheduler) side, 1 for draft runner side.
        device: CUDA device for this rank.
        nccl_port: TCP port for NCCL rendezvous via TCPStore.
        max_batch_size: Maximum concurrent requests (unused, reserved for
            future buffer pre-allocation).
        max_spec_k: Number of speculative draft steps (unused, reserved for
            future buffer pre-allocation).
        max_prefill_tokens: Maximum prefill tokens per batch (unused, reserved
            for future buffer pre-allocation).

    Returns:
        An initialized NcclDraftChannel ready for send/recv operations.
    """
    from sglang.srt.utils.common import init_custom_process_group

    store = TCPStore(
        host_name="127.0.0.1",
        port=nccl_port,
        world_size=2,
        is_master=(rank == 0),
    )

    with torch.cuda.device(device):
        pg = init_custom_process_group(
            backend="nccl",
            store=store,
            world_size=2,
            rank=rank,
            group_name="async_spec",
        )

    channel = NcclDraftChannel(group=pg, rank=rank, device=device)

    # Warmup: exchange a small tensor to trigger NCCL communicator init
    # and verify both sides are connected.
    warmup = torch.zeros(1, dtype=torch.int32, device=device)
    if rank == 0:
        dist.send(warmup, dst=1, group=pg)
        dist.recv(warmup, src=1, group=pg)
    else:
        dist.recv(warmup, src=0, group=pg)
        dist.send(warmup, dst=0, group=pg)
    torch.cuda.synchronize(device)

    logger.info(
        f"NcclDraftChannel ready: rank={rank}, device={device}, port={nccl_port}"
    )
    return channel
