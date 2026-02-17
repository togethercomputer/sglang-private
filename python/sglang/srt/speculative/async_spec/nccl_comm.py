import torch


class NcclDraftChannel:
    """Bidirectional NCCL P2P channel between local and remote draft process."""

    def __init__(self, rank, device, nccl_port, max_batch_size, max_spec_steps,
                hidden_size, vocab_size, max_context_len, max_num_draft_tokens):
        pass
        # Create a 2-rank NCCL process group (rank 0 = local, rank 1 = remote)
        # Pre-allocate fixed-size send/recv buffers for each operation to avoid
        # dynamic allocation on the critical path.

    def send_tensors(self, tensor_dict: dict[str, torch.Tensor]):
        """Send named tensors via NCCL. Packs into pre-allocated buffer."""
        # 1. Send metadata (num tensors, shapes, dtypes) via a small control tensor
        # 2. Pack all tensors contiguously into send_buffer
        # 3. dist.send(send_buffer, dst=peer_rank, group=self.group)

    def recv_tensors(self) -> dict[str, torch.Tensor]:
        """Receive named tensors via NCCL."""
        # 1. Receive control tensor with metadata
        # 2. dist.recv(recv_buffer, src=peer_rank, group=self.group)
        # 3. Unpack tensors from buffer

    def send_scalars(self, scalars: dict[str, int]):
        """Send small scalar metadata via a control tensor."""

    def recv_scalars(self) -> dict[str, int]:
        """Receive scalar metadata."""
