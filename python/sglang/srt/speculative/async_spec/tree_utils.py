"""Tree construction helpers for async speculative decoding.

Adapted from SSD's async_spec_helpers.py. These utilities help the draft
runner construct and manage the tree of speculative continuations.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def make_glue_decode_input_ids(
    draft_tokens: torch.Tensor,
    rec_tokens: torch.Tensor,
) -> torch.Tensor:
    """Build flat input IDs for the glue decode step.

    The glue decode runs K+1 tokens in parallel per sequence:
    - The recovery token (position 0)
    - K speculated tokens (positions 1..K)

    Args:
        draft_tokens: [B, K] speculated draft tokens
        rec_tokens: [B] recovery tokens from target verification

    Returns:
        glue_input_ids: [B*(K+1)] flat concatenation of rec_tokens and draft_tokens
    """
    assert draft_tokens.shape[0] == rec_tokens.shape[0]
    return torch.cat([rec_tokens.unsqueeze(1), draft_tokens], dim=1).view(-1)


@torch.inference_mode()
def get_forked_recovery_tokens_from_logits(
    logits: torch.Tensor,
    fan_out_list: list,
    cache_hits: torch.Tensor,
    returned_tokens: torch.Tensor,
    fan_out_list_miss: Optional[list] = None,
) -> torch.Tensor:
    """Extract forked alternative tokens from glue decode logits.

    After the glue decode, we sample top-k alternative tokens at each
    position to form the tree branches. Cache hits use fan_out_list,
    cache misses use fan_out_list_miss.

    At each position j (0..K-1), the token at position j+1 (returned_tokens[:, j+1])
    is suppressed (set to -inf) so we fork *alternatives* to what was already speculated.
    Position K (last) is not suppressed since there's no already-selected next token.

    Adapted from SSD's get_forked_recovery_tokens_from_logits.

    Args:
        logits: [B, K+1, V] logits from glue decode
        fan_out_list: [K+1] per-depth fan-out for cache hits
        cache_hits: [B] whether each req hit the cache
        returned_tokens: [B, K+1] the glue decode input tokens (to suppress)
        fan_out_list_miss: [K+1] per-depth fan-out for cache misses (defaults to fan_out_list)

    Returns:
        forked_tokens: [B, MQ_LEN] alternative tokens, where MQ_LEN = sum(fan_out_list)
    """
    if fan_out_list_miss is None:
        fan_out_list_miss = fan_out_list

    B, K_plus_1, V = logits.shape
    K = K_plus_1 - 1
    device = logits.device

    assert cache_hits.shape == (B,), f"cache_hits must have shape (B,), got {cache_hits.shape}"
    assert len(fan_out_list) == K + 1, f"fan_out_list must have length K+1={K+1}, got {len(fan_out_list)}"
    assert returned_tokens.shape == (B, K + 1), f"returned_tokens must have shape (B, K+1), got {returned_tokens.shape}"

    # Suppress returned tokens at positions 0..K-1 to avoid forking the same tokens.
    # At position j, suppress returned_tokens[:, j+1] (the token that was already selected
    # as the next token from this position).
    logits = logits.clone()
    logits[:, :-1, :] = logits[:, :-1, :].scatter(
        dim=2,
        index=returned_tokens[:, 1:].unsqueeze(2),
        value=float("-inf"),
    )

    # Compute top-k once at max fan-out, then mask per batch/position
    k_max = max(max(fan_out_list), max(fan_out_list_miss))
    _, topk_idx = torch.topk(logits, k_max, dim=-1)  # [B, K+1, k_max]

    # Build per-batch, per-position counts based on cache_hits
    hit_counts = torch.as_tensor(fan_out_list, device=device, dtype=torch.int64)  # [K+1]
    miss_counts = torch.as_tensor(fan_out_list_miss, device=device, dtype=torch.int64)  # [K+1]
    ch_bool = cache_hits.to(torch.bool).view(B, 1)  # [B, 1]
    counts_b = torch.where(
        ch_bool,
        hit_counts.view(1, -1).expand(B, -1),
        miss_counts.view(1, -1).expand(B, -1),
    )  # [B, K+1]

    # Build mask: for each (b, pos), keep the first counts_b[b, pos] top-k indices
    ar = torch.arange(k_max, device=device)
    mask = ar.view(1, 1, -1) < counts_b.view(B, K + 1, 1)  # [B, K+1, k_max]

    # Select and reshape to [B, MQ_LEN]
    mq_len = sum(fan_out_list)
    idxs_flat = topk_idx.masked_select(mask).view(B, mq_len)

    return idxs_flat


def apply_sampler_x_rescaling(
    probs: torch.Tensor,
    sampler_x: float,
    fan_out: int,
) -> torch.Tensor:
    """Apply rescaling to draft distribution probabilities.

    Rescales the probability distribution to make it sharper or flatter,
    which affects the diversity of tree branches.

    Args:
        probs: [B, V] probability distribution
        sampler_x: rescaling exponent (>1 sharper, <1 flatter)
        fan_out: number of branches (unused, kept for API compat)

    Returns:
        rescaled_probs: [B, V] rescaled probabilities
    """
    if sampler_x is None or sampler_x == 1.0:
        return probs

    rescaled = probs.pow(sampler_x)
    rescaled = rescaled / rescaled.sum(dim=-1, keepdim=True)
    return rescaled


@torch.inference_mode()
def compute_tree_lookahead(mq_len: int, K: int) -> int:
    """Compute total KV positions needed for glue decode + tree decode.

    Returns K+1 (glue decode) + K*MQ_LEN (tree decode steps).
    """
    return K + 1 + K * mq_len
