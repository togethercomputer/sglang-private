"""Tree construction helpers for async speculative decoding.

Adapted from SSD's async_spec_helpers.py. These utilities help the draft
runner construct and manage the tree of speculative continuations.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def make_glue_decode_input_ids(
    draft_tokens: torch.Tensor,
    rec_tokens: torch.Tensor,
) -> torch.Tensor:
    """Build input IDs for the glue decode step.

    The glue decode runs K+1 tokens in parallel per sequence:
    - The recovery token (position 0)
    - K speculated tokens (positions 1..K)

    Args:
        draft_tokens: [B, K] speculated draft tokens
        rec_tokens: [B] recovery tokens from target verification

    Returns:
        glue_input_ids: [B*(K+1)] flat concatenation matching SSD convention
    """
    return torch.cat([rec_tokens.unsqueeze(1), draft_tokens], dim=1).view(-1)


def get_forked_recovery_tokens_from_logits(
    logits: torch.Tensor,
    fan_out_list: list,
    temperatures: torch.Tensor,
    cache_hits: torch.Tensor,
    fan_out_list_miss: Optional[list] = None,
    returned_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Extract forked alternative tokens from glue decode logits.

    After the glue decode, we sample multiple alternative tokens at each
    position to form the tree branches. Cache hits use fan_out_list,
    cache misses use fan_out_list_miss.

    Matches SSD's vectorized implementation:
    - Masks returned_tokens with -inf to avoid re-selecting chain tokens
    - Uses topk over all K+1 positions simultaneously
    - Returns flat [B, MQ_LEN] output

    Args:
        logits: [B, K+1, V] logits from glue decode
        fan_out_list: per-depth fan-out for cache hits (K+1 entries)
        temperatures: [B] sampling temperatures
        cache_hits: [B] whether each req hit the cache
        fan_out_list_miss: per-depth fan-out for cache misses (K+1 entries)
        returned_tokens: [B, K+1] tokens already in the chain to exclude

    Returns:
        forked_tokens: [B, MQ_LEN] flat alternative tokens per position
    """
    if fan_out_list_miss is None:
        fan_out_list_miss = fan_out_list

    B, K_plus_1, V = logits.shape
    device = logits.device

    assert len(fan_out_list) == K_plus_1, (
        f"fan_out_list must have K+1={K_plus_1} entries, got {len(fan_out_list)}"
    )
    assert len(fan_out_list_miss) == K_plus_1, (
        f"fan_out_list_miss must have K+1={K_plus_1} entries, got {len(fan_out_list_miss)}"
    )

    # Mask returned tokens with -inf to avoid re-selecting chain tokens.
    # Don't touch the last position (K), only scatter on positions 0..K-1.
    # At position d, mask out returned_tokens[:, d+1] (the next-chain token).
    logits = logits.clone()
    if returned_tokens is not None:
        assert returned_tokens.shape == (B, K_plus_1), (
            f"returned_tokens must be (B, K+1), got {returned_tokens.shape}"
        )
        logits[:, :-1, :] = logits[:, :-1, :].scatter(
            dim=2,
            index=returned_tokens[:, 1:].unsqueeze(2),
            value=float("-inf"),
        )

    # Compute top-k once at max fanout, then mask per row/position
    k_max = max(max(fan_out_list), max(fan_out_list_miss))
    _, topk_idx = torch.topk(logits, k_max, dim=-1)  # [B, K+1, k_max]

    # Build per-b, per-(K+1) counts depending on cache_hits
    hit_counts = torch.as_tensor(fan_out_list, device=device, dtype=torch.int64)
    miss_counts = torch.as_tensor(fan_out_list_miss, device=device, dtype=torch.int64)
    ch_bool = cache_hits.to(torch.bool).view(B, 1)
    counts_b = torch.where(
        ch_bool,
        hit_counts.view(1, -1).expand(B, -1),
        miss_counts.view(1, -1).expand(B, -1),
    )  # [B, K+1]

    # Build mask: [B, K+1, k_max]
    ar = torch.arange(k_max, device=device)
    mask = ar.view(1, 1, -1) < counts_b.view(B, K_plus_1, 1)

    # Select tokens using mask -> [B, MQ_LEN]
    idxs_flat = topk_idx.masked_select(mask).view(B, -1)

    return idxs_flat


def compute_mq_len(fan_out_list: list) -> int:
    """Compute the total number of tree query tokens per sequence.

    MQ_LEN = sum of fan_out values across all depths. This is the number
    of alternative continuation tokens generated at each tree decode step.

    Args:
        fan_out_list: per-depth fan-out counts (e.g. [2, 2, 1] for K+1 depths)

    Returns:
        Total number of tree query tokens per sequence.
    """
    return sum(fan_out_list)


def apply_sampler_x_rescaling(
    probs: torch.Tensor,
    sampler_x: float,
    fan_out: int,
) -> torch.Tensor:
    """Apply rescaling to draft distribution probabilities.

    Matches SSD: multiplies top-(F+1) probabilities by sampler_x, then
    renormalizes. This makes the top tokens more or less likely relative
    to the rest.

    Args:
        probs: [..., V] probability distribution (any leading dims)
        sampler_x: rescaling factor for top-(fan_out+1) probabilities
        fan_out: number of branches used to determine top-k size

    Returns:
        rescaled_probs: [..., V] rescaled probabilities
    """
    if sampler_x is None or sampler_x == 1.0:
        return probs

    # Find top-(F+1) indices
    _, topk_indices = torch.topk(probs, fan_out + 1, dim=-1)

    # Create a mask for top positions
    topf_mask = torch.zeros_like(probs, dtype=torch.bool)
    topf_mask.scatter_(dim=-1, index=topk_indices, value=True)

    # Rescale top probs by sampler_x factor
    probs = torch.where(topf_mask, probs * sampler_x, probs)

    # Renormalize
    probs = probs / probs.sum(dim=-1, keepdim=True)

    return probs
