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

    This allows the draft model to process the full speculated sequence
    and then branch into alternative continuations.

    Args:
        draft_tokens: [B, K] speculated draft tokens
        rec_tokens: [B] recovery tokens from target verification

    Returns:
        glue_input_ids: [B, K+1] concatenation of rec_tokens and draft_tokens
    """
    return torch.cat([rec_tokens.unsqueeze(1), draft_tokens], dim=1)


def get_forked_recovery_tokens_from_logits(
    logits: torch.Tensor,
    fan_out_list: list,
    temperatures: torch.Tensor,
    cache_hits: torch.Tensor,
    fan_out_list_miss: Optional[list] = None,
) -> torch.Tensor:
    """Extract forked alternative tokens from glue decode logits.

    After the glue decode, we sample multiple alternative tokens at each
    position to form the tree branches. Cache hits use fan_out_list,
    cache misses use fan_out_list_miss.

    Args:
        logits: [B, K+1, V] logits from glue decode
        fan_out_list: per-depth fan-out for cache hits
        temperatures: [B] sampling temperatures
        cache_hits: [B] whether each req hit the cache
        fan_out_list_miss: per-depth fan-out for cache misses

    Returns:
        forked_tokens: [B, K, max_fan_out] alternative tokens per position
    """
    if fan_out_list_miss is None:
        fan_out_list_miss = fan_out_list

    B, K_plus_1, V = logits.shape
    K = K_plus_1 - 1
    max_fan_out = max(max(fan_out_list), max(fan_out_list_miss))
    device = logits.device

    forked_tokens = torch.zeros(B, K, max_fan_out, dtype=torch.long, device=device)

    for b in range(B):
        is_hit = cache_hits[b].item() > 0 if cache_hits is not None else True
        fo_list = fan_out_list if is_hit else fan_out_list_miss
        temp = temperatures[b].item()

        for k in range(K):
            fo = fo_list[k] if k < len(fo_list) else 1
            pos_logits = logits[b, k + 1]  # Skip recovery position

            if temp <= 0:
                # Greedy: take top-fo tokens
                topk_vals, topk_ids = torch.topk(pos_logits, min(fo, V))
                forked_tokens[b, k, :fo] = topk_ids[:fo]
            else:
                # Stochastic: sample fo tokens
                probs = F.softmax(pos_logits / temp, dim=-1)
                sampled = torch.multinomial(
                    probs, num_samples=min(fo, V), replacement=False
                )
                forked_tokens[b, k, :fo] = sampled[:fo]

    return forked_tokens


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
