"""Verification algorithm for async speculative decoding.

Adapted from SSD's verify.py. This is a pure tensor function that compares
target model logits against draft model logits to determine which speculated
tokens to accept.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _sample_from_logits(
    logits: torch.Tensor, temperature: torch.Tensor
) -> torch.Tensor:
    """Sample tokens from logits with per-sequence temperature.

    Args:
        logits: [B, V] logits
        temperature: [B] temperatures (0 = greedy)

    Returns:
        sampled tokens [B]
    """
    B = logits.shape[0]
    is_greedy = temperature <= 0

    if is_greedy.all():
        return logits.argmax(dim=-1)

    # For non-greedy: apply temperature and sample
    # For greedy: just argmax
    result = torch.empty(B, dtype=torch.long, device=logits.device)

    if is_greedy.any():
        greedy_mask = is_greedy
        result[greedy_mask] = logits[greedy_mask].argmax(dim=-1)

    if (~is_greedy).any():
        non_greedy_mask = ~is_greedy
        temp = temperature[non_greedy_mask].unsqueeze(-1)
        probs = F.softmax(logits[non_greedy_mask] / temp, dim=-1)
        result[non_greedy_mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return result


def verify(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    speculations: torch.Tensor,
    temperatures_target: torch.Tensor,
    temperatures_draft: torch.Tensor,
    cache_hits: Optional[torch.Tensor] = None,
    sampler_x: Optional[float] = None,
    async_fan_out: Optional[int] = None,
    jit_speculate: bool = False,
) -> Tuple[List[List[int]], List[int]]:
    """Verify speculated tokens against target model logits.

    Uses rejection sampling: for each position, accept the draft token if
    a uniform random sample is <= p(token)/q(token), where p is the target
    distribution and q is the draft distribution.

    For greedy decoding (temperature=0), simply check if the target model's
    top token matches the draft token.

    Args:
        logits_p: [B, K+1, V] target logits (K+1 positions: recovery + K speculated)
        logits_q: [B, K, V] draft logits for each speculated position
        speculations: [B, K+1] speculated tokens (recovery + K draft tokens)
        temperatures_target: [B] target sampling temperatures
        temperatures_draft: [B] draft sampling temperatures
        cache_hits: [B] whether each req hit tree cache (int64, 0 or 1)
        sampler_x: optional rescaling factor for draft distribution
        async_fan_out: fan-out parameter (unused in basic verify)
        jit_speculate: whether JIT speculation was used on cache miss

    Returns:
        accepted_suffixes: List[List[int]] - accepted token suffixes per request
        recovery_tokens: List[int] - next recovery token per request
    """
    B, K_plus_1, V = logits_p.shape
    K = K_plus_1 - 1
    device = logits_p.device

    accepted_suffixes = []
    recovery_tokens = []

    for b in range(B):
        temp_target = temperatures_target[b].item()
        temp_draft = temperatures_draft[b].item()
        is_greedy = temp_target <= 0

        accepted = []
        recovery_token = None

        for k in range(K + 1):
            draft_token = speculations[b, k].item()

            if is_greedy:
                # Greedy verification: check if target's argmax matches draft
                target_token = logits_p[b, k].argmax().item()

                if k == 0:
                    # Position 0 is the recovery token - always "accept" it
                    # (it was the target's output from last round)
                    accepted.append(draft_token)
                    if target_token != draft_token:
                        # Recovery token mismatch means we need to use target's token
                        recovery_token = target_token
                        break
                else:
                    if target_token == draft_token:
                        accepted.append(draft_token)
                    else:
                        # Reject: use target's token as recovery
                        recovery_token = target_token
                        break
            else:
                # Stochastic verification via rejection sampling
                if k == 0:
                    # Recovery token position
                    accepted.append(draft_token)
                    continue

                # Get target and draft probabilities for the draft token
                p_logits = logits_p[b, k]
                q_logits = logits_q[b, k - 1]  # draft logits are offset by 1

                p_probs = F.softmax(p_logits / temp_target, dim=-1)
                q_probs = F.softmax(q_logits / temp_draft, dim=-1)

                p_token = p_probs[draft_token].item()
                q_token = q_probs[draft_token].item()

                if q_token <= 0:
                    # Draft assigned zero probability - reject
                    recovery_token = _sample_from_logits(
                        p_logits.unsqueeze(0),
                        temperatures_target[b : b + 1],
                    ).item()
                    break

                accept_ratio = min(1.0, p_token / q_token)
                u = torch.rand(1, device=device).item()

                if u <= accept_ratio:
                    accepted.append(draft_token)
                else:
                    # Reject: sample from adjusted distribution
                    adjusted = torch.clamp(p_probs - q_probs, min=0)
                    adjusted_sum = adjusted.sum()
                    if adjusted_sum > 0:
                        adjusted = adjusted / adjusted_sum
                        recovery_token = torch.multinomial(
                            adjusted.unsqueeze(0), num_samples=1
                        ).item()
                    else:
                        recovery_token = _sample_from_logits(
                            p_logits.unsqueeze(0),
                            temperatures_target[b : b + 1],
                        ).item()
                    break

        # If all tokens accepted, sample recovery from last position's target logits
        if recovery_token is None:
            recovery_token = _sample_from_logits(
                logits_p[b, K].unsqueeze(0),
                temperatures_target[b : b + 1],
            ).item()

        # The accepted suffix excludes the recovery token (position 0)
        accepted_suffixes.append(accepted[1:] if len(accepted) > 1 else [])
        recovery_tokens.append(recovery_token)

    return accepted_suffixes, recovery_tokens
