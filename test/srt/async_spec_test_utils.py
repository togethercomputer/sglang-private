"""Test utilities for async speculative decoding.

Contains a standalone verify function for unit testing the acceptance/rejection
logic without requiring the full sglang runtime infrastructure.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _sample_from_logits(
    logits: torch.Tensor, temperature: torch.Tensor
) -> torch.Tensor:
    B = logits.shape[0]
    is_greedy = temperature <= 0

    if is_greedy.all():
        return logits.argmax(dim=-1)

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
    """Standalone verify for unit tests. See docstring in original verify.py."""
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
                target_token = logits_p[b, k].argmax().item()

                if k == 0:
                    accepted.append(draft_token)
                    if target_token != draft_token:
                        recovery_token = target_token
                        break
                else:
                    if target_token == draft_token:
                        accepted.append(draft_token)
                    else:
                        recovery_token = target_token
                        break
            else:
                if k == 0:
                    accepted.append(draft_token)
                    continue

                p_logits = logits_p[b, k]
                q_logits = logits_q[b, k - 1]

                p_probs = F.softmax(p_logits / temp_target, dim=-1)
                q_probs = F.softmax(q_logits / temp_draft, dim=-1)

                p_token = p_probs[draft_token].item()
                q_token = q_probs[draft_token].item()

                if q_token <= 0:
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

        if recovery_token is None:
            recovery_token = _sample_from_logits(
                logits_p[b, K].unsqueeze(0),
                temperatures_target[b : b + 1],
            ).item()

        accepted_suffixes.append(accepted[1:] if len(accepted) > 1 else [])
        recovery_tokens.append(recovery_token)

    return accepted_suffixes, recovery_tokens
