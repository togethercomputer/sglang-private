from __future__ import annotations

import pytest

import torch

from types import SimpleNamespace

from sglang.kernels.ops.speculative.dspark.dspark_accept import accept_sampling
from sglang.srt.speculative import dflash_worker_v2
from sglang.srt.speculative.dflash_worker_v2 import (
    DFlashDraftProposal,
    DFlashWorkerV2,
)


class FakeTpGroup:
    def __init__(self, gathered: torch.Tensor | None = None) -> None:
        self.world_size = 1 if gathered is None else 2
        self.gathered = gathered

    def all_gather(self, value: torch.Tensor, dim: int = -1) -> torch.Tensor:
        del dim
        return value if self.gathered is None else self.gathered.expand(value.shape[0], -1)


class FakeTpSync:
    def sync(self, site: object, value: torch.Tensor) -> torch.Tensor:
        del site
        return value


class FakeHead(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, vocab_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.org_vocab_size = vocab_size


def make_sampling_info(
    *,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    any_greedy: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        temperatures=temperatures,
        top_ks=top_ks,
        is_all_greedy=bool(torch.all(top_ks <= 1)),
        is_any_greedy=any_greedy,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
    )


def make_worker(block_size: int) -> DFlashWorkerV2:
    worker = object.__new__(DFlashWorkerV2)
    worker.block_size = block_size
    worker._tp_sync = FakeTpSync()
    return worker


@pytest.fixture(autouse=True)
def patch_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    set_parallel(monkeypatch, FakeTpGroup())


def set_parallel(monkeypatch: pytest.MonkeyPatch, group: FakeTpGroup) -> None:
    monkeypatch.setattr(
        dflash_worker_v2,
        "get_parallel",
        lambda: SimpleNamespace(tp_group=group),
    )


def test_dflash_proposals_sample_the_retained_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_parallel(monkeypatch, FakeTpGroup())
    torch.manual_seed(17)

    batch_size = 20000
    logits = torch.tensor([1.2, 0.3, -0.7], dtype=torch.float32)
    hidden = logits.expand(batch_size, -1).contiguous()
    head = FakeHead(torch.eye(3, dtype=torch.float32), vocab_size=3)
    sampling_info = make_sampling_info(
        temperatures=torch.ones(batch_size, 1),
        top_ks=torch.full((batch_size,), 1 << 30, dtype=torch.int32),
    )

    proposal = make_worker(2).sample_from_vocab_parallel_head(
        hidden_states=hidden,
        lm_head=head,
        batch_size=batch_size,
        sampling_info=sampling_info,
    )

    expected = torch.softmax(logits, dim=-1)
    observed = torch.bincount(proposal.tokens.flatten(), minlength=3) / batch_size
    torch.testing.assert_close(proposal.logits[0, 0], logits)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0.012)


def test_dflash_proposal_gathers_and_crops_tp_vocab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gathered = torch.tensor([[0.1, 0.2, 4.0, 9.0]], dtype=torch.float32)
    set_parallel(monkeypatch, FakeTpGroup(gathered))
    head = FakeHead(torch.tensor([[1.0], [2.0]]), vocab_size=3)
    head.shard_indices = SimpleNamespace(num_added_elements=0)
    sampling_info = make_sampling_info(
        temperatures=torch.ones(1, 1),
        top_ks=torch.ones(1, dtype=torch.int32),
        any_greedy=True,
    )

    proposal = make_worker(2).sample_from_vocab_parallel_head(
        hidden_states=torch.ones(1, 1),
        lm_head=head,
        batch_size=1,
        sampling_info=sampling_info,
    )

    torch.testing.assert_close(proposal.logits[0, 0], gathered[0, :3])
    assert proposal.tokens.item() == 2


def test_dflash_accept_receives_q_from_the_current_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, torch.Tensor] = {}

    def fake_accept_sampling(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        captured["draft_probs"] = kwargs["draft_probs"].clone()
        return torch.tensor([1]), torch.tensor([2]), torch.tensor([0])

    monkeypatch.setattr(dflash_worker_v2, "accept_sampling", fake_accept_sampling)
    logits = torch.tensor([[[2.0, 1.0, -1.0]]])
    proposal = DFlashDraftProposal(
        tokens=torch.tensor([[0]]),
        logits=logits,
        greedy_mask=torch.tensor([False]),
        temperatures=torch.tensor([2.0]),
    )
    sampling_info = make_sampling_info(
        temperatures=torch.tensor([[2.0]]),
        top_ks=torch.tensor([1 << 30], dtype=torch.int32),
    )

    accept_len, bonus = make_worker(2).accept_rejection_sampled_draft(
        candidates=torch.tensor([[2, 0]]),
        next_token_logits=torch.zeros(2, 3),
        proposal=proposal,
        sampling_info=sampling_info,
        draft_input=SimpleNamespace(),
    )

    expected = torch.softmax(logits.float() / 2.0, dim=-1)
    torch.testing.assert_close(captured["draft_probs"], expected)
    torch.testing.assert_close(accept_len, torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(bonus, torch.tensor([2], dtype=torch.int64))


def test_dflash_mixed_batch_keeps_greedy_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_accept_sampling(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del kwargs
        return torch.tensor([0, 1]), torch.tensor([4, 5]), torch.tensor([0, 0])

    def fake_greedy(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        return torch.tensor([1, 0]), torch.tensor([6, 7])

    monkeypatch.setattr(dflash_worker_v2, "accept_sampling", fake_accept_sampling)
    monkeypatch.setattr(
        dflash_worker_v2,
        "compute_dflash_correct_drafts_and_bonus",
        fake_greedy,
    )
    proposal = DFlashDraftProposal(
        tokens=torch.tensor([[1], [2]]),
        logits=torch.zeros(2, 1, 8),
        greedy_mask=torch.tensor([True, False]),
        temperatures=torch.ones(2),
    )
    sampling_info = make_sampling_info(
        temperatures=torch.ones(2, 1),
        top_ks=torch.tensor([1, 1 << 30], dtype=torch.int32),
        any_greedy=True,
    )

    accept_len, bonus = make_worker(2).accept_rejection_sampled_draft(
        candidates=torch.tensor([[0, 1], [0, 2]]),
        next_token_logits=torch.zeros(4, 8),
        proposal=proposal,
        sampling_info=sampling_info,
        draft_input=SimpleNamespace(),
    )

    torch.testing.assert_close(accept_len, torch.tensor([1, 1], dtype=torch.int32))
    torch.testing.assert_close(bonus, torch.tensor([6, 5], dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dflash_cuda_proposal_and_rejection_match_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = FakeTpGroup()
    set_parallel(monkeypatch, group)
    torch.manual_seed(23)
    device = torch.device("cuda")
    batch_size = 40000
    q = torch.tensor([0.7, 0.3], dtype=torch.float32, device=device)
    p = torch.tensor([0.2, 0.8], dtype=torch.float32, device=device)
    head = FakeHead(torch.eye(2, dtype=torch.float32, device=device), vocab_size=2)
    sampling_info = make_sampling_info(
        temperatures=torch.ones(batch_size, 1, device=device),
        top_ks=torch.full(
            (batch_size,),
            1 << 30,
            dtype=torch.int32,
            device=device,
        ),
    )
    worker = make_worker(2)
    proposal = worker.sample_from_vocab_parallel_head(
        hidden_states=q.log().expand(batch_size, -1).contiguous(),
        lm_head=head,
        batch_size=batch_size,
        sampling_info=sampling_info,
    )
    candidates = torch.stack(
        (torch.zeros_like(proposal.tokens.flatten()), proposal.tokens.flatten()),
        dim=1,
    )
    target_rows = torch.stack((p.log(), torch.tensor([0.5, 0.5], device=device).log()))
    target_logits = (
        target_rows.unsqueeze(0)
        .expand(batch_size, -1, -1)
        .reshape(batch_size * 2, 2)
        .contiguous()
    )
    accept_len, bonus = worker.accept_rejection_sampled_draft(
        candidates=candidates,
        next_token_logits=target_logits,
        proposal=proposal,
        sampling_info=sampling_info,
        draft_input=SimpleNamespace(),
    )
    proposals = proposal.tokens.flatten()
    proposal_observed = torch.bincount(proposals, minlength=2).float() / batch_size
    first_tokens = torch.where(accept_len > 0, proposals, bonus)
    observed = torch.bincount(first_tokens, minlength=2).float() / batch_size
    torch.testing.assert_close(proposal_observed, q, rtol=0, atol=0.015)
    torch.testing.assert_close(observed, p, rtol=0, atol=0.015)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dflash_rejection_one_hot_limit_accepts_and_rejects_exactly() -> None:
    device = torch.device("cuda")
    candidates = torch.tensor([[1, 0], [1, 0]], dtype=torch.int64, device=device)
    target_logits = torch.tensor(
        [
            [0.0, float("-inf")],
            [float("-inf"), 0.0],
            [float("-inf"), 0.0],
            [float("-inf"), 0.0],
        ],
        device=device,
    )
    draft_probs = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], device=device)
    sampling_info = SimpleNamespace(
        temperatures=torch.ones(2, 1, device=device),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
    )

    accept_len, bonus, unused_trim = accept_sampling(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=SimpleNamespace(),
        gamma=1,
        verify_num_draft_tokens=2,
        cutoff_verify_lens=None,
    )
    del unused_trim
    expected_len = torch.tensor([1, 0], dtype=torch.int32, device=device)
    torch.testing.assert_close(accept_len, expected_len)
    assert bonus[0].item() == 1
    assert bonus[1].item() == 1
