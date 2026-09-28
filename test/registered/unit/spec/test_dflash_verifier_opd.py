from __future__ import annotations

import pytest

import torch

from types import SimpleNamespace

from sglang.srt.speculative import dflash_worker_v2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2


def make_request(rid: str) -> SimpleNamespace:
    return SimpleNamespace(
        rid=rid,
        verifier_opd_stored=False,
        verifier_opd_context_hidden_states=None,
        verifier_opd_anchor_token_id=None,
        verifier_opd_hidden_states=None,
        verifier_opd_target=None,
    )


def make_training(*rids: str) -> SimpleNamespace:
    selected = set(rids)
    return SimpleNamespace(has_request=lambda rid: rid in selected)


@pytest.fixture(autouse=True)
def patch_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    parallel = SimpleNamespace(attn_tp_rank=0)
    monkeypatch.setattr(dflash_worker_v2, "get_parallel", lambda: parallel)


def test_verifier_opd_prefill_aligns_each_training_request() -> None:
    worker = object.__new__(DFlashWorkerV2)
    first = make_request("first")
    second = make_request("second")
    batch = SimpleNamespace(
        spec_training_info=make_training("second"),
        reqs=[first, second],
        extend_lens=[2, 3],
        prefix_lens=[4, 0],
    )
    features = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    output = SimpleNamespace(hidden_states=features)
    anchors = torch.tensor([7, 8])

    worker.capture_verifier_opd_prefill(
        batch=batch,
        logits_output=output,
        next_token_ids=anchors,
    )

    assert first.verifier_opd_context_hidden_states is None
    assert first.verifier_opd_anchor_token_id is None
    torch.testing.assert_close(second.verifier_opd_context_hidden_states, features[2:])
    torch.testing.assert_close(second.verifier_opd_anchor_token_id, anchors[1:])


def test_verifier_opd_prefill_rejects_cached_training_prefix() -> None:
    worker = object.__new__(DFlashWorkerV2)
    req = make_request("training")
    batch = SimpleNamespace(
        spec_training_info=make_training("training"),
        reqs=[req],
        extend_lens=[2],
        prefix_lens=[1],
    )
    output = SimpleNamespace(hidden_states=torch.zeros(2, 3))

    with pytest.raises(RuntimeError, match="disable_radix_cache"):
        worker.capture_verifier_opd_prefill(
            batch=batch,
            logits_output=output,
            next_token_ids=torch.tensor([7]),
        )


def test_verifier_opd_grade_uses_the_actual_proposal_rows() -> None:
    worker = object.__new__(DFlashWorkerV2)
    req = make_request("training")
    context = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    req.verifier_opd_context_hidden_states = context
    batch = SimpleNamespace(
        spec_training_info=make_training("training"),
        reqs=[req],
    )
    aux = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
    )
    grades = torch.tensor(
        [[100.0, 101.0], [200.0, 201.0], [300.0, 301.0], [400.0, 401.0]]
    )
    output = SimpleNamespace(hidden_states=aux, last_hidden_states=grades)

    worker.capture_verifier_opd_grade(
        batch=batch,
        logits_output=output,
        batch_size=1,
        block_size=4,
    )

    expected_features = torch.cat((context, aux[:1]))
    torch.testing.assert_close(req.verifier_opd_hidden_states, expected_features)
    torch.testing.assert_close(req.verifier_opd_target, grades[:3])


def test_verifier_opd_grade_skips_nontraining_requests() -> None:
    worker = object.__new__(DFlashWorkerV2)
    req = make_request("ordinary")
    batch = SimpleNamespace(
        spec_training_info=make_training("training"),
        reqs=[req],
    )
    output = SimpleNamespace(
        hidden_states=torch.zeros(3, 2),
        last_hidden_states=torch.ones(3, 2),
    )

    worker.capture_verifier_opd_grade(
        batch=batch,
        logits_output=output,
        batch_size=1,
        block_size=3,
    )

    assert req.verifier_opd_hidden_states is None
    assert req.verifier_opd_target is None
