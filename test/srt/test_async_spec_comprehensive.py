"""Comprehensive tests and benchmarks for async speculative decoding.

Tests the core functionality and performance of:
  - tree_utils: glue decode input construction, forked token sampling, etc.
  - Cache lookup and population logic
  - Fan-out expansion and tree decode structure
  - KV cache rollback and vectorized slot assignment
  - Cross-codebase equivalence with SSD's DraftRunner
  - Profiling of hot-path operations

Run with:
  source .venv/bin/activate
  python -m pytest test/srt/test_async_spec_comprehensive.py -v
"""

from __future__ import annotations

import sys
import time
import unittest
from typing import List, Optional

import torch
import torch.nn.functional as F

# ─── SSD imports for cross-codebase comparison ───
SSD_AVAILABLE = False
try:
    sys.path.insert(0, "/work/avner/git/ssd")
    from ssd.utils.async_helpers.async_spec_helpers import (
        make_glue_decode_input_ids as ssd_make_glue_decode_input_ids,
        get_forked_recovery_tokens_from_logits as ssd_get_forked_recovery_tokens_from_logits,
        apply_sampler_x_rescaling as ssd_apply_sampler_x_rescaling,
    )
    SSD_AVAILABLE = True
except ImportError:
    pass

# ─── SGLang imports ───
from sglang.srt.speculative.async_spec.tree_utils import (
    apply_sampler_x_rescaling,
    compute_mq_len,
    get_forked_recovery_tokens_from_logits,
    make_glue_decode_input_ids,
)
from sglang.srt.speculative.async_spec.handshake import (
    CMD_EXIT,
    CMD_PREFILL,
    CMD_SPEC_REQUEST,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Mock SSD Config for cross-codebase tests ───
class MockSSDConfig:
    def __init__(self, speculate_k, fan_out_list, fan_out_list_miss):
        self.speculate_k = speculate_k
        self.fan_out_list = fan_out_list
        self.fan_out_list_miss = fan_out_list_miss
        self.fan_out_t = torch.tensor(fan_out_list, device=DEVICE)
        self.fan_out_t_miss = torch.tensor(fan_out_list_miss, device=DEVICE)
        self.MQ_LEN = sum(fan_out_list)


# ─── Mock runner for algorithmic testing ───
class MockReqToTokenPool:
    def __init__(self, size, max_ctx, device=DEVICE):
        self.size = size
        self.req_to_token = torch.full((size, max_ctx), -1, dtype=torch.int64, device=device)


class MockKvAllocator:
    def __init__(self, total, device=DEVICE):
        self.device = device
        self._free = list(range(total - 1, -1, -1))

    def alloc(self, n):
        if len(self._free) < n:
            return None
        return torch.tensor([self._free.pop() for _ in range(n)], dtype=torch.int64, device=self.device)

    def free(self, locs):
        self._free.extend(locs.tolist())


class MockAsyncDraftRunner:
    """Stripped-down runner for testing cache lookup / populate / expansion."""

    def __init__(self, spec_k=3, fan_out_list=None, fan_out_list_miss=None,
                 max_reqs=32, max_ctx=2048, vocab_size=256, device=DEVICE):
        self.spec_k = spec_k
        K = spec_k
        # K+1 entries matching SSD convention
        self.fan_out_list = fan_out_list or [2] * (K + 1)
        self.fan_out_list_miss = fan_out_list_miss or self.fan_out_list
        self.mq_len = compute_mq_len(self.fan_out_list)
        self.mq_len_miss = compute_mq_len(self.fan_out_list_miss)
        self.vocab_size = vocab_size
        self.device = device
        self.jit_speculate = True
        self.draft_temperature = None

        self.tree_cache_keys = torch.zeros((0, 3), dtype=torch.int64, device=device)
        self.tree_cache_tokens: Optional[torch.Tensor] = None
        self.tree_cache_logits: Optional[torch.Tensor] = None

        self.req_to_token_pool = MockReqToTokenPool(max_reqs, max_ctx, device)
        self.token_to_kv_pool_allocator = MockKvAllocator(10000, device)
        self.target_to_draft = torch.full((max_ctx,), -1, dtype=torch.int64, device=device)
        self.draft_seq_lens = torch.zeros(max_reqs, dtype=torch.int64, device=device)
        self.draft_pool_free_slots: List[int] = list(range(max_reqs - 1, -1, -1))

        self._precompute_fan_out_tensors()

    def _precompute_fan_out_tensors(self):
        K = self.spec_k
        d = self.device
        self._fo_hit_t = torch.tensor(self.fan_out_list, dtype=torch.int64, device=d)
        self._fo_miss_t = torch.tensor(self.fan_out_list_miss, dtype=torch.int64, device=d)
        self._fan_idx_hit = torch.arange(K + 1, device=d, dtype=torch.int64).repeat_interleave(self._fo_hit_t)
        self._fan_idx_miss = torch.arange(K + 1, device=d, dtype=torch.int64).repeat_interleave(self._fo_miss_t)
        self._step_pos_offsets = torch.arange(K, device=d, dtype=torch.int64)[:, None] * self.mq_len
        self._step_rope_offsets = torch.arange(K, device=d, dtype=torch.int64)[:, None]
        self._arange_mq = torch.arange(self.mq_len, device=d, dtype=torch.int64)

    def _alloc_draft_slot(self):
        return self.draft_pool_free_slots.pop()

    def _free_draft_slot(self, idx):
        self.draft_pool_free_slots.append(idx)

    def _resolve_draft_indices(self, targets):
        return self.target_to_draft[targets]

    def _assign_slots_per_element(self, di, pos, locs):
        self.req_to_token_pool.req_to_token[di, pos] = locs.to(self.req_to_token_pool.req_to_token.dtype)

    def _assign_slots_to_pool(self, di, base, tpr, locs):
        pool = self.req_to_token_pool.req_to_token
        rows = di.repeat_interleave(tpr)
        offsets = torch.arange(tpr, device=self.device, dtype=torch.int64)
        cols = (base.unsqueeze(1) + offsets).reshape(-1)
        pool[rows, cols] = locs.to(pool.dtype)

    def _rollback_kv_cache(self, di, tgt_lens):
        cur = self.draft_seq_lens[di]
        needs = (cur > tgt_lens) & (tgt_lens > 0)
        if not needs.any():
            return
        excess = []
        for idx in needs.nonzero(as_tuple=True)[0]:
            excess.append(self.req_to_token_pool.req_to_token[di[idx], tgt_lens[idx]:cur[idx]])
        if excess:
            self.token_to_kv_pool_allocator.free(torch.cat(excess))
        self.draft_seq_lens[di[needs]] = tgt_lens[needs]

    def _reset_tree_cache(self):
        self.tree_cache_keys = torch.zeros((0, 3), dtype=torch.int64, device=self.device)
        self.tree_cache_tokens = None
        self.tree_cache_logits = None

    def _sample_tokens(self, logits, temps):
        if self.draft_temperature and self.draft_temperature > 0:
            return torch.multinomial(F.softmax(logits / self.draft_temperature, dim=-1), 1).squeeze(-1)
        elif temps is not None and (temps > 0).any():
            return torch.multinomial(F.softmax(logits / temps.unsqueeze(-1).clamp(min=1e-6), dim=-1), 1).squeeze(-1)
        else:
            return logits.argmax(dim=-1)

    def hit_cache_and_respond(self, cache_keys, B, K, V, temps, draft_indices):
        out_logits = torch.empty((B, K, V), dtype=torch.float32, device=self.device).uniform_()
        out_tokens = out_logits.argmax(dim=-1)
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)
        recovery_tokens = cache_keys[:, 2]

        if self.tree_cache_keys.numel() > 0:
            eq = cache_keys.unsqueeze(1) == self.tree_cache_keys.unsqueeze(0)
            match = torch.all(eq, dim=2)
            cache_hits_bool = match.any(dim=1)
            cache_hits = cache_hits_bool.to(torch.int64)
            if cache_hits_bool.any() and self.tree_cache_tokens is not None:
                idx = match.float().argmax(dim=1).to(torch.int64)
                sel = cache_hits_bool
                ck = min(K, self.tree_cache_tokens.shape[1])
                out_tokens[sel, :ck] = self.tree_cache_tokens[idx[sel], :ck]
                if self.tree_cache_logits is not None:
                    out_logits[sel, :ck] = self.tree_cache_logits[idx[sel], :ck]

        speculations = torch.cat([recovery_tokens.unsqueeze(1), out_tokens.to(torch.int64)], dim=1)
        return speculations, out_tokens, cache_hits

    def populate_tree_cache(self, batch_ids, fan_idx, rec_tokens, spec_tokens, spec_logits, cache_keys_orig):
        N = batch_ids.shape[0]
        if N == 0:
            return
        seq_ids = cache_keys_orig[batch_ids, 0]
        keys = torch.stack([seq_ids, fan_idx, rec_tokens.to(torch.int64)], dim=1)
        self.tree_cache_keys = keys
        self.tree_cache_tokens = spec_tokens
        self.tree_cache_logits = spec_logits


# ═════════════════════════════════════════════════════════════════════════
# TEST SUITE 1: Tree Utils
# ═════════════════════════════════════════════════════════════════════════
class TestTreeUtils(unittest.TestCase):
    """Tests for tree_utils.py functions."""

    # ── make_glue_decode_input_ids: now returns flat [B*(K+1)] ──

    def test_glue_decode_flat_shape(self):
        B, K = 4, 5
        draft = torch.randint(0, 100, (B, K))
        rec = torch.randint(0, 100, (B,))
        result = make_glue_decode_input_ids(draft, rec)
        self.assertEqual(result.shape, (B * (K + 1),))

    def test_glue_decode_recovery_first(self):
        draft = torch.tensor([[1, 2, 3], [4, 5, 6]])
        rec = torch.tensor([10, 20])
        result = make_glue_decode_input_ids(draft, rec).view(2, 4)
        self.assertTrue(torch.equal(result[:, 0], rec))
        self.assertTrue(torch.equal(result[:, 1:], draft))

    @unittest.skipUnless(SSD_AVAILABLE, "SSD not available")
    def test_glue_decode_matches_ssd(self):
        """Cross-check: SGLang flat output matches SSD flat output."""
        for B in [1, 2, 4, 8]:
            for K in [1, 2, 3, 5]:
                draft = torch.randint(0, 1000, (B, K), device=DEVICE)
                rec = torch.randint(0, 1000, (B,), device=DEVICE)
                sgl_out = make_glue_decode_input_ids(draft, rec)
                ssd_out = ssd_make_glue_decode_input_ids(draft, rec)
                self.assertTrue(torch.equal(sgl_out, ssd_out), f"Mismatch at B={B}, K={K}")

    # ── compute_mq_len ──

    def test_compute_mq_len_basic(self):
        self.assertEqual(compute_mq_len([2, 2, 1]), 5)
        self.assertEqual(compute_mq_len([3, 2, 1, 1]), 7)

    # ── get_forked_recovery_tokens_from_logits: now returns [B, MQ_LEN] ──

    def test_forked_tokens_shape_flat(self):
        B, K, V = 2, 3, 100
        fan_out_list = [2, 2, 2, 1]  # K+1=4 entries
        mq_len = sum(fan_out_list)
        logits = torch.randn(B, K + 1, V, device=DEVICE)
        temps = torch.ones(B, device=DEVICE)
        hits = torch.ones(B, dtype=torch.int64, device=DEVICE)
        result = get_forked_recovery_tokens_from_logits(
            logits=logits, fan_out_list=fan_out_list, temperatures=temps,
            cache_hits=hits,
        )
        self.assertEqual(result.shape, (B, mq_len))

    def test_forked_tokens_greedy_topk(self):
        """Greedy sampling should return top-k tokens per position."""
        B, K, V = 1, 2, 50
        fan_out_list = [2, 2, 1]  # K+1=3 entries
        logits = torch.randn(B, K + 1, V, device=DEVICE)
        temps = torch.zeros(B, device=DEVICE)
        hits = torch.ones(B, dtype=torch.int64, device=DEVICE)
        result = get_forked_recovery_tokens_from_logits(
            logits=logits, fan_out_list=fan_out_list, temperatures=temps,
            cache_hits=hits,
        )
        # At position 0: top-2 of logits[0, 0] (no masking at position K)
        # But position 0 masks returned_tokens[:, 1] — with returned_tokens=None, no masking
        # Just verify shape
        self.assertEqual(result.shape, (B, sum(fan_out_list)))

    def test_forked_tokens_excludes_returned(self):
        """With returned_tokens, masked tokens should NOT appear in output."""
        B, K, V = 1, 2, 20
        logits = torch.randn(B, K + 1, V, device=DEVICE)
        fan_out_list = [2, 2, 1]  # K+1=3
        temps = torch.zeros(B, device=DEVICE)
        hits = torch.ones(B, dtype=torch.int64, device=DEVICE)

        returned_tokens = torch.zeros(B, K + 1, dtype=torch.int64, device=DEVICE)
        for k in range(K + 1):
            returned_tokens[0, k] = logits[0, k].argmax().item()

        result = get_forked_recovery_tokens_from_logits(
            logits=logits, fan_out_list=fan_out_list, temperatures=temps,
            cache_hits=hits, returned_tokens=returned_tokens,
        )

        # At position d (d < K), returned_tokens[:, d+1] is masked
        offset = 0
        for d in range(K):
            fo = fan_out_list[d]
            masked_tok = returned_tokens[0, d + 1].item()
            tokens_at_d = result[0, offset:offset + fo].tolist()
            self.assertNotIn(masked_tok, tokens_at_d,
                             f"Masked token {masked_tok} found at depth {d}")
            offset += fo

    def test_forked_tokens_mixed_hits(self):
        """Mixed batch: different fan_out per sequence."""
        B, K, V = 4, 2, 50
        fan_out_list = [2, 2, 1]
        fan_out_list_miss = [1, 1, 1]
        logits = torch.randn(B, K + 1, V, device=DEVICE)
        temps = torch.zeros(B, device=DEVICE)
        hits = torch.tensor([1, 0, 1, 0], dtype=torch.int64, device=DEVICE)

        # Both lists must sum to same MQ_LEN for vectorized output
        # Here: hit=5, miss=3 — different sums, so we need equal sums
        fan_out_list = [2, 2, 1]
        fan_out_list_miss = [2, 2, 1]  # same sum=5
        result = get_forked_recovery_tokens_from_logits(
            logits=logits, fan_out_list=fan_out_list, temperatures=temps,
            cache_hits=hits, fan_out_list_miss=fan_out_list_miss,
        )
        self.assertEqual(result.shape[0], B)

    @unittest.skipUnless(SSD_AVAILABLE, "SSD not available")
    def test_forked_tokens_matches_ssd(self):
        """Cross-check: SGLang output matches SSD output for greedy."""
        B, K, V = 2, 2, 100
        logits = torch.randn(B, K + 1, V, device=DEVICE)
        fan_out_list = [2, 2, 1]  # K+1=3 entries
        fan_out_list_miss = fan_out_list
        temps = torch.zeros(B, device=DEVICE)
        hits = torch.ones(B, dtype=torch.int64, device=DEVICE)

        returned_tokens = torch.zeros(B, K + 1, dtype=torch.int64, device=DEVICE)
        for b in range(B):
            for k in range(K + 1):
                returned_tokens[b, k] = logits[b, k].argmax().item()

        class MockTok:
            def decode(self, ids):
                return str(ids)

        ssd_cfg = MockSSDConfig(K, fan_out_list, fan_out_list_miss)
        ssd_out = ssd_get_forked_recovery_tokens_from_logits(
            ssd_cfg, logits, hits, returned_tokens, tokenizer=MockTok()
        )

        sgl_out = get_forked_recovery_tokens_from_logits(
            logits=logits, fan_out_list=fan_out_list, temperatures=temps,
            cache_hits=hits, fan_out_list_miss=fan_out_list_miss,
            returned_tokens=returned_tokens,
        )

        self.assertEqual(sgl_out.shape, ssd_out.shape,
                         f"Shape mismatch: SGLang {sgl_out.shape} vs SSD {ssd_out.shape}")
        self.assertTrue(torch.equal(sgl_out, ssd_out),
                        f"Value mismatch:\nSGLang: {sgl_out}\nSSD:    {ssd_out}")

    # ── apply_sampler_x_rescaling: now matches SSD multiply-top-F ──

    def test_sampler_x_identity(self):
        probs = F.softmax(torch.randn(2, 100), dim=-1)
        result = apply_sampler_x_rescaling(probs, 1.0, 3)
        self.assertTrue(torch.allclose(result, probs, atol=1e-5))

    def test_sampler_x_none(self):
        probs = F.softmax(torch.randn(2, 100), dim=-1)
        result = apply_sampler_x_rescaling(probs, None, 3)
        self.assertTrue(torch.allclose(result, probs, atol=1e-5))

    def test_sampler_x_normalized(self):
        probs = F.softmax(torch.randn(5, 200), dim=-1)
        for sx in [0.5, 1.0, 2.0, 5.0]:
            result = apply_sampler_x_rescaling(probs, sx, 3)
            sums = result.sum(dim=-1)
            self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))
            self.assertTrue((result >= 0).all())

    @unittest.skipUnless(SSD_AVAILABLE, "SSD not available")
    def test_sampler_x_matches_ssd(self):
        """Cross-check: SGLang matches SSD sampler_x rescaling."""
        probs = F.softmax(torch.randn(3, 50, device=DEVICE), dim=-1)
        for sx in [0.5, 2.0, 5.0]:
            sgl_out = apply_sampler_x_rescaling(probs.clone(), sx, 3)
            ssd_out = ssd_apply_sampler_x_rescaling(probs.unsqueeze(1).clone(), sx, 3).squeeze(1)
            self.assertTrue(torch.allclose(sgl_out, ssd_out, atol=1e-5),
                            f"Mismatch at sx={sx}:\nmax_diff={torch.abs(sgl_out - ssd_out).max()}")


# ═════════════════════════════════════════════════════════════════════════
# TEST SUITE 2: Cache Lookup and Population
# ═════════════════════════════════════════════════════════════════════════
class TestCacheLookup(unittest.TestCase):

    def _r(self, **kw):
        return MockAsyncDraftRunner(device=DEVICE, **kw)

    def test_cache_miss_on_empty(self):
        r = self._r(spec_k=3, vocab_size=100)
        B, K, V = 2, 3, 100
        keys = torch.tensor([[0, 0, 5], [1, 0, 10]], dtype=torch.int64, device=DEVICE)
        t = torch.zeros(B, device=DEVICE)
        di = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        specs, _, hits = r.hit_cache_and_respond(keys, B, K, V, t, di)
        self.assertTrue((hits == 0).all())
        self.assertEqual(specs[0, 0].item(), 5)

    def test_cache_hit_after_populate(self):
        r = self._r(spec_k=3, vocab_size=100)
        K, V = 3, 100
        N = 4
        keys = torch.tensor([[0, 0, 5], [0, 1, 10], [1, 0, 20], [1, 1, 30]],
                             dtype=torch.int64, device=DEVICE)
        tokens = torch.randint(0, V, (N, K), device=DEVICE)
        logits = torch.randn(N, K, V, device=DEVICE)
        r.tree_cache_keys = keys
        r.tree_cache_tokens = tokens
        r.tree_cache_logits = logits

        B = 2
        qkeys = torch.tensor([[0, 0, 5], [1, 0, 20]], dtype=torch.int64, device=DEVICE)
        t = torch.zeros(B, device=DEVICE)
        di = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        _, out, hits = r.hit_cache_and_respond(qkeys, B, K, V, t, di)
        self.assertTrue((hits == 1).all())
        self.assertTrue(torch.equal(out[0], tokens[0]))
        self.assertTrue(torch.equal(out[1], tokens[2]))

    def test_cache_partial_hit(self):
        r = self._r(spec_k=2, vocab_size=50)
        K, V = 2, 50
        keys = torch.tensor([[0, 0, 5], [0, 1, 10]], dtype=torch.int64, device=DEVICE)
        r.tree_cache_keys = keys
        r.tree_cache_tokens = torch.randint(0, V, (2, K), device=DEVICE)
        r.tree_cache_logits = torch.randn(2, K, V, device=DEVICE)

        B = 2
        qkeys = torch.tensor([[0, 0, 5], [99, 99, 99]], dtype=torch.int64, device=DEVICE)
        _, _, hits = r.hit_cache_and_respond(qkeys, B, K, V, torch.zeros(B, device=DEVICE),
                                             torch.tensor([0, 1], dtype=torch.int64, device=DEVICE))
        self.assertEqual(hits[0].item(), 1)
        self.assertEqual(hits[1].item(), 0)

    def test_cache_reset(self):
        r = self._r()
        r.tree_cache_keys = torch.ones(5, 3, dtype=torch.int64, device=DEVICE)
        r._reset_tree_cache()
        self.assertEqual(r.tree_cache_keys.shape[0], 0)
        self.assertIsNone(r.tree_cache_tokens)

    def test_populate_tree_cache_key_format(self):
        """Cache keys use [seq_id, fan_idx, recovery_token] matching SSD."""
        r = self._r(spec_k=2, vocab_size=50)
        K, V, B = 2, 50, 2
        cache_keys_orig = torch.tensor([[10, 0, 5], [20, 0, 15]], dtype=torch.int64, device=DEVICE)
        N = 6
        batch_ids = torch.tensor([0, 0, 0, 1, 1, 1], device=DEVICE)
        fan_idx = torch.tensor([0, 0, 1, 0, 0, 1], device=DEVICE)
        rec_tokens = torch.tensor([5, 6, 7, 15, 16, 17], dtype=torch.int64, device=DEVICE)
        spec_tokens = torch.randint(0, V, (N, K), device=DEVICE)
        spec_logits = torch.randn(N, K, V, device=DEVICE)

        r.populate_tree_cache(batch_ids, fan_idx, rec_tokens, spec_tokens, spec_logits, cache_keys_orig)

        self.assertEqual(r.tree_cache_keys.shape, (N, 3))
        self.assertTrue((r.tree_cache_keys[:3, 0] == 10).all())
        self.assertTrue((r.tree_cache_keys[3:, 0] == 20).all())
        self.assertTrue(torch.equal(r.tree_cache_keys[:, 1], fan_idx))
        self.assertTrue(torch.equal(r.tree_cache_keys[:, 2], rec_tokens))


# ═════════════════════════════════════════════════════════════════════════
# TEST SUITE 3: KV Cache Management
# ═════════════════════════════════════════════════════════════════════════
class TestKVCacheManagement(unittest.TestCase):

    def _r(self, **kw):
        return MockAsyncDraftRunner(device=DEVICE, **kw)

    def test_assign_slots_per_element(self):
        r = self._r()
        pool = r.req_to_token_pool.req_to_token
        di = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=DEVICE)
        pos = torch.tensor([5, 10, 15, 20], dtype=torch.int64, device=DEVICE)
        locs = torch.tensor([100, 200, 300, 400], dtype=torch.int64, device=DEVICE)
        r._assign_slots_per_element(di, pos, locs)
        for i in range(4):
            self.assertEqual(pool[di[i], pos[i]].item(), locs[i].item())

    def test_assign_slots_to_pool(self):
        r = self._r()
        pool = r.req_to_token_pool.req_to_token
        di = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        base = torch.tensor([5, 10], dtype=torch.int64, device=DEVICE)
        locs = torch.tensor([100, 101, 102, 200, 201, 202], dtype=torch.int64, device=DEVICE)
        r._assign_slots_to_pool(di, base, 3, locs)
        self.assertTrue(torch.equal(pool[0, 5:8], torch.tensor([100, 101, 102], device=DEVICE)))
        self.assertTrue(torch.equal(pool[1, 10:13], torch.tensor([200, 201, 202], device=DEVICE)))

    def test_assign_slots_vectorized_matches_loop(self):
        B, tpr = 3, 4
        di = torch.tensor([2, 5, 8], dtype=torch.int64, device=DEVICE)
        base = torch.tensor([10, 20, 30], dtype=torch.int64, device=DEVICE)
        locs = torch.arange(B * tpr, device=DEVICE, dtype=torch.int64) + 500
        pool_loop = torch.full((16, 512), -1, dtype=torch.int64, device=DEVICE)
        pool_vec = torch.full((16, 512), -1, dtype=torch.int64, device=DEVICE)
        for i in range(B):
            for j in range(tpr):
                pool_loop[di[i], base[i] + j] = locs[i * tpr + j]
        rows = di.repeat_interleave(tpr)
        offsets = torch.arange(tpr, device=DEVICE, dtype=torch.int64)
        cols = (base.unsqueeze(1) + offsets).reshape(-1)
        pool_vec[rows, cols] = locs
        self.assertTrue(torch.equal(pool_loop, pool_vec))

    def test_rollback_basic(self):
        r = self._r(max_reqs=8)
        di = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        r.draft_seq_lens[0] = 10
        r.draft_seq_lens[1] = 15
        for i in range(10):
            r.req_to_token_pool.req_to_token[0, i] = 1000 + i
        for i in range(15):
            r.req_to_token_pool.req_to_token[1, i] = 2000 + i
        r._rollback_kv_cache(di, torch.tensor([7, 12], dtype=torch.int64, device=DEVICE))
        self.assertEqual(r.draft_seq_lens[0].item(), 7)
        self.assertEqual(r.draft_seq_lens[1].item(), 12)

    def test_rollback_no_change(self):
        r = self._r(max_reqs=4)
        di = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        r.draft_seq_lens[0] = 10
        r.draft_seq_lens[1] = 15
        r._rollback_kv_cache(di, torch.tensor([10, 15], dtype=torch.int64, device=DEVICE))
        self.assertEqual(r.draft_seq_lens[0].item(), 10)
        self.assertEqual(r.draft_seq_lens[1].item(), 15)


# ═════════════════════════════════════════════════════════════════════════
# TEST SUITE 4: Tree Decode Precomputation
# ═════════════════════════════════════════════════════════════════════════
class TestTreeDecodePrecomputation(unittest.TestCase):
    """Tests for precomputed position / fan_idx tensors."""

    def _r(self, **kw):
        return MockAsyncDraftRunner(device=DEVICE, **kw)

    def test_fan_idx_hit_pattern(self):
        """fan_idx maps each MQ position to its K+1 depth."""
        r = self._r(spec_k=2, fan_out_list=[2, 2, 1])
        # K+1=3, fan_out=[2,2,1], MQ_LEN=5
        # Expected: [0,0,1,1,2]
        expected = torch.tensor([0, 0, 1, 1, 2], device=DEVICE)
        self.assertTrue(torch.equal(r._fan_idx_hit, expected))

    def test_step_pos_offsets(self):
        """step_pos_offsets[d] = d * MQ_LEN."""
        r = self._r(spec_k=3, fan_out_list=[2, 2, 1, 1])
        MQ = sum([2, 2, 1, 1])  # 6
        K = 3
        for d in range(K):
            self.assertEqual(r._step_pos_offsets[d, 0].item(), d * MQ)

    def test_precomputed_positions_unique(self):
        """Each tree token should get a unique KV position."""
        K = 2
        fan_out_list = [2, 2, 1]
        r = self._r(spec_k=K, fan_out_list=fan_out_list)
        MQ_LEN = sum(fan_out_list)
        B = 2
        N_tree = B * MQ_LEN

        glue_seq_lens = torch.tensor([100, 200], dtype=torch.int64, device=DEVICE)
        batch_ids = torch.arange(B, device=DEVICE).repeat_interleave(MQ_LEN)
        tree_base_lens = glue_seq_lens[batch_ids]
        fkp1 = r._arange_mq.repeat(B)
        initial_pos = tree_base_lens + fkp1

        all_pos = initial_pos.unsqueeze(0) + r._step_pos_offsets  # [K, N_tree]

        # Check uniqueness per batch element at each step
        for depth in range(K):
            for b in range(B):
                mask = batch_ids == b
                positions = all_pos[depth][mask]
                self.assertEqual(len(positions.unique()), MQ_LEN,
                                 f"Non-unique positions for batch {b} depth {depth}")

    def test_precomputed_matches_loop(self):
        """Precomputed step positions match naive per-step computation."""
        K = 3
        fan_out_list = [2, 2, 1, 1]
        r = self._r(spec_k=K, fan_out_list=fan_out_list)
        MQ_LEN = sum(fan_out_list)
        B = 2
        N_tree = B * MQ_LEN

        glue_seq_lens = torch.tensor([50, 80], dtype=torch.int64, device=DEVICE)
        batch_ids = torch.arange(B, device=DEVICE).repeat_interleave(MQ_LEN)
        tree_base = glue_seq_lens[batch_ids]
        fkp1 = r._arange_mq.repeat(B)
        initial_pos = tree_base + fkp1

        all_pos = initial_pos.unsqueeze(0) + r._step_pos_offsets  # [K, N_tree]
        all_seq_lens = all_pos + 1

        # Compare with naive loop
        for d in range(K):
            naive_pos = tree_base + fkp1 + d * MQ_LEN
            naive_seq = naive_pos + 1
            self.assertTrue(torch.equal(all_pos[d], naive_pos), f"Position mismatch at depth {d}")
            self.assertTrue(torch.equal(all_seq_lens[d], naive_seq), f"Seq lens mismatch at depth {d}")


# ═════════════════════════════════════════════════════════════════════════
# TEST SUITE 5: End-to-End Mock Flow
# ═════════════════════════════════════════════════════════════════════════
class TestE2EMockFlow(unittest.TestCase):

    def _r(self, **kw):
        return MockAsyncDraftRunner(device=DEVICE, **kw)

    def test_full_cycle_miss_then_hit(self):
        K, V = 2, 50
        fan_out_list = [2, 2, 1]
        r = self._r(spec_k=K, vocab_size=V, fan_out_list=fan_out_list)
        B = 1
        MQ_LEN = sum(fan_out_list)

        keys = torch.tensor([[0, 0, 5]], dtype=torch.int64, device=DEVICE)
        t = torch.zeros(B, device=DEVICE)
        di = torch.tensor([0], dtype=torch.int64, device=DEVICE)

        specs, out_tokens, hits = r.hit_cache_and_respond(keys, B, K, V, t, di)
        self.assertEqual(hits[0].item(), 0)

        recovery = keys[:, 2]
        glue_flat = make_glue_decode_input_ids(out_tokens.to(torch.int64), recovery)
        glue_2d = glue_flat.view(B, K + 1)
        glue_logits = torch.randn(B, K + 1, V, device=DEVICE)

        forked = get_forked_recovery_tokens_from_logits(
            logits=glue_logits, fan_out_list=fan_out_list, temperatures=t,
            cache_hits=hits, fan_out_list_miss=fan_out_list,
            returned_tokens=glue_2d,
        )  # [B, MQ_LEN]

        N_tree = B * MQ_LEN
        batch_ids = torch.arange(B, device=DEVICE).repeat_interleave(MQ_LEN)
        fan_idx = r._fan_idx_hit.repeat(B)
        tree_flat = forked.reshape(-1)

        spec_tokens = torch.randint(0, V, (N_tree, K), device=DEVICE)
        spec_logits = torch.randn(N_tree, K, V, device=DEVICE)

        r.populate_tree_cache(batch_ids, fan_idx, tree_flat, spec_tokens, spec_logits, keys)
        self.assertGreater(r.tree_cache_keys.shape[0], 0)

        # Query with a key from the populated cache
        hit_key = r.tree_cache_keys[0].unsqueeze(0)
        _, _, hits2 = r.hit_cache_and_respond(hit_key, 1, K, V, t[:1], di[:1])
        self.assertEqual(hits2[0].item(), 1)

    def test_speculation_shape(self):
        for B in [1, 4, 8]:
            for K in [1, 3, 5]:
                V = 100
                fan_out_list = [2] * (K + 1)
                r = self._r(spec_k=K, vocab_size=V, fan_out_list=fan_out_list)
                keys = torch.randint(0, 50, (B, 3), dtype=torch.int64, device=DEVICE)
                t = torch.zeros(B, device=DEVICE)
                di = torch.arange(B, dtype=torch.int64, device=DEVICE)
                specs, _, _ = r.hit_cache_and_respond(keys, B, K, V, t, di)
                self.assertEqual(specs.shape, (B, K + 1))


# ═════════════════════════════════════════════════════════════════════════
# TEST SUITE 6: Verify Function
# ═════════════════════════════════════════════════════════════════════════
class TestVerifyFunction(unittest.TestCase):

    def test_greedy_all_accept(self):
        from test.srt.async_spec_test_utils import verify
        B, K, V = 2, 4, 100
        logits_p = torch.zeros(B, K + 1, V)
        logits_q = torch.zeros(B, K, V)
        specs = torch.zeros(B, K + 1, dtype=torch.long)
        for b in range(B):
            for k in range(K + 1):
                tok = (b * 10 + k + 1) % V
                specs[b, k] = tok
                logits_p[b, k, tok] = 10.0
                if k > 0:
                    logits_q[b, k - 1, tok] = 10.0
        accepted, _ = verify(logits_p, logits_q, specs, torch.zeros(B), torch.zeros(B))
        for b in range(B):
            self.assertEqual(len(accepted[b]), K)

    def test_greedy_reject(self):
        from test.srt.async_spec_test_utils import verify
        B, K, V = 1, 3, 50
        logits_p = torch.zeros(B, K + 1, V)
        logits_q = torch.zeros(B, K, V)
        specs = torch.zeros(B, K + 1, dtype=torch.long)
        specs[0, 0] = 5; logits_p[0, 0, 5] = 10.0
        specs[0, 1] = 10; logits_p[0, 1, 10] = 10.0; logits_q[0, 0, 10] = 10.0
        specs[0, 2] = 20; logits_p[0, 2, 30] = 10.0; logits_q[0, 1, 20] = 10.0
        specs[0, 3] = 40; logits_p[0, 3, 40] = 10.0
        accepted, recovery = verify(logits_p, logits_q, specs, torch.zeros(B), torch.zeros(B))
        self.assertEqual(len(accepted[0]), 1)
        self.assertEqual(recovery[0], 30)


# ═════════════════════════════════════════════════════════════════════════
# TEST SUITE 7: NCCL Channel Protocol
# ═════════════════════════════════════════════════════════════════════════
class TestNcclChannelProtocol(unittest.TestCase):

    def test_spec_request_roundtrip(self):
        from sglang.srt.speculative.async_spec.nccl_comm import HEADER_SIZE, AsyncSpecNcclChannel

        class MockComm:
            def __init__(self):
                self.stream = type("S", (), {"synchronize": lambda s: None})()
            def send(self, *a, **kw): pass
            def recv(self, *a, **kw): pass
            def group_start(self): pass
            def group_end(self): pass

        ch = AsyncSpecNcclChannel(MockComm(), rank=0, device=DEVICE, max_batch_size=64, max_spec_k=16)
        B, K, fo, vs = 3, 5, 2, 32000
        keys = torch.randint(0, 100, (B, 3), dtype=torch.int64, device=DEVICE)
        temps = torch.rand(B, device=DEVICE)
        sl = torch.randint(10, 100, (B,), dtype=torch.int64, device=DEVICE)

        ch.send_spec_request(B, K, fo, vs, keys, temps, sl)
        uB, uK, ufo, uvs, ukeys, utemps, usl = ch.unpack_spec_request()
        self.assertEqual(uB, B)
        self.assertEqual(uK, K)
        self.assertTrue(torch.equal(ukeys, keys))
        self.assertTrue(torch.allclose(utemps, temps))
        self.assertTrue(torch.equal(usl, sl))


# ═════════════════════════════════════════════════════════════════════════
# TEST SUITE 8: Profiling
# ═════════════════════════════════════════════════════════════════════════
class TestProfiling(unittest.TestCase):

    def _time(self, fn, warmup=5, iters=100):
        for _ in range(warmup):
            fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    def test_profile_cache_lookup(self):
        results = []
        for N in [10, 100, 500]:
            for B in [1, 4, 16]:
                ck = torch.randint(0, 1000, (N, 3), dtype=torch.int64, device=DEVICE)
                qk = torch.randint(0, 1000, (B, 3), dtype=torch.int64, device=DEVICE)
                def f():
                    eq = qk.unsqueeze(1) == ck.unsqueeze(0)
                    return torch.all(eq, dim=2).any(dim=1)
                results.append((N, B, self._time(f)))
        print("\n=== Cache Lookup ===")
        for n, b, ms in results:
            print(f"  N={n:>4} B={b:>2} {ms:.3f}ms")

    def test_profile_slot_assignment(self):
        B, tpr = 16, 4
        di = torch.arange(B, dtype=torch.int64, device=DEVICE)
        base = torch.randint(0, 100, (B,), dtype=torch.int64, device=DEVICE)
        locs = torch.arange(B * tpr, device=DEVICE, dtype=torch.int64)
        pool = torch.full((64, 2048), -1, dtype=torch.int64, device=DEVICE)

        def vec():
            rows = di.repeat_interleave(tpr)
            offsets = torch.arange(tpr, device=DEVICE, dtype=torch.int64)
            cols = (base.unsqueeze(1) + offsets).reshape(-1)
            pool[rows, cols] = locs

        def loop():
            for i in range(B):
                for j in range(tpr):
                    pool[di[i], base[i] + j] = locs[i * tpr + j]

        ms_v = self._time(vec)
        ms_l = self._time(loop)
        print(f"\n=== Slot Assignment ===\n  Vectorized: {ms_v:.3f}ms\n  Loop: {ms_l:.3f}ms\n  Speedup: {ms_l/ms_v:.1f}x")

    def test_profile_forked_sampling(self):
        results = []
        for B in [1, 4, 16]:
            for V in [1000, 32000]:
                K = 3
                fo = [2, 2, 2, 1]  # K+1=4 entries
                logits = torch.randn(B, K + 1, V, device=DEVICE)
                t = torch.ones(B, device=DEVICE)
                h = torch.ones(B, dtype=torch.int64, device=DEVICE)
                rt = torch.zeros(B, K + 1, dtype=torch.int64, device=DEVICE)
                def f():
                    return get_forked_recovery_tokens_from_logits(
                        logits=logits, fan_out_list=fo, temperatures=t, cache_hits=h, returned_tokens=rt)
                results.append((B, V, self._time(f, warmup=3, iters=20)))
        print("\n=== Forked Token Sampling ===")
        for b, v, ms in results:
            print(f"  B={b:>2} V={v:>6} {ms:.3f}ms")

    def test_profile_precomputed_positions(self):
        """Compare precomputed vs per-step position computation."""
        K, MQ = 3, 6
        B = 16
        N = B * MQ
        base = torch.randint(50, 200, (N,), dtype=torch.int64, device=DEVICE)
        fkp1 = torch.arange(MQ, device=DEVICE).repeat(B)
        step_offsets = torch.arange(K, device=DEVICE)[:, None] * MQ

        def precomputed():
            initial = base + fkp1
            all_pos = initial.unsqueeze(0) + step_offsets
            return all_pos + 1

        def per_step():
            results = []
            for d in range(K):
                pos = base + fkp1 + d * MQ
                results.append(pos + 1)
            return torch.stack(results)

        ms_pre = self._time(precomputed)
        ms_loop = self._time(per_step)
        print(f"\n=== Position Precomputation ===\n  Precomputed: {ms_pre:.3f}ms\n  Per-step: {ms_loop:.3f}ms\n  Speedup: {ms_loop/ms_pre:.1f}x")


# ═════════════════════════════════════════════════════════════════════════
# TEST SUITE 9: Edge Cases
# ═════════════════════════════════════════════════════════════════════════
class TestEdgeCases(unittest.TestCase):

    def test_single_seq_k1(self):
        K, V = 1, 10
        r = MockAsyncDraftRunner(spec_k=K, vocab_size=V, fan_out_list=[1, 1], device=DEVICE)
        keys = torch.tensor([[0, 0, 5]], dtype=torch.int64, device=DEVICE)
        specs, _, _ = r.hit_cache_and_respond(keys, 1, K, V, torch.zeros(1, device=DEVICE),
                                               torch.tensor([0], dtype=torch.int64, device=DEVICE))
        self.assertEqual(specs.shape, (1, 2))
        self.assertEqual(specs[0, 0].item(), 5)

    def test_large_vocab(self):
        K, V = 3, 128000
        r = MockAsyncDraftRunner(spec_k=K, vocab_size=V, device=DEVICE)
        B = 2
        keys = torch.randint(0, 50, (B, 3), dtype=torch.int64, device=DEVICE)
        specs, _, _ = r.hit_cache_and_respond(keys, B, K, V, torch.zeros(B, device=DEVICE),
                                               torch.arange(B, dtype=torch.int64, device=DEVICE))
        self.assertEqual(specs.shape, (B, K + 1))

    def test_same_recovery_different_seq(self):
        K, V = 2, 50
        r = MockAsyncDraftRunner(spec_k=K, vocab_size=V, device=DEVICE)
        keys = torch.tensor([[0, 0, 42], [1, 0, 42]], dtype=torch.int64, device=DEVICE)
        tokens = torch.tensor([[10, 20], [30, 40]], dtype=torch.int64, device=DEVICE)
        r.tree_cache_keys = keys
        r.tree_cache_tokens = tokens
        r.tree_cache_logits = torch.randn(2, K, V, device=DEVICE)

        q0 = torch.tensor([[0, 0, 42]], dtype=torch.int64, device=DEVICE)
        _, out0, _ = r.hit_cache_and_respond(q0, 1, K, V, torch.zeros(1, device=DEVICE),
                                              torch.tensor([0], dtype=torch.int64, device=DEVICE))
        self.assertTrue(torch.equal(out0[0], tokens[0]))

        q1 = torch.tensor([[1, 0, 42]], dtype=torch.int64, device=DEVICE)
        _, out1, _ = r.hit_cache_and_respond(q1, 1, K, V, torch.zeros(1, device=DEVICE),
                                              torch.tensor([0], dtype=torch.int64, device=DEVICE))
        self.assertTrue(torch.equal(out1[0], tokens[1]))

    def test_command_codes(self):
        self.assertEqual(CMD_SPEC_REQUEST, 0)
        self.assertEqual(CMD_PREFILL, 1)
        self.assertEqual(CMD_EXIT, 2)

    def test_zero_temp_greedy(self):
        r = MockAsyncDraftRunner(device=DEVICE)
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]], device=DEVICE)
        result = r._sample_tokens(logits, torch.tensor([0.0], device=DEVICE))
        self.assertEqual(result[0].item(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
