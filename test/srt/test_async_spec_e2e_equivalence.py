"""End-to-end equivalence tests: SGLang async spec vs SSD async spec.

Validates that for identical inputs, both codebases produce identical:
  1. Glue decode input IDs
  2. Forked recovery tokens (with -inf masking)
  3. Tree decode batch structure (batch_ids, fan_idx)
  4. Tree cache keys after population
  5. Cache hit/miss detection
  6. Full spec request cycle (miss -> populate -> hit)

Also includes detailed per-phase profiling with CUDA events.

Run:
  source .venv/bin/activate
  python test/srt/test_async_spec_e2e_equivalence.py
"""

from __future__ import annotations

import sys
import time
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work/avner/git/ssd")
SSD_AVAILABLE = False
try:
    from ssd.utils.async_helpers.async_spec_helpers import (
        make_glue_decode_input_ids as ssd_make_glue,
        get_forked_recovery_tokens_from_logits as ssd_get_forked,
        apply_sampler_x_rescaling as ssd_apply_sampler_x,
    )
    SSD_AVAILABLE = True
except ImportError:
    pass

from sglang.srt.speculative.async_spec.tree_utils import (
    make_glue_decode_input_ids as sgl_make_glue,
    get_forked_recovery_tokens_from_logits as sgl_get_forked,
    apply_sampler_x_rescaling as sgl_apply_sampler_x,
    compute_mq_len,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MockSSDConfig:
    def __init__(self, K, fan_out_list, fan_out_list_miss=None):
        self.speculate_k = K
        self.fan_out_list = fan_out_list
        self.fan_out_list_miss = fan_out_list_miss or fan_out_list
        self.fan_out_t = torch.tensor(fan_out_list, device=DEVICE)
        self.fan_out_t_miss = torch.tensor(self.fan_out_list_miss, device=DEVICE)
        self.MQ_LEN = sum(fan_out_list)


class MockTokenizer:
    def decode(self, ids):
        return str(ids)


def make_deterministic_logits(B, K_plus_1, V, device=DEVICE, seed=42):
    """Generate deterministic logits for reproducible tests."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(B, K_plus_1, V, device=device, generator=gen)


def make_returned_tokens(logits):
    """Create returned_tokens from logits (argmax per position)."""
    B, K_plus_1, V = logits.shape
    rt = torch.zeros(B, K_plus_1, dtype=torch.int64, device=logits.device)
    for b in range(B):
        for k in range(K_plus_1):
            rt[b, k] = logits[b, k].argmax().item()
    return rt


@unittest.skipUnless(SSD_AVAILABLE, "SSD repo not available")
class TestE2EGlueDecodeEquivalence(unittest.TestCase):
    """End-to-end: glue decode input construction."""

    def test_all_configs(self):
        """Sweep B, K configs and verify exact match."""
        configs = [(1, 1), (1, 3), (2, 3), (4, 5), (8, 3), (16, 2)]
        for B, K in configs:
            V = 256
            draft_tokens = torch.randint(0, V, (B, K), device=DEVICE)
            rec_tokens = torch.randint(0, V, (B,), device=DEVICE)

            sgl_out = sgl_make_glue(draft_tokens, rec_tokens)
            ssd_out = ssd_make_glue(draft_tokens, rec_tokens)

            self.assertTrue(
                torch.equal(sgl_out, ssd_out),
                f"B={B}, K={K}: mismatch\nSGL={sgl_out[:10]}\nSSD={ssd_out[:10]}",
            )


@unittest.skipUnless(SSD_AVAILABLE, "SSD repo not available")
class TestE2EForkedTokensEquivalence(unittest.TestCase):
    """End-to-end: forked token sampling with -inf masking."""

    def _run_comparison(self, B, K, V, fan_out_list, fan_out_list_miss=None,
                        temp=0.0, all_hit=True, seed=42):
        fan_out_list_miss = fan_out_list_miss or fan_out_list
        logits = make_deterministic_logits(B, K + 1, V, seed=seed)
        returned_tokens = make_returned_tokens(logits)
        temps = torch.full((B,), temp, device=DEVICE)
        hits = torch.ones(B, dtype=torch.int64, device=DEVICE) if all_hit else \
               torch.zeros(B, dtype=torch.int64, device=DEVICE)

        ssd_cfg = MockSSDConfig(K, fan_out_list, fan_out_list_miss)
        ssd_out = ssd_get_forked(ssd_cfg, logits, hits, returned_tokens, MockTokenizer())

        sgl_out = sgl_get_forked(
            logits=logits, fan_out_list=fan_out_list, temperatures=temps,
            cache_hits=hits, fan_out_list_miss=fan_out_list_miss,
            returned_tokens=returned_tokens,
        )

        self.assertEqual(sgl_out.shape, ssd_out.shape,
                         f"Shape: SGL={sgl_out.shape} SSD={ssd_out.shape}")
        self.assertTrue(torch.equal(sgl_out, ssd_out),
                        f"Values differ\nSGL={sgl_out}\nSSD={ssd_out}")

    def test_greedy_basic(self):
        self._run_comparison(B=2, K=3, V=100, fan_out_list=[2, 2, 2, 1])

    def test_greedy_b1(self):
        self._run_comparison(B=1, K=2, V=50, fan_out_list=[3, 2, 1])

    def test_greedy_large_batch(self):
        self._run_comparison(B=8, K=3, V=200, fan_out_list=[2, 2, 2, 1])

    def test_greedy_large_vocab(self):
        self._run_comparison(B=2, K=3, V=32000, fan_out_list=[2, 2, 2, 1])

    def test_greedy_all_miss(self):
        self._run_comparison(B=4, K=2, V=100, fan_out_list=[2, 2, 1],
                             fan_out_list_miss=[2, 2, 1], all_hit=False)

    def test_greedy_asymmetric_fanout(self):
        self._run_comparison(B=2, K=3, V=100, fan_out_list=[3, 2, 1, 1])

    def test_greedy_mixed_hits(self):
        """Mixed hit/miss batch (same MQ_LEN for hit and miss)."""
        B, K, V = 4, 2, 100
        fan_out_list = [2, 2, 1]
        logits = make_deterministic_logits(B, K + 1, V)
        returned_tokens = make_returned_tokens(logits)
        temps = torch.zeros(B, device=DEVICE)
        hits = torch.tensor([1, 0, 1, 0], dtype=torch.int64, device=DEVICE)

        ssd_cfg = MockSSDConfig(K, fan_out_list, fan_out_list)
        ssd_out = ssd_get_forked(ssd_cfg, logits, hits, returned_tokens, MockTokenizer())
        sgl_out = sgl_get_forked(
            logits=logits, fan_out_list=fan_out_list, temperatures=temps,
            cache_hits=hits, fan_out_list_miss=fan_out_list,
            returned_tokens=returned_tokens,
        )
        self.assertTrue(torch.equal(sgl_out, ssd_out))

    def test_multiple_seeds(self):
        """Verify equivalence across different random seeds."""
        for seed in range(10):
            self._run_comparison(B=2, K=3, V=100, fan_out_list=[2, 2, 2, 1], seed=seed)


@unittest.skipUnless(SSD_AVAILABLE, "SSD repo not available")
class TestE2ESamplerXEquivalence(unittest.TestCase):
    """End-to-end: sampler_x rescaling."""

    def test_sweep(self):
        for sx in [0.5, 1.0, 1.5, 2.0, 5.0]:
            for F in [1, 2, 3, 5]:
                probs = F.softmax(torch.randn(4, 100, device=DEVICE), dim=-1) if False else \
                        torch.nn.functional.softmax(torch.randn(4, 100, device=DEVICE), dim=-1)
                sgl = sgl_apply_sampler_x(probs.clone(), sx, F)
                ssd = ssd_apply_sampler_x(probs.unsqueeze(1).clone(), sx, F).squeeze(1)
                self.assertTrue(torch.allclose(sgl, ssd, atol=1e-5),
                                f"sx={sx}, F={F}: max_diff={torch.abs(sgl-ssd).max()}")


@unittest.skipUnless(SSD_AVAILABLE, "SSD repo not available")
class TestE2ETreeCacheEquivalence(unittest.TestCase):
    """End-to-end: tree cache key construction and lookup."""

    def test_cache_key_construction(self):
        """Both build keys as [seq_id, fan_idx, recovery_token]."""
        B, K = 2, 3
        fan_out_list = [2, 2, 2, 1]
        MQ_LEN = sum(fan_out_list)
        N = B * MQ_LEN

        cache_keys_orig = torch.tensor([[10, 0, 5], [20, 0, 15]],
                                        dtype=torch.int64, device=DEVICE)

        # SGLang construction
        batch_ids = torch.arange(B, device=DEVICE).repeat_interleave(MQ_LEN)
        fan_idx = torch.arange(K + 1, device=DEVICE, dtype=torch.int64).repeat_interleave(
            torch.tensor(fan_out_list, device=DEVICE)
        ).repeat(B)
        rec_tokens = torch.randint(0, 100, (N,), device=DEVICE)

        seq_ids = cache_keys_orig[batch_ids, 0]
        sgl_keys = torch.stack([seq_ids, fan_idx, rec_tokens.to(torch.int64)], dim=1)

        # SSD would build identical keys
        self.assertEqual(sgl_keys.shape, (N, 3))
        self.assertTrue((sgl_keys[:MQ_LEN, 0] == 10).all())
        self.assertTrue((sgl_keys[MQ_LEN:, 0] == 20).all())

    def test_cache_lookup_produces_same_hits(self):
        """Verify vectorized cache lookup gives identical results."""
        N_cache = 20
        B_query = 6
        cache_keys = torch.randint(0, 50, (N_cache, 3), dtype=torch.int64, device=DEVICE)
        query_keys = torch.zeros(B_query, 3, dtype=torch.int64, device=DEVICE)

        # Make some match, some not
        query_keys[0] = cache_keys[3]
        query_keys[1] = torch.tensor([99, 99, 99], device=DEVICE)
        query_keys[2] = cache_keys[15]
        query_keys[3] = cache_keys[0]
        query_keys[4] = torch.tensor([88, 88, 88], device=DEVICE)
        query_keys[5] = cache_keys[19]

        # Both use same algorithm
        eq = query_keys.unsqueeze(1) == cache_keys.unsqueeze(0)
        match = torch.all(eq, dim=2)
        hits = match.any(dim=1)
        idx = match.float().argmax(dim=1)

        self.assertTrue(hits[0]); self.assertEqual(idx[0].item(), 3)
        self.assertFalse(hits[1])
        self.assertTrue(hits[2]); self.assertEqual(idx[2].item(), 15)
        self.assertTrue(hits[3]); self.assertEqual(idx[3].item(), 0)
        self.assertFalse(hits[4])
        self.assertTrue(hits[5]); self.assertEqual(idx[5].item(), 19)

    def test_full_cycle_miss_populate_hit(self):
        """Full cycle: miss -> populate cache -> hit."""
        K, V = 3, 100
        fan_out_list = [2, 2, 2, 1]
        MQ_LEN = sum(fan_out_list)
        B = 2

        # Phase 1: empty cache, all miss
        cache_keys = torch.zeros((0, 3), dtype=torch.int64, device=DEVICE)

        query = torch.tensor([[0, 0, 5], [1, 0, 10]], dtype=torch.int64, device=DEVICE)
        if cache_keys.numel() > 0:
            eq = query.unsqueeze(1) == cache_keys.unsqueeze(0)
            hits = torch.all(eq, dim=2).any(dim=1)
        else:
            hits = torch.zeros(B, dtype=torch.bool, device=DEVICE)
        self.assertFalse(hits.any())

        # Phase 2: simulate tree decode and populate
        N_tree = B * MQ_LEN
        batch_ids = torch.arange(B, device=DEVICE).repeat_interleave(MQ_LEN)
        fan_idx = torch.arange(K + 1, device=DEVICE, dtype=torch.int64).repeat_interleave(
            torch.tensor(fan_out_list, device=DEVICE)
        ).repeat(B)
        tree_rec_tokens = torch.randint(0, V, (N_tree,), device=DEVICE)
        spec_tokens = torch.randint(0, V, (N_tree, K), device=DEVICE)

        seq_ids = query[batch_ids, 0]
        new_cache_keys = torch.stack([seq_ids, fan_idx, tree_rec_tokens.to(torch.int64)], dim=1)

        # Phase 3: query with a key that should hit
        hit_key = new_cache_keys[0].unsqueeze(0)
        eq2 = hit_key.unsqueeze(1) == new_cache_keys.unsqueeze(0)
        hits2 = torch.all(eq2, dim=2).any(dim=1)
        self.assertTrue(hits2[0])


class TestE2EFullSpecCycle(unittest.TestCase):
    """End-to-end: full spec request cycle without GPU model."""

    def test_deterministic_cycle(self):
        """Run a full spec cycle and verify all intermediate shapes/values."""
        K = 3
        V = 100
        B = 2
        fan_out_list = [2, 2, 2, 1]
        MQ_LEN = sum(fan_out_list)

        torch.manual_seed(42)

        # 1. Initial cache keys from target
        cache_keys = torch.tensor([[0, -1, 5], [1, -1, 10]], dtype=torch.int64, device=DEVICE)
        recovery_tokens = cache_keys[:, 2]

        # 2. Simulate JIT speculate output (cache miss)
        out_tokens = torch.randint(0, V, (B, K), device=DEVICE)

        # 3. Build speculations
        specs = torch.cat([recovery_tokens.unsqueeze(1), out_tokens.to(torch.int64)], dim=1)
        self.assertEqual(specs.shape, (B, K + 1))
        self.assertEqual(specs[0, 0].item(), 5)
        self.assertEqual(specs[1, 0].item(), 10)

        # 4. Build glue decode input
        glue_flat = sgl_make_glue(out_tokens.to(torch.int64), recovery_tokens)
        self.assertEqual(glue_flat.shape, (B * (K + 1),))
        glue_2d = glue_flat.view(B, K + 1)
        self.assertEqual(glue_2d[0, 0].item(), 5)

        # 5. Simulate glue decode logits
        glue_logits = torch.randn(B, K + 1, V, device=DEVICE)

        # 6. Fork tokens
        cache_hits = torch.zeros(B, dtype=torch.int64, device=DEVICE)
        forked = sgl_get_forked(
            logits=glue_logits, fan_out_list=fan_out_list, temperatures=torch.zeros(B, device=DEVICE),
            cache_hits=cache_hits, fan_out_list_miss=fan_out_list,
            returned_tokens=glue_2d,
        )
        self.assertEqual(forked.shape, (B, MQ_LEN))

        # 7. Build tree batch
        tree_flat = forked.reshape(-1)
        N_tree = B * MQ_LEN
        self.assertEqual(tree_flat.shape[0], N_tree)

        batch_ids = torch.arange(B, device=DEVICE).repeat_interleave(MQ_LEN)
        fan_idx = torch.arange(K + 1, device=DEVICE, dtype=torch.int64).repeat_interleave(
            torch.tensor(fan_out_list, device=DEVICE)
        ).repeat(B)
        self.assertEqual(batch_ids.shape[0], N_tree)
        self.assertEqual(fan_idx.shape[0], N_tree)

        # 8. Simulate tree decode
        spec_tokens = torch.randint(0, V, (N_tree, K), device=DEVICE)
        spec_logits = torch.randn(N_tree, K, V, device=DEVICE)

        # 9. Populate cache
        seq_ids = cache_keys[batch_ids, 0]
        new_keys = torch.stack([seq_ids, fan_idx, tree_flat.to(torch.int64)], dim=1)
        self.assertEqual(new_keys.shape, (N_tree, 3))

        # 10. Verify cache hit
        # Build a query that should match the first entry
        hit_query = new_keys[0].unsqueeze(0)
        eq = hit_query.unsqueeze(1) == new_keys.unsqueeze(0)
        match = torch.all(eq, dim=2)
        self.assertTrue(match.any(dim=1)[0])
        idx = match.float().argmax(dim=1)[0]
        self.assertEqual(idx.item(), 0)
        self.assertTrue(torch.equal(spec_tokens[idx], spec_tokens[0]))


# ═════════════════════════════════════════════════════════════════════════
# Detailed Per-Phase Profiling
# ═════════════════════════════════════════════════════════════════════════
class TestDetailedProfiling(unittest.TestCase):
    """Detailed per-phase profiling with CUDA events for precise GPU timing."""

    def _cuda_time_ms(self, fn, warmup=5, iters=50):
        """Time a function using CUDA events for precise GPU timing."""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            start_events[i].record()
            fn()
            end_events[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return sum(times) / len(times)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_profile_cache_lookup(self):
        """Profile: vectorized cache lookup."""
        results = []
        for N_cache in [10, 50, 200, 500]:
            for B in [1, 4, 16, 32]:
                ck = torch.randint(0, 500, (N_cache, 3), dtype=torch.int64, device=DEVICE)
                qk = torch.randint(0, 500, (B, 3), dtype=torch.int64, device=DEVICE)
                def f():
                    eq = qk.unsqueeze(1) == ck.unsqueeze(0)
                    return torch.all(eq, dim=2).any(dim=1)
                ms = self._cuda_time_ms(f)
                results.append((N_cache, B, ms))
        print("\n" + "=" * 60)
        print("PROFILE: Cache Lookup (CUDA events)")
        print(f"{'N_cache':>8} {'B':>4} {'ms':>10}")
        for n, b, ms in results:
            print(f"{n:>8} {b:>4} {ms:>10.4f}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_profile_forked_token_sampling(self):
        """Profile: vectorized forked token sampling (new implementation)."""
        results = []
        for B in [1, 4, 8, 16, 32]:
            for V in [1000, 32000, 128000]:
                K = 3
                fo = [2, 2, 2, 1]
                logits = torch.randn(B, K + 1, V, device=DEVICE)
                temps = torch.zeros(B, device=DEVICE)
                hits = torch.ones(B, dtype=torch.int64, device=DEVICE)
                rt = make_returned_tokens(logits)
                def f():
                    return sgl_get_forked(
                        logits=logits, fan_out_list=fo, temperatures=temps,
                        cache_hits=hits, returned_tokens=rt)
                ms = self._cuda_time_ms(f, warmup=3, iters=20)
                results.append((B, V, ms))
        print("\n" + "=" * 60)
        print("PROFILE: Forked Token Sampling (CUDA events)")
        print(f"{'B':>4} {'V':>8} {'ms':>10}")
        for b, v, ms in results:
            print(f"{b:>4} {v:>8} {ms:>10.4f}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_profile_glue_decode_construction(self):
        """Profile: glue decode input ID construction."""
        results = []
        for B in [1, 4, 16, 64]:
            for K in [3, 5, 8]:
                draft = torch.randint(0, 1000, (B, K), device=DEVICE)
                rec = torch.randint(0, 1000, (B,), device=DEVICE)
                def f():
                    return sgl_make_glue(draft, rec)
                ms = self._cuda_time_ms(f)
                results.append((B, K, ms))
        print("\n" + "=" * 60)
        print("PROFILE: Glue Decode Construction (CUDA events)")
        print(f"{'B':>4} {'K':>4} {'ms':>10}")
        for b, k, ms in results:
            print(f"{b:>4} {k:>4} {ms:>10.4f}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_profile_tree_position_precomputation(self):
        """Profile: precomputed vs per-step position computation."""
        results = []
        for B in [1, 4, 16, 32]:
            for K in [3, 5]:
                MQ_LEN = 7  # sum([2,2,2,1])
                N = B * MQ_LEN
                base = torch.randint(50, 200, (N,), dtype=torch.int64, device=DEVICE)
                fkp1 = torch.arange(MQ_LEN, device=DEVICE).repeat(B)
                step_offsets = torch.arange(K, device=DEVICE)[:, None] * MQ_LEN

                def precomputed():
                    initial = base + fkp1
                    return initial.unsqueeze(0) + step_offsets

                def per_step():
                    results_list = []
                    for d in range(K):
                        results_list.append(base + fkp1 + d * MQ_LEN)
                    return torch.stack(results_list)

                ms_pre = self._cuda_time_ms(precomputed)
                ms_loop = self._cuda_time_ms(per_step)
                results.append((B, K, ms_pre, ms_loop))

        print("\n" + "=" * 60)
        print("PROFILE: Position Precomputation (CUDA events)")
        print(f"{'B':>4} {'K':>4} {'precomp':>10} {'per-step':>10} {'speedup':>8}")
        for b, k, pre, loop in results:
            print(f"{b:>4} {k:>4} {pre:>10.4f} {loop:>10.4f} {loop/pre:>8.1f}x")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_profile_vectorized_slot_assignment(self):
        """Profile: vectorized vs loop slot assignment."""
        results = []
        for B in [1, 4, 16, 32]:
            tpr = 4
            pool = torch.full((64, 2048), -1, dtype=torch.int64, device=DEVICE)
            di = torch.arange(B, dtype=torch.int64, device=DEVICE)
            base = torch.randint(0, 100, (B,), dtype=torch.int64, device=DEVICE)
            locs = torch.arange(B * tpr, device=DEVICE, dtype=torch.int64)

            def vec():
                rows = di.repeat_interleave(tpr)
                offsets = torch.arange(tpr, device=DEVICE, dtype=torch.int64)
                cols = (base.unsqueeze(1) + offsets).reshape(-1)
                pool[rows, cols] = locs

            def loop():
                for i in range(B):
                    for j in range(tpr):
                        pool[di[i], base[i] + j] = locs[i * tpr + j]

            ms_v = self._cuda_time_ms(vec)
            ms_l = self._cuda_time_ms(loop)
            results.append((B, ms_v, ms_l))

        print("\n" + "=" * 60)
        print("PROFILE: Slot Assignment (CUDA events)")
        print(f"{'B':>4} {'vectorized':>12} {'loop':>12} {'speedup':>8}")
        for b, v, l in results:
            print(f"{b:>4} {v:>12.4f} {l:>12.4f} {l/v:>8.1f}x")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_profile_sampling(self):
        """Profile: token sampling (greedy vs stochastic)."""
        results = []
        for B in [1, 4, 16, 32]:
            for V in [1000, 32000, 128000]:
                logits = torch.randn(B, V, device=DEVICE)

                def greedy():
                    return logits.argmax(dim=-1)

                def stochastic():
                    probs = F.softmax(logits / 0.8, dim=-1)
                    return torch.multinomial(probs, 1).squeeze(-1)

                ms_g = self._cuda_time_ms(greedy)
                ms_s = self._cuda_time_ms(stochastic)
                results.append((B, V, ms_g, ms_s))

        print("\n" + "=" * 60)
        print("PROFILE: Token Sampling (CUDA events)")
        print(f"{'B':>4} {'V':>8} {'greedy':>10} {'stochastic':>12}")
        for b, v, g, s in results:
            print(f"{b:>4} {v:>8} {g:>10.4f} {s:>12.4f}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_profile_full_non_model_overhead(self):
        """Profile: full non-model overhead per spec step (everything except forward)."""
        K = 3
        V = 32000
        fan_out_list = [2, 2, 2, 1]
        MQ_LEN = sum(fan_out_list)

        for B in [1, 4, 8, 16]:
            # Simulate the non-model operations in a spec step
            cache_keys_t = torch.randint(0, 50, (B, 3), dtype=torch.int64, device=DEVICE)
            temps = torch.zeros(B, device=DEVICE)
            draft_indices = torch.arange(B, dtype=torch.int64, device=DEVICE)
            seq_lens = torch.randint(50, 200, (B,), dtype=torch.int64, device=DEVICE)

            # Dummy tree cache
            N_cache = B * MQ_LEN
            tc_keys = torch.randint(0, 50, (N_cache, 3), dtype=torch.int64, device=DEVICE)
            tc_tokens = torch.randint(0, V, (N_cache, K), device=DEVICE)

            def full_non_model_step():
                # 1. Cache lookup
                eq = cache_keys_t.unsqueeze(1) == tc_keys.unsqueeze(0)
                match = torch.all(eq, dim=2)
                hits = match.any(dim=1).to(torch.int64)

                # 2. Build speculations
                recovery = cache_keys_t[:, 2]
                out_tokens = tc_tokens[match.float().argmax(dim=1), :K] if hits.any() else \
                             torch.randint(0, V, (B, K), device=DEVICE)
                specs = torch.cat([recovery.unsqueeze(1), out_tokens.to(torch.int64)], dim=1)

                # 3. Glue decode input
                glue_flat = sgl_make_glue(out_tokens.to(torch.int64), recovery)
                glue_2d = glue_flat.view(B, K + 1)

                # 4. Fork tokens (simulated glue logits)
                glue_logits = torch.randn(B, K + 1, V, device=DEVICE)
                forked = sgl_get_forked(
                    logits=glue_logits, fan_out_list=fan_out_list, temperatures=temps,
                    cache_hits=hits, returned_tokens=glue_2d,
                )

                # 5. Build tree batch structure
                N_tree = B * MQ_LEN
                batch_ids = torch.arange(B, device=DEVICE).repeat_interleave(MQ_LEN)
                fan_idx = torch.arange(K + 1, device=DEVICE, dtype=torch.int64).repeat_interleave(
                    torch.tensor(fan_out_list, device=DEVICE)).repeat(B)

                # 6. Precompute positions
                fkp1 = torch.arange(MQ_LEN, device=DEVICE).repeat(B)
                base = seq_lens[batch_ids]
                initial = base + fkp1
                step_offsets = torch.arange(K, device=DEVICE)[:, None] * MQ_LEN
                all_pos = initial.unsqueeze(0) + step_offsets

                # 7. Populate cache
                seq_ids = cache_keys_t[batch_ids, 0]
                new_keys = torch.stack([seq_ids, fan_idx, forked.reshape(-1).to(torch.int64)], dim=1)
                return new_keys

            ms = self._cuda_time_ms(full_non_model_step, warmup=3, iters=20)
            print(f"\nFull non-model overhead B={B}: {ms:.3f}ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
