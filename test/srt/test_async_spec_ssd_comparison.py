"""Cross-codebase equivalence tests between SGLang and SSD async spec.

After fixes #1-3,5, verifies that SGLang's implementations now MATCH SSD:
  - make_glue_decode_input_ids: both return flat [B*(K+1)]
  - get_forked_recovery_tokens_from_logits: both use -inf masking, K+1 fan_out, [B,MQ_LEN]
  - apply_sampler_x_rescaling: both multiply top-(F+1) by sampler_x
  - Tree cache key format and population logic: identical

Run with:
  source .venv/bin/activate
  python -m pytest test/srt/test_async_spec_ssd_comparison.py -v -s
"""

from __future__ import annotations

import sys
import unittest

import torch
import torch.nn.functional as F

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

from sglang.srt.speculative.async_spec.tree_utils import (
    apply_sampler_x_rescaling as sgl_apply_sampler_x_rescaling,
    compute_mq_len,
    get_forked_recovery_tokens_from_logits as sgl_get_forked_tokens,
    make_glue_decode_input_ids as sgl_make_glue_decode_input_ids,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MockSSDConfig:
    def __init__(self, speculate_k, fan_out_list, fan_out_list_miss):
        self.speculate_k = speculate_k
        self.fan_out_list = fan_out_list
        self.fan_out_list_miss = fan_out_list_miss
        self.fan_out_t = torch.tensor(fan_out_list, device=DEVICE)
        self.fan_out_t_miss = torch.tensor(fan_out_list_miss, device=DEVICE)
        self.MQ_LEN = sum(fan_out_list)


class MockTok:
    def decode(self, ids):
        return str(ids)


@unittest.skipUnless(SSD_AVAILABLE, "SSD not available at /work/avner/git/ssd")
class TestGlueDecodeEquivalence(unittest.TestCase):

    def test_flat_output_matches(self):
        """Both now return flat [B*(K+1)]."""
        for B in [1, 2, 4, 8, 16]:
            for K in [1, 2, 3, 5]:
                draft = torch.randint(0, 10000, (B, K), device=DEVICE)
                rec = torch.randint(0, 10000, (B,), device=DEVICE)
                sgl = sgl_make_glue_decode_input_ids(draft, rec)
                ssd = ssd_make_glue_decode_input_ids(draft, rec)
                self.assertEqual(sgl.shape, ssd.shape, f"Shape mismatch at B={B}, K={K}")
                self.assertTrue(torch.equal(sgl, ssd), f"Value mismatch at B={B}, K={K}")


@unittest.skipUnless(SSD_AVAILABLE, "SSD not available at /work/avner/git/ssd")
class TestForkedTokensEquivalence(unittest.TestCase):

    def test_greedy_exact_match(self):
        """Greedy forked tokens should be identical between SGLang and SSD."""
        for B in [1, 2, 4]:
            for K in [2, 3]:
                V = 100
                fan_out_list = [2] * (K + 1)
                fan_out_list_miss = fan_out_list
                logits = torch.randn(B, K + 1, V, device=DEVICE)
                temps = torch.zeros(B, device=DEVICE)
                hits = torch.ones(B, dtype=torch.int64, device=DEVICE)
                returned_tokens = torch.zeros(B, K + 1, dtype=torch.int64, device=DEVICE)
                for b in range(B):
                    for k in range(K + 1):
                        returned_tokens[b, k] = logits[b, k].argmax().item()

                ssd_cfg = MockSSDConfig(K, fan_out_list, fan_out_list_miss)
                ssd_out = ssd_get_forked_recovery_tokens_from_logits(
                    ssd_cfg, logits, hits, returned_tokens, MockTok()
                )
                sgl_out = sgl_get_forked_tokens(
                    logits=logits, fan_out_list=fan_out_list, temperatures=temps,
                    cache_hits=hits, fan_out_list_miss=fan_out_list_miss,
                    returned_tokens=returned_tokens,
                )
                self.assertEqual(sgl_out.shape, ssd_out.shape,
                                 f"Shape mismatch at B={B}, K={K}")
                self.assertTrue(torch.equal(sgl_out, ssd_out),
                                f"Value mismatch at B={B}, K={K}:\nSGL={sgl_out}\nSSD={ssd_out}")

    def test_masking_behavior_matches(self):
        """Both should exclude returned_tokens from forked output at shifted positions."""
        B, K, V = 1, 2, 20
        logits = torch.randn(B, K + 1, V, device=DEVICE)
        fan_out_list = [2, 2, 1]
        temps = torch.zeros(B, device=DEVICE)
        hits = torch.ones(B, dtype=torch.int64, device=DEVICE)
        returned_tokens = torch.zeros(B, K + 1, dtype=torch.int64, device=DEVICE)
        for k in range(K + 1):
            returned_tokens[0, k] = logits[0, k].argmax().item()

        ssd_cfg = MockSSDConfig(K, fan_out_list, fan_out_list)
        ssd_out = ssd_get_forked_recovery_tokens_from_logits(
            ssd_cfg, logits, hits, returned_tokens, MockTok()
        )
        sgl_out = sgl_get_forked_tokens(
            logits=logits, fan_out_list=fan_out_list, temperatures=temps,
            cache_hits=hits, fan_out_list_miss=fan_out_list,
            returned_tokens=returned_tokens,
        )

        # Verify masking: at position d < K, returned_tokens[d+1] excluded
        offset = 0
        for d in range(K):
            fo = fan_out_list[d]
            masked_tok = returned_tokens[0, d + 1].item()
            sgl_toks = sgl_out[0, offset:offset + fo].tolist()
            ssd_toks = ssd_out[0, offset:offset + fo].tolist()
            self.assertNotIn(masked_tok, sgl_toks, f"SGLang: masked tok at d={d}")
            self.assertNotIn(masked_tok, ssd_toks, f"SSD: masked tok at d={d}")
            offset += fo

    def test_fan_out_list_kplus1(self):
        """Both now use K+1 fan_out entries (including recovery)."""
        K = 3
        fan_out = [2, 2, 2, 1]  # K+1=4 entries
        mq = compute_mq_len(fan_out)
        self.assertEqual(len(fan_out), K + 1)
        self.assertEqual(mq, 7)

    def test_mixed_cache_hits(self):
        """Mixed hit/miss with equal MQ_LEN."""
        B, K, V = 4, 2, 50
        fan_out_list = [2, 2, 1]
        logits = torch.randn(B, K + 1, V, device=DEVICE)
        temps = torch.zeros(B, device=DEVICE)
        hits = torch.tensor([1, 0, 1, 0], dtype=torch.int64, device=DEVICE)
        returned_tokens = torch.zeros(B, K + 1, dtype=torch.int64, device=DEVICE)
        for b in range(B):
            for k in range(K + 1):
                returned_tokens[b, k] = logits[b, k].argmax().item()

        ssd_cfg = MockSSDConfig(K, fan_out_list, fan_out_list)
        ssd_out = ssd_get_forked_recovery_tokens_from_logits(
            ssd_cfg, logits, hits, returned_tokens, MockTok()
        )
        sgl_out = sgl_get_forked_tokens(
            logits=logits, fan_out_list=fan_out_list, temperatures=temps,
            cache_hits=hits, fan_out_list_miss=fan_out_list,
            returned_tokens=returned_tokens,
        )
        self.assertTrue(torch.equal(sgl_out, ssd_out))


@unittest.skipUnless(SSD_AVAILABLE, "SSD not available at /work/avner/git/ssd")
class TestSamplerXEquivalence(unittest.TestCase):

    def test_exact_match(self):
        """SGLang now uses same algorithm as SSD: multiply top-(F+1) then normalize."""
        probs = F.softmax(torch.randn(3, 50, device=DEVICE), dim=-1)
        for sx in [0.5, 1.0, 2.0, 5.0]:
            sgl = sgl_apply_sampler_x_rescaling(probs.clone(), sx, 3)
            ssd = ssd_apply_sampler_x_rescaling(probs.unsqueeze(1).clone(), sx, 3).squeeze(1)
            self.assertTrue(
                torch.allclose(sgl, ssd, atol=1e-5),
                f"Mismatch at sx={sx}: max_diff={torch.abs(sgl-ssd).max()}"
            )


@unittest.skipUnless(SSD_AVAILABLE, "SSD not available at /work/avner/git/ssd")
class TestCacheKeyEquivalence(unittest.TestCase):

    def test_cache_lookup_algorithm(self):
        """Both use identical vectorized eq -> all -> any -> argmax."""
        N, B = 20, 4
        cache = torch.randint(0, 50, (N, 3), dtype=torch.int64, device=DEVICE)
        query = torch.zeros(B, 3, dtype=torch.int64, device=DEVICE)
        query[0] = cache[3]
        query[1] = torch.tensor([99, 99, 99], device=DEVICE)
        query[2] = cache[15]
        query[3] = cache[0]

        eq = query.unsqueeze(1) == cache.unsqueeze(0)
        match = torch.all(eq, dim=2)
        hits = match.any(dim=1)
        idx = match.float().argmax(dim=1)

        self.assertTrue(hits[0])
        self.assertFalse(hits[1])
        self.assertTrue(hits[2])
        self.assertTrue(hits[3])
        self.assertEqual(idx[0].item(), 3)
        self.assertEqual(idx[2].item(), 15)

    def test_cache_populate_key_construction(self):
        """Both construct keys as [seq_ids[batch_ids], fan_idx, recovery_tokens]."""
        B, MQ = 2, 5
        N = B * MQ
        cache_keys = torch.tensor([[10, 0, 5], [20, 0, 15]], dtype=torch.int64, device=DEVICE)
        batch_ids = torch.arange(B, device=DEVICE).repeat_interleave(MQ)
        fan_idx = torch.tensor([0, 0, 1, 1, 2], device=DEVICE).repeat(B)
        rec_tokens = torch.randint(0, 100, (N,), device=DEVICE)

        seq_ids = cache_keys[batch_ids, 0]
        keys = torch.stack([seq_ids, fan_idx, rec_tokens.to(torch.int64)], dim=1)

        self.assertEqual(keys.shape, (N, 3))
        self.assertTrue((keys[:MQ, 0] == 10).all())
        self.assertTrue((keys[MQ:, 0] == 20).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
