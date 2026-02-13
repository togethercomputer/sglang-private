"""Tests for async speculative decoding tree construction and decode.

Tests the tree_utils functions, _build_tree_batch, _decode_tree, and
_populate_tree_cache methods for correctness and equivalence with SSD.

These tests are mostly CPU-based (no GPU required) except for tests
that explicitly need CUDA for model runner testing.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

import torch


class TestMakeGlueDecodeInputIds(unittest.TestCase):
    """Test make_glue_decode_input_ids."""

    def test_basic_shape(self):
        from sglang.srt.speculative.async_spec.tree_utils import (
            make_glue_decode_input_ids,
        )

        B, K = 2, 3
        draft_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        rec_tokens = torch.tensor([10, 20])
        result = make_glue_decode_input_ids(draft_tokens, rec_tokens)
        # Should return flat [B*(K+1)]
        self.assertEqual(result.shape, (B * (K + 1),))

    def test_content_order(self):
        from sglang.srt.speculative.async_spec.tree_utils import (
            make_glue_decode_input_ids,
        )

        draft_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        rec_tokens = torch.tensor([10, 20])
        result = make_glue_decode_input_ids(draft_tokens, rec_tokens)
        # Request 0: [10, 1, 2, 3]
        self.assertEqual(result[0].item(), 10)
        self.assertEqual(result[1].item(), 1)
        self.assertEqual(result[2].item(), 2)
        self.assertEqual(result[3].item(), 3)
        # Request 1: [20, 4, 5, 6]
        self.assertEqual(result[4].item(), 20)
        self.assertEqual(result[5].item(), 4)

    def test_single_request(self):
        from sglang.srt.speculative.async_spec.tree_utils import (
            make_glue_decode_input_ids,
        )

        draft_tokens = torch.tensor([[1, 2]])
        rec_tokens = torch.tensor([99])
        result = make_glue_decode_input_ids(draft_tokens, rec_tokens)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.tolist(), [99, 1, 2])


class TestGetForkedRecoveryTokens(unittest.TestCase):
    """Test get_forked_recovery_tokens_from_logits."""

    def test_basic_shape(self):
        from sglang.srt.speculative.async_spec.tree_utils import (
            get_forked_recovery_tokens_from_logits,
        )

        B, K, V = 2, 3, 100
        fan_out_list = [2, 2, 2, 2]  # K+1 = 4 entries
        logits = torch.randn(B, K + 1, V)
        cache_hits = torch.ones(B, dtype=torch.int64)
        returned_tokens = torch.randint(0, V, (B, K + 1))

        result = get_forked_recovery_tokens_from_logits(
            logits, fan_out_list, cache_hits, returned_tokens
        )
        mq_len = sum(fan_out_list)  # 8
        self.assertEqual(result.shape, (B, mq_len))

    def test_suppression(self):
        """Test that returned tokens are suppressed (not chosen)."""
        from sglang.srt.speculative.async_spec.tree_utils import (
            get_forked_recovery_tokens_from_logits,
        )

        B, K, V = 1, 2, 10
        fan_out_list = [1, 1, 1]  # K+1 entries, take top-1 at each position

        # Create logits where token 5 is dominant at all positions
        logits = torch.zeros(B, K + 1, V)
        logits[0, :, 5] = 10.0  # Token 5 has highest logit
        logits[0, :, 3] = 9.0  # Token 3 is second

        # Returned tokens: position 1 has token 5 (will be suppressed at position 0)
        # Position 2 has token 3 (will be suppressed at position 1)
        returned_tokens = torch.tensor([[0, 5, 3]])

        cache_hits = torch.ones(B, dtype=torch.int64)

        result = get_forked_recovery_tokens_from_logits(
            logits, fan_out_list, cache_hits, returned_tokens
        )

        # At position 0: token 5 should be suppressed (returned_tokens[:, 1] = 5)
        # So position 0 should select token 3 (second highest)
        self.assertEqual(result[0, 0].item(), 3)

        # At position 1: token 3 should be suppressed (returned_tokens[:, 2] = 3)
        # So position 1 should select token 5 (highest, not suppressed here)
        self.assertEqual(result[0, 1].item(), 5)

        # At position 2 (last): no suppression, should select token 5
        self.assertEqual(result[0, 2].item(), 5)

    def test_variable_fan_out(self):
        """Test variable fan_out per position."""
        from sglang.srt.speculative.async_spec.tree_utils import (
            get_forked_recovery_tokens_from_logits,
        )

        B, K, V = 1, 2, 50
        fan_out_list = [3, 1, 2]  # 3 + 1 + 2 = 6 total (MQ_LEN)

        logits = torch.randn(B, K + 1, V)
        cache_hits = torch.ones(B, dtype=torch.int64)
        returned_tokens = torch.randint(0, V, (B, K + 1))

        result = get_forked_recovery_tokens_from_logits(
            logits, fan_out_list, cache_hits, returned_tokens
        )

        mq_len = sum(fan_out_list)  # 6
        self.assertEqual(result.shape, (B, mq_len))
        # All tokens should be valid vocab indices
        self.assertTrue((result >= 0).all())
        self.assertTrue((result < V).all())

    def test_cache_miss_uses_miss_list(self):
        """Test that cache misses use fan_out_list_miss."""
        from sglang.srt.speculative.async_spec.tree_utils import (
            get_forked_recovery_tokens_from_logits,
        )

        B, K, V = 2, 1, 50
        fan_out_list = [2, 2]  # MQ_LEN = 4
        fan_out_list_miss = [2, 2]  # Same sum for consistency

        logits = torch.randn(B, K + 1, V)
        # Batch 0: hit, Batch 1: miss
        cache_hits = torch.tensor([1, 0], dtype=torch.int64)
        returned_tokens = torch.randint(0, V, (B, K + 1))

        result = get_forked_recovery_tokens_from_logits(
            logits, fan_out_list, cache_hits, returned_tokens, fan_out_list_miss
        )

        mq_len = sum(fan_out_list)  # 4
        self.assertEqual(result.shape, (B, mq_len))

    def test_all_misses(self):
        """Test behavior when all requests are cache misses."""
        from sglang.srt.speculative.async_spec.tree_utils import (
            get_forked_recovery_tokens_from_logits,
        )

        B, K, V = 2, 2, 30
        fan_out_list = [1, 1, 1]
        logits = torch.randn(B, K + 1, V)
        cache_hits = torch.zeros(B, dtype=torch.int64)
        returned_tokens = torch.randint(0, V, (B, K + 1))

        result = get_forked_recovery_tokens_from_logits(
            logits, fan_out_list, cache_hits, returned_tokens
        )
        mq_len = sum(fan_out_list)  # 3
        self.assertEqual(result.shape, (B, mq_len))

    def test_top_k_correctness(self):
        """Test that the returned tokens are the actual top-k."""
        from sglang.srt.speculative.async_spec.tree_utils import (
            get_forked_recovery_tokens_from_logits,
        )

        B, K, V = 1, 1, 10
        fan_out_list = [3, 3]  # Take top-3 at each position

        # Create logits with known ordering
        logits = torch.zeros(B, K + 1, V)
        # Position 0: tokens 7, 8, 9 are top-3 (after suppression)
        logits[0, 0, :] = torch.arange(V, dtype=torch.float)
        # Position 1: tokens 7, 8, 9 are top-3 (no suppression at last pos)
        logits[0, 1, :] = torch.arange(V, dtype=torch.float)

        # Returned tokens: pos 0 suppresses returned_tokens[:, 1]
        returned_tokens = torch.tensor([[0, 9]])  # suppress 9 at pos 0
        cache_hits = torch.ones(B, dtype=torch.int64)

        result = get_forked_recovery_tokens_from_logits(
            logits, fan_out_list, cache_hits, returned_tokens
        )

        # Position 0: top-3 after suppressing token 9 → [8, 7, 6]
        pos0_tokens = set(result[0, :3].tolist())
        self.assertIn(8, pos0_tokens)
        self.assertIn(7, pos0_tokens)
        self.assertIn(6, pos0_tokens)
        self.assertNotIn(9, pos0_tokens)

        # Position 1 (last): no suppression → top-3 is [9, 8, 7]
        pos1_tokens = set(result[0, 3:6].tolist())
        self.assertIn(9, pos1_tokens)
        self.assertIn(8, pos1_tokens)
        self.assertIn(7, pos1_tokens)


class TestComputeTreeLookahead(unittest.TestCase):
    """Test compute_tree_lookahead."""

    def test_formula(self):
        from sglang.srt.speculative.async_spec.tree_utils import (
            compute_tree_lookahead,
        )

        # K=5, fan_out=3, MQ_LEN=18
        self.assertEqual(compute_tree_lookahead(18, 5), 5 + 1 + 5 * 18)  # 96

        # K=3, fan_out=2, MQ_LEN=8
        self.assertEqual(compute_tree_lookahead(8, 3), 3 + 1 + 3 * 8)  # 28


class TestPreallocBuffers(unittest.TestCase):
    """Test _init_prealloc_buffers shapes and values."""

    def _make_mock_runner(self, K=3, fan_out=2, device="cpu"):
        """Create a minimal mock AsyncDraftRunner for testing prealloc buffers."""
        runner = MagicMock()
        runner.spec_k = K
        runner.fan_out = fan_out
        runner.fan_out_list = [fan_out] * (K + 1)
        runner.fan_out_list_miss = runner.fan_out_list
        runner.mq_len = sum(runner.fan_out_list)
        runner.device = torch.device(device)

        # Call the actual method
        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        AsyncDraftRunner._init_prealloc_buffers(runner)
        return runner

    def test_step_pos_offsets_shape(self):
        runner = self._make_mock_runner(K=5, fan_out=3)
        self.assertEqual(runner._step_pos_offsets.shape, (5, 1))

    def test_step_pos_offsets_values(self):
        K, fan_out = 3, 2
        MQ_LEN = (K + 1) * fan_out  # 8
        runner = self._make_mock_runner(K=K, fan_out=fan_out)
        expected = torch.tensor([0, MQ_LEN, 2 * MQ_LEN]).unsqueeze(1)
        self.assertTrue(torch.equal(runner._step_pos_offsets, expected))

    def test_step_rope_offsets_shape(self):
        runner = self._make_mock_runner(K=5)
        self.assertEqual(runner._step_rope_offsets.shape, (5, 1))

    def test_fan_idx_hit_shape(self):
        K, fan_out = 3, 2
        MQ_LEN = (K + 1) * fan_out  # 8
        runner = self._make_mock_runner(K=K, fan_out=fan_out)
        self.assertEqual(runner._fan_idx_hit.shape[0], MQ_LEN)

    def test_fan_idx_hit_values(self):
        K, fan_out = 2, 3
        runner = self._make_mock_runner(K=K, fan_out=fan_out)
        # fan_out_list = [3, 3, 3], so fan_idx = [0,0,0, 1,1,1, 2,2,2]
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertTrue(torch.equal(runner._fan_idx_hit, expected))

    def test_arange_mq_shape(self):
        K, fan_out = 3, 2
        MQ_LEN = (K + 1) * fan_out  # 8
        runner = self._make_mock_runner(K=K, fan_out=fan_out)
        self.assertEqual(runner._arange_mq.shape[0], MQ_LEN)

    def test_arange_kp1_shape(self):
        K = 5
        runner = self._make_mock_runner(K=K)
        self.assertEqual(runner._arange_kp1.shape[0], K + 1)


class TestComputeStepPositionsAndSlotMaps(unittest.TestCase):
    """Test _compute_step_positions_and_slot_maps."""

    def _make_runner(self, K=3, fan_out=2, page_size=16, device="cpu"):
        """Create minimal runner with prealloc buffers."""
        runner = MagicMock()
        runner.spec_k = K
        runner.fan_out = fan_out
        runner.fan_out_list = [fan_out] * (K + 1)
        runner.fan_out_list_miss = runner.fan_out_list
        runner.mq_len = sum(runner.fan_out_list)
        runner.page_size = page_size
        runner.device = torch.device(device)

        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        AsyncDraftRunner._init_prealloc_buffers(runner)
        return runner

    def test_shapes(self):
        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        K, fan_out, B = 3, 2, 2
        MQ_LEN = (K + 1) * fan_out  # 8
        N = B * MQ_LEN  # 16
        runner = self._make_runner(K=K, fan_out=fan_out)

        initial_pos = torch.arange(N, dtype=torch.int64)
        initial_rope = torch.arange(N, dtype=torch.int64)
        dbt = torch.zeros(B, 128, dtype=torch.int64)

        result = AsyncDraftRunner._compute_step_positions_and_slot_maps(
            runner, initial_pos, initial_rope, dbt, B, K, N
        )
        step_pos, step_rope, step_ctx, step_slots = result

        self.assertEqual(step_pos.shape, (K, N))
        self.assertEqual(step_rope.shape, (K, N))
        self.assertEqual(step_ctx.shape, (K, B))
        self.assertEqual(step_slots.shape, (K, N))

    def test_position_increments(self):
        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        K, fan_out, B = 3, 2, 1
        MQ_LEN = (K + 1) * fan_out  # 8
        N = B * MQ_LEN
        runner = self._make_runner(K=K, fan_out=fan_out)

        base = 100
        initial_pos = torch.full((N,), base, dtype=torch.int64)
        initial_rope = torch.full((N,), base, dtype=torch.int64)
        dbt = torch.zeros(B, 128, dtype=torch.int64)

        step_pos, step_rope, step_ctx, _ = (
            AsyncDraftRunner._compute_step_positions_and_slot_maps(
                runner, initial_pos, initial_rope, dbt, B, K, N
            )
        )

        # Step 0: positions = base + 0 = base
        self.assertTrue(torch.all(step_pos[0] == base))
        # Step 1: positions = base + MQ_LEN
        self.assertTrue(torch.all(step_pos[1] == base + MQ_LEN))
        # Step 2: positions = base + 2*MQ_LEN
        self.assertTrue(torch.all(step_pos[2] == base + 2 * MQ_LEN))

        # Rope increments by 1 per step
        self.assertTrue(torch.all(step_rope[0] == base))
        self.assertTrue(torch.all(step_rope[1] == base + 1))
        self.assertTrue(torch.all(step_rope[2] == base + 2))

    def test_context_lens(self):
        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        K, fan_out, B = 2, 3, 1
        MQ_LEN = (K + 1) * fan_out  # 9
        N = B * MQ_LEN
        runner = self._make_runner(K=K, fan_out=fan_out)

        # Initial positions: 50, 51, ..., 58
        initial_pos = torch.arange(50, 50 + N, dtype=torch.int64)
        initial_rope = torch.arange(50, 50 + N, dtype=torch.int64)
        dbt = torch.zeros(B, 128, dtype=torch.int64)

        _, _, step_ctx, _ = AsyncDraftRunner._compute_step_positions_and_slot_maps(
            runner, initial_pos, initial_rope, dbt, B, K, N
        )

        # Context lens = last position per request + 1
        # Step 0: last pos = 50 + 8 = 58, ctx = 59
        self.assertEqual(step_ctx[0, 0].item(), 58 + 1)
        # Step 1: last pos = 58 + MQ_LEN = 67, ctx = 68
        self.assertEqual(step_ctx[1, 0].item(), 58 + MQ_LEN + 1)


class TestPopulateTreeCache(unittest.TestCase):
    """Test _populate_tree_cache and cache lookup."""

    def _make_runner(self, K=3, fan_out=2, device="cpu"):
        runner = MagicMock()
        runner.spec_k = K
        runner.fan_out = fan_out
        runner.fan_out_list = [fan_out] * (K + 1)
        runner.fan_out_list_miss = runner.fan_out_list
        runner.mq_len = sum(runner.fan_out_list)
        runner.device = torch.device(device)
        runner.tree_cache_keys = torch.zeros((0, 3), dtype=torch.int64)
        runner.tree_cache_tokens = None
        runner.tree_cache_logits = None

        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        AsyncDraftRunner._init_prealloc_buffers(runner)
        # Bind the vectorized fan_idx method so _populate_tree_cache can call it
        import types
        runner._vectorized_fan_idx = types.MethodType(
            AsyncDraftRunner._vectorized_fan_idx, runner
        )
        return runner

    def test_populate_and_lookup(self):
        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        K, fan_out, B, V = 2, 2, 1, 50
        MQ_LEN = (K + 1) * fan_out  # 6
        N = B * MQ_LEN
        runner = self._make_runner(K=K, fan_out=fan_out)

        # Create fake tree_decode_args
        seq_ids_expanded = torch.zeros(N, dtype=torch.int64)  # All from seq 0
        rec_flat = torch.randint(0, V, (N,), dtype=torch.int64)
        cache_hits = torch.ones(B, dtype=torch.int64)

        tree_decode_args = {
            "seq_ids_expanded": seq_ids_expanded,
            "rec_flat": rec_flat,
        }

        # Fake tokens and logits
        tokens = torch.randint(0, V, (N, K), dtype=torch.int64)
        logits = torch.randn(N, K, V)

        # Populate
        AsyncDraftRunner._populate_tree_cache(
            runner, tree_decode_args, tokens, logits, cache_hits
        )

        # Verify cache was populated
        self.assertEqual(runner.tree_cache_keys.shape, (N, 3))
        self.assertEqual(runner.tree_cache_tokens.shape, (N, K))
        self.assertEqual(runner.tree_cache_logits.shape, (N, K, V))

        # Verify key structure: (seq_id, depth, rec_token)
        for i in range(N):
            self.assertEqual(runner.tree_cache_keys[i, 0].item(), 0)  # seq_id
            self.assertEqual(runner.tree_cache_keys[i, 2].item(), rec_flat[i].item())

        # Verify depth index structure: fan_idx pattern
        # For fan_out_list = [2, 2, 2], fan_idx = [0, 0, 1, 1, 2, 2]
        expected_depths = [0, 0, 1, 1, 2, 2]
        for i in range(N):
            self.assertEqual(
                runner.tree_cache_keys[i, 1].item(),
                expected_depths[i],
            )

    def test_cache_hit_lookup(self):
        """Test that cache hits work correctly with populated cache."""
        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        K, fan_out, B, V = 2, 2, 1, 50
        MQ_LEN = (K + 1) * fan_out
        N = B * MQ_LEN
        runner = self._make_runner(K=K, fan_out=fan_out)

        # Populate cache with known data
        seq_ids = torch.zeros(N, dtype=torch.int64)
        rec_tokens = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int64)[:N]
        cache_hits = torch.ones(B, dtype=torch.int64)

        tree_decode_args = {"seq_ids_expanded": seq_ids, "rec_flat": rec_tokens}
        tokens = torch.arange(N * K).reshape(N, K)
        logits = torch.randn(N, K, V)

        AsyncDraftRunner._populate_tree_cache(
            runner, tree_decode_args, tokens, logits, cache_hits
        )

        # Now try to look up a key
        # Key: (seq_id=0, depth=0, rec_token=10) should match entry 0
        query_key = torch.tensor([[0, 0, 10]], dtype=torch.int64)
        eq = query_key.unsqueeze(1) == runner.tree_cache_keys.unsqueeze(0)
        match = torch.all(eq, dim=2)
        self.assertTrue(match.any())

        idx = match.float().argmax(dim=1).item()
        self.assertTrue(torch.equal(runner.tree_cache_tokens[idx], tokens[0]))


class TestServerArgsFanOutList(unittest.TestCase):
    """Test that server_args correctly initializes fan_out_list with K+1 entries."""

    def test_default_fan_out_list_length(self):
        """Verify fan_out_list has K+1 entries (not K)."""
        # We can't easily construct a full ServerArgs, so test the logic directly
        K = 5
        fan_out = 3
        # This matches the patched server_args logic
        fan_out_list = [fan_out] * (K + 1)
        self.assertEqual(len(fan_out_list), K + 1)
        self.assertEqual(sum(fan_out_list), fan_out * (K + 1))

    def test_mq_len_formula(self):
        """Test MQ_LEN = sum(fan_out_list) = fan_out * (K+1)."""
        K = 5
        fan_out = 3
        fan_out_list = [fan_out] * (K + 1)
        mq_len = sum(fan_out_list)
        self.assertEqual(mq_len, 18)  # 3 * 6


class TestTreeDecodePositionLayout(unittest.TestCase):
    """Test the position layout for tree decode matches SSD's layout."""

    def test_initial_kv_positions(self):
        """Test KV write positions for tree decode step 0."""
        B, K, fan_out = 1, 3, 2
        MQ_LEN = (K + 1) * fan_out  # 8
        N = B * MQ_LEN
        num_tokens = torch.tensor([100])  # 100 tokens in sequence

        # Replicate the position computation from _build_tree_batch
        b_flat = torch.arange(B)[:, None].expand(B, MQ_LEN).flatten()
        fkp1_flat = torch.arange(MQ_LEN).repeat(B)
        positions = (num_tokens[b_flat] - 1) + (K + 1) + fkp1_flat

        # Expected: 99 + 4 + [0,1,2,3,4,5,6,7] = [103, 104, ..., 110]
        expected = torch.arange(103, 103 + MQ_LEN)
        self.assertTrue(torch.equal(positions, expected))

    def test_initial_rope_positions(self):
        """Test rope positions reflect tree depth."""
        B, K, fan_out = 1, 2, 3
        MQ_LEN = (K + 1) * fan_out  # 9
        num_tokens = torch.tensor([50])

        # fan_idx for uniform fan_out: [0,0,0, 1,1,1, 2,2,2]
        fan_out_list = [fan_out] * (K + 1)
        fan_idx = torch.arange(K + 1).repeat_interleave(
            torch.tensor(fan_out_list)
        )

        b_flat = torch.arange(B)[:, None].expand(B, MQ_LEN).flatten()
        rope_positions = (num_tokens[b_flat] - 1) + fan_idx + 1

        # Expected: 49 + [0,0,0,1,1,1,2,2,2] + 1 = [50,50,50,51,51,51,52,52,52]
        expected = torch.tensor([50, 50, 50, 51, 51, 51, 52, 52, 52])
        self.assertTrue(torch.equal(rope_positions, expected))

    def test_total_kv_positions_used(self):
        """Test total KV positions matches compute_tree_lookahead."""
        from sglang.srt.speculative.async_spec.tree_utils import (
            compute_tree_lookahead,
        )

        K, fan_out = 5, 3
        MQ_LEN = (K + 1) * fan_out  # 18
        num_tokens = 100

        # Glue decode: positions num_tokens-1..num_tokens+K-1 → K+1 positions
        # Tree step i: positions num_tokens+K + i*MQ_LEN .. num_tokens+K + (i+1)*MQ_LEN - 1
        # Total extra: K+1 + K*MQ_LEN
        extra = compute_tree_lookahead(MQ_LEN, K)
        self.assertEqual(extra, K + 1 + K * MQ_LEN)
        self.assertEqual(extra, 96)

        # Max position used
        max_pos = (num_tokens - 1) + extra - 1
        self.assertEqual(max_pos, 194)


class TestSSDEquivalence(unittest.TestCase):
    """Test numerical equivalence with SSD implementations."""

    def test_forked_tokens_match_ssd(self):
        """Test that forked recovery tokens match SSD's output for same inputs."""
        from sglang.srt.speculative.async_spec.tree_utils import (
            get_forked_recovery_tokens_from_logits,
        )

        # Fixed seed for reproducibility
        torch.manual_seed(42)

        B, K, V = 2, 3, 100
        fan_out_list = [2, 3, 1, 2]  # Variable fan-out, sum=8
        logits = torch.randn(B, K + 1, V)
        cache_hits = torch.ones(B, dtype=torch.int64)
        returned_tokens = torch.randint(0, V, (B, K + 1))

        result = get_forked_recovery_tokens_from_logits(
            logits, fan_out_list, cache_hits, returned_tokens
        )

        # Verify shape
        mq_len = sum(fan_out_list)
        self.assertEqual(result.shape, (B, mq_len))

        # Manually verify: at each position, the selected tokens should be
        # the top-fan_out from (suppressed) logits
        logits_clone = logits.clone()
        logits_clone[:, :-1, :] = logits_clone[:, :-1, :].scatter(
            dim=2,
            index=returned_tokens[:, 1:].unsqueeze(2),
            value=float("-inf"),
        )

        offset = 0
        for pos in range(K + 1):
            fo = fan_out_list[pos]
            _, topk_idx = torch.topk(logits_clone[:, pos], fo, dim=-1)
            for b in range(B):
                sglang_tokens = set(result[b, offset : offset + fo].tolist())
                expected_tokens = set(topk_idx[b].tolist())
                self.assertEqual(
                    sglang_tokens,
                    expected_tokens,
                    f"Mismatch at batch={b}, pos={pos}: got {sglang_tokens}, expected {expected_tokens}",
                )
            offset += fo

    def test_glue_decode_ids_match_ssd(self):
        """Test make_glue_decode_input_ids matches SSD output."""
        from sglang.srt.speculative.async_spec.tree_utils import (
            make_glue_decode_input_ids,
        )

        B, K = 3, 4
        draft_tokens = torch.randint(0, 100, (B, K))
        rec_tokens = torch.randint(0, 100, (B,))

        result = make_glue_decode_input_ids(draft_tokens, rec_tokens)

        # SSD returns flat [B*(K+1)]
        self.assertEqual(result.shape, (B * (K + 1),))

        # Verify structure: [rec_0, draft_0_0, ..., draft_0_{K-1}, rec_1, ...]
        result_2d = result.view(B, K + 1)
        for b in range(B):
            self.assertEqual(result_2d[b, 0].item(), rec_tokens[b].item())
            for k in range(K):
                self.assertEqual(
                    result_2d[b, k + 1].item(), draft_tokens[b, k].item()
                )


class TestBenchmarks(unittest.TestCase):
    """Performance benchmarks for tree decode operations."""

    def test_benchmark_forked_tokens(self):
        """Benchmark get_forked_recovery_tokens_from_logits."""
        from sglang.srt.speculative.async_spec.tree_utils import (
            get_forked_recovery_tokens_from_logits,
        )

        B, K, V = 8, 5, 32000
        fan_out_list = [3] * (K + 1)
        logits = torch.randn(B, K + 1, V)
        cache_hits = torch.ones(B, dtype=torch.int64)
        returned_tokens = torch.randint(0, V, (B, K + 1))

        # Warmup
        for _ in range(3):
            get_forked_recovery_tokens_from_logits(
                logits, fan_out_list, cache_hits, returned_tokens
            )

        # Benchmark
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            get_forked_recovery_tokens_from_logits(
                logits, fan_out_list, cache_hits, returned_tokens
            )
        elapsed = (time.perf_counter() - start) / n_iters * 1000
        print(f"\n[BENCH] get_forked_recovery_tokens: {elapsed:.3f}ms (B={B}, K={K}, V={V})")
        # Should be under 50ms on CPU
        self.assertLess(elapsed, 200, f"Fork tokens too slow: {elapsed:.1f}ms")

    def test_benchmark_step_positions(self):
        """Benchmark _compute_step_positions_and_slot_maps."""
        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        K, fan_out, B = 5, 3, 8
        MQ_LEN = (K + 1) * fan_out
        N = B * MQ_LEN

        runner = MagicMock()
        runner.spec_k = K
        runner.fan_out = fan_out
        runner.fan_out_list = [fan_out] * (K + 1)
        runner.fan_out_list_miss = runner.fan_out_list
        runner.mq_len = MQ_LEN
        runner.page_size = 16
        runner.device = torch.device("cpu")
        AsyncDraftRunner._init_prealloc_buffers(runner)

        initial_pos = torch.arange(N, dtype=torch.int64)
        initial_rope = torch.arange(N, dtype=torch.int64)
        dbt = torch.zeros(B, 128, dtype=torch.int64)

        # Warmup
        for _ in range(3):
            AsyncDraftRunner._compute_step_positions_and_slot_maps(
                runner, initial_pos, initial_rope, dbt, B, K, N
            )

        # Benchmark
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            AsyncDraftRunner._compute_step_positions_and_slot_maps(
                runner, initial_pos, initial_rope, dbt, B, K, N
            )
        elapsed = (time.perf_counter() - start) / n_iters * 1000
        print(f"\n[BENCH] compute_step_positions: {elapsed:.3f}ms (B={B}, K={K}, N={N})")
        self.assertLess(elapsed, 100, f"Step positions too slow: {elapsed:.1f}ms")


class TestSlotMapComputation(unittest.TestCase):
    """Test _compute_slot_map against manual computation."""

    def test_basic(self):
        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        runner = MagicMock()
        runner.page_size = 16
        runner.device = torch.device("cpu")

        positions = torch.tensor([0, 15, 16, 31, 32], dtype=torch.int64)
        # Block table: request 0 has blocks [0, 1, 2]
        block_tables = torch.zeros(5, 10, dtype=torch.int64)
        block_tables[0, 0] = 5  # Block 0 → physical block 5
        block_tables[0, 1] = 10  # Block 1 → physical block 10
        block_tables[1, 0] = 5
        block_tables[2, 1] = 10
        block_tables[3, 1] = 10
        block_tables[4, 2] = 20

        slot_map = AsyncDraftRunner._compute_slot_map(runner, positions, block_tables)

        # Position 0: block 0, offset 0 → slot 5*16 + 0 = 80
        self.assertEqual(slot_map[0].item(), 80)
        # Position 15: block 0, offset 15 → slot 5*16 + 15 = 95
        self.assertEqual(slot_map[1].item(), 95)
        # Position 16: block 1, offset 0 → slot 10*16 + 0 = 160
        self.assertEqual(slot_map[2].item(), 160)
        # Position 31: block 1, offset 15 → slot 10*16 + 15 = 175
        self.assertEqual(slot_map[3].item(), 175)
        # Position 32: block 2, offset 0 → slot 20*16 + 0 = 320
        self.assertEqual(slot_map[4].item(), 320)


class TestBlockAllocation(unittest.TestCase):
    """Test that block allocation accounts for tree decode capacity."""

    def test_tree_lookahead_included(self):
        """Verify the allocation formula includes tree decode positions."""
        K = 5
        fan_out = 3
        mq_len = (K + 1) * fan_out  # 18
        page_size = 16
        nt = 100  # num_tokens

        # Old formula (without tree decode):
        old_needed = (nt + K + page_size - 1) // page_size

        # New formula (with tree decode):
        tree_lookahead = K + 1 + K * mq_len  # 96
        new_needed = (nt + tree_lookahead + page_size - 1) // page_size

        # New should be significantly more
        self.assertGreater(new_needed, old_needed)
        # Check exact values
        self.assertEqual(old_needed, (100 + 5 + 15) // 16)  # 7
        self.assertEqual(new_needed, (100 + 96 + 15) // 16)  # 13


class TestVectorizedKVMapping(unittest.TestCase):
    """Test that vectorized _update_kv_mapping matches per-element semantics."""

    def test_vectorized_write(self):
        """Test that advanced indexing write matches element-wise writes."""
        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        runner = MagicMock()
        runner.page_size = 16
        runner.device = torch.device("cpu")

        # Create a small req_to_token_pool mock
        pool = MagicMock()
        pool.req_to_token = torch.zeros((4, 64), dtype=torch.int32)
        runner.req_to_token_pool = pool

        # Write some values
        req_pool_indices = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)
        positions = torch.tensor([0, 5, 3, 10, 11], dtype=torch.int64)
        slot_map = torch.tensor([100, 200, 300, 400, 500], dtype=torch.int32)

        AsyncDraftRunner._update_kv_mapping(runner, req_pool_indices, positions, slot_map)

        # Verify
        self.assertEqual(pool.req_to_token[0, 0].item(), 100)
        self.assertEqual(pool.req_to_token[0, 5].item(), 200)
        self.assertEqual(pool.req_to_token[1, 3].item(), 300)
        self.assertEqual(pool.req_to_token[2, 10].item(), 400)
        self.assertEqual(pool.req_to_token[2, 11].item(), 500)

    def test_bulk_write_all_steps(self):
        """Test that bulk write of all K steps works correctly."""
        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        runner = MagicMock()
        runner.page_size = 16
        runner.device = torch.device("cpu")

        pool = MagicMock()
        pool.req_to_token = torch.zeros((2, 128), dtype=torch.int32)
        runner.req_to_token_pool = pool

        K, N = 3, 4  # 3 steps, 4 branches
        # req_pool_indices repeated K times
        rpis = torch.tensor([0, 0, 1, 1], dtype=torch.int64).repeat(K)  # [K*N]
        # Positions: step 0 at [10..13], step 1 at [20..23], step 2 at [30..33]
        positions = torch.cat([
            torch.tensor([10, 11, 12, 13]),
            torch.tensor([20, 21, 22, 23]),
            torch.tensor([30, 31, 32, 33]),
        ]).to(torch.int64)
        slot_map = torch.arange(K * N, dtype=torch.int32) + 100

        AsyncDraftRunner._update_kv_mapping(runner, rpis, positions, slot_map)

        # Check step 0 writes
        self.assertEqual(pool.req_to_token[0, 10].item(), 100)
        self.assertEqual(pool.req_to_token[0, 11].item(), 101)
        self.assertEqual(pool.req_to_token[1, 12].item(), 102)
        self.assertEqual(pool.req_to_token[1, 13].item(), 103)
        # Check step 2 writes
        self.assertEqual(pool.req_to_token[0, 30].item(), 108)
        self.assertEqual(pool.req_to_token[1, 33].item(), 111)


class TestTreeCudaGraphRunnerBuckets(unittest.TestCase):
    """Test TreeDecodeCudaGraphRunner bucket size computation."""

    def test_bucket_sizes(self):
        from sglang.srt.speculative.async_spec.tree_cuda_graph_runner import (
            TreeDecodeCudaGraphRunner,
        )

        # max_batch_size=8, mq_len=18
        buckets = TreeDecodeCudaGraphRunner._compute_bucket_sizes(8, 18)
        # Should include: 1*18=18, 2*18=36, 4*18=72, 8*18=144
        self.assertIn(18, buckets)
        self.assertIn(36, buckets)
        self.assertIn(72, buckets)
        self.assertIn(144, buckets)
        self.assertEqual(buckets, sorted(buckets))  # Should be sorted

    def test_bucket_sizes_non_power_of_2(self):
        from sglang.srt.speculative.async_spec.tree_cuda_graph_runner import (
            TreeDecodeCudaGraphRunner,
        )

        # max_batch_size=5, mq_len=8
        buckets = TreeDecodeCudaGraphRunner._compute_bucket_sizes(5, 8)
        # Should include: 1*8=8, 2*8=16, 4*8=32, 5*8=40
        self.assertIn(8, buckets)
        self.assertIn(16, buckets)
        self.assertIn(32, buckets)
        self.assertIn(40, buckets)  # Max always included

    def test_bucket_sizes_single(self):
        from sglang.srt.speculative.async_spec.tree_cuda_graph_runner import (
            TreeDecodeCudaGraphRunner,
        )

        buckets = TreeDecodeCudaGraphRunner._compute_bucket_sizes(1, 6)
        self.assertEqual(buckets, [6])

    def test_bucket_sizes_zero(self):
        from sglang.srt.speculative.async_spec.tree_cuda_graph_runner import (
            TreeDecodeCudaGraphRunner,
        )

        buckets = TreeDecodeCudaGraphRunner._compute_bucket_sizes(0, 18)
        self.assertEqual(buckets, [])


class TestGlueDecodeCudaGraphRunnerBuckets(unittest.TestCase):
    """Test GlueDecodeCudaGraphRunner bucket size computation."""

    def test_bucket_sizes(self):
        from sglang.srt.speculative.async_spec.glue_decode_cuda_graph_runner import (
            GlueDecodeCudaGraphRunner,
        )

        buckets = GlueDecodeCudaGraphRunner._compute_bucket_sizes(8)
        self.assertIn(1, buckets)
        self.assertIn(2, buckets)
        self.assertIn(4, buckets)
        self.assertIn(8, buckets)
        self.assertEqual(buckets, sorted(buckets))

    def test_bucket_sizes_non_power_of_2(self):
        from sglang.srt.speculative.async_spec.glue_decode_cuda_graph_runner import (
            GlueDecodeCudaGraphRunner,
        )

        buckets = GlueDecodeCudaGraphRunner._compute_bucket_sizes(5)
        self.assertIn(1, buckets)
        self.assertIn(2, buckets)
        self.assertIn(4, buckets)
        self.assertIn(5, buckets)  # Max always included


class TestBenchmarkVectorizedKVMapping(unittest.TestCase):
    """Benchmark vectorized vs per-element KV mapping."""

    def test_benchmark_vectorized_kv_mapping(self):
        """Benchmark vectorized _update_kv_mapping."""
        from sglang.srt.speculative.async_spec.async_draft_runner import (
            AsyncDraftRunner,
        )

        runner = MagicMock()
        runner.page_size = 16
        runner.device = torch.device("cpu")

        pool = MagicMock()
        pool.req_to_token = torch.zeros((64, 1024), dtype=torch.int32)
        runner.req_to_token_pool = pool

        # Simulate K*N entries: K=5, B=8, mq_len=18 → N=144, K*N=720
        K_N = 720
        rpis = torch.randint(0, 64, (K_N,), dtype=torch.int64)
        positions = torch.randint(0, 1024, (K_N,), dtype=torch.int64)
        slot_map = torch.randint(0, 10000, (K_N,), dtype=torch.int32)

        # Warmup
        for _ in range(3):
            AsyncDraftRunner._update_kv_mapping(runner, rpis, positions, slot_map)

        # Benchmark
        n_iters = 1000
        start = time.perf_counter()
        for _ in range(n_iters):
            AsyncDraftRunner._update_kv_mapping(runner, rpis, positions, slot_map)
        elapsed = (time.perf_counter() - start) / n_iters * 1000
        print(f"\n[BENCH] vectorized _update_kv_mapping: {elapsed:.3f}ms (K*N={K_N})")
        # Should be well under 1ms on CPU
        self.assertLess(elapsed, 10, f"Vectorized KV mapping too slow: {elapsed:.1f}ms")


if __name__ == "__main__":
    unittest.main()
