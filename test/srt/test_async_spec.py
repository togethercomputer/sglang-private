"""Tests for asynchronous speculative decoding.

Tests the ASYNC_SPEC algorithm which runs the draft model on a dedicated GPU
in a separate process, communicating via NCCL.

NOTE: These tests require at least 2 GPUs. The target model runs on GPU 0
and the draft model runs on GPU 1.
"""

import unittest

import torch

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    verify,
)


def _get_num_gpus():
    """Get number of available CUDA GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


@unittest.skipIf(_get_num_gpus() < 2, "Requires at least 2 GPUs")
class TestAsyncSpecUnit(CustomTestCase):
    """Unit tests for async spec components that don't require full server."""

    def test_verify_greedy_all_accept(self):
        """Test verification with greedy decoding where all tokens match."""

        B, K, V = 2, 3, 100
        # Create logits where argmax matches speculations
        logits_p = torch.zeros(B, K + 1, V)
        logits_q = torch.zeros(B, K, V)
        speculations = torch.zeros(B, K + 1, dtype=torch.long)

        for b in range(B):
            for k in range(K + 1):
                token = (b * 10 + k) % V
                speculations[b, k] = token
                logits_p[b, k, token] = 10.0  # High logit = argmax
                if k > 0:
                    logits_q[b, k - 1, token] = 10.0

        temperatures_target = torch.zeros(B)  # greedy
        temperatures_draft = torch.zeros(B)

        accepted_suffixes, recovery_tokens = verify(
            logits_p=logits_p,
            logits_q=logits_q,
            speculations=speculations,
            temperatures_target=temperatures_target,
            temperatures_draft=temperatures_draft,
        )

        # All tokens should be accepted
        for b in range(B):
            self.assertEqual(len(accepted_suffixes[b]), K)

    def test_verify_greedy_reject_at_position(self):
        """Test verification with greedy decoding where token 2 mismatches."""

        B, K, V = 1, 3, 100
        logits_p = torch.zeros(B, K + 1, V)
        logits_q = torch.zeros(B, K, V)
        speculations = torch.zeros(B, K + 1, dtype=torch.long)

        # Position 0 (recovery): matches
        speculations[0, 0] = 5
        logits_p[0, 0, 5] = 10.0

        # Position 1: matches
        speculations[0, 1] = 10
        logits_p[0, 1, 10] = 10.0
        logits_q[0, 0, 10] = 10.0

        # Position 2: mismatch (draft says 20, target says 30)
        speculations[0, 2] = 20
        logits_p[0, 2, 30] = 10.0  # target prefers 30
        logits_q[0, 1, 20] = 10.0

        # Position 3: shouldn't matter
        speculations[0, 3] = 40
        logits_p[0, 3, 40] = 10.0

        temperatures_target = torch.zeros(B)
        temperatures_draft = torch.zeros(B)

        accepted_suffixes, recovery_tokens = verify(
            logits_p=logits_p,
            logits_q=logits_q,
            speculations=speculations,
            temperatures_target=temperatures_target,
            temperatures_draft=temperatures_draft,
        )

        # Should accept position 1 only (position 0 is recovery, position 2 rejected)
        self.assertEqual(len(accepted_suffixes[0]), 1)
        self.assertEqual(accepted_suffixes[0][0], 10)
        self.assertEqual(recovery_tokens[0], 30)

    def test_verify_empty_batch(self):
        """Test verification with empty inputs."""

        B, K, V = 0, 3, 100
        logits_p = torch.zeros(B, K + 1, V)
        logits_q = torch.zeros(B, K, V)
        speculations = torch.zeros(B, K + 1, dtype=torch.long)
        temperatures_target = torch.zeros(B)
        temperatures_draft = torch.zeros(B)

        accepted_suffixes, recovery_tokens = verify(
            logits_p=logits_p,
            logits_q=logits_q,
            speculations=speculations,
            temperatures_target=temperatures_target,
            temperatures_draft=temperatures_draft,
        )

        self.assertEqual(len(accepted_suffixes), 0)
        self.assertEqual(len(recovery_tokens), 0)

    def test_command_codes(self):
        """Test the command code constants."""
        from sglang.srt.speculative.async_spec.handshake import (
            CMD_EXIT,
            CMD_PREFILL,
            CMD_SPEC_REQUEST,
        )

        # Test command codes
        self.assertEqual(CMD_SPEC_REQUEST, 0)
        self.assertEqual(CMD_PREFILL, 1)
        self.assertEqual(CMD_EXIT, 2)

    def test_tree_utils(self):
        """Test tree utility functions."""
        from sglang.srt.speculative.async_spec.tree_utils import (
            apply_sampler_x_rescaling,
            make_glue_decode_input_ids,
        )

        # Test make_glue_decode_input_ids (returns flat [B*(K+1)])
        draft_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        rec_tokens = torch.tensor([10, 20])
        result = make_glue_decode_input_ids(draft_tokens, rec_tokens)
        self.assertEqual(result.shape, (8,))  # 2 * (3+1) = 8
        result_2d = result.view(2, 4)
        self.assertEqual(result_2d[0, 0].item(), 10)
        self.assertEqual(result_2d[1, 0].item(), 20)

        # Test apply_sampler_x_rescaling (identity)
        probs = torch.softmax(torch.randn(2, 100), dim=-1)
        result = apply_sampler_x_rescaling(probs, 1.0, 3)
        self.assertTrue(torch.allclose(result, probs, atol=1e-5))


@unittest.skipIf(_get_num_gpus() < 2, "Requires at least 2 GPUs")
class TestAsyncSpecDraftRunner(CustomTestCase):
    """Test the draft runner via Engine (NCCL requires full dist init)."""

    def test_draft_runner_via_engine(self):
        """Test draft runner starts and communicates correctly through Engine."""
        import sglang as sgl

        # Starting the engine implicitly starts the draft runner process
        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            speculative_algorithm="ASYNC_SPEC",
            speculative_draft_model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            speculative_num_steps=3,
            speculative_async_fan_out=2,
            mem_fraction_static=0.5,
            log_level="info",
        )

        try:
            # If engine starts without error, draft runner is working
            prompt = "Hello"
            output = engine.generate(prompt, {"temperature": 0, "max_new_tokens": 5})
            self.assertGreater(len(output["text"]), 0)
        finally:
            engine.shutdown()


@unittest.skipIf(_get_num_gpus() < 2, "Requires at least 2 GPUs")
class TestAsyncSpecE2E(CustomTestCase):
    """End-to-end test for async speculative decoding using the Engine API."""

    def test_basic_generation(self):
        """Test that async spec can generate text."""
        import sglang as sgl

        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            speculative_algorithm="ASYNC_SPEC",
            speculative_draft_model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            speculative_num_steps=3,
            speculative_async_fan_out=2,
            mem_fraction_static=0.5,
            log_level="info",
        )

        try:
            prompt = "The capital of France is"
            output = engine.generate(prompt, {"temperature": 0, "max_new_tokens": 20})
            text = output["text"]
            print(f"Generated: {text}")
            self.assertGreater(len(text), 0)
        finally:
            engine.shutdown()

    def test_batch_generation(self):
        """Test batch generation with async spec."""
        import sglang as sgl

        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            speculative_algorithm="ASYNC_SPEC",
            speculative_draft_model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            speculative_num_steps=3,
            speculative_async_fan_out=2,
            mem_fraction_static=0.5,
            log_level="info",
        )

        try:
            prompts = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
            ]
            params = {"temperature": 0, "max_new_tokens": 30}

            outputs = engine.generate(prompts, params)
            for prompt, output in zip(prompts, outputs):
                text = output["text"]
                print(f"Prompt: {prompt}")
                print(f"Generated: {text}")
                self.assertGreater(len(text), 0)
        finally:
            engine.shutdown()

    def test_max_tokens_respected(self):
        """Test that max_tokens is respected."""
        import sglang as sgl

        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            speculative_algorithm="ASYNC_SPEC",
            speculative_draft_model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            speculative_num_steps=3,
            mem_fraction_static=0.5,
            log_level="info",
        )

        try:
            prompt = "Write a long essay about AI"
            output = engine.generate(prompt, {"temperature": 0, "max_new_tokens": 5})
            completion_tokens = output["meta_info"]["completion_tokens"]
            print(f"Completion tokens: {completion_tokens}")
            self.assertLessEqual(completion_tokens, 5)
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
