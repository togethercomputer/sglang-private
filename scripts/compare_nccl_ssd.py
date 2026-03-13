#!/usr/bin/env python3
"""
Launch SSD engine with NCCL logging enabled to capture async spec decode data flow.
Uses the same prompt as SGLang warmup: "The capital city of France is"

Usage:
    SSD_NCCL_LOG=1 python scripts/compare_nccl_ssd.py
"""

import os

# Ensure NCCL logging is on
os.environ["SSD_NCCL_LOG"] = "1"

from ssd import LLM, SamplingParams


def main():
    prompt = "The capital city of France is"

    llm = LLM(
        "meta-llama/Llama-3.2-1B-Instruct",
        draft="meta-llama/Llama-3.2-1B-Instruct",
        draft_async=True,
        speculate=True,
        speculate_k=6,
        num_gpus=2,
        verbose=True,
        max_num_seqs=1,
        max_steps=100,
        jit_speculate=False,
        max_model_len=4096,
        kvcache_block_size=1,
    )

    sampling_params = [SamplingParams(
        temperature=0.0,
        max_new_tokens=64,
    )]

    print(f"\n{'#' * 80}", flush=True)
    print(f"# GENERATING WITH PROMPT: '{prompt}'", flush=True)
    print(f"{'#' * 80}\n", flush=True)

    outputs, metrics = llm.generate([prompt], sampling_params, use_tqdm=False)

    print(f"\n{'#' * 80}", flush=True)
    print(f"# OUTPUT", flush=True)
    print(f"{'#' * 80}", flush=True)
    if outputs:
        print(f"Text: {outputs[0]['text']}", flush=True)
        print(f"Token IDs: {outputs[0]['token_ids']}", flush=True)
    else:
        print(f"No outputs (max_steps={llm.config.max_steps} may have cut short)", flush=True)

    llm.exit()


if __name__ == "__main__":
    main()
