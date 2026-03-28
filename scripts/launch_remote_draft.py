#!/usr/bin/env python3
"""
Standalone launcher for the SSD async draft runner on a remote node.

This script is used for cross-node async speculative decoding, where the
target model runs on one node (via SGLang) and the draft model runs on
a separate node.

Usage:
    python scripts/launch_remote_draft.py \
        --draft-model-path /path/to/draft/model \
        --target-host <target_node_ip> \
        --nccl-port <port> \
        --gpu-id 0 \
        --speculate-k 5 \
        --kv-cache-size <num_blocks> \
        --max-model-len 4096 \
        --max-running-requests 64 \
        --fan-out 3

The target SGLang server should be started with:
    --speculative-async-target-host <this_node_or_target_ip>
    --speculative-async-port <same_port>
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch SSD async draft runner for cross-node speculative decoding"
    )
    parser.add_argument(
        "--draft-model-path",
        type=str,
        required=True,
        help="Path to the draft model (local path or HuggingFace model ID)",
    )
    parser.add_argument(
        "--target-host",
        type=str,
        required=True,
        help="Hostname/IP of the target node running the SGLang server",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="NCCL port for cross-node communication (must match target's --speculative-async-port)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to use on this node (default: 0)",
    )
    parser.add_argument(
        "--speculate-k",
        type=int,
        default=5,
        help="Number of speculative decoding steps (must match target)",
    )
    parser.add_argument(
        "--kv-cache-size",
        type=int,
        default=-1,
        help="Number of KV cache blocks (should match target's token_to_kv_pool.size). "
        "If -1, auto-allocate based on GPU memory.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model sequence length (must match target's --context-length)",
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=64,
        help="Maximum number of concurrent requests (must match target)",
    )
    parser.add_argument(
        "--fan-out",
        type=int,
        default=3,
        help="Async speculation fan-out factor (must match target's --speculative-async-fan-out)",
    )
    parser.add_argument(
        "--fan-out-list",
        type=int,
        nargs="+",
        default=None,
        help="Per-depth fan-out list for cache hits (must match target)",
    )
    parser.add_argument(
        "--fan-out-list-miss",
        type=int,
        nargs="+",
        default=None,
        help="Per-depth fan-out list for cache misses (must match target)",
    )
    parser.add_argument(
        "--jit-speculate",
        action="store_true",
        default=False,
        help="Enable JIT speculation on cache miss (must match target)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Fraction of GPU memory to use",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Disable CUDA graphs for debugging",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    # Eagle/Phoenix-specific options
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to target model tokenizer (required for EAGLE/Phoenix variants)",
    )
    parser.add_argument(
        "--d-model-target",
        type=int,
        default=None,
        help="Target model hidden size (required for EAGLE/Phoenix variants)",
    )
    parser.add_argument(
        "--use-phoenix",
        action="store_true",
        default=False,
        help="Use Phoenix draft model architecture instead of EAGLE",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve HuggingFace model ID to local cache path if needed
    draft_model_path = args.draft_model_path
    if not os.path.exists(draft_model_path):
        from huggingface_hub import snapshot_download
        print(f"Downloading draft model: {draft_model_path}")
        draft_model_path = snapshot_download(draft_model_path)
        print(f"Downloaded to: {draft_model_path}")

    from ssd.config import Config
    from ssd.engine.draft_runner import DraftRunner

    use_phoenix = args.use_phoenix
    use_eagle = args.tokenizer_path is not None and not use_phoenix

    config = Config(
        draft=draft_model_path,
        model=draft_model_path,
        num_gpus=2,  # NCCL world size: target (rank 0) + draft (rank 1)
        speculate=True,
        speculate_k=args.speculate_k,
        draft_async=True,
        async_fan_out=args.fan_out,
        fan_out_list=args.fan_out_list,
        fan_out_list_miss=args.fan_out_list_miss,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tokenizer_path=args.tokenizer_path,
        d_model_target=args.d_model_target,
        kvcache_block_size=1,
        num_kvcache_blocks=args.kv_cache_size,
        max_num_seqs=args.max_running_requests,
        max_model_len=args.max_model_len,
        jit_speculate=args.jit_speculate,
        async_nccl_port=args.port,
        async_nccl_host=args.target_host,
        communicate_logits=False,
        communicate_cache_hits=False,
        use_eagle=use_eagle,
        use_phoenix=use_phoenix,
        enforce_eager=args.enforce_eager,
        verbose=args.verbose,
    )

    print(f"Starting cross-node draft runner:")
    print(f"  Draft model: {draft_model_path}")
    print(f"  Target host: {args.target_host}:{args.port}")
    print(f"  GPU: {args.gpu_id}")
    print(f"  Speculate K: {args.speculate_k}")
    print(f"  Fan-out: {args.fan_out}")
    print(f"  KV cache blocks: {args.kv_cache_size}")
    print(f"  EAGLE: {use_eagle}, Phoenix: {use_phoenix}")

    # DraftRunner.__init__ will:
    # 1. Load the model
    # 2. Connect to target's TCPStore via async_nccl_host:async_nccl_port
    # 3. Form NCCL process group
    # 4. Warm up and capture CUDA graphs
    # 5. Send num_kvcache_blocks via NCCL (since init_q=None in cross-node mode)
    # 6. Enter the draft_loop() to serve speculation requests
    DraftRunner(config, rank=args.gpu_id, init_q=None)


if __name__ == "__main__":
    main()
