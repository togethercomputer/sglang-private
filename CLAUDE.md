# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a fast serving framework for large language models (LLMs) and vision language models (VLMs). It provides an OpenAI-compatible API server with advanced features like RadixAttention for prefix caching, speculative decoding, and structured generation.

## Build & Install

```bash
# Install from source (development mode)
pip install -e "python[dev]"

# Install sgl-kernel from source (CUDA kernels)
cd sgl-kernel && make build
```

## Running Tests

Tests use Python's `unittest` framework. Most test files are standalone scripts (not pytest-discovered).

```bash
# Run a single test file
python3 test/srt/test_srt_endpoint.py

# Run a specific test case
python3 test/srt/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Run a CI test suite
python3 test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu
```

Tests register themselves for CI via decorators:
```python
from sglang.test.ci.ci_register import register_cuda_ci
register_cuda_ci(est_time=80, suite="stage-b-test-small-1-gpu")
```

## Linting & Formatting

```bash
# Run all checks (pre-commit must be installed: pip install pre-commit)
pre-commit run --all-files
```

Tools: `black` (formatting), `isort` (imports, profile=black), `ruff` (F401/F821 only), `clang-format` v18 (C++/CUDA), `codespell`.

## Architecture

### Multi-Process Runtime

The serving runtime (`python/sglang/srt/`) runs as multiple processes communicating via ZMQ:

```
HTTP Server (FastAPI, entrypoints/http_server.py)
       │
       ▼
TokenizerManager (managers/tokenizer_manager.py)
  - Text → token IDs, multimodal input processing
       │ ZMQ
       ▼
Scheduler (managers/scheduler.py)
  - Core orchestration: batching, KV cache (RadixCache), request lifecycle
  - Uses mixin classes for modularity (DP attention, pipeline parallelism, etc.)
       │
       ▼
ModelRunner (model_executor/model_runner.py)
  - Forward passes, attention backends, CUDA graph execution
       │
       ▼
DetokenizerManager (managers/detokenizer_manager.py)
  - Token IDs → text, streaming responses
```

### Key Data Flow

1. **GenerateReqInput** (`managers/io_struct.py`) — raw user request with text/images
2. **Req** (`managers/schedule_batch.py`) — internal request with KV cache state, token IDs
3. **ScheduleBatch** (`managers/schedule_batch.py`) — CPU-side batch of requests for scheduling
4. **ForwardBatch** (`model_executor/forward_batch_info.py`) — GPU-resident tensors with `ForwardMode` (PREFILL, EXTEND, DECODE)

### Directory Layout

- `python/sglang/srt/` — Serving runtime (the inference engine)
  - `entrypoints/` — Engine API, HTTP/gRPC servers
  - `managers/` — TokenizerManager, Scheduler, DetokenizerManager
  - `model_executor/` — ModelRunner, CudaGraphRunner, forward batch logic
  - `models/` — Model implementations (HuggingFace-compatible)
  - `layers/` — Attention, MoE, quantization, sampling layers
  - `mem_cache/` — RadixCache (prefix-sharing KV cache), memory pools
  - `speculative/` — Speculative decoding (EAGLE, ngram, async)
  - `server_args.py` — All server configuration (`ServerArgs` dataclass)
- `python/sglang/lang/` — Frontend DSL for structured generation (`sgl.function`, `sgl.gen`)
- `sgl-kernel/` — C++/CUDA kernel library (custom attention, sampling kernels)
- `test/` — Tests organized by `srt/`, `lang/`, `unit/`

### Configuration

All server options are in `ServerArgs` (`python/sglang/srt/server_args.py`). This is a large dataclass covering model loading, GPU config, batching, caching, LoRA, quantization, speculative decoding, and distributed execution.

### Scheduler Mixins

The Scheduler class uses mixins for feature modularity:
- `SchedulerOutputProcessorMixin` — post-processing model outputs
- `SchedulerDPAttnMixin` — data-parallel attention
- `SchedulerPPMixin` — pipeline parallelism
- `SchedulerMultiplexMixin` — request multiplexing

## Utility Scripts

```bash
scripts/killall_sglang.sh     # Kill all SGLang processes
scripts/ensure_vram_clear.sh  # Verify GPU memory is freed
```
