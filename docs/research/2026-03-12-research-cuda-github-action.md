# Research: CUDA GitHub Action Worker

**Date:** 2026-03-12
**Issue:** samschlegel/meshcore-utils#2 — Setup a CUDA GitHub Action worker
**Related:** samschlegel/meshcore-utils#3 — Move GPU verification bits into a test

## Problem Statement

The project currently has CI for CPU-only builds (Ubuntu) and Metal GPU builds (macOS), but no CUDA CI. The CUDA code path (`--features cuda`) is untested in CI, meaning regressions can go unnoticed.

## Current CI State

The existing `.github/workflows/rust.yml` has two jobs:

1. **build** (ubuntu-latest) — CPU-only `cargo build` + `cargo test`
2. **build-macos** (macos-latest) — CPU build, Metal build (`--features metal`), tests, and GPU verification via `./target/release/mc-keygen --verify A`

## CUDA Build Requirements

The project uses `cudarc` v0.19 with the `cuda-13010` feature flag. Building with `--features cuda` requires:

- **CUDA Toolkit 13.1** — provides `nvcc` (NVRTC compiler) and CUDA runtime libraries
- **NVIDIA driver** — only needed for *execution* on a real GPU; not needed for compilation
- The CUDA kernel (`cuda/vanity_kernel.cu`) is compiled at runtime via NVRTC (`cudarc::nvrtc::compile_ptx_with_opts`), so `nvcc` must be available at build time for cudarc to link against the CUDA driver API

## Options for CUDA CI

### Option A: Compile-Only on Standard Runner (Recommended for Phase 1)

Use `Jimver/cuda-toolkit` action to install the CUDA Toolkit on a standard `ubuntu-latest` runner, then build with `--features cuda`.

**Pros:**
- Free (uses standard GitHub-hosted runners)
- Validates compilation, linking, and that the CUDA feature doesn't regress
- Simple setup — single action + cargo build
- No GPU hardware needed for compilation

**Cons:**
- Cannot run the binary or execute GPU verification (no GPU hardware)
- Cannot test kernel launch, memory allocation, or result correctness

**Implementation:**
```yaml
- uses: Jimver/cuda-toolkit@v0.2.30
  with:
    cuda: '13.1.0'
- run: cargo build --release --features cuda --verbose
- run: cargo test --verbose
```

### Option B: GitHub-Hosted GPU Runners (Full Verification)

Use GitHub's managed GPU runners (`gpu-t4-4-core`) with a Tesla T4 GPU.

**Pros:**
- Full end-to-end testing including kernel execution and GPU vs CPU verification
- Managed by GitHub — no infrastructure to maintain
- Tesla T4 supports CUDA compute capability 7.5

**Cons:**
- Costs $0.07/min (not included in free tier)
- Requires organization billing setup with a credit card
- Requires setting up a custom runner with the NVIDIA GPU-Optimized Image
- T4 is an older GPU (Turing architecture) but sufficient for correctness testing

**Implementation:**
```yaml
runs-on: [gpu-t4-4-core]
# GPU image comes with CUDA pre-installed
steps:
  - uses: actions/checkout@v4
  - run: cargo build --release --features cuda --verbose
  - run: cargo test --verbose
  - run: ./target/release/mc-keygen --verify A
```

### Option C: Self-Hosted GPU Runner

Set up a self-hosted runner on a machine with an NVIDIA GPU.

**Pros:**
- Full control over hardware and software stack
- Can use any GPU (RTX 4090, etc.)
- No per-minute costs beyond hardware/electricity

**Cons:**
- Requires maintaining infrastructure
- Security considerations for public repos (arbitrary code execution)
- Single point of failure

### Option D: Conditional/Label-Triggered GPU Testing

Combine Options A and B: always run compile-only checks on standard runners, but trigger GPU execution tests only on specific labels or manual dispatch.

**Pros:**
- Cost-effective — GPU minutes only used when needed
- Fast feedback on compilation for every PR
- Full verification available on demand

**Cons:**
- More complex workflow configuration
- GPU tests could be forgotten/skipped

## Recommendation

**Phase 1 (this PR):** Option A — compile-only CUDA build on standard `ubuntu-latest` runners using `Jimver/cuda-toolkit@v0.2.30`. This catches compilation regressions at zero cost.

**Phase 2 (future):** Option D — add a GPU runner job (triggered by label or manual dispatch) for full end-to-end verification with `--verify`. This requires organization billing setup.

## Technical Details

### Jimver/cuda-toolkit Action

- **Version:** v0.2.30 (latest stable)
- **CUDA version:** 13.1.0 (matches cudarc feature `cuda-13010`)
- **Method:** `local` (default) downloads full toolkit; `network` allows sub-package selection
- **Sub-packages (network only):** Can install only `nvcc` and required libraries to speed up installation
- **Outputs:** `cuda` (version), `CUDA_PATH` (install path)

### cudarc Linking

cudarc uses dynamic loading by default — it `dlopen`s CUDA libraries at runtime. This means:
- Compilation succeeds as long as CUDA headers/stubs are available
- Runtime execution fails gracefully if no GPU is present (returns an error, doesn't crash)
- The `verify_gpu_keygen()` function will return `CudaError::NoCudaDevice` on runners without a GPU

### Workflow Structure

The new `build-cuda` job should:
1. Install CUDA Toolkit 13.1
2. Build with `cargo build --release --features cuda --verbose`
3. Run `cargo test --verbose` (existing CPU tests still pass with cuda feature enabled)
4. Optionally (with GPU): run `./target/release/mc-keygen --verify A`

## Sources

- [Jimver/cuda-toolkit GitHub Action](https://github.com/Jimver/cuda-toolkit)
- [GitHub Actions GPU hosted runners GA announcement](https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available/)
- [GitHub Action workflows with GPUs (Tim Head)](https://betatim.github.io/posts/github-action-with-gpu/)
- [CUDA Toolkit 13.1 Downloads](https://developer.nvidia.com/cuda-downloads)
