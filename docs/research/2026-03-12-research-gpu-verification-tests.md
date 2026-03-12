# Research: Move GPU Verification into Rust Tests

**Date:** 2026-03-12
**Issue:** [#3 — GPU verification as a Rust test](https://github.com/samschlegel/meshcore-utils/issues/3)
**Related:** PR #6 (CUDA CI job)

## Current State

### GPU Verification Functions

Both backends implement identical `verify_gpu_keygen()` standalone functions:

- **CUDA** (`src/gpu.rs:225-325`): Compiles kernel, generates 64 deterministic test seeds, launches `verify_keygen` kernel, compares GPU output against CPU `generate_keypair()`. Returns `Result<(), CudaError>`.
- **Metal** (`src/metal_gpu.rs:206-307`): Same algorithm using Metal API. Returns `Result<(), MetalError>`.

### How Verification is Invoked Today

Via CLI `--verify` flag in `main.rs:411-428`:
```rust
if cli.verify {
    eprint!("Compiling GPU kernel and running verification... ");
    #[cfg(feature = "cuda")]
    let result = gpu::verify_gpu_keygen();
    // ...
    match result {
        Ok(()) => { eprintln!("PASSED"); std::process::exit(0); }
        Err(e) => { eprintln!("FAILED: {}", e); std::process::exit(1); }
    }
}
```

CI invokes this in `build-macos` job: `./target/release/mc-keygen --verify A`

### Existing Test Structure

Tests use standard `#[cfg(test)] mod tests` inline in source files:
- `src/keygen.rs:40-86` — 5 unit tests for key generation
- `src/search.rs:384-496` — 9 unit tests for prefix matching and search

No `tests/` directory exists. No integration tests.

## CI Hardware Constraints

| Job | Runner | GPU Hardware | Can Execute GPU Kernels |
|-----|--------|-------------|------------------------|
| `build` | ubuntu-latest | None | No |
| `build-cuda` | ubuntu-latest | CUDA toolkit only, no GPU | No |
| `build-macos` | macos-latest | Apple Silicon Metal GPU | **Yes** |

## Analysis

### Where Should Tests Live?

The repo follows a pattern of inline `#[cfg(test)]` modules within each source file. GPU tests should follow this convention by adding test modules to `gpu.rs` and `metal_gpu.rs`.

### Handling No-GPU Environments

CUDA CI runners have the toolkit (compiler) but no GPU device. When `CudaSearcher::new()` or `compile_kernel()` fails with `CudaError::NoCudaDevice`, tests must treat this as a skip, not a failure.

Rust's built-in test framework doesn't have a native "skip" mechanism like `#[ignore]`. The idiomatic approaches are:

1. **Early return on no-device** — Test function detects no GPU and returns `Ok(())` with a printed message. This is the simplest and most common pattern in hardware-dependent Rust crates (e.g., `wgpu`, `vulkano`).
2. **`#[ignore]` + CI `--include-ignored`** — Mark tests `#[ignore]` and only run them on GPU runners. Downside: tests never run automatically where GPUs exist unless CI explicitly opts in.
3. **Compile-only test** — A test that only calls `compile_kernel()` and skips execution. Useful for CUDA CI to verify kernel compilation without needing a device.

**Recommendation:** Use approach (1) for full verification tests (early-return on NoCudaDevice/NoMetalDevice), combined with a separate compile-only test for CUDA that verifies kernel compilation succeeds even without a GPU device.

### What Tests to Add

For each backend (`gpu.rs` and `metal_gpu.rs`):

1. **`verify_gpu_keygen_matches_cpu`** — Calls existing `verify_gpu_keygen()`. On `NoCudaDevice`/`NoMetalDevice`, prints a message and returns Ok. On any other error, fails the test. This is the core regression test.

For CUDA only:
2. **`cuda_kernel_compiles`** — Calls `compile_kernel()` (currently private, needs `pub(crate)` or test inside module). Verifies NVRTC compilation succeeds. This can run on CUDA CI without a GPU *if* the compilation step doesn't require a device context — however, looking at `compile_kernel()` in `gpu.rs:53-80`, it calls `CudaContext::new(0)` first which requires a device. So this test would also need the early-return pattern.

Actually, looking more carefully at `compile_kernel()`:
```rust
fn compile_kernel() -> Result<(Arc<CudaModule>, Arc<CudaStream>), CudaError> {
    let ctx = CudaContext::new(0).map_err(|e| { ... CudaError::NoCudaDevice ... })?;
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SRC, ...)?;
    ...
}
```

The NVRTC compilation (`compile_ptx_with_opts`) is separate from device context creation and could theoretically run without a GPU. But `compile_kernel()` bundles both together. To test compilation separately, we'd need to extract the NVRTC call — but that would be a refactor beyond the scope of this issue. The current `compile_kernel()` approach requires a device.

### CI Workflow Changes

- `build-cuda`: Change `cargo test --verbose` to `cargo test --features cuda --verbose` so the CUDA test module compiles and runs (with early-return for no device).
- `build-macos`: Change `cargo test --verbose` to `cargo test --features metal --verbose` and remove the separate `./target/release/mc-keygen --verify A` step since the test covers it.

### Impact on `--verify` CLI Flag

The `--verify` CLI flag in `main.rs` should remain — it's useful for users to manually verify their GPU setup. The test simply calls the same underlying function.

## Summary

- Add `#[cfg(test)] mod tests` to `gpu.rs` and `metal_gpu.rs`
- Each module gets a `verify_gpu_keygen_matches_cpu` test that calls the existing verification function
- Tests early-return with a printed message when no GPU device is available
- CI workflow updated to pass feature flags to `cargo test`
- Remove redundant `--verify A` step from macOS CI (test covers it)
- Keep `--verify` CLI flag for manual use
