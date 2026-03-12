# Plan: Move GPU Verification into Rust Tests

**Date:** 2026-03-12
**Issue:** [#3](https://github.com/samschlegel/meshcore-utils/issues/3)
**Research:** [docs/research/2026-03-12-research-gpu-verification-tests.md](../research/2026-03-12-research-gpu-verification-tests.md)

## Problem

GPU verification (`verify_gpu_keygen()`) is only accessible via the `--verify` CLI flag. It isn't integrated into the Rust test framework, so `cargo test` doesn't catch GPU keygen regressions. CI works around this by invoking the binary directly (`./target/release/mc-keygen --verify A`), which is fragile and inconsistent with how the rest of the project is tested.

## What Changes

Add `#[cfg(test)] mod tests` to `src/gpu.rs` and `src/metal_gpu.rs` with tests that call the existing `verify_gpu_keygen()` functions. Update CI to run `cargo test` with appropriate feature flags.

## Changes

### 1. Add test module to `src/gpu.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_gpu_keygen_matches_cpu() {
        // cudarc panics if the CUDA shared library isn't installed at all
        // (e.g. CI runners without GPU drivers). catch_unwind lets us
        // treat that the same as NoCudaDevice — a graceful skip.
        let result = std::panic::catch_unwind(verify_gpu_keygen);
        match result {
            Ok(Ok(())) => {}
            Ok(Err(CudaError::NoCudaDevice)) => {
                eprintln!("no CUDA device available, skipping GPU verification");
            }
            Ok(Err(e)) => panic!("GPU keygen verification failed: {}", e),
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("");
                if msg.contains("libcuda") || msg.contains("libnvcuda") {
                    eprintln!("CUDA driver not installed, skipping GPU verification");
                } else {
                    std::panic::resume_unwind(payload);
                }
            }
        }
    }
}
```

The test calls the existing `verify_gpu_keygen()`. If no CUDA device is present (standard CI runners), it prints a skip message and passes. `catch_unwind` is needed because `cudarc` panics when the CUDA shared library isn't installed at all (before `CudaError::NoCudaDevice` can be returned). The panic payload is inspected to only skip CUDA library-loading failures — any other panic is re-raised. Any other error (compilation failure, key mismatch) fails the test.

### 2. Add test module to `src/metal_gpu.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_gpu_keygen_matches_cpu() {
        match verify_gpu_keygen() {
            Ok(()) => {}
            Err(MetalError::NoMetalDevice) => {
                eprintln!("no Metal device available, skipping GPU verification");
            }
            Err(e) => panic!("GPU keygen verification failed: {}", e),
        }
    }
}
```

Same pattern as CUDA. On macOS CI runners (which have Metal GPUs), this will execute full verification.

### 3. Update CI workflow (`.github/workflows/rust.yml`)

**`build-cuda` job:** Change test step to pass the `cuda` feature flag:
```yaml
    - name: Run tests
      run: cargo test --features cuda --verbose
```

Currently it runs `cargo test --verbose` which doesn't compile the CUDA module at all, meaning the test we're adding would never run.

**`build-macos` job:** Change test step to pass the `metal` feature flag and remove the redundant `--verify` binary invocation:
```yaml
    - name: Run tests
      run: cargo test --features metal --verbose
```

Remove:
```yaml
    - name: Verify Metal GPU keygen
      run: ./target/release/mc-keygen --verify A
```

The `cargo test --features metal` step now covers GPU verification through the test framework.

### 4. No changes to `--verify` CLI flag

The `--verify` flag in `main.rs` remains for manual user verification. The tests call the same underlying function.

## Expected Outcome

- `cargo test --features cuda` compiles and runs the CUDA test (skips gracefully without GPU hardware)
- `cargo test --features metal` compiles and runs the Metal test (full verification on macOS runners)
- `cargo test` (no features) is unaffected — GPU modules don't compile
- CI catches GPU keygen regressions automatically through standard `cargo test`
- The separate `--verify A` binary invocation is removed from CI

## TODO

### Phase 1: Add GPU test modules
- [x] Add `#[cfg(test)] mod tests` with `verify_gpu_keygen_matches_cpu` test to `src/gpu.rs`
- [x] Add `#[cfg(test)] mod tests` with `verify_gpu_keygen_matches_cpu` test to `src/metal_gpu.rs`
- [x] Verify `cargo test --features cuda` compiles locally (graceful skip on no device)

### Phase 2: Update CI workflow
- [x] Update `build-cuda` job: change `cargo test --verbose` to `cargo test --features cuda --verbose`
- [x] Update `build-macos` job: change `cargo test --verbose` to `cargo test --features metal --verbose`
- [x] Remove `Verify Metal GPU keygen` step (`./target/release/mc-keygen --verify A`) from `build-macos` job

### Phase 3: Update documentation
- [x] Update `docs/gpu.md` Verification section to document that GPU verification now runs via `cargo test --features cuda` / `cargo test --features metal`, and that `--verify` remains available for manual use
- [x] Review `README.md` for any testing-related updates needed (no changes needed — `--verify` already documented under options)

### Phase 4: Validate
- [x] Run `cargo test --features cuda --verbose` on Linux to confirm CUDA test compiles and skips gracefully (no GPU driver)
- [x] Run `cargo test --features metal --verbose` on macOS to confirm Metal test runs full GPU verification (16/16 passed)
- [x] Security audit (see below)
- [ ] Commit and open PR against `master` (dependent on PR #6)

### Platform constraints

- **Metal:** The `metal` crate requires Apple framework linking — `cargo build --features metal` fails on Linux with `link kind "framework" is only supported on Apple targets`. Metal tests can only be validated on macOS (locally or via CI `build-macos` job).
- **CUDA:** Compiles on both Linux and macOS, but `cudarc` panics at runtime if no CUDA shared library is present. The CUDA test uses `catch_unwind` to handle this gracefully.

## Acceptance Criteria

1. `cargo test --features cuda --verbose` passes on CUDA CI (no GPU, graceful skip)
2. `cargo test --features metal --verbose` passes on macOS CI (full GPU verification)
3. GPU keygen mismatches cause test failures, not silent passes
4. No changes to existing CPU-only test behavior
5. `--verify` CLI flag continues to work for manual use
