# Plan: CUDA GitHub Action Worker (Phase 1 — Compile-Only)

**Date:** 2026-03-12
**Issue:** samschlegel/meshcore-utils#2
**Research:** docs/research/2026-03-12-research-cuda-github-action.md

## What Changes

Add a `build-cuda` job to `.github/workflows/rust.yml` that installs CUDA 13.1 on `ubuntu-latest` and builds with `--features cuda`.

## Implementation

Add the following job after the existing `build` job:

```yaml
build-cuda:
  runs-on: ubuntu-latest

  steps:
  - uses: actions/checkout@v4
  - uses: Jimver/cuda-toolkit@v0.2.30
    id: cuda-toolkit
    with:
      cuda: '13.1.0'
  - name: Build (CUDA)
    run: cargo build --release --features cuda --verbose
  - name: Run tests
    run: cargo test --verbose
```

## Notes

- `--release` mirrors the macOS Metal job pattern (release build for feature-gated GPU code)
- `cargo test` runs existing CPU tests to ensure the cuda feature flag doesn't break them
- No `--verify` step — no GPU hardware on standard runners
- CUDA Toolkit 13.1.0 matches the cudarc `cuda-13010` feature in Cargo.toml
