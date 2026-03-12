# GPU Acceleration

## Performance

Measured throughput on real hardware:

| | Throughput | Hardware |
|-|------------|----------|
| **CPU** | ~1.3M keys/sec | AMD Ryzen 9 7950X3D (32 threads) |
| **GPU** | ~68M keys/sec | NVIDIA RTX 4090 (24GB) |

The GPU is roughly **50x faster** than the CPU on this hardware. The speedup comes from running the full Ed25519 pipeline (SHA-512, scalar multiply, point compress, prefix check) across thousands of GPU threads in parallel.

## Building with CUDA

```bash
cargo build --release --features cuda
```

Requires the NVIDIA CUDA Toolkit to be installed. The CUDA kernel is compiled at runtime via NVRTC, so `nvrtc` and `cuda` shared libraries must be on PATH.

The `cudarc` dependency is configured for CUDA 13.1 by default. To target a different CUDA version, change the `cuda-13010` feature in `Cargo.toml` to match your installation (e.g. `cuda-12060` for CUDA 12.6).

## How GPU mode works

When `--gpu` is passed, the entire keygen pipeline (SHA-512, scalar clamping, Ed25519 scalar multiplication, point compression, prefix check) runs on the GPU. A single host thread loops: launch kernel batch, sync, check for match, repeat. The TUI progress display works identically for both CPU and GPU modes.

The Ed25519 field arithmetic uses the ref10 implementation (radix-2^25.5, int32[10] limbs) from [solana-perf-libs](https://github.com/solana-labs/solana-perf-libs), which is a CUDA adaptation of the [SUPERCOP](https://bench.cr.yp.to/supercop.html) reference code. Scalar multiplication uses windowed lookup tables with 32 precomputed basepoint multiples.

## Verification

GPU keygen correctness is verified by comparing 64 deterministic test seeds against the CPU reference implementation. This runs automatically in CI via `cargo test`:

```bash
cargo test --features cuda    # CUDA (skips gracefully if no GPU device)
cargo test --features metal   # Metal (runs full verification on macOS)
```

For manual verification, use the `--verify` CLI flag:

```bash
mc-keygen --verify A
```

## Sources

- [solana-perf-libs](https://github.com/solana-labs/solana-perf-libs) — CUDA Ed25519 field arithmetic (ref10)
- [cudarc](https://github.com/coreylowman/cudarc) — safe Rust bindings for the CUDA driver API
