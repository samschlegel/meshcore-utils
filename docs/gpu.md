# GPU Acceleration

## Performance

Measured throughput on real hardware:

### NVIDIA CUDA

| | Throughput | Hardware |
|-|------------|----------|
| **CPU** | ~1.3M keys/sec | AMD Ryzen 9 7950X3D (32 threads) |
| **GPU** | ~68M keys/sec | NVIDIA RTX 4090 (24GB) |

The GPU is roughly **50x faster** than the CPU on this hardware.

### Apple Metal

| | Throughput | Hardware |
|-|------------|----------|
| **Hybrid (Metal + CPU)** | ~5.1M keys/sec | MacBook Pro M1 Max (high power mode, mains) |
| **Hybrid (Metal + CPU)** | ~2.9M keys/sec | MacBook Pro M1 Pro (low power mode, battery) |
| **CPU-only** | ~477K keys/sec | MacBook Pro M1 Pro (low power mode, battery) |

Metal hybrid mode provides roughly a **6x speedup** over CPU-only on the M1 Pro.

The speedup comes from running the full Ed25519 pipeline (SHA-512, scalar multiply, point compress, prefix check) across thousands of GPU threads in parallel.

## Building with CUDA

```bash
cargo build --release --features cuda
```

Requires the NVIDIA CUDA Toolkit to be installed. The CUDA kernel is compiled at runtime via NVRTC, so `nvrtc` and `cuda` shared libraries must be on PATH.

The `cudarc` dependency is configured for CUDA 13.1 by default. To target a different CUDA version, change the `cuda-13010` feature in `Cargo.toml` to match your installation (e.g. `cuda-12060` for CUDA 12.6).

## Building with Metal

```bash
cargo build --release --features metal
```

Requires macOS with Apple Silicon. Metal hybrid mode (GPU + CPU) is the default when the `metal` feature is enabled. Use `--cpu-only` to disable GPU acceleration.

## How GPU mode works

When built with a GPU feature (`cuda` or `metal`), the default mode is **hybrid** — GPU and CPU search threads run concurrently. Use `--cpu-only` to disable GPU acceleration, or `--gpu-only` to disable CPU threads.

The entire keygen pipeline (SHA-512, scalar clamping, Ed25519 scalar multiplication, point compression, prefix check) runs on the GPU. A single host thread loops: launch kernel batch, sync, check for match, repeat. The TUI progress display works identically across all modes.

The CUDA backend's Ed25519 field arithmetic uses the ref10 implementation (radix-2^25.5, int32[10] limbs) from [solana-perf-libs](https://github.com/solana-labs/solana-perf-libs), which is a CUDA adaptation of the [SUPERCOP](https://bench.cr.yp.to/supercop.html) reference code. Scalar multiplication uses windowed lookup tables with 32 precomputed basepoint multiples.

## Verification

Use `--verify` to cross-check GPU-generated keys against the CPU implementation. This is useful for validating correctness after modifying the CUDA kernel.

## Sources

- [solana-perf-libs](https://github.com/solana-labs/solana-perf-libs) — CUDA Ed25519 field arithmetic (ref10)
- [cudarc](https://github.com/coreylowman/cudarc) — safe Rust bindings for the CUDA driver API
- [metal](https://crates.io/crates/metal) — Rust bindings for Apple's Metal framework
