# Research: GPU Trait Abstraction

## Context

PR #1 review comment from @samschlegel requests that the hybrid search work with
*any* GPU backend, not just Metal. Currently `start_hybrid()` is hardcoded to
`MetalSearcher`. The CUDA path (`start_gpu`) is similarly hardcoded to
`crate::gpu::GpuSearcher`.

## Current State

### CUDA backend (`src/gpu.rs`)

```rust
pub struct GpuSearcher { /* CUDA device state */ }
pub struct GpuBatchResult { pub keys_checked: u64, pub keypair: Option<MeshCoreKeypair> }
pub enum GpuError { NoCudaDevice, CudaDriver(String), Compilation(String) }

impl GpuSearcher {
    pub fn new(prefixes: &[String]) -> Result<Self, GpuError>;
    pub fn search_batch(&mut self, base_nonce: u64) -> Result<GpuBatchResult, GpuError>;
}
pub fn verify_gpu_keygen() -> Result<(), GpuError>;
```

### Metal backend (`src/metal_gpu.rs`)

```rust
pub struct MetalSearcher { /* Metal device state */ }
pub struct MetalBatchResult { pub keys_checked: u64, pub keypair: Option<MeshCoreKeypair> }
pub enum MetalError { NoMetalDevice, Compilation(String), Runtime(String) }

impl MetalSearcher {
    pub fn new(prefixes: &[String]) -> Result<Self, MetalError>;
    pub fn search_batch(&self, base_nonce: u64) -> Result<MetalBatchResult, MetalError>;
}
pub fn verify_gpu_keygen() -> Result<(), MetalError>;
```

### Shared interface pattern

Both backends have identical batch result shapes (`keys_checked` + `keypair`) and
the same lifecycle: construct with prefixes, then call `search_batch` in a loop.

Key difference: CUDA takes `&mut self`, Metal takes `&self`. The trait should use
`&mut self` since each searcher instance is owned by a single thread and `&mut self`
is the more honest/permissive signature.

### How `search.rs` uses GPU backends

Three methods on `SearchHandle`:
- `start_gpu()` — CUDA-only, single GPU thread
- `start_metal()` — Metal-only, single GPU thread
- `start_hybrid()` — Metal GPU + CPU threads, hardcoded to `MetalSearcher`

The GPU dispatch loop in all three is nearly identical:
1. Random initial nonce
2. Build CPU-side `PrefixMatcher` vec for identifying which prefix matched
3. Loop: `search_batch(nonce)` → update atomics → check for match → advance nonce

### How `main.rs` uses the backends

- `#[cfg]` gates select which backend to use
- Metal auto-detects device and defaults to hybrid mode
- CUDA requires explicit `--gpu` flag
- `--verify` dispatches to the per-backend `verify_gpu_keygen()`

## Multi-GPU considerations

- CUDA: `CudaContext::new(device_id)` — natural multi-device support
- Metal: `Device::all()` returns all Metal devices (rare to have >1 on Mac)
- Each device gets its own searcher instance on its own thread
- `&mut self` is correct: one thread owns one searcher, no sharing needed

## Error type considerations

Both backends define their own error enums. The trait needs a unified error type.
Options:
1. `Box<dyn std::error::Error + Send>` — simplest, no new types
2. A new `GpuError` enum wrapping backend errors — more structured but adds boilerplate
3. `String` — too lossy

Recommendation: `Box<dyn std::error::Error + Send + Sync>` — the trait consumer
(`search.rs`) only ever formats errors with `eprintln!`, so full structure isn't needed.
This also keeps the trait free of backend-specific types.

## Feature flag interactions

- `default = []` — no GPU
- `cuda` — enables CUDA backend
- `metal` — enables Metal backend
- Both could theoretically be enabled (the reviewer wants this path open)
- Currently `main.rs` uses `#[cfg(all(feature = "metal", not(feature = "cuda")))]`
  which would need updating for dual-backend support
