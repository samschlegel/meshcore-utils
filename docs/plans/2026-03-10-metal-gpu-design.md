# Design: Apple Metal GPU Support for mc-keygen

**Date:** 2026-03-10
**Status:** Approved
**Approach:** Parallel Module (Approach A)

## Decisions

- **Hybrid mode is default on Apple Silicon** — when Metal GPU is detected, both CPU
  threads and GPU run concurrently. CUDA retains existing `--gpu` behavior unchanged.
- **Direct port of CUDA kernel** — mechanical translation to MSL, accepting the 64-bit
  performance penalty. Optimize to 32-bit limbs later if profiling warrants it.
- **Runtime MSL compilation** — kernel source included via `include_str!`, compiled at
  runtime with `newLibraryWithSource`. Matches CUDA's NVRTC approach.

## File Structure

### New Files
- `metal/vanity_kernel.metal` — MSL kernel (direct port of `cuda/vanity_kernel.cu`)
- `src/metal_gpu.rs` — Metal GPU wrapper (mirrors `src/gpu.rs`)

### Modified Files
- `Cargo.toml` — add `metal` feature flag and dependency
- `src/main.rs` — Metal-aware `--gpu` flag, hybrid auto-detection
- `src/search.rs` — add `start_metal()` and `start_hybrid()`

## 1. Feature Flags & Dependencies

```toml
[features]
default = []
cuda = ["dep:cudarc"]
metal = ["dep:metal"]

[dependencies]
metal = { version = "0.30", optional = true }
```

Conditional compilation:
```rust
// main.rs
#[cfg(feature = "metal")]
mod metal_gpu;

#[cfg(feature = "cuda")]
mod gpu;
```

The `--gpu` flag exists on both features but dispatches to the right backend. On `metal`
builds, default behavior (no `--gpu`) uses hybrid mode automatically when a Metal GPU
is detected.

## 2. MSL Kernel — `metal/vanity_kernel.metal`

Direct mechanical port of `cuda/vanity_kernel.cu` (~3100 lines). Translation rules:

### Type System
MSL has `uint8_t`, `int32_t`, `int64_t`, `uint64_t` natively — no changes needed.

### Function Qualifiers
| CUDA | MSL |
|------|-----|
| `__device__` | (no qualifier) |
| `__device__ __forceinline__` | `inline` |
| `extern "C" __global__` | `kernel` |
| `__constant__` arrays | `constant` address space |

### Kernel Signature
```metal
kernel void vanity_search(
    device uint8_t* result [[buffer(0)]],
    constant uint8_t* prefix_data [[buffer(1)]],
    constant uint32_t& prefix_count [[buffer(2)]],
    constant uint64_t& base_nonce [[buffer(3)]],
    constant uint64_t& iters_per_thread [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
)
```

### Thread ID
`blockIdx.x * blockDim.x + threadIdx.x` → `tid` via `[[thread_position_in_grid]]`
(Metal computes this automatically).

### Atomics
Found-flag CAS:
```cuda
// CUDA
unsigned int old = atomicCAS((unsigned int *)result, 0, 1);
```
becomes:
```metal
// MSL — atomic_exchange is simpler, same correctness (we only check old == 0)
device atomic_uint* flag = (device atomic_uint*)result;
uint old = atomic_exchange_explicit(flag, 1, memory_order_relaxed);
```

### Volatile Early-Exit Check
```cuda
// CUDA
if (*((volatile unsigned int *)result) != 0) return;
```
becomes:
```metal
// MSL
if (atomic_load_explicit((device atomic_uint*)result, memory_order_relaxed) != 0) return;
```

### rotr64
No intrinsic in MSL — manual implementation:
```metal
inline uint64_t rotr64(uint64_t x, uint32_t n) {
    return (x >> n) | (x << (64 - n));
}
```

### Precomputed Tables
~1400 lines of constant data. Copy-paste with `__constant__` → `constant`.

### Everything Else
`fe_*`, `ge_*`, `sha512_32`, `check_prefix`, `check_any_prefix` are plain C arithmetic
that ports directly — remove CUDA qualifiers and it's valid MSL.

### Verification Kernel
Same pattern: `kernel void verify_keygen(...)` with `[[buffer(N)]]` attributes and
`[[thread_position_in_grid]]`.

## 3. Rust Metal Wrapper — `src/metal_gpu.rs`

Mirrors `src/gpu.rs` but uses `metal-rs` APIs.

### Struct
```rust
use metal::{Device, MTLResourceOptions, MTLSize, ComputePipelineState, CommandQueue};

pub struct MetalSearcher {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,       // vanity_search kernel
    verify_pipeline: ComputePipelineState, // verify_keygen kernel
    result_buf: metal::Buffer,             // 100 bytes, shared mode
    prefix_buf: metal::Buffer,             // packed prefix data, shared mode
    prefix_count: u32,
    grid_size: u64,                        // total threads per dispatch
    threadgroup_size: u64,                 // threads per threadgroup
}
```

### Zero-Copy Buffers (Key Difference from CUDA)
The CUDA path does `clone_htod()` / `clone_dtoh()` every batch. Metal with unified
memory uses `MTLResourceStorageModeShared` — CPU and GPU access the same memory:

```rust
// CUDA approach (copies):
let prefix_dev = self.stream.clone_htod(&self.prefix_data);
let result_host = self.stream.clone_dtoh(&self.result_buf);

// Metal approach (zero-copy):
let result_buf = device.new_buffer(100, MTLResourceOptions::StorageModeShared);
// After GPU finishes, read the pointer directly:
let ptr = result_buf.contents() as *const u8;
```

Prefix data uploaded once at construction and reused (no per-batch copy).

### Kernel Compilation
```rust
fn compile_kernel(device: &Device) -> Result<metal::Library, MetalError> {
    let source = include_str!("../metal/vanity_kernel.metal");
    device.new_library_with_source(source, &metal::CompileOptions::new())
        .map_err(|e| MetalError::Compilation(e.to_string()))
}
```

### Batch Dispatch
```rust
pub fn search_batch(&mut self, base_nonce: u64) -> Result<MetalBatchResult, MetalError> {
    // Zero the result buffer (memset the shared pointer)
    unsafe { std::ptr::write_bytes(self.result_buf.contents() as *mut u8, 0, 100); }

    let cmd_buf = self.queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&self.pipeline);
    encoder.set_buffer(0, Some(&self.result_buf), 0);
    encoder.set_buffer(1, Some(&self.prefix_buf), 0);
    encoder.set_bytes(2, size_of::<u32>() as u64, &self.prefix_count as *const _ as *const _);
    encoder.set_bytes(3, size_of::<u64>() as u64, &base_nonce as *const _ as *const _);
    encoder.set_bytes(4, size_of::<u64>() as u64, &ITERS_PER_THREAD as *const _ as *const _);

    let grid = MTLSize::new(self.grid_size, 1, 1);
    let threadgroup = MTLSize::new(self.threadgroup_size, 1, 1);
    encoder.dispatch_threads(grid, threadgroup);
    encoder.end_encoding();

    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Read result directly from shared memory (zero-copy)
    let result_ptr = self.result_buf.contents() as *const u8;
    let result_slice = unsafe { std::slice::from_raw_parts(result_ptr, 100) };
    // ... parse found flag + keypair (same logic as gpu.rs)
}
```

### Grid Sizing
```rust
let max_threads = pipeline.max_total_threads_per_threadgroup();  // typically 1024
let thread_execution_width = pipeline.thread_execution_width();  // 32 (SIMD group)
let threadgroup_size = max_threads.min(256); // match CUDA BLOCK_SIZE
let grid_size = threadgroup_size * 64;       // tunable, start conservative
```

### Error Type
```rust
pub enum MetalError {
    NoMetalDevice,
    Compilation(String),
    Runtime(String),
}
```

### Verification
Same as CUDA — `verify_gpu_keygen()` runs 64 test seeds through GPU and CPU,
cross-checks byte-by-byte.

## 4. Hybrid Mode & CLI

### `search.rs` — New Methods

`start_metal()` — GPU-only Metal search, mirrors `start_gpu()`:
```rust
#[cfg(feature = "metal")]
pub fn start_metal(prefixes: &[String]) -> Result<Self, crate::metal_gpu::MetalError> {
    // Same pattern as start_gpu(): one thread, MetalSearcher dispatch loop
}
```

`start_hybrid()` — CPU threads + Metal GPU concurrently:
```rust
#[cfg(feature = "metal")]
pub fn start_hybrid(
    prefixes: &[String],
    cpu_threads: usize,
) -> Result<Self, crate::metal_gpu::MetalError> {
    let found = Arc::new(AtomicBool::new(false));
    let attempts = Arc::new(AtomicU64::new(0));
    let result = Arc::new(Mutex::new(None));
    let mut workers = Vec::new();

    // Spawn CPU workers — identical to start() loop bodies
    for _ in 0..cpu_threads {
        // ... clone arcs, spawn thread with generate_keypair loop
    }

    // Spawn Metal dispatch thread — identical to start_metal() loop body
    let metal_searcher = MetalSearcher::new(prefixes)?;
    workers.push(thread::spawn(move || {
        // ... Metal batch dispatch loop
    }));

    Ok(SearchHandle { found, attempts, result, start: Instant::now(), workers })
}
```

All workers (CPU and GPU) share the same `found`, `attempts`, and `result` atomics —
first to find a match wins.

### `main.rs` — CLI Changes

The `--gpu` flag becomes available on both features:
```rust
/// Use GPU acceleration (CUDA on Linux, Metal on macOS)
#[cfg(any(feature = "cuda", feature = "metal"))]
#[arg(long)]
gpu: bool,

/// Verify GPU keygen matches CPU
#[cfg(any(feature = "cuda", feature = "metal"))]
#[arg(long)]
verify: bool,
```

Dispatch logic:
```rust
// Determine GPU availability
#[cfg(feature = "cuda")]
let use_gpu = cli.gpu;
#[cfg(feature = "metal")]
let use_gpu = cli.gpu || metal::Device::system_default().is_some();
#[cfg(not(any(feature = "cuda", feature = "metal")))]
let use_gpu = false;

if use_gpu {
    #[cfg(feature = "cuda")]
    {
        // Existing CUDA path — GPU only, unchanged
        let handle = SearchHandle::start_gpu(&prefixes)?;
        // mode_label = "GPU"
    }
    #[cfg(feature = "metal")]
    {
        // Hybrid by default on Metal
        let handle = SearchHandle::start_hybrid(&prefixes, num_threads)?;
        // mode_label = "Metal GPU + N threads"
    }
} else {
    // CPU-only path — unchanged
}
```

TUI mode label shows `"Metal GPU + 8 threads"` so the user knows both are active.

## 5. Testing & Verification

- `verify_gpu_keygen()` in `metal_gpu.rs` generates 64 test seeds, runs them through
  both the Metal `verify_keygen` kernel and CPU `generate_keypair()`, cross-checks
  public and private keys byte-by-byte
- Accessible via `--verify` flag on Metal builds
- CI: add a `macos-latest` (Apple Silicon) GitHub Actions job that builds with
  `--features metal` and runs `--verify`
- CPU-side code (`SearchHandle`, `PrefixMatcher`) already has tests — no new unit
  tests needed beyond the verification kernel

## 64-bit Performance Note

Apple GPU has no native 64-bit ALUs. SHA-512 and `fe_mul` widening multiplies will be
~4-16x slower per operation than NVIDIA. Expected Metal GPU throughput on M1 is ~1-3M
keys/sec (vs ~68M on RTX 4090). This is still 3-7x faster than CPU-only (~400K keys/sec
on M1). The 32-bit limb optimization (Strategy B from research) is a follow-up if
profiling confirms field arithmetic as the bottleneck.
