# Plan: GPU Trait Abstraction

## Problem

PR #1 review: `start_hybrid()` and `start_gpu()` are hardcoded to specific GPU
backends. The reviewer wants a single code path that works with any GPU via a trait.

## Design

### 1. Unified batch result type (`search.rs`)

Replace `GpuBatchResult` and `MetalBatchResult` with a single type in `search.rs`:

```rust
pub struct GpuBatchResult {
    pub keys_checked: u64,
    pub keypair: Option<MeshCoreKeypair>,
}
```

Both backends already return exactly these fields.

### 2. `GpuSearcher` trait (`search.rs`)

```rust
pub trait GpuSearcher: Send {
    fn search_batch(
        &mut self,
        base_nonce: u64,
    ) -> Result<GpuBatchResult, Box<dyn std::error::Error + Send + Sync>>;

    fn device_name(&self) -> &str;
}
```

- `Send` bound: the searcher is moved into a `thread::spawn` closure.
- `&mut self`: correct for single-thread-per-device ownership.
- `Box<dyn Error>`: both backends only format errors for display, no need for
  structured error matching in the consumer.
- `device_name()`: for TUI display ("Metal GPU", "CUDA GPU (RTX 5090)", etc.)

### 3. Backend implementations

**Metal** (`metal_gpu.rs`):
```rust
impl GpuSearcher for MetalSearcher {
    fn search_batch(&mut self, base_nonce: u64) -> Result<GpuBatchResult, ...> {
        // existing logic, return unified GpuBatchResult
    }
    fn device_name(&self) -> &str { &self.device_name }
}
```

**CUDA** (`gpu.rs`):
```rust
impl GpuSearcher for CudaSearcher {
    fn search_batch(&mut self, base_nonce: u64) -> Result<GpuBatchResult, ...> {
        // existing logic, return unified GpuBatchResult
    }
    fn device_name(&self) -> &str { &self.device_name }
}
```

Each backend stores a `device_name: String` field populated during `new()`.

### 4. Replace `start_gpu`, `start_metal`, `start_hybrid` with two methods

**`start_gpu`** — GPU-only search with one or more GPU devices:
```rust
pub fn start_gpu(
    prefixes: &[String],
    gpu_searchers: Vec<Box<dyn GpuSearcher>>,
) -> Self
```

**`start_hybrid`** — CPU threads + GPU devices concurrently:
```rust
pub fn start_hybrid(
    prefixes: &[String],
    cpu_threads: usize,
    gpu_searchers: Vec<Box<dyn GpuSearcher>>,
) -> Self
```

Both accept `Vec<Box<dyn GpuSearcher>>` — one entry per device. Today that's
always a single-element vec, but multi-GPU is naturally supported.

The GPU dispatch loop (nonce init → search_batch → update atomics → check match)
is extracted into a shared helper used by both methods:

```rust
fn gpu_dispatch_loop(
    mut searcher: Box<dyn GpuSearcher>,
    matchers: Arc<Vec<(String, PrefixMatcher)>>,
    found: Arc<AtomicBool>,
    attempts: Arc<AtomicU64>,
    result: Arc<Mutex<Option<MatchResult>>>,
)
```

### 5. CLI changes (`main.rs`)

Replace `--gpu` flag with two explicit override flags:

```rust
/// Force CPU-only search (no GPU even if available)
#[cfg(any(feature = "cuda", feature = "metal"))]
#[arg(long, conflicts_with = "gpu_only")]
cpu_only: bool,

/// Force GPU-only search (no CPU threads)
#[cfg(any(feature = "cuda", feature = "metal"))]
#[arg(long, conflicts_with = "cpu_only")]
gpu_only: bool,
```

**Default behavior** (no flags): attempt GPU initialization; if it succeeds, run
hybrid mode (CPU + GPU). If GPU init fails (no device, no driver, etc.), fall back
to CPU-only with a warning to stderr. This applies to both CUDA and Metal builds
uniformly.

**`--gpu-only`**: fail hard if GPU initialization fails — the user explicitly
requested GPU and should know it's unavailable.

**`--cpu-only`**: skip GPU initialization entirely.

### 6. GPU initialization with graceful fallback

```rust
/// Attempt to initialize GPU searchers. Returns empty vec on failure.
/// Prints a warning to stderr if GPU init fails (unless --cpu-only).
fn try_init_gpu(prefixes: &[String]) -> Vec<Box<dyn search::GpuSearcher>> {
    #[cfg(feature = "metal")]
    {
        match metal_gpu::MetalSearcher::new(prefixes) {
            Ok(s) => return vec![Box::new(s)],
            Err(e) => {
                eprintln!("Warning: Metal GPU unavailable ({}), using CPU only", e);
                return vec![];
            }
        }
    }
    #[cfg(feature = "cuda")]
    {
        match gpu::CudaSearcher::new(prefixes) {
            Ok(s) => return vec![Box::new(s)],
            Err(e) => {
                eprintln!("Warning: CUDA GPU unavailable ({}), using CPU only", e);
                return vec![];
            }
        }
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    { vec![] }
}
```

### 7. Mode selection and dispatch

```rust
let gpu_searchers = if cpu_only {
    vec![]
} else {
    try_init_gpu(&prefixes)
};

let (handle, mode_label) = if gpu_only {
    if gpu_searchers.is_empty() {
        // User explicitly asked for GPU but none available — fatal
        print_colored_error("--gpu-only requested but no GPU available");
        std::process::exit(1);
    }
    let label = gpu_names_label(&gpu_searchers);
    (SearchHandle::start_gpu(&prefixes, gpu_searchers), label)
} else if gpu_searchers.is_empty() {
    // CPU-only (explicit via --cpu-only, or fallback from failed GPU init)
    let label = format!("{} threads{}", num_threads, prefix_count_label);
    (SearchHandle::start(&prefixes, num_threads), label)
} else {
    // Hybrid — default when GPU is available
    let gpu_label = gpu_names_label(&gpu_searchers);
    let label = format!("{} + {} threads{}", gpu_label, num_threads, prefix_count_label);
    (SearchHandle::start_hybrid(&prefixes, num_threads, gpu_searchers), label)
};
```

Mode label uses `device_name()` from the trait:
```rust
fn gpu_names_label(searchers: &[Box<dyn search::GpuSearcher>]) -> String {
    searchers.iter().map(|g| g.device_name()).collect::<Vec<_>>().join(", ")
}
```

### 8. `--verify` path

`verify_gpu_keygen()` stays as standalone functions on each backend module — it's a
one-shot diagnostic, not a search, so it doesn't need the trait. The `#[cfg]` dispatch
in `main.rs` for `--verify` stays as-is.

### 9. Rename `gpu::GpuSearcher` to `gpu::CudaSearcher`

The CUDA struct is currently named `GpuSearcher` which collides with the new trait
name. Rename to `CudaSearcher` for clarity. Similarly rename `GpuBatchResult` and
`GpuError` in the CUDA module to `CudaBatchResult` and `CudaError` (the batch result
type in `gpu.rs` becomes internal — the trait returns the unified `GpuBatchResult`
from `search.rs`).

## What's NOT changing

- Feature flags (`metal`, `cuda`) — Cargo limitation, can't unify
- GPU kernel code (`.metal`, `.cu`)
- `verify_gpu_keygen()` — stays per-backend
- CPU-only `SearchHandle::start()` — untouched
- `PrefixMatcher`, `SearchStats`, `SearchResult` — untouched
- Existing tests

## Files modified

| File | Change |
|------|--------|
| `src/search.rs` | Add `GpuBatchResult`, `GpuSearcher` trait, `gpu_dispatch_loop`, refactor to `start_gpu`/`start_hybrid` accepting `Vec<Box<dyn GpuSearcher>>`, remove `start_metal` |
| `src/gpu.rs` | Rename to `CudaSearcher`/`CudaError`, implement `GpuSearcher` trait, store `device_name` |
| `src/metal_gpu.rs` | Remove `MetalBatchResult`, implement `GpuSearcher` trait, store `device_name` |
| `src/main.rs` | Replace `--gpu` with `--cpu-only`/`--gpu-only`, add `try_init_gpu()` with graceful fallback, unified mode dispatch |
| `src/lib.rs` | No changes needed |

---

## TODO

### Phase 1: Trait and unified types in `search.rs`

- [x]1.1 Add `GpuBatchResult` struct to `search.rs` (fields: `keys_checked: u64`, `keypair: Option<MeshCoreKeypair>`)
- [x]1.2 Add `GpuSearcher` trait to `search.rs` (`search_batch(&mut self, u64) -> Result<GpuBatchResult, Box<dyn Error + Send + Sync>>`, `device_name(&self) -> &str`; bound `Send`)
- [x]1.3 Extract `gpu_dispatch_loop()` helper function from the duplicated GPU loop logic in `start_metal`/`start_gpu`
- [x]1.4 Rewrite `start_gpu()` to accept `Vec<Box<dyn GpuSearcher>>` and use `gpu_dispatch_loop()` (one thread per searcher)
- [x]1.5 Rewrite `start_hybrid()` to accept `Vec<Box<dyn GpuSearcher>>` + `cpu_threads: usize`, spawn CPU workers + GPU dispatch threads via `gpu_dispatch_loop()`
- [x]1.6 Remove `start_metal()` — now covered by `start_gpu`/`start_hybrid` with trait objects

### Phase 2: CUDA backend (`src/gpu.rs`)

- [x]2.1 Rename `GpuSearcher` → `CudaSearcher`, `GpuBatchResult` → `CudaBatchResult`, `GpuError` → `CudaError`
- [x]2.2 Add `device_name: String` field to `CudaSearcher`, populate in `new()` from CUDA device name query
- [x]2.3 Implement `search::GpuSearcher` trait for `CudaSearcher` — delegate to existing `search_batch()`, convert `CudaBatchResult` → `GpuBatchResult`, convert `CudaError` → `Box<dyn Error>`

### Phase 3: Metal backend (`src/metal_gpu.rs`)

- [x]3.1 Remove `MetalBatchResult` struct (replaced by `search::GpuBatchResult`)
- [x]3.2 Add `device_name: String` field to `MetalSearcher`, populate in `new()` from `device.name()`
- [x]3.3 Implement `search::GpuSearcher` trait for `MetalSearcher` — adapt existing `search_batch()` to return `GpuBatchResult` and `Box<dyn Error>`

### Phase 4: CLI and mode dispatch (`src/main.rs`)

- [x]4.1 Replace `--gpu` flag with `--cpu-only` and `--gpu-only` (both `#[cfg(any(feature = "cuda", feature = "metal"))]`, `conflicts_with` each other)
- [x]4.2 Add `try_init_gpu()` function — attempts backend init behind `#[cfg]`, returns `Vec<Box<dyn GpuSearcher>>`, warns to stderr on failure
- [x]4.3 Add `gpu_names_label()` helper using `device_name()` trait method
- [x]4.4 Replace the existing GPU/Metal/CPU dispatch blocks with unified mode selection: `--gpu-only` → `start_gpu` (fatal on empty), `--cpu-only` or empty vec → `start`, else → `start_hybrid`
- [x]4.5 Remove `use metal::Device as MetalDevice` import and its auto-detection logic (replaced by `try_init_gpu`)

### Phase 5: Verification

- [x]5.1 `cargo check` (no features) — compiles clean, zero warnings
- [x]5.2 `cargo check --features metal` — compiles clean, zero warnings
- [x]5.3 `cargo test` (no features) — all tests pass
- [x]5.4 `cargo test --features metal` — all tests pass
- [x]5.5 `mc-keygen --verify A` — 64/64 GPU vs CPU match (Metal build)
- [x]5.6 `mc-keygen --cpu-only A` — runs CPU-only mode, TUI shows "N threads"
- [x]5.7 `mc-keygen --gpu-only A` — runs GPU-only mode, TUI shows device name
- [x]5.8 `mc-keygen A` — auto-detects GPU, runs hybrid, TUI shows "Metal GPU + N threads"

### Phase 6: Documentation

- [x]6.1 Update README.md — replace `--gpu` flag references with `--cpu-only`/`--gpu-only`, document new default hybrid behavior and graceful fallback
