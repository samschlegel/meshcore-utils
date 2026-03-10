# Apple Metal GPU Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Apple Metal GPU compute support for vanity key generation on Apple Silicon, with hybrid CPU+GPU mode as the default.

**Architecture:** Parallel module approach — `src/metal_gpu.rs` + `metal/vanity_kernel.metal` mirror the existing CUDA path (`src/gpu.rs` + `cuda/vanity_kernel.cu`). Metal dispatch thread + CPU worker threads run concurrently via shared atomics.

**Tech Stack:** `metal` crate (metal-rs) v0.30 for Rust Metal bindings, Metal Shading Language (MSL) for GPU kernel, runtime MSL compilation via `newLibraryWithSource`.

**Design doc:** `docs/plans/2026-03-10-metal-gpu-design.md`

---

## Phase 1: Feature Flag & Dependency

### Task 1: Add `metal` feature flag to Cargo.toml

**Files:**
- Modify: `Cargo.toml:8-11` (features section), `Cargo.toml:22` (after cudarc dep)

**Step 1: Add the metal feature and dependency**

In `Cargo.toml`, add `metal` to features and add the `metal` dependency:

```toml
[features]
default = []
cuda = ["dep:cudarc"]
metal = ["dep:metal"]

[dependencies]
# ... existing deps ...
cudarc = { version = "0.19", optional = true, features = ["cuda-13010"] }
metal = { version = "0.30", optional = true }
```

**Step 2: Verify it compiles without the feature enabled**

Run: `cargo check`
Expected: compiles cleanly (metal dep is optional, not pulled in)

**Step 3: Verify it compiles with the feature enabled**

Run: `cargo check --features metal`
Expected: compiles cleanly (metal crate resolves, no code uses it yet)

**Step 4: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "Add metal feature flag and dependency for Apple GPU support"
```

---

## Phase 2: MSL Kernel

### Task 2: Port CUDA kernel to Metal Shading Language

**Files:**
- Create: `metal/vanity_kernel.metal`
- Reference: `cuda/vanity_kernel.cu` (source for port)

This is a mechanical translation of `cuda/vanity_kernel.cu` (~3435 lines) to MSL.
The kernel is a single self-contained file for runtime compilation (same as CUDA/NVRTC).

**Step 1: Create the MSL kernel file**

Port `cuda/vanity_kernel.cu` → `metal/vanity_kernel.metal` with these translations:

1. **Header**: Replace CUDA typedef block (lines 19-27) with MSL includes:
   ```metal
   #include <metal_stdlib>
   #include <metal_atomic>
   using namespace metal;
   ```
   MSL provides `uint8_t`, `int32_t`, `int64_t`, `uint64_t` natively.

2. **`in` is a reserved keyword in MSL**: The CUDA kernel uses `in` as a parameter name
   in `load_3(const unsigned char *in)` and `load_4(const unsigned char *in)`. Rename
   the parameter to `inp` in both functions.

3. **Function qualifiers**: Apply throughout the file:
   - `static uint64_t __device__` → `static uint64_t` (or just remove `__device__`)
   - `void __device__` → `void`
   - `__device__ __forceinline__` → `inline`
   - `static const __device__` → `constant static const`
   - `static const ge_precomp __device__` → `constant static const ge_precomp`
   - `__constant__` → `constant`

4. **`fe` typedef**: `typedef int32_t fe[10];` — keep as-is, valid MSL.

5. **All `fe_*` functions** (lines 71-1424): Remove `__device__` qualifier. Everything
   else (int32_t limbs, int64_t intermediates, carry propagation) is valid MSL as-is.
   Functions: `fe_0`, `fe_1`, `fe_add`, `fe_cmov`, `fe_copy`, `fe_frombytes`, `fe_mul`,
   `fe_neg`, `fe_pow22523`, `fe_sq`, `fe_sq2`, `fe_sub`, `fe_tobytes`.

6. **`ge_*` structs and functions** (lines 1426-3260): Remove `__device__` qualifier.
   Structs `ge_p2`, `ge_p3`, `ge_p1p1`, `ge_precomp`, `ge_cached` stay as-is.
   Functions: `ge_add`, `ge_maddsub`, `ge_madd`, `ge_msub`, `ge_p1p1_to_p2`,
   `ge_p1p1_to_p3`, `ge_p2_0`, `ge_p2_dbl`, `ge_p3_0`, `ge_p3_dbl`,
   `ge_p3_to_cached`, `ge_p3_tobytes`, `ge_tobytes`, `equal`, `negative`, `cmov`,
   `select`, `ge_scalarmult_base`, `ge_sub`, `ge_addsub`.

7. **Precomputed tables** (lines 1457-2853): Change `static const ge_precomp __device__`
   to `constant static const ge_precomp`. Two tables:
   - `Bi[8]` (8 entries, ~45 lines)
   - `base[32][8]` (256 entries, ~1350 lines)
   - `d`, `sqrtm1`, `d2` field element constants — same treatment.

8. **SHA-512 section** (lines 3265-3342):
   - `typedef unsigned long long u64;` → keep (or use `uint64_t`)
   - `typedef unsigned char u8;` → keep (or use `uint8_t`)
   - `__constant__ u64 K512[80]` → `constant u64 K512[80]`
   - `rotr64`: replace `__device__ __forceinline__` with `inline`. The implementation
     `(x >> n) | (x << (64 - n))` is valid MSL. Change parameter type from `int n`
     to `uint n` (MSL shift amounts should be unsigned).
   - `Sha_Ch`, `Sha_Maj`, `Sha_Sigma0`, `Sha_Sigma1`, `sha_sigma0`, `sha_sigma1`:
     replace `__device__ __forceinline__` with `inline`.
   - `load_be64`, `store_be64`: same treatment.
   - `sha512_32`: remove `__device__`, function body unchanged.

9. **`verify_keygen` kernel** (lines 3348-3364):
   ```metal
   kernel void verify_keygen(
       const device uint8_t* seeds [[buffer(0)]],
       device uint8_t* pubkeys [[buffer(1)]],
       device uint8_t* privkeys [[buffer(2)]],
       constant uint32_t& count [[buffer(3)]],
       uint tid [[thread_position_in_grid]])
   {
       if (tid >= count) return;
       const device uint8_t* seed = seeds + tid * 32;
       // ... rest of body unchanged (sha512_32, scalar clamp, ge_scalarmult_base, etc.)
   }
   ```
   Key change: `const u8 *seeds` → `const device uint8_t* seeds`, parameters use
   `[[buffer(N)]]` attributes, thread ID via `[[thread_position_in_grid]]`.

10. **`vanity_search` kernel** (lines 3404-3434):
    ```metal
    kernel void vanity_search(
        device uint8_t* result [[buffer(0)]],
        constant uint8_t* prefix_data [[buffer(1)]],
        constant uint32_t& prefix_count [[buffer(2)]],
        constant uint64_t& base_nonce [[buffer(3)]],
        constant uint64_t& iters_per_thread [[buffer(4)]],
        uint tid [[thread_position_in_grid]])
    {
        for (uint64_t iter = 0; iter < iters_per_thread; iter++) {
            // Early exit: atomic load instead of volatile cast
            if (atomic_load_explicit(
                    (device atomic_uint*)result, memory_order_relaxed) != 0)
                return;

            uint64_t idx = base_nonce + (uint64_t)tid * iters_per_thread + iter;
            // ... seed construction, sha512_32, scalar clamp unchanged ...

            // Atomic flag set: atomic_exchange instead of atomicCAS
            if (check_any_prefix(pubkey, prefix_data, prefix_count)) {
                uint old = atomic_exchange_explicit(
                    (device atomic_uint*)result, 1, memory_order_relaxed);
                if (old == 0) {
                    for (int i = 0; i < 32; i++) result[4 + i] = pubkey[i];
                    for (int i = 0; i < 32; i++) result[36 + i] = scalar[i];
                    for (int i = 0; i < 32; i++) result[68 + i] = hash[32 + i];
                }
                return;
            }
        }
    }
    ```

11. **`check_prefix` and `check_any_prefix`** (lines 3370-3401): Remove `__device__`,
    change pointer parameters to use `device` or `constant` address space as appropriate:
    - `check_prefix(const u8 *pubkey, const u8 *prefix_bytes, ...)` — pubkey is on stack
      (thread-local), prefix_bytes points into `constant` buffer. Since MSL doesn't
      allow implicit conversions between address spaces, and the pubkey is stack-local
      while prefix_bytes comes from `constant` buffer, use `thread const uint8_t*` for
      pubkey and `constant const uint8_t*` for prefix_bytes. Alternatively, keep both
      as `const device uint8_t*` if the compiler allows it from the constant buffer.
      Simplest: use generic pointers by passing data via thread-local copies or
      templating. **Pragmatic approach**: copy prefix bytes to a local array before
      calling `check_prefix`, so both pointers are thread-local.

    Actually, the simplest approach: in `vanity_search`, `pubkey` is a local `uint8_t[32]`
    (thread address space), and `prefix_data` is `constant`. Since `check_any_prefix`
    receives both, make it take explicit address spaces:
    ```metal
    int check_prefix(thread const uint8_t* pubkey,
                     constant const uint8_t* prefix_bytes,
                     uint prefix_nibbles) { ... }

    int check_any_prefix(thread const uint8_t* pubkey,
                         constant const uint8_t* prefix_data,
                         uint prefix_count) { ... }
    ```

**Step 2: Verify MSL compiles**

This will be verified in Task 3 when the Rust wrapper compiles and loads the kernel.
There is no standalone MSL compiler step needed — runtime compilation via
`new_library_with_source` will catch syntax errors.

**Step 3: Commit**

```bash
git add metal/vanity_kernel.metal
git commit -m "Port CUDA vanity kernel to Metal Shading Language

Mechanical translation of cuda/vanity_kernel.cu to MSL for Apple GPU.
Ed25519 field arithmetic, group operations, SHA-512, and vanity search
kernels ported with CUDA-to-MSL qualifier mapping. Uses int64_t directly
(accepting Apple GPU emulation penalty for initial implementation)."
```

---

## Phase 3: Rust Metal Wrapper

### Task 3: Create `src/metal_gpu.rs` with MetalSearcher

**Files:**
- Create: `src/metal_gpu.rs`
- Modify: `src/main.rs:1-5` (add conditional module declaration)
- Reference: `src/gpu.rs` (structural mirror)

**Step 1: Add conditional module declaration to `main.rs`**

After the existing `#[cfg(feature = "cuda")] mod gpu;` line (line 3), add:
```rust
#[cfg(feature = "metal")]
mod metal_gpu;
```

**Step 2: Create `src/metal_gpu.rs`**

```rust
use std::ffi::c_void;
use std::fmt;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize,
};

use crate::search::PrefixMatcher;
use crate::types::MeshCoreKeypair;

const KERNEL_SRC: &str = include_str!("../metal/vanity_kernel.metal");
const BLOCK_SIZE: u64 = 256;
const ITERS_PER_THREAD: u64 = 64;

pub struct MetalBatchResult {
    pub keys_checked: u64,
    pub keypair: Option<MeshCoreKeypair>,
}

pub struct MetalSearcher {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    result_buf: Buffer,
    prefix_buf: Buffer,
    prefix_count: u32,
    grid_size: u64,
    threadgroup_size: u64,
}

#[derive(Debug)]
pub enum MetalError {
    NoMetalDevice,
    Compilation(String),
    Runtime(String),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::NoMetalDevice => write!(f, "no Metal-capable GPU found"),
            MetalError::Compilation(e) => write!(f, "Metal kernel compilation error: {}", e),
            MetalError::Runtime(e) => write!(f, "Metal runtime error: {}", e),
        }
    }
}

impl std::error::Error for MetalError {}

fn compile_kernel(device: &Device) -> Result<metal::Library, MetalError> {
    let options = CompileOptions::new();
    device
        .new_library_with_source(KERNEL_SRC, &options)
        .map_err(|e| MetalError::Compilation(e.to_string()))
}

impl MetalSearcher {
    pub fn new(prefixes: &[String]) -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoMetalDevice)?;
        let library = compile_kernel(&device)?;

        let func = library
            .get_function("vanity_search", None)
            .map_err(|e| MetalError::Runtime(e.to_string()))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| MetalError::Runtime(e.to_string()))?;

        let queue = device.new_command_queue();

        // Determine grid sizing from pipeline properties
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let threadgroup_size = max_threads.min(BLOCK_SIZE);
        let grid_size = threadgroup_size * 64; // conservative, tunable

        // Pack prefix data — same format as CUDA path
        let prefix_count = prefixes.len() as u32;
        let mut prefix_data: Vec<u8> = Vec::new();
        prefix_data.extend_from_slice(&prefix_count.to_le_bytes());
        for p in prefixes {
            let matcher = PrefixMatcher::new(p);
            let mut bytes = matcher.full_bytes.clone();
            if let Some(nibble) = matcher.trailing_nibble {
                bytes.push(nibble << 4);
            }
            let nibbles = p.len() as u32;
            prefix_data.extend_from_slice(&nibbles.to_le_bytes());
            prefix_data.extend_from_slice(&bytes);
            while prefix_data.len() % 4 != 0 {
                prefix_data.push(0);
            }
        }

        // Create shared-mode buffers (zero-copy on Apple Silicon)
        let result_buf = device.new_buffer(100, MTLResourceOptions::StorageModeShared);
        let prefix_buf = device.new_buffer_with_data(
            prefix_data.as_ptr() as *const c_void,
            prefix_data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(MetalSearcher {
            queue,
            pipeline,
            result_buf,
            prefix_buf,
            prefix_count,
            grid_size,
            threadgroup_size,
        })
    }

    pub fn search_batch(&self, base_nonce: u64) -> Result<MetalBatchResult, MetalError> {
        let keys_checked = self.grid_size * ITERS_PER_THREAD;

        // Zero the result buffer
        unsafe {
            std::ptr::write_bytes(self.result_buf.contents() as *mut u8, 0, 100);
        }

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.result_buf), 0);
        encoder.set_buffer(1, Some(&self.prefix_buf), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &self.prefix_count as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u64>() as u64,
            &base_nonce as *const u64 as *const c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u64>() as u64,
            &ITERS_PER_THREAD as *const u64 as *const c_void,
        );

        let grid = MTLSize::new(self.grid_size, 1, 1);
        let threadgroup = MTLSize::new(self.threadgroup_size, 1, 1);
        encoder.dispatch_threads(grid, threadgroup);
        encoder.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read result from shared memory (zero-copy)
        let result_ptr = self.result_buf.contents() as *const u8;
        let result_host = unsafe { std::slice::from_raw_parts(result_ptr, 100) };

        let found_flag = u32::from_le_bytes([
            result_host[0],
            result_host[1],
            result_host[2],
            result_host[3],
        ]);

        let keypair = if found_flag != 0 {
            let mut public_key = [0u8; 32];
            let mut private_key = [0u8; 64];
            public_key.copy_from_slice(&result_host[4..36]);
            private_key.copy_from_slice(&result_host[36..100]);
            Some(MeshCoreKeypair {
                public_key,
                private_key,
            })
        } else {
            None
        };

        Ok(MetalBatchResult {
            keys_checked,
            keypair,
        })
    }
}

/// Run Metal GPU vs CPU verification on 64 test seeds.
pub fn verify_gpu_keygen() -> Result<(), MetalError> {
    use crate::keygen::generate_keypair;

    let device = Device::system_default().ok_or(MetalError::NoMetalDevice)?;
    let library = compile_kernel(&device)?;

    let verify_func = library
        .get_function("verify_keygen", None)
        .map_err(|e| MetalError::Runtime(e.to_string()))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&verify_func)
        .map_err(|e| MetalError::Runtime(e.to_string()))?;

    let queue = device.new_command_queue();

    let count: u32 = 64;
    let mut seeds_flat = vec![0u8; count as usize * 32];
    for i in 0..count as usize {
        seeds_flat[i * 32] = i as u8;
        seeds_flat[i * 32 + 1] = (i >> 8) as u8;
        seeds_flat[i * 32 + 4] = 0xDE;
        seeds_flat[i * 32 + 5] = 0xAD;
        seeds_flat[i * 32 + 31] = 0x42;
    }

    let seeds_buf = device.new_buffer_with_data(
        seeds_flat.as_ptr() as *const c_void,
        seeds_flat.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let pubkeys_buf =
        device.new_buffer((count as u64) * 32, MTLResourceOptions::StorageModeShared);
    let privkeys_buf =
        device.new_buffer((count as u64) * 64, MTLResourceOptions::StorageModeShared);

    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&seeds_buf), 0);
    encoder.set_buffer(1, Some(&pubkeys_buf), 0);
    encoder.set_buffer(2, Some(&privkeys_buf), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &count as *const u32 as *const c_void,
    );

    let grid = MTLSize::new(count as u64, 1, 1);
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    let threadgroup = MTLSize::new(max_threads.min(count as u64), 1, 1);
    encoder.dispatch_threads(grid, threadgroup);
    encoder.end_encoding();

    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let gpu_pubkeys =
        unsafe { std::slice::from_raw_parts(pubkeys_buf.contents() as *const u8, count as usize * 32) };
    let gpu_privkeys =
        unsafe { std::slice::from_raw_parts(privkeys_buf.contents() as *const u8, count as usize * 64) };

    let mut pass = 0;
    let mut fail = 0;
    for i in 0..count as usize {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&seeds_flat[i * 32..(i + 1) * 32]);
        let cpu_kp = generate_keypair(&seed);

        let gpu_pub = &gpu_pubkeys[i * 32..(i + 1) * 32];
        let gpu_priv = &gpu_privkeys[i * 64..(i + 1) * 64];

        let pub_match = gpu_pub == cpu_kp.public_key;
        let priv_match = gpu_priv == cpu_kp.private_key;

        if pub_match && priv_match {
            pass += 1;
        } else {
            fail += 1;
            eprintln!("MISMATCH seed #{}", i);
            eprintln!("  CPU pubkey:  {}", hex::encode_upper(&cpu_kp.public_key));
            eprintln!("  GPU pubkey:  {}", hex::encode_upper(gpu_pub));
            eprintln!("  CPU privkey: {}", hex::encode_upper(&cpu_kp.private_key));
            eprintln!("  GPU privkey: {}", hex::encode_upper(gpu_priv));
            if fail >= 5 {
                eprintln!("  (stopping after 5 mismatches)");
                break;
            }
        }
    }

    eprintln!("{}/{} keys matched (Metal GPU vs CPU)", pass, count);
    if fail > 0 {
        Err(MetalError::Runtime(format!(
            "{} of {} keys mismatched between Metal GPU and CPU",
            fail, count
        )))
    } else {
        Ok(())
    }
}
```

**Step 3: Verify compilation**

Run: `cargo check --features metal`
Expected: compiles cleanly. The MSL kernel is included as a string but not yet
compiled on GPU — that happens at runtime.

**Step 4: Commit**

```bash
git add src/metal_gpu.rs src/main.rs
git commit -m "Add Metal GPU wrapper with MetalSearcher and verification

Mirrors src/gpu.rs for Apple Metal. Uses zero-copy shared buffers,
runtime MSL compilation, and GPU-vs-CPU cross-checking for correctness."
```

---

## Phase 4: Verification Gate

### Task 4: Wire up `--verify` for Metal and run verification

**Files:**
- Modify: `src/main.rs:39-46` (verify flag), `src/main.rs:358-376` (verify dispatch)

**Step 1: Update CLI flags for Metal verify support**

In `src/main.rs`, change the `--gpu` and `--verify` flags from `#[cfg(feature = "cuda")]`
to `#[cfg(any(feature = "cuda", feature = "metal"))]`:

```rust
    /// Use GPU acceleration
    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[arg(long)]
    gpu: bool,

    /// Verify GPU keygen matches CPU (run 64 test seeds and compare)
    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[arg(long)]
    verify: bool,
```

**Step 2: Add Metal verify dispatch**

In `main()`, after the existing CUDA verify block (around line 363-376), add a Metal
verify block. Restructure the verify section to handle both features:

```rust
    #[cfg(any(feature = "cuda", feature = "metal"))]
    if cli.verify {
        eprint!("Compiling GPU kernel and running verification... ");
        #[cfg(feature = "cuda")]
        let result = gpu::verify_gpu_keygen()
            .map_err(|e| format!("{}", e));
        #[cfg(feature = "metal")]
        let result = metal_gpu::verify_gpu_keygen()
            .map_err(|e| format!("{}", e));
        match result {
            Ok(()) => {
                eprintln!("PASSED");
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("FAILED: {}", e);
                std::process::exit(1);
            }
        }
    }
```

Also update the `use_gpu` determination to support both features without conflicting:
```rust
    #[cfg(feature = "cuda")]
    let use_gpu = cli.gpu;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let use_gpu = cli.gpu;
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let use_gpu = false;
```

**Step 3: Build and run verification**

Run: `cargo build --release --features metal`
Then: `./target/release/mc-keygen --verify A`
Expected: `Compiling GPU kernel and running verification... 64/64 keys matched (Metal GPU vs CPU)\nPASSED`

This is the **critical correctness gate**. If any keys mismatch, the MSL kernel has a
bug that must be fixed before proceeding.

**Step 4: If verification fails**

Debug strategy:
1. Check which seed index failed — compare CPU vs GPU output byte-by-byte
2. Most likely causes: address space qualifier mismatch in MSL, `in` keyword conflict,
   missing `constant`/`device` qualifier on a pointer
3. Fix the MSL kernel, rebuild, re-run `--verify`
4. Do not proceed to Phase 5 until all 64/64 pass

**Step 5: Commit**

```bash
git add src/main.rs
git commit -m "Wire up --verify for Metal GPU and validate correctness

All 64 test seeds produce identical keypairs on Metal GPU and CPU."
```

---

## Phase 5: Search Integration

### Task 5: Add `start_metal()` and `start_hybrid()` to SearchHandle

**Files:**
- Modify: `src/search.rs` (add two new methods after `start_gpu()` at line 268)

**Step 1: Add `start_metal()` method**

After the `start_gpu()` method (line 268), add:

```rust
    #[cfg(feature = "metal")]
    pub fn start_metal(prefixes: &[String]) -> Result<Self, crate::metal_gpu::MetalError> {
        let found = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let result: Arc<Mutex<Option<MatchResult>>> = Arc::new(Mutex::new(None));

        let mut metal_searcher = crate::metal_gpu::MetalSearcher::new(prefixes)?;
        let prefixes_owned: Vec<String> = prefixes.to_vec();

        let found_clone = Arc::clone(&found);
        let attempts_clone = Arc::clone(&attempts);
        let result_clone = Arc::clone(&result);

        let worker = thread::spawn(move || {
            let mut nonce_bytes = [0u8; 8];
            OsRng.fill_bytes(&mut nonce_bytes);
            let mut base_nonce: u64 = u64::from_le_bytes(nonce_bytes);

            let matchers: Vec<(String, PrefixMatcher)> = prefixes_owned
                .iter()
                .map(|p| (p.clone(), PrefixMatcher::new(p)))
                .collect();

            while !found_clone.load(Ordering::Relaxed) {
                match metal_searcher.search_batch(base_nonce) {
                    Ok(batch_result) => {
                        attempts_clone.fetch_add(batch_result.keys_checked, Ordering::Relaxed);
                        if let Some(kp) = batch_result.keypair {
                            found_clone.store(true, Ordering::Relaxed);
                            let matched_prefix = matchers
                                .iter()
                                .find(|(_, m)| m.matches(&kp.public_key))
                                .map(|(p, _)| p.clone())
                                .unwrap_or_else(|| prefixes_owned[0].clone());
                            *result_clone.lock().unwrap() = Some(MatchResult {
                                keypair: kp,
                                matched_prefix,
                            });
                            return;
                        }
                        base_nonce = base_nonce.wrapping_add(batch_result.keys_checked);
                    }
                    Err(e) => {
                        eprintln!("Metal GPU error: {}", e);
                        return;
                    }
                }
            }
        });

        Ok(SearchHandle {
            found,
            attempts,
            result,
            start: Instant::now(),
            workers: vec![worker],
        })
    }
```

**Step 2: Add `start_hybrid()` method**

After `start_metal()`, add:

```rust
    #[cfg(feature = "metal")]
    pub fn start_hybrid(
        prefixes: &[String],
        cpu_threads: usize,
    ) -> Result<Self, crate::metal_gpu::MetalError> {
        let matchers: Arc<Vec<(String, PrefixMatcher)>> = Arc::new(
            prefixes
                .iter()
                .map(|p| (p.clone(), PrefixMatcher::new(p)))
                .collect(),
        );
        let found = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let result: Arc<Mutex<Option<MatchResult>>> = Arc::new(Mutex::new(None));

        let mut workers = Vec::with_capacity(cpu_threads + 1);

        // Spawn CPU workers — same loop body as start()
        for _ in 0..cpu_threads {
            let matchers = Arc::clone(&matchers);
            let found = Arc::clone(&found);
            let total_attempts = Arc::clone(&attempts);
            let result = Arc::clone(&result);

            workers.push(thread::spawn(move || {
                let mut seed = [0u8; 32];
                let mut local_count: u64 = 0;

                while !found.load(Ordering::Relaxed) {
                    OsRng.fill_bytes(&mut seed);
                    let kp = generate_keypair(&seed);
                    local_count += 1;

                    if local_count % BATCH_SIZE == 0 {
                        total_attempts.fetch_add(BATCH_SIZE, Ordering::Relaxed);
                    }

                    if should_skip(&kp.public_key) {
                        continue;
                    }

                    if let Some(matched) =
                        matchers.iter().find(|(_, m)| m.matches(&kp.public_key))
                    {
                        total_attempts.fetch_add(local_count % BATCH_SIZE, Ordering::Relaxed);
                        found.store(true, Ordering::Relaxed);
                        *result.lock().unwrap() = Some(MatchResult {
                            keypair: kp,
                            matched_prefix: matched.0.clone(),
                        });
                        return;
                    }
                }

                total_attempts.fetch_add(local_count % BATCH_SIZE, Ordering::Relaxed);
            }));
        }

        // Spawn Metal GPU dispatch thread
        let metal_searcher = crate::metal_gpu::MetalSearcher::new(prefixes)?;
        let prefixes_owned: Vec<String> = prefixes.to_vec();
        let found_clone = Arc::clone(&found);
        let attempts_clone = Arc::clone(&attempts);
        let result_clone = Arc::clone(&result);

        workers.push(thread::spawn(move || {
            let mut nonce_bytes = [0u8; 8];
            OsRng.fill_bytes(&mut nonce_bytes);
            let mut base_nonce: u64 = u64::from_le_bytes(nonce_bytes);

            let matchers: Vec<(String, PrefixMatcher)> = prefixes_owned
                .iter()
                .map(|p| (p.clone(), PrefixMatcher::new(p)))
                .collect();

            while !found_clone.load(Ordering::Relaxed) {
                match metal_searcher.search_batch(base_nonce) {
                    Ok(batch_result) => {
                        attempts_clone.fetch_add(batch_result.keys_checked, Ordering::Relaxed);
                        if let Some(kp) = batch_result.keypair {
                            found_clone.store(true, Ordering::Relaxed);
                            let matched_prefix = matchers
                                .iter()
                                .find(|(_, m)| m.matches(&kp.public_key))
                                .map(|(p, _)| p.clone())
                                .unwrap_or_else(|| prefixes_owned[0].clone());
                            *result_clone.lock().unwrap() = Some(MatchResult {
                                keypair: kp,
                                matched_prefix,
                            });
                            return;
                        }
                        base_nonce = base_nonce.wrapping_add(batch_result.keys_checked);
                    }
                    Err(e) => {
                        eprintln!("Metal GPU error: {}", e);
                        return;
                    }
                }
            }
        }));

        Ok(SearchHandle {
            found,
            attempts,
            result,
            start: Instant::now(),
            workers,
        })
    }
```

**Step 3: Verify compilation**

Run: `cargo check --features metal`
Expected: compiles cleanly.

**Step 4: Commit**

```bash
git add src/search.rs
git commit -m "Add start_metal() and start_hybrid() to SearchHandle

Hybrid mode spawns CPU worker threads plus a Metal GPU dispatch thread,
all sharing atomics for coordination. First match from either wins."
```

---

## Phase 6: CLI Integration

### Task 6: Wire up Metal hybrid mode in main.rs

**Files:**
- Modify: `src/main.rs:358-426` (GPU dispatch and main search logic)

**Step 1: Restructure the main dispatch logic**

Replace the current `use_gpu` determination and dispatch block (lines 358-426) with
a version that handles `cuda`, `metal`, and no-GPU builds:

```rust
    // Determine if GPU is available
    #[cfg(feature = "cuda")]
    let use_gpu = cli.gpu;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let use_gpu = cli.gpu || metal::Device::system_default().is_some();
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let use_gpu = false;

    if use_gpu {
        #[cfg(feature = "cuda")]
        {
            let handle = match SearchHandle::start_gpu(&prefixes) {
                Ok(h) => h,
                Err(e) => {
                    if cli.json {
                        eprintln!("Error: {}", e);
                    } else {
                        print_colored_error(&format!("{}", e));
                    }
                    std::process::exit(1);
                }
            };

            let mode_label = format!("GPU{}", prefix_count_label);
            if cli.json {
                let result = handle.finish();
                println!("{}", serde_json::to_string_pretty(&result).unwrap());
            } else {
                let result = match run_tui_loop(handle, &prefixes, expected, &mode_label) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("TUI error: {}, falling back to simple mode", e);
                        let handle = SearchHandle::start_gpu(&prefixes).unwrap();
                        handle.finish()
                    }
                };
                print_colored_result(&result);
            }
        }
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        {
            let handle = match SearchHandle::start_hybrid(&prefixes, num_threads) {
                Ok(h) => h,
                Err(e) => {
                    if cli.json {
                        eprintln!("Error: {}", e);
                    } else {
                        print_colored_error(&format!("{}", e));
                    }
                    std::process::exit(1);
                }
            };

            let mode_label = format!(
                "Metal GPU + {} threads{}",
                num_threads, prefix_count_label
            );
            if cli.json {
                let result = handle.finish();
                println!("{}", serde_json::to_string_pretty(&result).unwrap());
            } else {
                let result = match run_tui_loop(handle, &prefixes, expected, &mode_label) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("TUI error: {}, falling back to simple mode", e);
                        let handle = SearchHandle::start_hybrid(&prefixes, num_threads).unwrap();
                        handle.finish()
                    }
                };
                print_colored_result(&result);
            }
        }
    } else if cli.json {
        // ... existing CPU-only JSON path unchanged ...
    } else {
        // ... existing CPU-only TUI path unchanged ...
    }
```

Note: `#[cfg(all(feature = "metal", not(feature = "cuda")))]` ensures that if both
features are somehow enabled, CUDA takes priority (unlikely but prevents conflicts).

**Step 2: Add Metal device import for auto-detection**

At the top of `main.rs`, add:
```rust
#[cfg(feature = "metal")]
use metal::Device as MetalDevice;
```

And in the `use_gpu` determination, use `MetalDevice::system_default()` instead of
`metal::Device::system_default()`.

**Step 3: Verify full build and test**

Run: `cargo build --release --features metal`
Run: `cargo test --features metal`
Expected: all existing tests pass, binary builds.

**Step 4: Run a real vanity search with hybrid mode**

Run: `./target/release/mc-keygen A`
Expected: TUI shows `Metal GPU + N threads`, finds a key starting with `A` quickly.
Verify the TUI displays keys/sec from both CPU and GPU combined.

**Step 5: Commit**

```bash
git add src/main.rs
git commit -m "Integrate Metal hybrid mode into CLI

On Metal builds, hybrid CPU+GPU mode activates automatically when a
Metal device is detected. TUI shows 'Metal GPU + N threads'."
```

---

## Phase 7: CI

### Task 7: Add macOS build and verify job to GitHub Actions

**Files:**
- Modify: `.github/workflows/rust.yml`

**Step 1: Add macOS job**

Add a second job to the workflow that builds and tests on macOS with Metal:

```yaml
name: Rust

on:
  push:
    branches: [ "master" ]
  pull_request:
    branches: [ "master" ]

env:
  CARGO_TERM_COLOR: always

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build
      run: cargo build --verbose
    - name: Run tests
      run: cargo test --verbose

  build-macos:
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build (CPU only)
      run: cargo build --verbose
    - name: Build (Metal)
      run: cargo build --release --features metal --verbose
    - name: Run tests
      run: cargo test --verbose
    - name: Verify Metal GPU keygen
      run: ./target/release/mc-keygen --verify A
```

Note: `macos-latest` on GitHub Actions uses Apple Silicon (M-series) runners, which
have Metal GPU support. The `--verify` step validates that the MSL kernel produces
identical output to CPU.

**Step 2: Verify locally that the workflow YAML is valid**

Run: `cat .github/workflows/rust.yml` and confirm syntax.

**Step 3: Commit**

```bash
git add .github/workflows/rust.yml
git commit -m "Add macOS CI job for Metal GPU build and verification"
```

---

## Phase 8: Final Validation

### Task 8: End-to-end validation and cleanup

**Files:**
- All modified files (review pass)

**Step 1: Run full test suite**

Run: `cargo test --features metal`
Expected: all tests pass.

**Step 2: Run verification**

Run: `cargo run --release --features metal -- --verify A`
Expected: `64/64 keys matched (Metal GPU vs CPU)\nPASSED`

**Step 3: Run a real 4-character prefix search**

Run: `cargo run --release --features metal -- DEAD`
Expected: TUI shows `Metal GPU + N threads`, finds key starting with `DEAD`.
Note the keys/sec rate for comparison.

**Step 4: Run CPU-only for comparison**

Build without metal feature:
Run: `cargo run --release -- DEAD`
Expected: CPU-only mode, lower keys/sec.

**Step 5: Verify CUDA path is unaffected**

Run: `cargo check --features cuda`
Expected: compiles cleanly, no changes to CUDA code path.

Run: `cargo check` (no features)
Expected: compiles cleanly, CPU-only.

**Step 6: Commit any remaining fixes**

If any issues were found and fixed during validation, commit them.
