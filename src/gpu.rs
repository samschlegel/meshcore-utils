use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use rand::rngs::OsRng;
use rand::RngCore;

use crate::search::{advance_scalar, clamp_scalar, PrefixMatcher};
use crate::types::MeshCoreKeypair;

const KERNEL_SRC: &str = include_str!("../cuda/vanity_kernel.cu");
const BLOCK_SIZE: u32 = 256;
const ITERS_PER_THREAD: u64 = 256;

/// Result from a single GPU batch launch.
pub struct CudaBatchResult {
    pub keys_checked: u64,
    pub keypair: Option<MeshCoreKeypair>,
}

/// GPU vanity key searcher using CUDA.
pub struct CudaSearcher {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    count_func: CudaFunction,
    result_buf: CudaSlice<u8>,
    count_buf: CudaSlice<u8>,
    grid_size: u32,
    prefix_data: Vec<u8>,
    prefix_count: u32,
    device_name: String,
    /// Clamped 256-bit starting scalar. Each thread tests scalars
    /// `start_scalar + 8 * (tid * iters + iter)`; after each batch the host
    /// advances this by `8 * total_keys_checked` so successive batches cover
    /// fresh territory. Drawn fresh from OsRng on construction.
    start_scalar: [u8; 32],
}

/// GPU search errors.
#[derive(Debug)]
pub enum CudaError {
    NoCudaDevice,
    CudaDriver(String),
    Compilation(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::NoCudaDevice => write!(f, "no CUDA-capable GPU found"),
            CudaError::CudaDriver(e) => write!(f, "CUDA driver error: {}", e),
            CudaError::Compilation(e) => write!(f, "CUDA kernel compilation error: {}", e),
        }
    }
}

impl std::error::Error for CudaError {}

/// Compile the kernel and return (module, context, stream).
fn compile_kernel() -> Result<(Arc<CudaModule>, Arc<CudaStream>), CudaError> {
    let ctx = CudaContext::new(0).map_err(|e| {
        let msg = format!("{}", e);
        if msg.contains("no device") || msg.contains("not found") {
            CudaError::NoCudaDevice
        } else {
            CudaError::CudaDriver(msg)
        }
    })?;

    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        KERNEL_SRC,
        cudarc::nvrtc::CompileOptions {
            options: vec![
                "--device-as-default-execution-space".into(),
            ],
            ..Default::default()
        },
    )
    .map_err(|e| CudaError::Compilation(format!("{}", e)))?;

    let module = ctx
        .load_module(ptx)
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

    let stream = ctx.default_stream();
    Ok((module, stream))
}

impl CudaSearcher {
    /// Compile the CUDA kernel and prepare for searching.
    /// Packs prefixes into a buffer: [count: u32][nibbles0: u32][bytes0..padded to 4][nibbles1: u32][bytes1..]...]
    pub fn new(prefixes: &[String]) -> Result<Self, CudaError> {
        let (module, stream) = compile_kernel()?;

        let func = module
            .load_function("vanity_search")
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;
        let count_func = module
            .load_function("vanity_count_matches")
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

        let ctx = stream.context();
        let sm_count = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;
        let grid_size = (sm_count as u32) * 2;

        // Pack prefix data: [count: u32] then for each prefix: [nibbles: u32][bytes..padded to 4 alignment]
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
            // Pad to 4-byte alignment
            while prefix_data.len() % 4 != 0 {
                prefix_data.push(0);
            }
        }

        let result_buf = stream
            .alloc_zeros::<u8>(100)
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;
        let count_buf = stream
            .alloc_zeros::<u8>(4)
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

        let device_name = ctx
            .name()
            .map(|n| format!("CUDA GPU ({})", n))
            .unwrap_or_else(|_| "CUDA GPU".to_string());

        let mut start_scalar = [0u8; 32];
        OsRng.fill_bytes(&mut start_scalar);
        clamp_scalar(&mut start_scalar);

        Ok(CudaSearcher {
            stream,
            func,
            count_func,
            result_buf,
            count_buf,
            grid_size,
            prefix_data,
            prefix_count,
            device_name,
            start_scalar,
        })
    }

    /// Launch one batch of the count-matches kernel for benchmarking. Returns
    /// `(keys_checked, matches_found)`. Unlike `search_batch`, this kernel
    /// processes the FULL iters_per_thread on every thread (no early exit on
    /// match) and atomically tallies every prefix match it sees, so the
    /// returned counts are exact and the host-reported throughput can be
    /// cross-checked against observed match frequency.
    pub fn count_batch(&mut self) -> Result<(u64, u32), CudaError> {
        let total_threads = (self.grid_size * BLOCK_SIZE) as u64;
        let keys_checked = total_threads * ITERS_PER_THREAD;

        self.stream
            .memset_zeros(&mut self.count_buf)
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

        let prefix_dev = self
            .stream
            .clone_htod(&self.prefix_data)
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;
        let start_scalar_dev = self
            .stream
            .clone_htod(&self.start_scalar.to_vec())
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

        let cfg = LaunchConfig {
            grid_dim: (self.grid_size, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.count_func)
                .arg(&mut self.count_buf)
                .arg(&prefix_dev)
                .arg(&self.prefix_count)
                .arg(&start_scalar_dev)
                .arg(&ITERS_PER_THREAD)
                .launch(cfg)
        }
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

        self.stream
            .synchronize()
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

        let counter_host: Vec<u8> = self
            .stream
            .clone_dtoh(&self.count_buf)
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;
        let matches = u32::from_le_bytes([
            counter_host[0], counter_host[1], counter_host[2], counter_host[3],
        ]);

        advance_scalar(&mut self.start_scalar, 8u64.wrapping_mul(keys_checked));

        Ok((keys_checked, matches))
    }

    /// Launch one batch of GPU kernel and check for results.
    /// Uses the searcher's `start_scalar` (drawn from OsRng) and advances it
    /// by `8 * keys_checked` after the launch so subsequent batches cover
    /// fresh territory. The legacy `_base_nonce` arg is kept for trait
    /// compatibility and ignored on this backend.
    pub fn search_batch(&mut self, _base_nonce: u64) -> Result<CudaBatchResult, CudaError> {
        let total_threads = (self.grid_size * BLOCK_SIZE) as u64;
        let keys_checked = total_threads * ITERS_PER_THREAD;

        self.stream
            .memset_zeros(&mut self.result_buf)
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

        let prefix_dev = self
            .stream
            .clone_htod(&self.prefix_data)
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;
        let start_scalar_dev = self
            .stream
            .clone_htod(&self.start_scalar.to_vec())
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

        let cfg = LaunchConfig {
            grid_dim: (self.grid_size, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(&mut self.result_buf)
                .arg(&prefix_dev)
                .arg(&self.prefix_count)
                .arg(&start_scalar_dev)
                .arg(&ITERS_PER_THREAD)
                .launch(cfg)
        }
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

        self.stream
            .synchronize()
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

        let result_host: Vec<u8> = self
            .stream
            .clone_dtoh(&self.result_buf)
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

        let found_flag = u32::from_le_bytes([
            result_host[0],
            result_host[1],
            result_host[2],
            result_host[3],
        ]);

        let keypair = if found_flag != 0 {
            let mut public_key = [0u8; 32];
            public_key.copy_from_slice(&result_host[4..36]);
            let mut scalar = [0u8; 32];
            scalar.copy_from_slice(&result_host[36..68]);

            // Defensive: verify scalar*B == pubkey via curve25519-dalek. Cheap
            // (one scalarmult) and only runs on a match, but it catches kernel
            // bugs (off-by-one in match index, batched-inversion mistakes, etc.)
            // before a user trusts the resulting private key.
            {
                use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
                use curve25519_dalek::scalar::Scalar;
                let s = Scalar::from_bytes_mod_order(scalar);
                let cpu_pub = (&s * ED25519_BASEPOINT_TABLE).compress().to_bytes();
                if cpu_pub != public_key {
                    return Err(CudaError::CudaDriver(format!(
                        "GPU match validation failed: scalar·B != pubkey\n  scalar:  {}\n  GPU pub: {}\n  CPU pub: {}",
                        hex::encode_upper(scalar),
                        hex::encode_upper(public_key),
                        hex::encode_upper(cpu_pub),
                    )));
                }
            }

            // Prefix half of the expanded private key is just fresh random
            // bytes -- there's no derivation requirement on it. The user
            // gets one match per search, so a single OsRng draw here is fine.
            let mut prefix = [0u8; 32];
            OsRng.fill_bytes(&mut prefix);
            let mut private_key = [0u8; 64];
            private_key[..32].copy_from_slice(&scalar);
            private_key[32..].copy_from_slice(&prefix);
            Some(MeshCoreKeypair {
                public_key,
                private_key,
            })
        } else {
            None
        };

        // Advance start_scalar by 8 * keys_checked so the next batch covers
        // fresh scalars. (Only meaningful if no match was found; harmless
        // otherwise.)
        advance_scalar(&mut self.start_scalar, 8u64.wrapping_mul(keys_checked));

        Ok(CudaBatchResult {
            keys_checked,
            keypair,
        })
    }
}

impl crate::search::GpuSearcher for CudaSearcher {
    fn search_batch(
        &mut self,
        base_nonce: u64,
    ) -> Result<crate::search::GpuBatchResult, Box<dyn std::error::Error + Send + Sync>> {
        // The trait still calls this `base_nonce`; the CUDA backend ignores it
        // and draws/advances its own `start_scalar` from OsRng instead.
        let result = self.search_batch(base_nonce)?;
        Ok(crate::search::GpuBatchResult {
            keys_checked: result.keys_checked,
            keypair: result.keypair,
        })
    }

    fn count_batch(
        &mut self,
    ) -> Result<(u64, u32), Box<dyn std::error::Error + Send + Sync>> {
        Ok(CudaSearcher::count_batch(self)?)
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }
}

/// Run GPU vs CPU verification: launch the chain verifier with a fixed start
/// scalar, then for each step `i`, compute `(start + 8*i)·B` directly via
/// curve25519-dalek and compare. Validates the initial scalarmult AND that the
/// `+8B` chain stays in lockstep across many iterations.
pub fn verify_gpu_keygen() -> Result<(), CudaError> {
    use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
    use curve25519_dalek::scalar::Scalar;

    let (module, stream) = compile_kernel()?;

    let verify_func = module
        .load_function("verify_keygen")
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

    // Fixed (clamped) start scalar so the test is deterministic.
    let mut start_scalar = [0u8; 32];
    start_scalar[0] = 0x10;
    start_scalar[1] = 0x32;
    start_scalar[2] = 0x54;
    start_scalar[3] = 0x76;
    start_scalar[31] = 0x12;
    clamp_scalar(&mut start_scalar);

    let count: u32 = 64;

    let start_scalar_dev = stream
        .clone_htod(&start_scalar.to_vec())
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;
    let mut pubkeys_dev = stream
        .alloc_zeros::<u8>(count as usize * 32)
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;
    let mut scalars_dev = stream
        .alloc_zeros::<u8>(count as usize * 32)
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&verify_func)
            .arg(&start_scalar_dev)
            .arg(&mut pubkeys_dev)
            .arg(&mut scalars_dev)
            .arg(&count)
            .launch(cfg)
    }
    .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

    stream
        .synchronize()
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

    let gpu_pubkeys: Vec<u8> = stream
        .clone_dtoh(&pubkeys_dev)
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;
    let gpu_scalars: Vec<u8> = stream
        .clone_dtoh(&scalars_dev)
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

    let mut pass = 0;
    let mut fail = 0;
    let mut expected_scalar = start_scalar;
    for i in 0..count as usize {
        // Direct host-side scalarmult of expected_scalar.
        let scalar = Scalar::from_bytes_mod_order(expected_scalar);
        let cpu_pub = (&scalar * ED25519_BASEPOINT_TABLE).compress().to_bytes();

        let gpu_pub = &gpu_pubkeys[i * 32..(i + 1) * 32];
        let gpu_scalar = &gpu_scalars[i * 32..(i + 1) * 32];

        let pub_match = gpu_pub == cpu_pub;
        let scalar_match = gpu_scalar == expected_scalar;

        if pub_match && scalar_match {
            pass += 1;
        } else {
            fail += 1;
            eprintln!("MISMATCH idx #{}", i);
            eprintln!("  CPU pubkey:  {}", hex::encode_upper(cpu_pub));
            eprintln!("  GPU pubkey:  {}", hex::encode_upper(gpu_pub));
            eprintln!("  expected scalar: {}", hex::encode_upper(expected_scalar));
            eprintln!("  GPU scalar:      {}", hex::encode_upper(gpu_scalar));
            if fail >= 5 {
                eprintln!("  (stopping after 5 mismatches)");
                break;
            }
        }

        advance_scalar(&mut expected_scalar, 8);
    }

    eprintln!("{}/{} chain steps matched (GPU vs CPU)", pass, count);
    if fail > 0 {
        Err(CudaError::CudaDriver(format!(
            "{} of {} chain steps mismatched between GPU and CPU",
            fail, count
        )))
    } else {
        Ok(())
    }
}
