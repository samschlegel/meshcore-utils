use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

use crate::search::PrefixMatcher;
use crate::types::MeshCoreKeypair;

const KERNEL_SRC: &str = include_str!("../cuda/vanity_kernel.cu");
const BLOCK_SIZE: u32 = 256;
const ITERS_PER_THREAD: u64 = 64;

/// Result from a single GPU batch launch.
pub struct CudaBatchResult {
    pub keys_checked: u64,
    pub keypair: Option<MeshCoreKeypair>,
}

/// GPU vanity key searcher using CUDA.
pub struct CudaSearcher {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    result_buf: CudaSlice<u8>,
    grid_size: u32,
    prefix_data: Vec<u8>,
    prefix_count: u32,
    device_name: String,
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

        let device_name = ctx
            .name()
            .map(|n| format!("CUDA GPU ({})", n))
            .unwrap_or_else(|_| "CUDA GPU".to_string());

        Ok(CudaSearcher {
            stream,
            func,
            result_buf,
            grid_size,
            prefix_data,
            prefix_count,
            device_name,
        })
    }

    /// Launch one batch of GPU kernel and check for results.
    pub fn search_batch(&mut self, base_nonce: u64) -> Result<CudaBatchResult, CudaError> {
        let total_threads = (self.grid_size * BLOCK_SIZE) as u64;
        let keys_checked = total_threads * ITERS_PER_THREAD;

        self.stream
            .memset_zeros(&mut self.result_buf)
            .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

        let prefix_dev = self
            .stream
            .clone_htod(&self.prefix_data)
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
                .arg(&base_nonce)
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
        let result = self.search_batch(base_nonce)?;
        Ok(crate::search::GpuBatchResult {
            keys_checked: result.keys_checked,
            keypair: result.keypair,
        })
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }
}

/// Run GPU vs CPU verification on a set of test seeds.
/// Returns Ok(()) if all keys match, Err with details on mismatch.
pub fn verify_gpu_keygen() -> Result<(), CudaError> {
    use crate::keygen::generate_keypair;

    let (module, stream) = compile_kernel()?;

    let verify_func = module
        .load_function("verify_keygen")
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

    // Generate test seeds
    let count: u32 = 64;
    let mut seeds_flat = vec![0u8; count as usize * 32];
    for i in 0..count as usize {
        // Each seed: first 4 bytes = index, rest varies
        seeds_flat[i * 32] = i as u8;
        seeds_flat[i * 32 + 1] = (i >> 8) as u8;
        seeds_flat[i * 32 + 4] = 0xDE;
        seeds_flat[i * 32 + 5] = 0xAD;
        seeds_flat[i * 32 + 31] = 0x42;
    }

    // Upload seeds
    let seeds_dev = stream
        .clone_htod(&seeds_flat)
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;
    let mut pubkeys_dev = stream
        .alloc_zeros::<u8>(count as usize * 32)
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;
    let mut privkeys_dev = stream
        .alloc_zeros::<u8>(count as usize * 64)
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (count, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&verify_func)
            .arg(&seeds_dev)
            .arg(&mut pubkeys_dev)
            .arg(&mut privkeys_dev)
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
    let gpu_privkeys: Vec<u8> = stream
        .clone_dtoh(&privkeys_dev)
        .map_err(|e| CudaError::CudaDriver(format!("{}", e)))?;

    // Cross-check against CPU
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

    eprintln!("{}/{} keys matched (GPU vs CPU)", pass, count);
    if fail > 0 {
        Err(CudaError::CudaDriver(format!(
            "{} of {} keys mismatched between GPU and CPU",
            fail, count
        )))
    } else {
        Ok(())
    }
}
