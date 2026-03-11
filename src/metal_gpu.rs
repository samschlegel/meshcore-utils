use std::ffi::c_void;
use std::fmt;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize,
};
use rand::rngs::OsRng;
use rand::RngCore;

use crate::search::PrefixMatcher;
use crate::types::MeshCoreKeypair;

const KERNEL_SRC: &str = include_str!("../metal/vanity_kernel.metal");
const BLOCK_SIZE: u64 = 256;
const ITERS_PER_THREAD: u64 = 64;

pub struct MetalSearcher {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    result_buf: Buffer,
    prefix_buf: Buffer,
    salt_buf: Buffer,
    prefix_count: u32,
    grid_size: u64,
    threadgroup_size: u64,
    device_name: String,
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
        let device_name = format!("Metal GPU ({})", device.name());
        let library = compile_kernel(&device)?;

        let func = library
            .get_function("vanity_search", None)
            .map_err(|e| MetalError::Runtime(e.to_string()))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| MetalError::Runtime(e.to_string()))?;

        let queue = device.new_command_queue();

        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let threadgroup_size = max_threads.min(BLOCK_SIZE);
        let grid_size = threadgroup_size * 64;

        // Pack prefix data -- same format as CUDA path
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
        let salt_buf = device.new_buffer(24, MTLResourceOptions::StorageModeShared);

        Ok(MetalSearcher {
            queue,
            pipeline,
            result_buf,
            prefix_buf,
            salt_buf,
            prefix_count,
            grid_size,
            threadgroup_size,
            device_name,
        })
    }

    pub fn search_batch(&self, base_nonce: u64) -> Result<crate::search::GpuBatchResult, MetalError> {
        let keys_checked = self.grid_size * ITERS_PER_THREAD;

        // Zero the result buffer
        unsafe {
            std::ptr::write_bytes(self.result_buf.contents() as *mut u8, 0, 100);
        }

        // Generate fresh 24-byte random salt for seed entropy (bytes 8-31)
        unsafe {
            let salt_ptr = self.salt_buf.contents() as *mut u8;
            let salt_slice = std::slice::from_raw_parts_mut(salt_ptr, 24);
            OsRng.fill_bytes(salt_slice);
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
        encoder.set_buffer(5, Some(&self.salt_buf), 0);

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

        Ok(crate::search::GpuBatchResult {
            keys_checked,
            keypair,
        })
    }
}

impl crate::search::GpuSearcher for MetalSearcher {
    fn search_batch(
        &mut self,
        base_nonce: u64,
    ) -> Result<crate::search::GpuBatchResult, Box<dyn std::error::Error + Send + Sync>> {
        // MetalSearcher::search_batch takes &self, so this just delegates
        Ok(MetalSearcher::search_batch(self, base_nonce)?)
    }

    fn device_name(&self) -> &str {
        &self.device_name
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

    let gpu_pubkeys = unsafe {
        std::slice::from_raw_parts(pubkeys_buf.contents() as *const u8, count as usize * 32)
    };
    let gpu_privkeys = unsafe {
        std::slice::from_raw_parts(privkeys_buf.contents() as *const u8, count as usize * 64)
    };

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
