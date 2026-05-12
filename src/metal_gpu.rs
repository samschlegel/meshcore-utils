use std::ffi::c_void;
use std::fmt;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize,
};
use rand::rngs::OsRng;
use rand::RngCore;

use crate::search::{advance_scalar, clamp_scalar, PrefixMatcher};
use crate::types::MeshCoreKeypair;

const KERNEL_SRC: &str = include_str!("../metal/vanity_kernel.metal");
const BLOCK_SIZE: u64 = 256;
const ITERS_PER_THREAD: u64 = 256;

pub struct MetalSearcher {
    queue: CommandQueue,
    search_pipeline: ComputePipelineState,
    count_pipeline: ComputePipelineState,
    result_buf: Buffer,
    count_buf: Buffer,
    prefix_buf: Buffer,
    start_scalar_buf: Buffer,
    prefix_count: u32,
    grid_size: u64,
    threadgroup_size: u64,
    device_name: String,
    /// Clamped 256-bit starting scalar. Each thread tests scalars
    /// `start_scalar + 8 * (tid * iters + iter)`; after each batch the host
    /// advances this by `8 * total_keys_checked`. Drawn fresh from OsRng on
    /// construction.
    start_scalar: [u8; 32],
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

fn make_pipeline(
    device: &Device,
    library: &metal::Library,
    name: &str,
) -> Result<ComputePipelineState, MetalError> {
    let func = library
        .get_function(name, None)
        .map_err(|e| MetalError::Runtime(e.to_string()))?;
    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| MetalError::Runtime(e.to_string()))
}

impl MetalSearcher {
    pub fn new(prefixes: &[String]) -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoMetalDevice)?;
        let device_name = format!("Metal GPU ({})", device.name());
        let library = compile_kernel(&device)?;

        let search_pipeline = make_pipeline(&device, &library, "vanity_search")?;
        let count_pipeline = make_pipeline(&device, &library, "vanity_count_matches")?;

        let queue = device.new_command_queue();

        let max_threads = search_pipeline.max_total_threads_per_threadgroup();
        let threadgroup_size = max_threads.min(BLOCK_SIZE);
        let grid_size = threadgroup_size * 64;

        // Pack prefix data -- same format as CUDA path.
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

        // Shared-mode buffers (zero-copy on Apple Silicon).
        let result_buf = device.new_buffer(100, MTLResourceOptions::StorageModeShared);
        let count_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let prefix_buf = device.new_buffer_with_data(
            prefix_data.as_ptr() as *const c_void,
            prefix_data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let start_scalar_buf = device.new_buffer(32, MTLResourceOptions::StorageModeShared);

        let mut start_scalar = [0u8; 32];
        OsRng.fill_bytes(&mut start_scalar);
        clamp_scalar(&mut start_scalar);

        Ok(MetalSearcher {
            queue,
            search_pipeline,
            count_pipeline,
            result_buf,
            count_buf,
            prefix_buf,
            start_scalar_buf,
            prefix_count,
            grid_size,
            threadgroup_size,
            device_name,
            start_scalar,
        })
    }

    /// Copy the current start_scalar into its device-visible buffer.
    fn upload_start_scalar(&self) {
        unsafe {
            let ptr = self.start_scalar_buf.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(self.start_scalar.as_ptr(), ptr, 32);
        }
    }

    /// Launch one batch of the search kernel. Legacy `_base_nonce` arg is
    /// kept for trait compatibility and ignored -- the searcher tracks its
    /// own start_scalar state.
    pub fn search_batch(
        &mut self,
        _base_nonce: u64,
    ) -> Result<crate::search::GpuBatchResult, MetalError> {
        let keys_checked = self.grid_size * ITERS_PER_THREAD;

        // Zero the result buffer.
        unsafe {
            std::ptr::write_bytes(self.result_buf.contents() as *mut u8, 0, 100);
        }
        self.upload_start_scalar();

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.search_pipeline);
        encoder.set_buffer(0, Some(&self.result_buf), 0);
        encoder.set_buffer(1, Some(&self.prefix_buf), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &self.prefix_count as *const u32 as *const c_void,
        );
        encoder.set_buffer(3, Some(&self.start_scalar_buf), 0);
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

        let result_ptr = self.result_buf.contents() as *const u8;
        let result_host = unsafe { std::slice::from_raw_parts(result_ptr, 100) };

        let found_flag = u32::from_le_bytes([
            result_host[0], result_host[1], result_host[2], result_host[3],
        ]);

        let keypair = if found_flag != 0 {
            let mut public_key = [0u8; 32];
            public_key.copy_from_slice(&result_host[4..36]);
            let mut scalar = [0u8; 32];
            scalar.copy_from_slice(&result_host[36..68]);

            // Defensive: verify scalar*B == pubkey via curve25519-dalek before
            // handing the user a private key derived from GPU output.
            {
                use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
                use curve25519_dalek::scalar::Scalar;
                let s = Scalar::from_bytes_mod_order(scalar);
                let cpu_pub = (&s * ED25519_BASEPOINT_TABLE).compress().to_bytes();
                if cpu_pub != public_key {
                    return Err(MetalError::Runtime(format!(
                        "GPU match validation failed: scalar·B != pubkey\n  scalar:  {}\n  GPU pub: {}\n  CPU pub: {}",
                        hex::encode_upper(scalar),
                        hex::encode_upper(public_key),
                        hex::encode_upper(cpu_pub),
                    )));
                }
            }

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

        advance_scalar(&mut self.start_scalar, 8u64.wrapping_mul(keys_checked));

        Ok(crate::search::GpuBatchResult {
            keys_checked,
            keypair,
        })
    }

    /// Launch one batch of the count-matches kernel for benchmarking. Returns
    /// `(keys_checked, matches_found)`. Like the CUDA variant, this never
    /// exits early and atomically tallies every match.
    pub fn count_batch(&mut self) -> Result<(u64, u32), MetalError> {
        let keys_checked = self.grid_size * ITERS_PER_THREAD;

        unsafe {
            std::ptr::write_bytes(self.count_buf.contents() as *mut u8, 0, 4);
        }
        self.upload_start_scalar();

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.count_pipeline);
        encoder.set_buffer(0, Some(&self.count_buf), 0);
        encoder.set_buffer(1, Some(&self.prefix_buf), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &self.prefix_count as *const u32 as *const c_void,
        );
        encoder.set_buffer(3, Some(&self.start_scalar_buf), 0);
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

        let count_ptr = self.count_buf.contents() as *const u8;
        let count_host = unsafe { std::slice::from_raw_parts(count_ptr, 4) };
        let matches = u32::from_le_bytes([count_host[0], count_host[1], count_host[2], count_host[3]]);

        advance_scalar(&mut self.start_scalar, 8u64.wrapping_mul(keys_checked));

        Ok((keys_checked, matches))
    }
}

impl crate::search::GpuSearcher for MetalSearcher {
    fn search_batch(
        &mut self,
        base_nonce: u64,
    ) -> Result<crate::search::GpuBatchResult, Box<dyn std::error::Error + Send + Sync>> {
        Ok(MetalSearcher::search_batch(self, base_nonce)?)
    }

    fn count_batch(
        &mut self,
    ) -> Result<(u64, u32), Box<dyn std::error::Error + Send + Sync>> {
        Ok(MetalSearcher::count_batch(self)?)
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }
}

/// Run Metal GPU vs CPU verification: launch the chain verifier with a fixed
/// start scalar, then for each step `i`, compute `(start + 8*i)·B` directly
/// via curve25519-dalek and compare. Validates the initial scalarmult AND
/// that the `+8B` chain stays in lockstep across many iterations.
pub fn verify_gpu_keygen() -> Result<(), MetalError> {
    use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
    use curve25519_dalek::scalar::Scalar;

    let device = Device::system_default().ok_or(MetalError::NoMetalDevice)?;
    let library = compile_kernel(&device)?;
    let pipeline = make_pipeline(&device, &library, "verify_keygen")?;
    let queue = device.new_command_queue();

    // Fixed (clamped) start scalar so the test is deterministic.
    let mut start_scalar = [0u8; 32];
    start_scalar[0] = 0x10;
    start_scalar[1] = 0x32;
    start_scalar[2] = 0x54;
    start_scalar[3] = 0x76;
    start_scalar[31] = 0x12;
    clamp_scalar(&mut start_scalar);

    let count: u32 = 64;

    let start_scalar_buf = device.new_buffer_with_data(
        start_scalar.as_ptr() as *const c_void,
        32,
        MTLResourceOptions::StorageModeShared,
    );
    let pubkeys_buf =
        device.new_buffer((count as u64) * 32, MTLResourceOptions::StorageModeShared);
    let scalars_buf =
        device.new_buffer((count as u64) * 32, MTLResourceOptions::StorageModeShared);

    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&start_scalar_buf), 0);
    encoder.set_buffer(1, Some(&pubkeys_buf), 0);
    encoder.set_buffer(2, Some(&scalars_buf), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &count as *const u32 as *const c_void,
    );

    // Chain verifier is single-threaded.
    let grid = MTLSize::new(1, 1, 1);
    let threadgroup = MTLSize::new(1, 1, 1);
    encoder.dispatch_threads(grid, threadgroup);
    encoder.end_encoding();

    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let gpu_pubkeys = unsafe {
        std::slice::from_raw_parts(pubkeys_buf.contents() as *const u8, count as usize * 32)
    };
    let gpu_scalars = unsafe {
        std::slice::from_raw_parts(scalars_buf.contents() as *const u8, count as usize * 32)
    };

    let mut pass = 0;
    let mut fail = 0;
    let mut expected_scalar = start_scalar;
    for i in 0..count as usize {
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

    eprintln!("{}/{} chain steps matched (Metal GPU vs CPU)", pass, count);
    if fail > 0 {
        Err(MetalError::Runtime(format!(
            "{} of {} chain steps mismatched between Metal GPU and CPU",
            fail, count
        )))
    } else {
        Ok(())
    }
}
