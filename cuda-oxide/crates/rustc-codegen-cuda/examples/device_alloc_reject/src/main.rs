//! NEGATIVE probe (expected to FAIL to build): heap allocation on device.
//! Evidence for the matrix row: no GlobalAlloc on device -> loud failure.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{cuda_module, kernel, thread};
// (thread used inside mod kernels via use super::* path below)

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn heap_vec(out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            // Heap allocation on device: expected to fail loudly.
            let mut v: Vec<u32> = Vec::new();
            v.push(1);
            v.push(2);
            out[0] = v[0] + v[1];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let s0 = ctx.default_stream();
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    let module = kernels::load(&ctx)?;
    let mut out = DeviceBuffer::zeroed(&s0, 1)?;
    // SAFETY: launch shape matches.
    unsafe { module.heap_vec((s0).as_ref(), cfg, &mut out) }?;
    s0.synchronize()?;
    Ok(())
}
