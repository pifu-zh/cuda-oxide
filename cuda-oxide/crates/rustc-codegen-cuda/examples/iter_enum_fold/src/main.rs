use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{cuda_module, kernel, thread};
use cuda_host::cuda_module as _;

#[cuda_module]
mod kernels {
    use super::*;

    // The previously-rejected chain: enumerate().fold() -- Enumerate::fold
    // builds nested closure struct constants (the exact shape the MIR
    // importer rejected before the Closure-constant fallback).
    #[kernel]
    pub fn enum_fold(data: &[u32], out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            out[0] = data
                .iter()
                .enumerate()
                .fold(0u32, |acc, (i, &v)| acc + i as u32 + v);
        }
    }

    // And the sum() variant through the same adapter chain.
    #[kernel]
    pub fn enum_sum(data: &[u32], out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            out[0] = data.iter().enumerate().map(|(i, &v)| i as u32 + v).sum();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let s0 = ctx.default_stream();
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    let module = kernels::load(&ctx)?;
    let data: Vec<u32> = vec![7, 3, 12];
    let dev = DeviceBuffer::from_host(&s0, &data)?;
    let mut out = DeviceBuffer::zeroed(&s0, 1)?;
    let mut host_out = vec![0u32; 1];
    // SAFETY: launch shape matches; buffers cover accesses.
    unsafe { module.enum_fold((s0).as_ref(), cfg, &dev, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    let want: u32 = data.iter().enumerate().fold(0, |a, (i, &v)| a + i as u32 + v);
    println!("fold: device={} host={}", host_out[0], want);
    assert_eq!(host_out[0], want);
    // SAFETY: as above.
    unsafe { module.enum_sum((s0).as_ref(), cfg, &dev, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    let want2: u32 = data.iter().enumerate().map(|(i, &v)| i as u32 + v).sum();
    println!("sum: device={} host={}", host_out[0], want2);
    assert_eq!(host_out[0], want2);
    Ok(())
}
