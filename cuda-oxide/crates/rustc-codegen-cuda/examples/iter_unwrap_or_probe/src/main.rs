use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{cuda_module, kernel, thread};
use cuda_host::cuda_module as _;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn unwrap_or_probe(data: &[u32], out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            let min: u32 = data.iter().min().copied().unwrap_or(0);
            let max: u32 = data.iter().max().copied().unwrap_or(0);
            let pos: u32 = data.iter().position(|&x| x == 99).unwrap_or(7) as u32;
            let via_map_or: u32 = data.first().map_or(9, |&v| v.wrapping_add(1));
            let via_else: u32 = data.first().copied().unwrap_or_else(|| 42);
            out[0] = min
                .wrapping_mul(1_000_000)
                .wrapping_add(max.wrapping_mul(1_000))
                .wrapping_add(pos)
                .wrapping_add(via_map_or)
                .wrapping_add(via_else);
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
    unsafe { module.unwrap_or_probe((s0).as_ref(), cfg, &dev, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    let min: u32 = data.iter().min().copied().unwrap_or(0);
    let max: u32 = data.iter().max().copied().unwrap_or(0);
    let pos: u32 = data.iter().position(|&x| x == 99).unwrap_or(7) as u32;
    let via_map_or: u32 = data.first().map_or(9, |&v| v.wrapping_add(1));
    let via_else: u32 = data.first().copied().unwrap_or_else(|| 42);
    let want: u32 = min
        .wrapping_mul(1_000_000)
        .wrapping_add(max.wrapping_mul(1_000))
        .wrapping_add(pos)
        .wrapping_add(via_map_or)
        .wrapping_add(via_else);
    println!("device={} host={}", host_out[0], want);
    assert_eq!(host_out[0], want);
    Ok(())
}
