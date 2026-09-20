//! [PORT gfx1030 Stage3-1] Minimal shared-memory + block-barrier smoke.
//!
//! Two kernels, no mbarrier machinery:
//! 1. `shared_neighbor_test`: every thread writes its tid into a
//!    `SharedArray`, `sync_threads` (lowers to `llvm.amdgcn.s.barrier` on the
//!    amdgcn path), then reads the neighbor's slot. Verifies LDS globals and
//!    the block barrier.
//! 2. `block_dim_test`: reads `blockDim.x` (the `ntid_x` kernarg appended by
//!    the amdgcn prep pass and supplied by the Stage-2 host append) and
//!    writes `tid < block_size` flags, cross-checked after a barrier.
//!
//! Build and run with:
//!   CUDA_OXIDE_TARGET=gfx1030 cargo oxide run barrier_smoke

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

const N: usize = 1024;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn shared_neighbor_test(mut out: DisjointSlice<u32>) {
        static mut DATA: SharedArray<u32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let block_size = thread::blockDim_x();
        let gid = thread::index_1d();

        // Scatter: each thread publishes its tid in shared memory.
        unsafe {
            DATA[tid as usize] = tid;
        }
        thread::sync_threads();

        // Gather the neighbor's value with wraparound inside the block.
        let neighbor_idx = ((tid + 1) % block_size) as usize;
        let neighbor_val = unsafe { DATA[neighbor_idx] };

        // Expected: [1, 2, ..., 255, 0] per 256-thread block.
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = neighbor_val;
        }
    }

    #[kernel]
    pub fn block_dim_test(mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let block_size = thread::blockDim_x();
        let gid = thread::index_1d();

        thread::sync_threads();

        // blockDim.x must arrive as the appended ntid_x kernarg (256).
        let value = if tid < block_size { block_size } else { 0 };
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = value;
        }
    }
}

fn main() {
    println!("=== barrier_smoke (gfx1030 shared memory + sync_threads) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    let mut expected = vec![0u32; N];
    for block in 0..N / 256 {
        for tid in 0..256usize {
            expected[block * 256 + tid] = ((tid + 1) % 256) as u32;
        }
    }

    // Kernel 1: shared-memory neighbor exchange across a barrier.
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    unsafe {
        module.shared_neighbor_test(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev)
    }
    .expect("shared_neighbor_test launch failed");
    let got = out_dev.to_host_vec(&stream).unwrap();
    let bad = got.iter().zip(&expected).filter(|(a, b)| a != b).count();
    println!("shared_neighbor_test: {}", if bad == 0 { "PASS" } else { "FAIL" });
    if bad != 0 {
        for i in 0..N {
            if got[i] != expected[i] {
                println!("  first mismatch @[{i}]: got {} want {}", got[i], expected[i]);
                break;
            }
        }
    }

    // Kernel 2: blockDim.x via the appended ntid_x kernarg.
    let mut out_dev2 = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    unsafe {
        module.block_dim_test(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev2)
    }
    .expect("block_dim_test launch failed");
    let got2 = out_dev2.to_host_vec(&stream).unwrap();
    let bad2 = got2.iter().filter(|&&v| v != 256).count();
    println!("block_dim_test:       {}", if bad2 == 0 { "PASS" } else { "FAIL" });

    if bad == 0 && bad2 == 0 {
        println!("\n✓ SUCCESS: barrier_smoke passed!");
    } else {
        println!("\n✗ FAILED");
        std::process::exit(1);
    }
}
