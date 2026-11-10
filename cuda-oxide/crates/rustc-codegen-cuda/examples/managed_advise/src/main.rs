/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Managed-memory advise/prefetch probe: cuMemAdvise_v2 + cuMemPrefetchAsync_v2
//! through the HIP shim (hipMemAdvise/hipMemPrefetchAsync). Managed semantics
//! were already verified in runtime_sync; this closes the advise/prefetch API
//! surface. Advice that the platform cannot honor comes back as a driver
//! error and is reported as such (CUDA portability guidance: treat Err as
//! "unsupported here"), not silently swallowed.

use cuda_core::simt::LaunchConfig;
use cuda_core::sys;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_core::{malloc_managed, mem_advise, mem_prefetch_async};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: usize = 1024;

#[cuda_module]
mod kernels {
    use super::*;

    /// Bumps every managed element by one (validates device-side access).
    #[kernel]
    pub fn bump(mut managed: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(o) = managed.get_mut(idx) {
            *o = o.wrapping_add(1);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== managed-memory advise/prefetch probe ===");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Managed allocation, attach-host spelling (portable default).
    let ptr = unsafe { malloc_managed(N * 4, sys::CUmemAttach_flags_enum_CU_MEM_ATTACH_HOST) }
        .map_err(|e| format!("malloc_managed failed: {e}"))?;
    println!("managed alloc: {ptr:#x} ({} B)", N * 4);

    // Write host-side, launch a kernel against the managed pointer.
    let initial: Vec<u32> = (0..N as u32).collect();
    unsafe { std::ptr::copy_nonoverlapping(initial.as_ptr(), ptr as *mut u32, N) };

    let module = kernels::load(&ctx)?;
    let mut managed = unsafe { DeviceBuffer::<u32>::from_raw_parts(ptr, N, ctx.clone()) };
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: launch shape covers the N-element allocation.
    unsafe { module.bump((stream).as_ref(), cfg, &mut managed) }?;
    stream.synchronize()?;
    let after_kernel = managed.to_host_vec(&stream)?;
    assert_eq!(after_kernel[0], 1, "device bump must be visible");
    println!("kernel roundtrip through managed memory: OK");

    // Advise: read-mostly on device 0. On HIP-backed stacks this can be
    // unsupported — report, don't swallow.
    let mut loc: sys::CUmemLocation = unsafe { std::mem::zeroed() };
    loc.type_ = sys::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE;
    unsafe { sys::set_mem_location_id(&mut loc, 0) };
    let read_mostly = sys::CUmem_advise_enum_CU_MEM_ADVISE_SET_READ_MOSTLY;
    match unsafe { mem_advise(ptr, N * 4, read_mostly, loc) } {
        Ok(()) => println!("mem_advise (SET_READ_MOSTLY, dev 0): OK"),
        Err(e) => println!(
            "mem_advise: Err({e}) — advice unsupported on this stack (treated per CUDA portability guidance)"
        ),
    }
    // The canonical unsupported-on-many-stacks one, same policy.
    let accessed_by = sys::CUmem_advise_enum_CU_MEM_ADVISE_SET_ACCESSED_BY;
    match unsafe { mem_advise(ptr, N * 4, accessed_by, loc) } {
        Ok(()) => println!("mem_advise (SET_ACCESSED_BY, dev 0): OK"),
        Err(e) => println!(
            "mem_advise: Err({e}) — advice unsupported on this stack (treated per CUDA portability guidance)"
        ),
    }

    // Prefetch to device 0 (where the kernel will run again).
    match unsafe { mem_prefetch_async(ptr, N * 4, loc, stream.cu_stream()) } {
        Ok(()) => {
            stream.synchronize()?;
            println!("mem_prefetch_async (dev 0): OK");
        }
        Err(e) => println!("mem_prefetch_async: Err({e}) — unsupported on this stack"),
    }

    // Second bump after advise+prefetch: correctness unchanged.
    // SAFETY: launch shape covers the N-element allocation.
    unsafe { module.bump((stream).as_ref(), cfg, &mut managed) }?;
    stream.synchronize()?;
    let final_vec = managed.to_host_vec(&stream)?;
    assert_eq!(
        final_vec[0], 2,
        "second bump must be visible after advise/prefetch"
    );
    println!("post-advise roundtrip: OK");
    println!("=== PASS ===");
    Ok(())
}
