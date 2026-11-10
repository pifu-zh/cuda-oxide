/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! gpu_assert! message-form probe: the failing assert's message (with
//! file:line context) renders through the ockl printf transport; the
//! faulting wave exits cleanly. The passing path produces no output.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, gpu_assert, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Thread 3 fails the assert; threads 0-2 pass silently.
    #[kernel]
    pub fn assert_probe(values: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let raw = idx.get();
        let v = values[raw];
        gpu_assert!(v < 100, "value exceeded limit");
        if let Some(o) = out.get_mut(idx) {
            *o = v;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    // Passing case: no output, results written.
    let values = DeviceBuffer::from_host(&stream, &[1u32, 2, 3, 4])?;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, 4)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (4, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: launch shape covers the buffers.
    unsafe { module.assert_probe(&stream, cfg, &values, &mut out) }?;
    stream.synchronize()?;
    let got = out.to_host_vec(&stream)?;
    assert_eq!(got, [1, 2, 3, 4], "passing case must copy through");
    println!("passing case: OK (no assert output)");

    // Failing case: thread 3 (v=300) reports and the wave exits cleanly.
    let bad = DeviceBuffer::from_host(&stream, &[1u32, 2, 3, 300])?;
    let mut out2 = DeviceBuffer::<u32>::zeroed(&stream, 4)?;
    println!("--- launching failing assert (expect gpu_assert output above) ---");
    // SAFETY: launch shape covers the buffers; the failing assert exits the
    // wave cleanly by design.
    unsafe { module.assert_probe(&stream, cfg, &bad, &mut out2) }?;
    stream.synchronize()?;
    println!("failing case: launch/sync OK (clean exit)");
    Ok(())
}
