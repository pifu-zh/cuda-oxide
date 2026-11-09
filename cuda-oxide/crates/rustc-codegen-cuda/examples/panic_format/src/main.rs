/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU format-arg panic-message test.
//!
//! A kernel panics with `panic!("... {} ...", x)` style messages when a
//! runtime condition holds. The literal-message panic path reports exact
//! bytes; this example exercises the format-arg form (multiple pieces +
//! runtime arguments), which the pipeline renders through the ockl printf
//! transport at the dropped-panic site before the clean wave exit.
//!
//! Run: cargo oxide run panic_format
//! Expect on stdout: `panic: value = 7 / limit = 100` (rendered on card).

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Panics with a format-arg message on the out-of-range element.
    #[kernel]
    pub fn check_limit(values: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let raw = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let v = values[raw];
            if v > 100 {
                panic!("value = {} / limit = {}", v, 100u32);
            }
            *o = v;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU Format-Arg Panic Test ===\n");

    let ctx = cuda_core::CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // In-range values copy through, no panic.
    const N: usize = 8;
    let ok_host: Vec<u32> = (0..N).map(|i| (i * 10) as u32).collect();
    let values_ok = cuda_core::DeviceBuffer::from_host(&stream, &ok_host).unwrap();
    let mut out_ok = cuda_core::DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = cuda_core::simt::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: launch shape matches the buffer lengths; one thread per element.
    unsafe { module.check_limit(&stream, cfg, &values_ok, &mut out_ok) }
        .expect("in-range launch failed");
    stream.synchronize().expect("in-range sync failed");
    let got = out_ok.to_host_vec(&stream).unwrap();
    println!("in-range copy: {:?}", got);
    assert_eq!(got, ok_host, "in-range copy mismatch");
    println!("in-range case: OK\n");

    // The panicking case: element 2 exceeds the limit. The wave exits
    // cleanly; the rendered message must have reached stdout above.
    let bad_host: Vec<u32> = vec![1, 2, 300, 4];
    let values_bad = cuda_core::DeviceBuffer::from_host(&stream, &bad_host).unwrap();
    let mut out_bad = cuda_core::DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
    let cfg_bad = cuda_core::simt::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (4 as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    println!("--- launching panicking kernel (expect 'panic: value = 300 ...' above) ---");
    // SAFETY: launch shape matches the buffer lengths; the panic exits
    // cleanly by design.
    let _ = unsafe { module.check_limit(&stream, cfg_bad, &values_bad, &mut out_bad) };
    let _ = stream.synchronize();
    println!("=== done ===");
    Ok(())
}
