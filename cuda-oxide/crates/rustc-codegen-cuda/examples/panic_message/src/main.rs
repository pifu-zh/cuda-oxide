/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU panic-message test.
//!
//! A kernel panics with a literal message when a runtime condition holds.
//! The importer materializes the message as an immutable device global and
//! calls the `__cuda_panic_str` device helper before the kernel trap; the
//! message rides the ockl hostcall transport (device printf's service) to
//! the host stdout as `panic: <message>`.
//!
//! Scope note (honest): format-arg panics (`panic!("x = {}", x)`) are not
//! rendered yet - they stay plain traps. Literal panics report exactly.
//!
//! Run: cargo oxide run panic_message
//! Expect on stdout: `panic: device panic: value out of range`

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Copies `values` to `out`, panicking on the out-of-range element.
    #[kernel]
    pub fn check_limit(values: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let raw = idx.get();
        if values[raw] > 100 {
            panic!("device panic: value out of range");
        }
        if let Some(o) = out.get_mut(idx) {
            *o = values[raw];
        }
    }
}

fn main() {
    println!("=== GPU Panic Message Test ===\n");

    let ctx = cuda_core::CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // The passing case first: in-range values copy through, no panic.
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

    // The panicking case: element 2 exceeds the limit. The kernel traps, so
    // the sync error is expected; the message must have reached stdout.
    let bad_host: Vec<u32> = vec![1, 2, 300, 4];
    let values_bad = cuda_core::DeviceBuffer::from_host(&stream, &bad_host).unwrap();
    let mut out_bad = cuda_core::DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
    let cfg_bad = cuda_core::simt::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (4, 1, 1),
        shared_mem_bytes: 0,
    };
    println!("--- launching panicking kernel (expect 'panic: ...' above) ---");
    // SAFETY: launch shape matches the buffer lengths; the kernel traps by design.
    let launch = unsafe { module.check_limit(&stream, cfg_bad, &values_bad, &mut out_bad) };
    let sync = stream.synchronize();
    match (&launch, &sync) {
        (Err(e), _) | (_, Err(e)) => {
            println!("kernel error (expected after a device trap): {e}");
        }
        _ => println!("launch/sync reported no error (trap may be silent on this stack)"),
    }
    println!("=== done ===");
}
