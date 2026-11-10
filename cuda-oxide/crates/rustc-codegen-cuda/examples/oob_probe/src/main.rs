/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Core-machinery panic-message test: the panics the compiler raises
//! itself (slice bounds checks, integer division/remainder by zero) never
//! appear as a source `panic!`, so the kernel macro's format-arg rewrite
//! cannot reach them. The pipeline handles these at the MIR assert
//! terminators instead: on the failure edge the matching device helper
//! renders core's message text through the ockl printf transport and the
//! wave exits cleanly (no queue abort). This example asserts the clean
//! exits; the rendered messages appear on stdout above the result lines.
//!
//! Messages mirror `core::panicking` (rustc 1.99):
//! - bounds check: `index out of bounds: the len is {len} but the index is
//!   {index}` — one report per faulting thread;
//! - division: `attempt to divide by zero`;
//! - remainder: `attempt to calculate the remainder with a divisor of zero`.
//!
//! Scope: unit-return kernels; non-unit kernels, device functions, and
//! core panics without a structured assert (none today in this set) keep
//! the plain trap. See docs/research/2026-11-09-format-arg-panic.md.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Reads out of bounds for thread >= 4 (runtime index, bounds check
    /// cannot fold away).
    #[kernel]
    pub fn read_oob(values: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let raw = idx.get();
        if raw >= 4 {
            let v = values[raw];
            if let Some(o) = out.get_mut(idx) {
                *o = v;
            }
        }
    }

    /// Thread 3 divides by a runtime zero (the check cannot fold away).
    #[kernel]
    pub fn div_zero(seed: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let raw = idx.get();
        let a = 100u32 + raw as u32;
        let b = (seed[raw & 3] ^ seed[raw & 3]) + (raw as u32 & 0); // always 0
        if raw == 3 {
            let v = a / b;
            if let Some(o) = out.get_mut(idx) {
                *o = v;
            }
        }
    }

    /// Thread 3 takes a remainder by a runtime zero.
    #[kernel]
    pub fn rem_zero(seed: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let raw = idx.get();
        let a = 100u32 + raw as u32;
        let b = (seed[raw & 3] ^ seed[raw & 3]) + (raw as u32 & 0); // always 0
        if raw == 3 {
            let v = a % b;
            if let Some(o) = out.get_mut(idx) {
                *o = v;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = cuda_core::CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).map_err(|e| {
        eprintln!("FAIL at load: {e}");
        e
    })?;
    let values =
        cuda_core::DeviceBuffer::from_host(&stream, &[10u32, 20, 30, 40]).map_err(|e| {
            eprintln!("FAIL at from_host: {e}");
            e
        })?;
    let mut out = cuda_core::DeviceBuffer::<u32>::zeroed(&stream, 8).map_err(|e| {
        eprintln!("FAIL at zeroed: {e}");
        e
    })?;
    let cfg = cuda_core::simt::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (8, 1, 1),
        shared_mem_bytes: 0,
    };
    println!("launching (threads 4..8 read out of bounds; expect four reports, clean exit)");
    let launch = unsafe { module.read_oob(&stream, cfg, &values, &mut out) };
    let sync = stream.synchronize();
    match (&launch, &sync) {
        (Ok(()), Ok(())) => println!("oob: clean exit (no queue abort)"),
        _ => {
            eprintln!("OOB REGRESSION: launch err {launch:?} sync err {sync:?}");
            return Err(Box::<dyn std::error::Error>::from(
                "oob launch/sync must succeed (clean exit)",
            ));
        }
    }

    // Division/remainder by zero: thread 3 only.
    let seed = cuda_core::DeviceBuffer::from_host(&stream, &[1u32, 2, 3, 4])?;
    let mut d_out = cuda_core::DeviceBuffer::<u32>::zeroed(&stream, 4)?;
    let cfg4 = cuda_core::simt::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (4, 1, 1),
        shared_mem_bytes: 0,
    };
    println!("launching div-by-zero (thread 3; expect one report, clean exit)");
    let dl = unsafe { module.div_zero(&stream, cfg4, &seed, &mut d_out) };
    let ds = stream.synchronize();
    match (&dl, &ds) {
        (Ok(()), Ok(())) => println!("div: clean exit (no queue abort)"),
        _ => {
            eprintln!("DIV REGRESSION: launch err {dl:?} sync err {ds:?}");
            return Err(Box::<dyn std::error::Error>::from(
                "div launch/sync must succeed (clean exit)",
            ));
        }
    }
    println!("launching rem-by-zero (thread 3; expect one report, clean exit)");
    let rl = unsafe { module.rem_zero(&stream, cfg4, &seed, &mut d_out) };
    let rs = stream.synchronize();
    match (&rl, &rs) {
        (Ok(()), Ok(())) => println!("rem: clean exit (no queue abort)"),
        _ => {
            eprintln!("REM REGRESSION: launch err {rl:?} sync err {rs:?}");
            return Err(Box::<dyn std::error::Error>::from(
                "rem launch/sync must succeed (clean exit)",
            ));
        }
    }
    Ok(())
}
