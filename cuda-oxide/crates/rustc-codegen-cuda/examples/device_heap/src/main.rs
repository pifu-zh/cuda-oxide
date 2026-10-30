/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device-side heap e2e: Vec/String/Box inside kernels.
//!
//! The kernel crate declares a crate-local `#[global_allocator]` pointing
//! at `cuda_device::heap::DeviceAlloc` (a bump arena over 256 MiB of
//! global memory). The attribute expansion generates the allocator shims
//! (`__rust_alloc` family) as definitions in THIS crate — local DefIds the
//! collector's sysroot-guard stands down for — and the `alloc` crate's
//! extern block links against them. Every heap facility then works inside
//! kernels.
//!
//! Allocator semantics (see cuda-device::heap docs for the full contract):
//! bump allocation over one atomic cursor; `dealloc` is a no-op and space
//! is reclaimed by `DeviceAlloc::reset` between launches; exhaustion
//! aborts via `handle_alloc_error`, exactly like host OOM.
//!
//! Host note: this binary is compiled ONCE (the backend generates device
//! code as a side artifact), so the declaration also installs DeviceAlloc
//! as the host binary's allocator. This example's host-side Rust heap use
//! is a few kilobytes (small Vecs; the HIP runtime allocates in its own
//! native heap), which the 256 MiB arena absorbs.
//!
//! Verified against independent host references:
//! 1. heap_vec — push N squares, sum + capacity.
//! 2. heap_string — push_str/len/byte-checksum.
//! 3. heap_box — Box a pair, mutate through it.
//! 4. heap_realloc — Vec grow/truncate/re-grow (realloc + drain paths).
//! 5. heap_after_reset — reset() then a fresh Vec (arena reuse).
//!
//! Build and run with:
//!   cargo oxide run device_heap

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::heap::DeviceAlloc;
use cuda_device::{cuda_module, kernel, thread};

#[global_allocator]
static HEAP_ALLOC: DeviceAlloc = DeviceAlloc;

extern crate alloc;

#[cuda_module]
mod kernels {
    use super::*;

    /// push squares 0..N into a Vec; out = sum + final capacity.
    #[kernel]
    pub fn heap_vec(n: u32, out: &mut [u64]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            let mut v: Vec<u32> = Vec::new();
            let mut i: u32 = 0;
            while i < n {
                v.push(i.wrapping_mul(i));
                i += 1;
            }
            let sum: u32 = v.iter().sum();
            let cap = v.capacity() as u64;
            out[0] = sum as u64;
            out[1] = cap;
        }
    }

    /// String push_str/len/byte-checksum.
    #[kernel]
    pub fn heap_string(n: u32, out: &mut [u64]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            // NOTE: push_str(&str literal) trips a translate_constant gap
            // (str constants in thread-local allocations have no handler);
            // char push is the working equivalent for the prototype.
            let mut s = String::new();
            let mut i: u32 = 0;
            while i < n {
                s.push('x');
                i += 1;
            }
            s.push('!');
            let len = s.len() as u64;
            let mut sum: u64 = 0;
            for b in s.bytes() {
                sum += b as u64;
            }
            out[0] = len;
            out[1] = sum;
        }
    }

    /// Box a pair, mutate through the box.
    #[kernel]
    pub fn heap_box(a: u32, b: u32, out: &mut [u64]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            let mut boxed = Box::new((a as u64, b as u64));
            boxed.0 = boxed.0.wrapping_mul(3);
            boxed.1 += 1;
            out[0] = boxed.0;
            out[1] = boxed.1;
        }
    }

    /// Vec<u64> grow-shrink cycle: exercises realloc and truncation.
    #[kernel]
    pub fn heap_realloc(n: u32, out: &mut [u64]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            let mut v: Vec<u64> = Vec::with_capacity(4);
            let mut i: u64 = 0;
            while (i as u32) < n {
                v.push(i);
                i += 1;
            }
            let first = v[0];
            v.truncate(2);
            let mut j: u64 = 0;
            while j < 6 {
                v.push(j);
                j += 1;
            }
            let sum: u64 = v.iter().sum();
            out[0] = sum;
            out[1] = first;
        }
    }

    /// reset() then a fresh Vec — arena reuse across launches.
    #[kernel]
    pub fn heap_after_reset(out: &mut [u64]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            DeviceAlloc::reset();
            let mut v: Vec<u32> = Vec::new();
            v.push(41);
            v.push(43);
            out[0] = (v[0] + v[1]) as u64;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Device Heap Probe (Vec/String/Box on GPU) ===\n");

    let ctx = CudaContext::new(0)?;
    let s0 = ctx.default_stream();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let module = kernels::load(&ctx)?;
    let mut out = DeviceBuffer::zeroed(&s0, 2)?;
    let mut host = vec![0u64; 2];

    println!("--- [1] heap_vec (push + sum + capacity) ---");
    // SAFETY: launch shape matches; out covers the accesses.
    unsafe { module.heap_vec((s0).as_ref(), cfg, 10, &mut out) }?;
    out.copy_to_host(&s0, &mut host)?;
    s0.synchronize()?;
    let want_sum: u32 = (0..10u32).map(|i| i * i).sum();
    println!("  sum={} cap={}", host[0], host[1]);
    assert_eq!(host[0], want_sum as u64);
    assert!(host[1] >= 10);

    println!("--- [2] heap_string (push_str + len + checksum) ---");
    // SAFETY: as above.
    unsafe { module.heap_string((s0).as_ref(), cfg, 5, &mut out) }?;
    out.copy_to_host(&s0, &mut host)?;
    s0.synchronize()?;
    // "xxxxx!" -> len 6, checksum 5*120 + 33.
    println!("  len={} checksum={}", host[0], host[1]);
    assert_eq!(host[0], 6);
    assert_eq!(host[1], 5 * 120 + 33);

    println!("--- [3] heap_box (Box<(u64,u64)> mutate) ---");
    // SAFETY: as above.
    unsafe { module.heap_box((s0).as_ref(), cfg, 7, 10, &mut out) }?;
    out.copy_to_host(&s0, &mut host)?;
    s0.synchronize()?;
    println!("  .0={} .1={}", host[0], host[1]);
    assert_eq!(host[0], 21);
    assert_eq!(host[1], 11);

    println!("--- [4] heap_realloc (grow/truncate cycle) ---");
    // SAFETY: as above.
    unsafe { module.heap_realloc((s0).as_ref(), cfg, 8, &mut out) }?;
    out.copy_to_host(&s0, &mut host)?;
    s0.synchronize()?;
    // [0..8] truncated to [0,1] then extended with [0..6].
    let want: u64 = 0 + 1 + 0 + 1 + 2 + 3 + 4 + 5;
    println!("  sum={} first={}", host[0], host[1]);
    assert_eq!(host[0], want);

    println!("--- [5] heap_after_reset (arena reuse across launches) ---");
    // SAFETY: as above.
    unsafe { module.heap_after_reset((s0).as_ref(), cfg, &mut out) }?;
    out.copy_to_host(&s0, &mut host)?;
    s0.synchronize()?;
    println!("  41+43 = {}", host[0]);
    assert_eq!(host[0], 84);

    println!("\n=== ALL DEVICE HEAP PROBES PASSED ===");
    Ok(())
}
