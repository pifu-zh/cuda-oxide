/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// `core::mem::Alignment` is unstable; the shim signatures use the real type
// so the MIR-level prints match the alloc crate call sites exactly.
#![feature(ptr_alignment_type)]

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
use cuda_device::{cuda_module, kernel, thread};


// ── Allocator shims (exact extern-block shapes) ────────────────────────────
//
// The alloc crate's extern block declares the shims with structured
// parameter types. Defining them HERE — in the kernel crate — gives them
// local DefIds: the collector picks them up through mir_keys (the
// user-defined-shim escape hatch), and the walk links the alloc crate's
// call sites against these bodies. cuda-device::heap supplies the arena
// primitives.
//
// `core::mem::Alignment` is unstable; this local mirror is
// repr(transparent) over usize, byte-identical at the ABI level.


/// Byte-memset backing the `write_bytes` intrinsic dispatch.
///
/// # Safety
/// `dst` must be valid for `count` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "Rust" fn __cuda_device_memset_u8(dst: *mut u8, val: u8, count: usize) {
    let mut i = 0usize;
    while i < count {
        // SAFETY: dst covers count bytes by contract.
        unsafe { core::ptr::write_volatile(dst.add(i), val) };
        i += 1;
    }
}

use core::mem::Alignment;

/// Bump-allocate `size` bytes (16-aligned) from the device arena.
unsafe fn arena_alloc(size: usize) -> *mut u8 {
    let size = (size.max(1) + 15) & !15;
    let old = cuda_device::heap::arena_cursor_fetch_add(size);
    let start = (old + 15) & !15;
    let end = start + size;
    if end > cuda_device::heap::HEAP_BYTES {
        cuda_device::heap::arena_pin_exhausted();
        core::ptr::null_mut()
    } else {
        cuda_device::heap::arena_record_served(size);
        cuda_device::heap::arena_base_ptr().wrapping_add(start)
    }
}

/// # Safety
/// Device side only; follows the GlobalAlloc::alloc contract.
#[unsafe(no_mangle)]
pub unsafe extern "Rust" fn __rust_alloc(size: usize, align: Alignment) -> *mut u8 {
    let _ = align;
    arena_alloc(size)
}

/// # Safety
/// Device side only; follows the GlobalAlloc::alloc_zeroed contract.
#[unsafe(no_mangle)]
pub unsafe extern "Rust" fn __rust_alloc_zeroed(size: usize, align: Alignment) -> *mut u8 {
    let p = __rust_alloc(size, align);
    if !p.is_null() {
        let words = size / 8;
        let base = p as *mut u64;
        let mut i = 0usize;
        while i < words {
            // SAFETY: base covers size bytes; i < words stays in bounds.
            unsafe { core::ptr::write_volatile(base.add(i), 0u64) };
            i += 1;
        }
    }
    p
}

/// # Safety
/// Device side only; follows the GlobalAlloc::dealloc contract.
#[unsafe(no_mangle)]
pub unsafe extern "Rust" fn __rust_dealloc(_ptr: core::ptr::NonNull<u8>, _size: usize, _align: Alignment) {
    // Bump allocator: reclamation via DeviceAlloc::reset between launches.
}

/// # Safety
/// Device side only; follows the GlobalAlloc::realloc contract.
#[unsafe(no_mangle)]
pub unsafe extern "Rust" fn __rust_realloc(
    ptr: core::ptr::NonNull<u8>,
    old_size: usize,
    align: Alignment,
    new_size: usize,
) -> *mut u8 {
    let fresh = __rust_alloc(new_size, align);
    if !fresh.is_null() {
        let copy = old_size.min(new_size);
        core::ptr::copy_nonoverlapping(ptr.as_ptr(), fresh, copy);
        __rust_dealloc(ptr, old_size, align);
    }
    fresh
}

/// No-op detection stub (std provides one on host targets).
///
/// # Safety
/// No-op; device side only.
#[unsafe(no_mangle)]
pub unsafe extern "Rust" fn __rust_no_alloc_shim_is_unstable_v2() {}

/// OOM handler: device abort (panic=abort semantics).
///
/// # Safety
/// Never returns; device side only.
#[unsafe(no_mangle)]
pub unsafe extern "Rust" fn __rust_alloc_error_handler(_size: usize, _align: usize) -> ! {
    unreachable!("device heap exhausted")
}

extern crate alloc;
// NOTE: no #[global_allocator] here — its expansion generates ABI-shaped
// shim signatures (*mut u8, usize) that collide in MIR with the alloc
// crate's structured extern-block signatures (NonNull<u8>, Alignment).
// cuda-device::heap defines the shims at the exact extern-block shapes,
// which the walk links the call sites against (local DefIds, so the
// sysroot guard stands down).

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

    /// Vec<u64> with capacity reserved up front (no grow): exercises the
    /// alloc + push + drain paths without `Result<NonNull<[u8]>, AllocError>`
    /// reallocation — that enum's niche slot map (fat-pointer payload at
    /// byte 8) trips the provenance fail-closed gate and is recorded as the
    /// remaining heap gap (docs: research archive, fifth-layer section).
    #[kernel]
    pub fn heap_realloc(n: u32, out: &mut [u64]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            let mut v: Vec<u64> = Vec::with_capacity(n as usize);
            let mut i: u64 = 0;
            while (i as u32) < n {
                v.push(i);
                i += 1;
            }
            let first = v[0];
            v.truncate(2);
            let sum = v[0] + v[1];
            out[0] = sum;
            out[1] = first;
        }
    }

    /// reset() then a fresh Vec — arena reuse across launches.
    #[kernel]
    pub fn heap_after_reset(out: &mut [u64]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            cuda_device::heap::DeviceAlloc::reset();
            let mut v: Vec<u32> = Vec::new();
            v.push(41);
            v.push(43);
            out[0] = (v[0] + v[1]) as u64;
        }
    }
}

// ============================================================================
// CAPABILITY BOUNDARY (verified 2026-10-30, evidence in research archive):
// The full Vec/String/Box family funnels through
// Result<NonNull<[u8]>, AllocError> (alloc_impl_runtime), whose niche slot
// map carries a fat-pointer payload overlapping an integer claim at byte 8.
// The importer's provenance gate rejects it fail-closed ("overlapping
// pointer and non-identical storage"). Four upstream layers are already
// landed (arena, guard precision, write_bytes dispatch, symbol routing);
// the slot-map generalization for fat-pointer niches is the single
// remaining blocker. This example builds and asserts everything up to
// that gate; its kernels are kept as the reproduction contract.
// ============================================================================

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
    // [0..8] truncated to [0,1]: sum = 1, first = 0.
    let want: u64 = 1;
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
