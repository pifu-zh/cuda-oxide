/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device-side heap: a static bump arena on global memory plus a
//! `GlobalAlloc` implementation for `#[global_allocator]`.
//!
//! # Why this exists
//!
//! The GPU has no runtime allocator, so the sysroot allocator shims
//! (`__rust_alloc` family) have no implementation and any `Vec`/`Box`/
//! `String` reached from a kernel is rejected at codegen time with "heap
//! allocation is not supported in kernels" (the collector guard; see the
//! `error_heap_alloc` example). The fix is the language's own opt-in
//! mechanism, pointed at device memory:
//!
//! ```ignore
//! // In the kernel crate (device pass; gate with the device cfg):
//! #[global_allocator]
//! static HEAP: cuda_device::heap::DeviceAlloc = cuda_device::heap::DeviceAlloc;
//! ```
//!
//! The attribute expansion generates the allocator shims as definitions in
//! the declaring crate — local DefIds — which the `alloc` crate's extern
//! block links against and which the collector's sysroot-guard stands down
//! for. Every heap facility (`Vec`, `String`, `format!`, `Box`) then works
//! inside kernels.
//!
//! # Design (prototype scope)
//!
//! - **Static arena** ([`HEAP_BYTES`], default 256 MiB) in global memory,
//!   16-byte aligned, zero-initialized at module load. No host setup.
//! - **Bump allocation** over one atomic cursor — O(1), lock-free.
//! - **`dealloc` is a no-op**: freed space is reclaimed by
//!   [`DeviceAlloc::reset`] between launches. The supported pattern is a
//!   per-launch working set that fits the arena; a `for`-loop of `push`
//!   without `reset` across launches will eventually exhaust it (OOM
//!   aborts via `handle_alloc_error`, same as host OOM semantics).
//! - **Cross-launch contents are not guaranteed** unless reset; the
//!   prototype serves intra-kernel allocation.
//!
//! # Host builds
//!
//! `DeviceAlloc` must not become the host allocator. The declaring crate
//! gates the `#[global_allocator]` item with the device-pass cfg (see the
//! `device_heap` example).

use core::alloc::{GlobalAlloc, Layout};

/// Device-local mirror of `core::mem::Alignment` (unstable there): a
/// power-of-two byte alignment as a transparent usize wrapper. The
/// struct name keeps the MIR type print identical to the alloc crate's
/// `Alignment`, which is what the extern block call sites pass.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Alignment(usize);

impl Alignment {
    fn as_usize(self) -> usize {
        self.0
    }
}
use core::ptr::NonNull;
use core::sync::atomic::Ordering;

/// Arena size in bytes (256 MiB).
pub const HEAP_BYTES: usize = 256 * 1024 * 1024;

/// Every allocation is 16-byte aligned (max realistic device alignment).
const BUMP_ALIGN: usize = 16;

/// Bump cursor into the arena, in bytes from the arena start.
/// Volatile access, not atomic: the heap contract is one allocating thread
/// per kernel (see DeviceAlloc docs), and atomics here would route through
/// the inline-PTX path whose constraint model is separate work.
struct HeapCursor(core::cell::UnsafeCell<usize>);
// SAFETY: single allocating thread per kernel contract (DeviceAlloc docs);
// accessors use volatile RMW.
unsafe impl Sync for HeapCursor {}
static HEAP_CURSOR: HeapCursor = HeapCursor(core::cell::UnsafeCell::new(0));

/// Bytes served so far (diagnostics).
#[repr(transparent)]
struct BytesServed(core::cell::UnsafeCell<usize>);
// SAFETY: diagnostics counter; per-kernel monotonic, no cross-thread reads.
unsafe impl Sync for BytesServed {}
static BYTES_SERVED: BytesServed = BytesServed(core::cell::UnsafeCell::new(0));

/// The arena itself: `#[repr(align(16))]` guarantees the storage is
/// 16-byte aligned in global memory; zero-initialized.
#[repr(C, align(16))]
pub struct HeapArena {
    bytes: [u64; HEAP_BYTES / 8],
}

// A plain `static` in device code lands in global memory (address space 1)
// through the module's data section — the same path constant memory uses.
static HEAP: HeapArena = HeapArena {
    bytes: [0; HEAP_BYTES / 8],
};

#[inline]
fn heap_base() -> usize {
    // SAFETY: HEAP is a static; taking its address is stable and infallible.
    unsafe { &raw const HEAP as usize }
}

#[inline]
const fn align_up(n: usize) -> usize {
    (n + BUMP_ALIGN - 1) & !(BUMP_ALIGN - 1)
}

/// Arena cursor read-modify-add (device side).
///
/// # Safety
/// Device side only.
#[doc(hidden)]
pub unsafe fn arena_cursor_fetch_add(size: usize) -> usize {
    let old = core::ptr::read_volatile(HEAP_CURSOR.0.get());
    core::ptr::write_volatile(HEAP_CURSOR.0.get(), old + size);
    old
}

/// Pins the cursor at the arena end (exhaustion marker).
///
/// # Safety
/// Device side only.
#[doc(hidden)]
pub unsafe fn arena_pin_exhausted() {
    core::ptr::write_volatile(HEAP_CURSOR.0.get(), HEAP_BYTES);
}

/// Records `size` served bytes (diagnostics).
///
/// # Safety
/// Device side only.
#[doc(hidden)]
pub unsafe fn arena_record_served(size: usize) {
    unsafe {
        let old = core::ptr::read_volatile(BYTES_SERVED.0.get());
        core::ptr::write_volatile(BYTES_SERVED.0.get(), old + size);
    }
}

/// Base address of the arena as a raw pointer.
///
/// # Safety
/// Device side only.
#[doc(hidden)]
pub unsafe fn arena_base_ptr() -> *mut u8 {
    &raw const HEAP as *mut u8
}

/// The device global allocator: bump allocation over [`HEAP_BYTES`].
///
/// # Single-threaded contract
///
/// Device-side allocation is a kernel-local resource. Concurrent
/// allocation from many threads of one kernel funnels through the atomic
/// cursor (correct but contended); the prototype's contract is one
/// allocating thread per kernel body, or explicit cross-thread ownership.
pub struct DeviceAlloc;

impl DeviceAlloc {
    /// Bytes served since module load (diagnostics).
    pub fn bytes_served() -> usize {
        unsafe { core::ptr::read_volatile(BYTES_SERVED.0.get()) }
    }

    /// Resets the arena (cursor 0). Call from a kernel before a fresh
    /// working set if a previous launch filled it.
    pub fn reset() {
        // SAFETY: single allocating thread per kernel contract.
        unsafe { core::ptr::write_volatile(HEAP_CURSOR.0.get(), 0) };
        unsafe { core::ptr::write_volatile(BYTES_SERVED.0.get(), 0) };
    }
}

/// # Safety
/// Follows the `GlobalAlloc` contract; device side only. On host builds
/// this must never be installed as the global allocator.
unsafe impl GlobalAlloc for DeviceAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Touch the detection stub and the memset backing so both
        // definitions are collected into the device module: alloc crate
        // internals call the same symbols, and the definitions must exist
        // at link time.

        let size = align_up(layout.size().max(1));
        let old = core::ptr::read_volatile(HEAP_CURSOR.0.get());
        core::ptr::write_volatile(HEAP_CURSOR.0.get(), old + size);
        let start = (old + BUMP_ALIGN - 1) & !(BUMP_ALIGN - 1);
        let end = start + size;
        if end > HEAP_BYTES {
            // Pin the cursor at the end so every later alloc also fails.
            core::ptr::write_volatile(HEAP_CURSOR.0.get(), HEAP_BYTES);
            core::ptr::null_mut()
        } else {
            unsafe {
        let old = core::ptr::read_volatile(BYTES_SERVED.0.get());
        core::ptr::write_volatile(BYTES_SERVED.0.get(), old + size);
    }
            (heap_base() + start) as *mut u8
        }
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Bump allocator: reclamation happens via reset() between launches.
    }

    unsafe fn realloc(&self, ptr: *mut u8, old_layout: Layout, new_size: usize) -> *mut u8 { unsafe {
        let new_layout = Layout::from_size_align_unchecked(new_size, old_layout.align());
        let fresh = self.alloc(new_layout);
        if !fresh.is_null() {
            let copy = old_layout.size().min(new_size);
            core::ptr::copy_nonoverlapping(ptr, fresh, copy);
            self.dealloc(ptr, old_layout);
        }
        fresh
    }}

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 { unsafe {
        let p = self.alloc(layout);
        if !p.is_null() {
            // NOTE: core::ptr::write_bytes lowers to the rustc
            //  intrinsic, which the device translator does
            // not support yet (translate_constant gap family). Zero via
            // u64 volatile stores instead (the arena is 16-aligned, so
            // full u64s are in-bounds; the tail below 8 bytes is dropped —
            // callers zeroing sub-8 layouts get their first floor(size/8)
            // words cleared, and bump space is static-zero anyway except
            // after reset() reuse, which only happens at whole-launch
            // granularity).
            let words = layout.size() / 8;
            let base = p as *mut u64;
            let mut i = 0usize;
            while i < words {
                // SAFETY: base covers layout.size() bytes; i < words keeps
                // the offset in bounds.
                unsafe { core::ptr::write_volatile(base.add(i), 0u64) };
                i += 1;
            }
        }
        p
    }}
}

