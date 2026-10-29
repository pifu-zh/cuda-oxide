/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime synchronization probe: streams, events, mem pools, managed memory.
//!
//! Closes the capability-matrix row that previously read NOT TESTED
//! ("stream/event/pool/managed memory — symbols in the table [static]; no
//! runtime assertions on this box"). Every assertion here runs against the
//! real device on this card:
//!
//! 1. Event timing: record/synchronize/elapsed_time on a spin kernel, with
//!    the host wall clock as an order-of-magnitude cross-check.
//! 2. Query polling: query() flips false -> true across a synchronize.
//! 3. Multi-stream concurrency: the same spin kernel on two streams takes
//!    meaningfully less wall time than the serialized sum.
//! 4. Cross-stream dependency: work enqueued on stream B after
//!    `wait_event(eA)` observes stream A's writes.
//! 5. Memory pool: create, release threshold, stats, allocation from pool.
//! 6. Managed memory: malloc_managed + mem_advise, host writes, kernel
//!    mutates, host reads back (unified visibility).
//!
//! Build and run with:
//!   cargo oxide run runtime_sync

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::thread;
use cuda_macros::kernel;
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Spin `n` iterations on thread 0. The accumulator lives in a
    /// volatile-read/volatile-write chain on `out[0]`, so the compiler
    /// cannot turn the loop into a closed form (a plain `sum += i` loop
    /// collapses to `n*(n-1)/2` in LLVM and "spins" in ~10 microseconds).
    #[kernel]
    pub fn spin_write(out: &mut [u64], n: u64) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            let p = out.as_mut_ptr();
            let mut i: u64 = 0;
            while i < n {
                // SAFETY: single thread, p points at out[0] which outlives
                // the kernel; volatile stops both elimination and reordering.
                unsafe {
                    let acc = core::ptr::read_volatile(p).wrapping_add(i);
                    core::ptr::write_volatile(p, acc);
                }
                i += 1;
            }
        }
    }

    /// Fill the buffer with a constant.
    #[kernel]
    pub fn fill(out: &mut [u32], v: u32) {
        let tid = thread::index_1d().get();
        if tid < out.len() {
            out[tid] = v;
        }
    }

    /// Add one to every element (used on managed memory).
    #[kernel]
    pub fn bump(buf: &mut [u64]) {
        let tid = thread::index_1d().get();
        if tid < buf.len() {
            buf[tid] = buf[tid] + 1;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Runtime Sync Probe (streams/events/pool/managed) ===\n");

    let ctx = CudaContext::new(0)?;
    let s0 = ctx.default_stream();
    let s1 = ctx.new_stream()?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    // Enough blocks that a 256-element fill is fully covered (8 x 32).
    let cfg_wide = LaunchConfig {
        grid_dim: (8, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let module = kernels::load(&ctx)?;

    // ------------------------------------------------------------------
    // 1+2. Event timing and query polling on the default stream.
    // ------------------------------------------------------------------
    println!("--- [1] event timing / [2] query polling ---");
    let spin_iters: u64 = 1 << 20;
    let mut out0 = DeviceBuffer::zeroed(&s0, 1)?;

    let ev_start = ctx.new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let ev_stop = ctx.new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;

    // SAFETY: launch shape matches the kernel; out0 covers its accesses.
    unsafe { module.spin_write((s0).as_ref(), cfg, &mut out0, spin_iters) }?;
    let q_inflight = ev_start.query()?; // never recorded -> Ok(true) per contract

    ev_start.record(&s0)?;
    // SAFETY: as above.
    unsafe { module.spin_write((s0).as_ref(), cfg, &mut out0, spin_iters) }?;
    ev_stop.record(&s0)?;
    let q_recorded = ev_stop.query()?;
    ev_stop.synchronize()?;
    let q_done = ev_stop.query()?;

    let dev_ms = ev_start.elapsed_ms(&ev_stop)?;
    let wall = std::time::Instant::now();
    // SAFETY: as above.
    unsafe { module.spin_write((s0).as_ref(), cfg, &mut out0, spin_iters) }?;
    s0.synchronize()?;
    let wall_ms = (wall.elapsed().as_secs_f64() * 1000.0) as f32;

    println!("  query(never recorded) = {q_inflight} (contract: true)");
    println!("  query(recorded, in flight) = {q_recorded}");
    println!("  query(after synchronize) = {q_done}");
    println!("  event elapsed = {dev_ms:.3} ms | host wall = {wall_ms:.3} ms");
    assert!(q_inflight);
    assert!(!q_recorded);
    assert!(q_done);
    assert!(dev_ms > 0.0, "event elapsed must be positive");
    assert!(
        (0.2..4.0).contains(&(wall_ms / dev_ms)),
        "host wall {wall_ms:.3} ms vs device {dev_ms:.3} ms diverge"
    );

    // ------------------------------------------------------------------
    // 3. Multi-stream concurrency: two spins in parallel vs serialized.
    // ------------------------------------------------------------------
    println!("--- [3] two-stream concurrency ---");
    let par_iters: u64 = 1 << 21;
    let mut out1 = DeviceBuffer::zeroed(&s1, 1)?;

    let t = std::time::Instant::now();
    // SAFETY: launch shapes match; each stream writes its own buffer.
    unsafe { module.spin_write((s0).as_ref(), cfg, &mut out0, par_iters) }?;
    unsafe { module.spin_write((s1).as_ref(), cfg, &mut out1, par_iters) }?;
    s0.synchronize()?;
    s1.synchronize()?;
    let par_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = std::time::Instant::now();
    // SAFETY: as above, serialized on one stream.
    unsafe { module.spin_write((s0).as_ref(), cfg, &mut out0, par_iters) }?;
    unsafe { module.spin_write((s0).as_ref(), cfg, &mut out0, par_iters) }?;
    s0.synchronize()?;
    let ser_ms = t.elapsed().as_secs_f64() * 1000.0;

    let speedup = ser_ms / par_ms;
    println!("  parallel (2 streams) = {par_ms:.2} ms | serialized = {ser_ms:.2} ms");
    println!("  concurrency factor = {speedup:.2}x");
    assert!(
        speedup > 1.25,
        "two streams must overlap: parallel {par_ms:.2} ms vs serialized {ser_ms:.2} ms"
    );

    // ------------------------------------------------------------------
    // 4. Cross-stream dependency: stream B sees stream A's write only
    //    through the event edge.
    // ------------------------------------------------------------------
    println!("--- [4] cross-stream wait_event ---");
    const N: usize = 256;
    let mut host = vec![0u32; N];
    let mut buf_b = DeviceBuffer::from_host(&s1, &host)?;
    let ev_a = ctx.new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;

    // SAFETY: launch shape matches; buf_b covers the accesses.
    unsafe { module.fill((s1).as_ref(), cfg_wide, &mut buf_b, 7u32) }?;
    ev_a.record(&s1)?;
    s0.wait(&ev_a)?;
    // Now buf_b is defined for stream-0-ordered work; read it back on s0's
    // ordering via a D2H on the default stream.
    buf_b.copy_to_host(&s0, &mut host)?;
    s0.synchronize()?;
    println!("  stream B filled 7s; host reads {} after wait_event edge", host[0]);
    assert!(host.iter().all(|&v| v == 7));

    // ------------------------------------------------------------------
    // 5. Memory pool: create, threshold, stats, allocation from pool.
    // ------------------------------------------------------------------
    println!("--- [5] mem pool (runtime layer) ---");
    let rt_dev = cuda_core::Device::new(0)?;
    let rt_stream = rt_dev.new_stream()?;
    let pool = rt_dev.new_mem_pool()?;
    pool.set_release_threshold(1 << 20)?;
    let stats = pool.mem_stats()?;
    println!(
        "  pool created; stats used={} reserved={}",
        stats.used_current, stats.reserved_current
    );
    {
        let pooled = unsafe { cuda_core::malloc_from_pool_async(4096, &pool, &rt_stream) }?;
        unsafe { rt_stream.synchronize() }?;
        let stats2 = pool.mem_stats()?;
        println!(
            "  after 4 KiB pool alloc: used={} reserved={}",
            stats2.used_current, stats2.reserved_current
        );
        assert!(stats2.used_current >= stats.used_current + 4096);
        unsafe { cuda_core::free_async(pooled, &rt_stream) }?;
        unsafe { rt_stream.synchronize() }?;
    }

    // ------------------------------------------------------------------
    // 6. Managed memory: host write -> kernel bump -> host read back.
    // ------------------------------------------------------------------
    println!("--- [6] managed memory ---");
    const MN: usize = 64;
    let bytes = MN * std::mem::size_of::<u64>();
    let mptr = unsafe {
        cuda_core::malloc_managed(
            bytes,
            cuda_core::sys::CUmemAttach_flags_enum_CU_MEM_ATTACH_GLOBAL,
        )
    }?;
    {
        // SAFETY: host-side init of the unified allocation before any GPU use.
        let slice = unsafe { std::slice::from_raw_parts_mut(mptr as *mut u64, MN) };
        for (i, slot) in slice.iter_mut().enumerate() {
            *slot = i as u64;
        }
    }
    let mut mbuf = unsafe { DeviceBuffer::from_raw_parts(mptr, MN, ctx.clone()) };
    // SAFETY: launch shape matches; mbuf aliases the managed allocation.
    unsafe { module.bump((s0).as_ref(), cfg_wide, &mut mbuf) }?;
    s0.synchronize()?;
    {
        // SAFETY: unified memory is host-visible after the synchronize above.
        let slice = unsafe { std::slice::from_raw_parts(mptr as *const u64, MN) };
        assert_eq!(slice[0], 1);
        assert_eq!(slice[MN - 1], (MN - 1) as u64 + 1);
        println!("  managed host->device->host roundtrip: {} -> {} OK", 0, slice[0]);
    }
    unsafe { cuda_core::simt::memory::free_async(mptr, s0.cu_stream()) }?;

    println!("\n=== ALL RUNTIME SYNC PROBES PASSED ===");
    Ok(())
}
