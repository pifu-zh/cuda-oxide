/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Iterator-adapter probe: the allocation-free subset of the iterator
//! machinery on the device, with its precise support boundary.
//!
//! Closes the allocation-free half of the capability-matrix row that read
//! NOT TESTED ("iterators/Vec/formatting — facilities that depend on
//! device-side allocation"). Each kernel checks one adapter against an
//! independent host reference. The heap half (Vec/Box/String) is covered by
//! the negative example device_alloc_reject: no GlobalAlloc on device, the
//! failure is a loud compile error ("heap allocation is not supported in
//! kernels"), never a silent miscompute.
//!
//! VERIFIED WORKING (all values checked against host references):
//!   iter/map/sum, min/max (via match), filter+count, zip, position,
//!   enumerate + `for` loop, slice fold (manual accumulation).
//!
//! NOT SUPPORTED (loud codegen failure, root cause in the MIR importer):
//!   - `enumerate().fold()/sum()` chains: Enumerate::fold builds a nested
//!     closure struct constant ("Expected ADT Rust type for struct constant,
//!     got RigidTy(Closure(...))" from core/src/iter/adapters/rev.rs) — the
//!     importer cannot materialize closure-typed struct constants.
//!   - `Option::unwrap_or`/`map_or` on iterator results: routes through
//!     Option::map_or whose Fn-trait dispatch is rejected ("calling
//!     Result::Err through Fn/FnOnce is not yet supported"). Plain `match`
//!     is the working substitute (used for min/max/position below).
//!
//! Build and run with:
//!   cargo oxide run iterator_adapters

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// map + sum: 2*x summed over the slice.
    #[kernel]
    pub fn map_sum(data: &[u32], out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            out[0] = data.iter().map(|&x| x.wrapping_mul(2)).sum();
        }
    }

    /// min/max via value-shaped unwrap_or: `copied()` converts the
    /// `Option<&T>` from min/max into `Option<T>`, whose unwrap_or dispatches
    /// through the plain branch (verified on device — the reference-shaped
    /// `unwrap_or(&0)` form is what routed into Fn-trait dispatch).
    #[kernel]
    pub fn min_max(data: &[u32], out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            let min: u32 = data.iter().min().copied().unwrap_or(0);
            let max: u32 = data.iter().max().copied().unwrap_or(0);
            out[0] = min.wrapping_mul(1_000_000).wrapping_add(max);
        }
    }

    /// map_or + unwrap_or_else with value-shaped/non-capturing closures
    /// (2026-10-31: verified working on device — the earlier "Fn-trait
    /// dispatch rejected" boundary applies to reference-shaped defaults).
    #[kernel]
    pub fn map_or_probe(data: &[u32], out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            let via_map_or: u32 = data.first().map_or(9, |&v| v.wrapping_add(1));
            let via_else: u32 = data.first().copied().unwrap_or_else(|| 42);
            out[0] = via_map_or.wrapping_mul(1_000).wrapping_add(via_else);
        }
    }

    /// filter + count.
    #[kernel]
    pub fn filter_count(data: &[u32], threshold: u32, out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            out[0] = data.iter().filter(|&&x| x > threshold).count() as u32;
        }
    }

    /// zip + map + sum (with a range iterator on the left).
    #[kernel]
    pub fn zip_probe(data: &[u32], out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            out[0] = (0..data.len() as u32)
                .zip(data.iter())
                .map(|(i, &v)| i + v)
                .sum();
        }
    }

    /// position via match.
    #[kernel]
    pub fn position_probe(data: &[u32], needle: u32, out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            out[0] = match data.iter().position(|&x| x == needle) {
                Some(p) => p as u32,
                None => u32::MAX,
            };
        }
    }

    /// enumerate + `for` loop (the working enumerate form; the fold/sum
    /// chain through Enumerate is the unsupported one — see module docs).
    #[kernel]
    pub fn enum_loop(data: &[u32], out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            let mut acc: u32 = 0;
            for (i, &v) in data.iter().enumerate() {
                acc = acc + i as u32 + v;
            }
            out[0] = acc;
        }
    }

    /// enumerate + fold + sum: previously the boundary (Enumerate::fold
    /// builds nested closure struct constants; the Closure-constant
    /// zero-synthesis fallback in the importer resolves them).
    #[kernel]
    pub fn enum_fold(data: &[u32], out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            out[0] = data
                .iter()
                .enumerate()
                .fold(0u32, |acc, (i, &v)| acc + i as u32 + v);
        }
    }

    #[kernel]
    pub fn enum_sum(data: &[u32], out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            out[0] = data
                .iter()
                .enumerate()
                .map(|(i, &v)| i as u32 + v)
                .sum();
        }
    }

    /// slice fold WITHOUT enumerate (works; Enumerate::fold does not).
    /// Deliberately not `.sum()`: the fold form is the tested subject.
    #[allow(clippy::unnecessary_fold)]
    #[kernel]
    pub fn fold_plain(data: &[u32], out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            out[0] = data.iter().fold(0u32, |acc, &v| acc + v);
        }
    }
}

/// Independent host references (same expressions, host side).
mod host_ref {
    pub fn map_sum(data: &[u32]) -> u32 {
        data.iter().map(|&x| x.wrapping_mul(2)).sum()
    }
    pub fn min_max(data: &[u32]) -> u32 {
        let min = data.iter().min().copied().unwrap_or(0);
        let max = data.iter().max().copied().unwrap_or(0);
        min.wrapping_mul(1_000_000).wrapping_add(max)
    }
    pub fn map_or_probe(data: &[u32]) -> u32 {
        let via_map_or: u32 = data.first().map_or(9, |&v| v.wrapping_add(1));
        let via_else: u32 = data.first().copied().unwrap_or_else(|| 42);
        via_map_or.wrapping_mul(1_000).wrapping_add(via_else)
    }
    pub fn filter_count(data: &[u32], threshold: u32) -> u32 {
        data.iter().filter(|&&x| x > threshold).count() as u32
    }
    pub fn zip_probe(data: &[u32]) -> u32 {
        (0..data.len() as u32)
            .zip(data.iter())
            .map(|(i, &v)| i + v)
            .sum()
    }
    pub fn position_probe(data: &[u32], needle: u32) -> u32 {
        data.iter().position(|&x| x == needle).unwrap_or(usize::MAX) as u32
    }
    pub fn enum_loop(data: &[u32]) -> u32 {
        let mut acc = 0;
        for (i, &v) in data.iter().enumerate() {
            acc += i as u32 + v;
        }
        acc
    }
    pub fn fold_plain(data: &[u32]) -> u32 {
        data.iter().fold(0, |acc, &v| acc + v)
    }
    pub fn enum_fold(data: &[u32]) -> u32 {
        data.iter()
            .enumerate()
            .fold(0u32, |acc, (i, &v)| acc + i as u32 + v)
    }
    pub fn enum_sum(data: &[u32]) -> u32 {
        data.iter().enumerate().map(|(i, &v)| i as u32 + v).sum()
    }
}

fn check(name: &str, got: u32, want: u32) -> Result<(), Box<dyn std::error::Error>> {
    println!("  {name}: device={got} host={want}");
    if got != want {
        return Err(format!("{name}: device {got} != host {want}").into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Device Iterator Adapter Probe ===\n");

    let ctx = CudaContext::new(0)?;
    let s0 = ctx.default_stream();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let module = kernels::load(&ctx)?;

    let data: Vec<u32> = vec![7, 3, 12, 5, 8, 21, 4, 9];
    let dev = DeviceBuffer::from_host(&s0, &data)?;
    let mut out = DeviceBuffer::zeroed(&s0, 1)?;
    let mut host_out = vec![0u32; 1];

    println!("--- allocation-free adapters (each vs host reference) ---");

    // SAFETY: launch shape matches; buffers cover the accesses.
    unsafe { module.map_sum((s0).as_ref(), cfg, &dev, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    check("map_sum", host_out[0], host_ref::map_sum(&data))?;

    // SAFETY: as above.
    unsafe { module.min_max((s0).as_ref(), cfg, &dev, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    check("min_max", host_out[0], host_ref::min_max(&data))?;

    // SAFETY: as above.
    unsafe { module.map_or_probe((s0).as_ref(), cfg, &dev, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    check("map_or_probe", host_out[0], host_ref::map_or_probe(&data))?;

    // SAFETY: as above.
    unsafe { module.filter_count((s0).as_ref(), cfg, &dev, 6, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    check(
        "filter_count(>6)",
        host_out[0],
        host_ref::filter_count(&data, 6),
    )?;

    // SAFETY: as above.
    unsafe { module.zip_probe((s0).as_ref(), cfg, &dev, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    check("zip_probe", host_out[0], host_ref::zip_probe(&data))?;

    // SAFETY: as above.
    unsafe { module.position_probe((s0).as_ref(), cfg, &dev, 21, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    check(
        "position_probe(21)",
        host_out[0],
        host_ref::position_probe(&data, 21),
    )?;

    // SAFETY: as above.
    unsafe { module.enum_loop((s0).as_ref(), cfg, &dev, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    check("enum_loop", host_out[0], host_ref::enum_loop(&data))?;

    // SAFETY: as above.
    unsafe { module.enum_fold((s0).as_ref(), cfg, &dev, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    check("enum_fold", host_out[0], host_ref::enum_fold(&data))?;

    // SAFETY: as above.
    unsafe { module.enum_sum((s0).as_ref(), cfg, &dev, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    check("enum_sum", host_out[0], host_ref::enum_sum(&data))?;

    // SAFETY: as above.
    unsafe { module.fold_plain((s0).as_ref(), cfg, &dev, &mut out) }?;
    out.copy_to_host(&s0, &mut host_out)?;
    s0.synchronize()?;
    check("fold_plain", host_out[0], host_ref::fold_plain(&data))?;

    println!("\n=== ALL DEVICE ITERATOR ADAPTER PROBES PASSED ===");
    Ok(())
}
