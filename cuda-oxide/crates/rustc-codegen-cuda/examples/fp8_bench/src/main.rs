/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Software FP8/FP4 conversion throughput benchmark (RDNA2 has no fp8/fp4
//! conversion hardware; the conversions are the pure software bit
//! manipulation from `cuda_device::fp8`).
//!
//! Measurement: each conversion kernel is graded against a same-footprint
//! `copy_f32` kernel using the §8 triad protocol — wall-clock warmup to
//! steady clocks, a fresh H2D input upload before each timed block (the
//! re-irrigation), event timing over 10-launch blocks, best block reported.
//! Effective bandwidth = nominal DRAM traffic / time; the nominal traffic is
//! 5 B per element (1 B fp8 in + 4 B f32 out, or the reverse — the packed
//! forms move the same bytes per element).
//!
//! Grading frame (the 60/70/80/90 substitution scale): on hardware with fp8
//! conversion units the conversion is free, so the replaced vendor path is
//! memory-bound. A software conversion that keeps up with the copy stream
//! hides entirely behind memory; kernel-bandwidth / copy-bandwidth is the
//! grade. Correctness is fp8_decode's job (404,872 exhaustive checks); the
//! values here only need to cover the pattern classes so no branch dominates.
//!
//! Working set: 128 Mi elements per direction (512 MiB on the f32 side) —
//! the output stream alone cycles the 128 MiB L2 several times per rep, so
//! no rep runs out of cache.
//!
//! Run: cargo oxide run fp8_bench

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::fp8::{
    e4m3_to_f32, e4m3x2_to_f32x2, e5m2_to_f32, e5m2x2_to_f32x2, f32_to_e4m3_rne_satfinite,
    f32_to_e5m2_rne, f32x2_to_e4m3x2_rne_satfinite, f32x2_to_e5m2x2_rne,
};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const ELEMENTS: usize = 128 << 20; // 128 Mi elements per direction
const THREADS: u32 = 1024;
const REPS: usize = 10;
const BLOCKS: usize = 3;
const WARMUP_SECS: u64 = 2;

#[cuda_module]
mod kernels {
    use super::*;

    /// Memory-bound reference: out[i] = in[i] (8 B/element of traffic).
    #[kernel]
    pub fn copy_f32(input: &[f32], mut out: DisjointSlice<f32>) {
        let raw = thread::index_1d().get() as usize;
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = input[raw];
        }
    }

    /// u8 fp8 stream -> f32 stream (1 B in + 4 B out per element).
    #[kernel]
    pub fn decode_e4m3(input: &[u8], mut out: DisjointSlice<f32>) {
        let raw = thread::index_1d().get() as usize;
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = e4m3_to_f32(input[raw]);
        }
    }

    /// u8 fp8 stream -> f32 stream (1 B in + 4 B out per element).
    #[kernel]
    pub fn decode_e5m2(input: &[u8], mut out: DisjointSlice<f32>) {
        let raw = thread::index_1d().get() as usize;
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = e5m2_to_f32(input[raw]);
        }
    }

    /// f32 stream -> u8 fp8 stream (4 B in + 1 B out per element).
    #[kernel]
    pub fn encode_e4m3(input: &[f32], mut out: DisjointSlice<u8>) {
        let raw = thread::index_1d().get() as usize;
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f32_to_e4m3_rne_satfinite(input[raw]);
        }
    }

    /// f32 stream -> u8 fp8 stream (4 B in + 1 B out per element).
    #[kernel]
    pub fn encode_e5m2(input: &[f32], mut out: DisjointSlice<u8>) {
        let raw = thread::index_1d().get() as usize;
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f32_to_e5m2_rne(input[raw]);
        }
    }

    /// Packed decode: one u16 carries two fp8 values (5 B/element overall).
    #[kernel]
    pub fn decode_e4m3x2(
        input: &[u16],
        mut out_x: DisjointSlice<f32>,
        mut out_y: DisjointSlice<f32>,
    ) {
        let raw = thread::index_1d().get() as usize;
        let (x, y) = e4m3x2_to_f32x2(input[raw]);
        if let Some(o) = out_x.get_mut(thread::index_1d()) {
            *o = x;
        }
        if let Some(o) = out_y.get_mut(thread::index_1d()) {
            *o = y;
        }
    }

    /// Packed decode: one u16 carries two fp8 values (5 B/element overall).
    #[kernel]
    pub fn decode_e5m2x2(
        input: &[u16],
        mut out_x: DisjointSlice<f32>,
        mut out_y: DisjointSlice<f32>,
    ) {
        let raw = thread::index_1d().get() as usize;
        let (x, y) = e5m2x2_to_f32x2(input[raw]);
        if let Some(o) = out_x.get_mut(thread::index_1d()) {
            *o = x;
        }
        if let Some(o) = out_y.get_mut(thread::index_1d()) {
            *o = y;
        }
    }

    /// Packed encode: two f32 streams pack into one u16 (5 B/element).
    #[kernel]
    pub fn encode_e4m3x2(a: &[f32], b: &[f32], mut out: DisjointSlice<u16>) {
        let raw = thread::index_1d().get() as usize;
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f32x2_to_e4m3x2_rne_satfinite(a[raw], b[raw]);
        }
    }

    /// Packed encode: two f32 streams pack into one u16 (5 B/element).
    #[kernel]
    pub fn encode_e5m2x2(a: &[f32], b: &[f32], mut out: DisjointSlice<u16>) {
        let raw = thread::index_1d().get() as usize;
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f32x2_to_e5m2x2_rne(a[raw], b[raw]);
        }
    }
}

/// Deterministic byte spread covering every fp8 pattern class: normals,
/// denormals, the NaN byte, the saturation zone, all signs. (Throughput does
/// not depend on values; correctness is fp8_decode's job.)
fn fill_byte_spread(seed: u64, out: &mut [u8]) {
    let mut s = seed | 1;
    for b in out.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (s >> 33) as u8;
    }
}

/// Deterministic f32 spread across the encode envelope (0..=512, both
/// signs) — covers normal rounding, the E4M3 satfinite clamp, the E5M2
/// overflow-to-Inf threshold, and sub-1.0 denormal-target magnitudes.
fn fill_f32_spread(seed: u64, out: &mut [f32]) {
    let mut s = seed | 1;
    for v in out.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 33) as u32;
        let mag = (u & 0x7FFF_FFFF) as f32 / 4_194_304.0; // 0 ..= 512
        let bits = (u & 0x8000_0000) | (mag.to_bits() & 0x7FFF_FFFF);
        *v = f32::from_bits(bits);
    }
}

/// One family measurement: 2 s of launch-loop warmup, then BLOCKS timed
/// blocks of REPS launches with a fresh H2D-irrigated input before each
/// block (the triad re-irrigation). Returns the best block's per-launch
/// average in milliseconds.
fn measure<B>(
    ctx: &CudaContext,
    stream: &CudaStream,
    ev_start: &cuda_core::simt::CudaEvent,
    ev_stop: &cuda_core::simt::CudaEvent,
    fresh_input: &mut dyn FnMut() -> Result<B, Box<dyn std::error::Error>>,
    launch: &mut dyn FnMut(&B) -> Result<(), Box<dyn std::error::Error>>,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Warmup on the first block's input until the clocks settle.
    let first = fresh_input()?;
    let start = std::time::Instant::now();
    while start.elapsed().as_secs() < WARMUP_SECS {
        launch(&first)?;
    }
    stream.synchronize()?;
    drop(first);

    let mut best_ms = f64::INFINITY;
    for _ in 0..BLOCKS {
        let input = fresh_input()?;
        stream.synchronize()?;
        ev_start.record(stream)?;
        for _ in 0..REPS {
            launch(&input)?;
        }
        ev_stop.record(stream)?;
        stream.synchronize()?;
        let ms = ev_start.elapsed_ms(&ev_stop)? as f64 / REPS as f64;
        if ms < best_ms {
            best_ms = ms;
        }
        drop(input);
    }
    Ok(best_ms)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU FP8/FP4 conversion throughput benchmark ===");
    println!(
        "working set: {} Mi elements/direction (512 MiB f32 side); \
         {} blocks x {} reps, best block; warmup {}s\n",
        ELEMENTS >> 20,
        BLOCKS,
        REPS,
        WARMUP_SECS
    );

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let ev_start = ctx.new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let ev_stop = ctx.new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;

    let grid = LaunchConfig {
        grid_dim: ((ELEMENTS as u32 + THREADS - 1) / THREADS, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let packed_elements = ELEMENTS / 2;
    let packed_grid = LaunchConfig {
        grid_dim: ((packed_elements as u32 + THREADS - 1) / THREADS, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    // Host-side sources; `fresh_input` uploads them per block (H2D
    // re-irrigation).
    let mut host_u8 = vec![0u8; ELEMENTS];
    fill_byte_spread(0x9E37_79B9, &mut host_u8);
    let mut host_f32_a = vec![0f32; ELEMENTS];
    fill_f32_spread(0x51ED_270B, &mut host_f32_a);
    let mut host_f32_b = vec![0f32; ELEMENTS];
    fill_f32_spread(0x2545_F491, &mut host_f32_b);
    let mut host_u16 = vec![0u16; packed_elements];
    fill_byte_spread(0x7F4A_7C15, unsafe {
        std::slice::from_raw_parts_mut(host_u16.as_mut_ptr().cast::<u8>(), host_u16.len() * 2)
    });

    // Nominal DRAM traffic per rep, in bytes.
    let copy_traffic = (ELEMENTS * 8) as f64;
    let conv_traffic = (ELEMENTS * 5) as f64;
    let packed_conv_traffic = (ELEMENTS * 5) as f64;

    let mut output_f32 = DeviceBuffer::<f32>::zeroed(&stream, ELEMENTS)?;
    let mut output_u8 = DeviceBuffer::<u8>::zeroed(&stream, ELEMENTS)?;
    let mut output_u16 = DeviceBuffer::<u16>::zeroed(&stream, packed_elements)?;
    let mut out_x = DeviceBuffer::<f32>::zeroed(&stream, packed_elements)?;
    let mut out_y = DeviceBuffer::<f32>::zeroed(&stream, packed_elements)?;

    let mut results: Vec<(&'static str, f64, f64)> = Vec::new(); // name, GB/s, ms

    // ---- reference: same-footprint copy -----------------------------------
    {
        let ms = measure(
            &ctx,
            &stream,
            &ev_start,
            &ev_stop,
            &mut || Ok(DeviceBuffer::from_host(&stream, &host_f32_a)?),
            &mut |input| {
                // SAFETY: launch shape covers the ELEMENTS-element buffers.
                unsafe { module.copy_f32(&stream, grid, input, &mut output_f32) }
                    .map_err(Into::into)
            },
        )?;
        results.push(("copy_f32 (reference)", copy_traffic / ms / 1e6, ms));
    }

    // ---- scalar decode / encode -------------------------------------------
    {
        let ms = measure(
            &ctx,
            &stream,
            &ev_start,
            &ev_stop,
            &mut || Ok(DeviceBuffer::from_host(&stream, &host_u8)?),
            &mut |input| {
                // SAFETY: launch shape covers the buffers.
                unsafe { module.decode_e4m3(&stream, grid, input, &mut output_f32) }
                    .map_err(Into::into)
            },
        )?;
        results.push(("decode_e4m3", conv_traffic / ms / 1e6, ms));
    }
    {
        let ms = measure(
            &ctx,
            &stream,
            &ev_start,
            &ev_stop,
            &mut || Ok(DeviceBuffer::from_host(&stream, &host_u8)?),
            &mut |input| {
                // SAFETY: launch shape covers the buffers.
                unsafe { module.decode_e5m2(&stream, grid, input, &mut output_f32) }
                    .map_err(Into::into)
            },
        )?;
        results.push(("decode_e5m2", conv_traffic / ms / 1e6, ms));
    }
    {
        let ms = measure(
            &ctx,
            &stream,
            &ev_start,
            &ev_stop,
            &mut || Ok(DeviceBuffer::from_host(&stream, &host_f32_a)?),
            &mut |input| {
                // SAFETY: launch shape covers the buffers.
                unsafe { module.encode_e4m3(&stream, grid, input, &mut output_u8) }
                    .map_err(Into::into)
            },
        )?;
        results.push(("encode_e4m3", conv_traffic / ms / 1e6, ms));
    }
    {
        let ms = measure(
            &ctx,
            &stream,
            &ev_start,
            &ev_stop,
            &mut || Ok(DeviceBuffer::from_host(&stream, &host_f32_a)?),
            &mut |input| {
                // SAFETY: launch shape covers the buffers.
                unsafe { module.encode_e5m2(&stream, grid, input, &mut output_u8) }
                    .map_err(Into::into)
            },
        )?;
        results.push(("encode_e5m2", conv_traffic / ms / 1e6, ms));
    }

    // ---- packed x2 families -------------------------------------------------
    {
        let ms = measure(
            &ctx,
            &stream,
            &ev_start,
            &ev_stop,
            &mut || Ok(DeviceBuffer::from_host(&stream, &host_u16)?),
            &mut |input| {
                // SAFETY: launch shape covers the buffers.
                unsafe { module.decode_e4m3x2(&stream, packed_grid, input, &mut out_x, &mut out_y) }
                    .map_err(Into::into)
            },
        )?;
        results.push(("decode_e4m3x2 (packed)", packed_conv_traffic / ms / 1e6, ms));
    }
    {
        let ms = measure(
            &ctx,
            &stream,
            &ev_start,
            &ev_stop,
            &mut || Ok(DeviceBuffer::from_host(&stream, &host_u16)?),
            &mut |input| {
                // SAFETY: launch shape covers the buffers.
                unsafe { module.decode_e5m2x2(&stream, packed_grid, input, &mut out_x, &mut out_y) }
                    .map_err(Into::into)
            },
        )?;
        results.push(("decode_e5m2x2 (packed)", packed_conv_traffic / ms / 1e6, ms));
    }
    {
        let ms = measure(
            &ctx,
            &stream,
            &ev_start,
            &ev_stop,
            &mut || {
                // Both sides re-irrigated together (two H2D uploads).
                let a = DeviceBuffer::from_host(&stream, &host_f32_a)?;
                let b = DeviceBuffer::from_host(&stream, &host_f32_b)?;
                Ok((a, b))
            },
            &mut |pair: &(DeviceBuffer<f32>, DeviceBuffer<f32>)| {
                // SAFETY: launch shape covers the buffers.
                unsafe {
                    module.encode_e4m3x2(&stream, packed_grid, &pair.0, &pair.1, &mut output_u16)
                }
                .map_err(Into::into)
            },
        )?;
        results.push(("encode_e4m3x2 (packed)", packed_conv_traffic / ms / 1e6, ms));
    }
    {
        let ms = measure(
            &ctx,
            &stream,
            &ev_start,
            &ev_stop,
            &mut || {
                let a = DeviceBuffer::from_host(&stream, &host_f32_a)?;
                let b = DeviceBuffer::from_host(&stream, &host_f32_b)?;
                Ok((a, b))
            },
            &mut |pair: &(DeviceBuffer<f32>, DeviceBuffer<f32>)| {
                // SAFETY: launch shape covers the buffers.
                unsafe {
                    module.encode_e5m2x2(&stream, packed_grid, &pair.0, &pair.1, &mut output_u16)
                }
                .map_err(Into::into)
            },
        )?;
        results.push(("encode_e5m2x2 (packed)", packed_conv_traffic / ms / 1e6, ms));
    }

    println!("\nresults (best block, nominal traffic):");
    let copy_gbps = results[0].1;
    for (name, gbps, ms) in &results {
        let grade = gbps / copy_gbps * 100.0;
        println!("  {name:26} {gbps:8.1} GB/s  ({ms:7.3} ms)  grade {grade:5.1}%");
    }

    Ok(())
}
