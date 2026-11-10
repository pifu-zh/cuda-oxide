/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Software FP8/FP4 <-> f32 conversion test (RDNA2 has no fp8/fp4 hardware).
//!
//! Scalar families: the kernels exhaustively decode every E4M3/E5M2 byte
//! (256 each) and every E2M1 nibble (16), and encode a host-generated spread
//! of f32 test values (boundaries, exact RNE midpoints, one-ulp-off cases,
//! specials) with the three encoders.
//!
//! Packed x2 families (CUDA `__nv_fp8x2_*` / `__nv_fp4x2_*` semantics, lane 0
//! in the low byte/nibble): every packed pattern is decoded exhaustively
//! (2^16 for each FP8 pair type, 2^8 for FP4 x2), every packed pattern is
//! round-tripped decode->encode against the per-lane host reference (which
//! also pins the NaN canonicalization), and the RNE midpoint/boundary spread
//! is encoded as lane-mixed pairs (forward x mirrored) so a lane swap or
//! wrong shift cannot hide.
//!
//! The host cross-checks everything against independent reference
//! implementations and reports PASS/FAIL per family.
//!
//! Run: cargo oxide run fp8_decode

use cuda_device::fp8::{
    e2m1_to_f32, e2m1x2_to_f32x2, e4m3_to_f32, e4m3x2_to_f32x2, e4m3x4_to_f32x4, e5m2_to_f32,
    e5m2x2_to_f32x2, e5m2x4_to_f32x4, f32_to_e2m1_rne_satfinite, f32_to_e4m3_rne_satfinite,
    f32_to_e5m2_rne, f32x2_to_e2m1x2_rne_satfinite, f32x2_to_e4m3x2_rne_satfinite,
    f32x2_to_e5m2x2_rne, f32x4_to_e4m3x4_rne_satfinite, f32x4_to_e5m2x4_rne,
};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Decodes every E4M3 byte pattern (idx 0..256) to f32.
    #[kernel]
    pub fn decode_e4m3(mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = e4m3_to_f32(idx as u8);
        }
    }

    /// Decodes every E5M2 byte pattern (idx 0..256) to f32.
    #[kernel]
    pub fn decode_e5m2(mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = e5m2_to_f32(idx as u8);
        }
    }

    /// Decodes every E2M1 nibble pattern (idx 0..16) to f32.
    #[kernel]
    pub fn decode_e2m1(mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = e2m1_to_f32(idx as u8);
        }
    }

    /// Encodes N f32 test values with the E4M3 encoder (RNE + satfinite).
    #[kernel]
    pub fn encode_e4m3(values: &[f32], mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f32_to_e4m3_rne_satfinite(values[idx]);
        }
    }

    /// Encodes N f32 test values with the E5M2 encoder (RNE, overflow -> Inf).
    #[kernel]
    pub fn encode_e5m2(values: &[f32], mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f32_to_e5m2_rne(values[idx]);
        }
    }

    /// Encodes N f32 test values with the E2M1 encoder (RNE + satfinite).
    #[kernel]
    pub fn encode_e2m1(values: &[f32], mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f32_to_e2m1_rne_satfinite(values[idx]);
        }
    }

    // ---- packed x2 families ------------------------------------------------

    /// Decodes every E4M3 x2 pattern (idx 0..65536) into the (x, y) lanes.
    #[kernel]
    pub fn decode_e4m3x2(
        values: &[u16],
        mut out_x: DisjointSlice<f32>,
        mut out_y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d().get();
        let (x, y) = e4m3x2_to_f32x2(values[idx]);
        if let Some(o) = out_x.get_mut(thread::index_1d()) {
            *o = x;
        }
        if let Some(o) = out_y.get_mut(thread::index_1d()) {
            *o = y;
        }
    }

    /// Decodes every E5M2 x2 pattern (idx 0..65536) into the (x, y) lanes.
    #[kernel]
    pub fn decode_e5m2x2(
        values: &[u16],
        mut out_x: DisjointSlice<f32>,
        mut out_y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d().get();
        let (x, y) = e5m2x2_to_f32x2(values[idx]);
        if let Some(o) = out_x.get_mut(thread::index_1d()) {
            *o = x;
        }
        if let Some(o) = out_y.get_mut(thread::index_1d()) {
            *o = y;
        }
    }

    /// Decodes every E2M1 x2 pattern (idx 0..256) into the (x, y) lanes.
    #[kernel]
    pub fn decode_e2m1x2(
        values: &[u8],
        mut out_x: DisjointSlice<f32>,
        mut out_y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d().get();
        let (x, y) = e2m1x2_to_f32x2(values[idx]);
        if let Some(o) = out_x.get_mut(thread::index_1d()) {
            *o = x;
        }
        if let Some(o) = out_y.get_mut(thread::index_1d()) {
            *o = y;
        }
    }

    /// Round-trips every E4M3 x2 pattern: decode -> encode (per-lane RNE +
    /// satfinite; NaN lanes canonicalize).
    #[kernel]
    pub fn roundtrip_e4m3x2(values: &[u16], mut out: DisjointSlice<u16>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            let (x, y) = e4m3x2_to_f32x2(values[idx]);
            *o = f32x2_to_e4m3x2_rne_satfinite(x, y);
        }
    }

    /// Round-trips every E5M2 x2 pattern: decode -> encode (per-lane RNE,
    /// IEEE overflow; NaN lanes canonicalize).
    #[kernel]
    pub fn roundtrip_e5m2x2(values: &[u16], mut out: DisjointSlice<u16>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            let (x, y) = e5m2x2_to_f32x2(values[idx]);
            *o = f32x2_to_e5m2x2_rne(x, y);
        }
    }

    /// Round-trips every E2M1 x2 pattern: decode -> encode (per-lane RNE +
    /// satfinite).
    #[kernel]
    pub fn roundtrip_e2m1x2(values: &[u8], mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            let (x, y) = e2m1x2_to_f32x2(values[idx]);
            *o = f32x2_to_e2m1x2_rne_satfinite(x, y);
        }
    }

    /// Encodes (a[i], b[i]) f32 pairs into packed E4M3 x2. The host pairs the
    /// midpoint spread forward x mirrored so lane mix-ups cannot hide.
    #[kernel]
    pub fn encode_pairs_e4m3x2(a: &[f32], b: &[f32], mut out: DisjointSlice<u16>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f32x2_to_e4m3x2_rne_satfinite(a[idx], b[idx]);
        }
    }

    /// Encodes (a[i], b[i]) f32 pairs into packed E5M2 x2.
    #[kernel]
    pub fn encode_pairs_e5m2x2(a: &[f32], b: &[f32], mut out: DisjointSlice<u16>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f32x2_to_e5m2x2_rne(a[idx], b[idx]);
        }
    }

    /// Encodes (a[i], b[i]) f32 pairs into packed E2M1 x2.
    #[kernel]
    pub fn encode_pairs_e2m1x2(a: &[f32], b: &[f32], mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f32x2_to_e2m1x2_rne_satfinite(a[idx], b[idx]);
        }
    }

    // ---- packed x4 families ------------------------------------------------
    // The x4 form composes two x2 halves; the x2 layers are exhaustively
    // verified above, so here a sampled sweep (stride over the 2^32 space
    // plus an all-byte-patterns block in the low half) checks only the
    // composition: byte order (x lowest, w highest) and shift placement.

    /// Decodes sampled u32 patterns into the four f32 lanes.
    #[kernel]
    pub fn decode_e4m3x4(
        values: &[u32],
        mut out0: DisjointSlice<f32>,
        mut out1: DisjointSlice<f32>,
        mut out2: DisjointSlice<f32>,
        mut out3: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d().get();
        let (x, y, z, w) = e4m3x4_to_f32x4(values[idx]);
        if let Some(o) = out0.get_mut(thread::index_1d()) {
            *o = x;
        }
        if let Some(o) = out1.get_mut(thread::index_1d()) {
            *o = y;
        }
        if let Some(o) = out2.get_mut(thread::index_1d()) {
            *o = z;
        }
        if let Some(o) = out3.get_mut(thread::index_1d()) {
            *o = w;
        }
    }

    /// Decodes sampled u32 patterns into the four f32 lanes.
    #[kernel]
    pub fn decode_e5m2x4(
        values: &[u32],
        mut out0: DisjointSlice<f32>,
        mut out1: DisjointSlice<f32>,
        mut out2: DisjointSlice<f32>,
        mut out3: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d().get();
        let (x, y, z, w) = e5m2x4_to_f32x4(values[idx]);
        if let Some(o) = out0.get_mut(thread::index_1d()) {
            *o = x;
        }
        if let Some(o) = out1.get_mut(thread::index_1d()) {
            *o = y;
        }
        if let Some(o) = out2.get_mut(thread::index_1d()) {
            *o = z;
        }
        if let Some(o) = out3.get_mut(thread::index_1d()) {
            *o = w;
        }
    }

    /// Encodes four f32 streams into packed E4M3 x4.
    #[kernel]
    pub fn encode_e4m3x4(a: &[f32], b: &[f32], c: &[f32], d: &[f32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f32x4_to_e4m3x4_rne_satfinite(a[idx], b[idx], c[idx], d[idx]);
        }
    }

    /// Encodes four f32 streams into packed E5M2 x4.
    #[kernel]
    pub fn encode_e5m2x4(a: &[f32], b: &[f32], c: &[f32], d: &[f32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d().get();
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f32x4_to_e5m2x4_rne(a[idx], b[idx], c[idx], d[idx]);
        }
    }
}

// ---------------------------------------------------------------------------
// Host reference implementations (independent of the device code: the decode
// builds f32 fields arithmetically; the encode walks the decode table with
// explicit distance/tie logic).
// ---------------------------------------------------------------------------

fn host_e4m3_to_f32(bits: u8) -> f32 {
    let s = (bits >> 7) & 1;
    let e = (bits >> 3) & 0xF;
    let m = bits & 0x7;
    let mag = if e == 0 {
        (m as f32) * 2f32.powi(-9)
    } else if e == 0xF && m == 0x7 {
        f32::NAN
    } else {
        2f32.powi(e as i32 - 7) * (1.0 + (m as f32) / 8.0)
    };
    if s == 1 { -mag } else { mag }
}

fn host_e5m2_to_f32(bits: u8) -> f32 {
    let s = (bits >> 7) & 1;
    let e = (bits >> 2) & 0x1F;
    let m = bits & 0x3;
    let mag = if e == 0 {
        (m as f32) * 2f32.powi(-16)
    } else if e == 0x1F && m != 0 {
        f32::NAN
    } else if e == 0x1F {
        f32::INFINITY
    } else {
        2f32.powi(e as i32 - 15) * (1.0 + (m as f32) / 4.0)
    };
    if s == 1 { -mag } else { mag }
}

fn host_e2m1_to_f32(bits: u8) -> f32 {
    let s = (bits >> 3) & 1;
    let e = (bits >> 1) & 0x3;
    let m = bits & 0x1;
    // Denormal (e == 0): m * 2^0 without the implicit leading one —
    // i.e. 0.0 / 0.5. Normals (e >= 1): 2^(e-1) * (1 + m/2).
    let mag = match (e, m) {
        (0, 0) => 0.0,
        (0, 1) => 0.5,
        _ => 2f32.powi(e as i32 - 1) * (1.0 + (m as f32) / 2.0),
    };
    if s == 1 { -mag } else { mag }
}

/// Host RNE encode reference: nearest pattern by decoded value, ties to the
/// even pattern (pattern bit 0). NaN/Inf inputs handled per the same rules
/// the device encoders document.
fn host_encode_reference(
    x: f32,
    mags: &[(u32, f32)], // (low-bit pattern, magnitude) sorted ascending
    sat_max: f32,        // max finite magnitude (f32::INFINITY = never clamp)
    nan_byte: u8,
    inf_byte: u8,
    sign_shift: u32, // 7 for the byte formats, 3 for the E2M1 nibble
    inf_threshold: f32, // at/above this, RNE rounds to Inf (E5M2); the
                     // satfinite families pass f32::INFINITY (never)
) -> u8 {
    let s = (x.to_bits() >> 31) as u8;
    if x.is_nan() {
        return (s << sign_shift) | nan_byte;
    }
    if x.is_infinite() {
        return (s << sign_shift) | inf_byte;
    }
    let ax = x.abs();
    // Overflow: at/above the RNE threshold the result leaves the finite
    // range (E5M2 -> Inf); the satfinite families clamp to the max finite
    // from max_finite upward (their threshold is infinity = never).
    if inf_threshold.is_finite() && ax >= inf_threshold {
        return (s << sign_shift) | inf_byte;
    }
    if sat_max.is_finite() && ax >= sat_max {
        return (s << sign_shift) | (mags.last().unwrap().0 as u8);
    }
    // Nearest over the sorted magnitude list; ties to the even pattern.
    let mut best = mags[0].0 as u8;
    for (i, (pat, val)) in mags.iter().enumerate() {
        match mags.get(i + 1) {
            Some((npat, nval)) if ax > *val && ax < *nval => {
                let da = ax - val;
                let db = nval - ax;
                best = if da < db {
                    *pat as u8
                } else if db < da {
                    *npat as u8
                } else if pat & 1 == 0 {
                    *pat as u8 // tie: even LSB
                } else {
                    *npat as u8
                };
                break;
            }
            Some(_) => {
                if (ax - val).abs() <= f32::EPSILON * val.abs().max(1.0) {
                    best = *pat as u8; // exact hit
                    break;
                }
            }
            None => best = *pat as u8, // at/above the last finite
        }
    }
    (best & 0x7F) | (s << sign_shift)
}

fn build_e4m3_table() -> Vec<(u32, f32)> {
    let mut t: Vec<(u32, f32)> = (0u16..256)
        .map(|b| b as u8)
        .filter(|b| (*b & 0x7F) != 0x7F) // drop the NaN byte
        .map(|b| (u32::from(b & 0x7F), host_e4m3_to_f32(b).abs()))
        .collect();
    t.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    t
}

fn build_e5m2_table() -> Vec<(u32, f32)> {
    let mut t: Vec<(u32, f32)> = (0u16..256)
        .map(|b| b as u8)
        .filter(|b| {
            let v = host_e5m2_to_f32(*b);
            !v.is_nan() && !v.is_infinite()
        })
        .map(|b| (u32::from(b & 0x7F), host_e5m2_to_f32(b).abs()))
        .collect();
    t.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    t
}

fn build_e2m1_table() -> Vec<(u32, f32)> {
    let mut t: Vec<(u32, f32)> = (0u16..16)
        .map(|b| b as u8)
        .map(|b| (u32::from(b & 0x7), host_e2m1_to_f32(b).abs()))
        .collect();
    t.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    t
}

/// The f32 test spread: boundaries, exact midpoints between adjacent
/// candidates (RNE ties), one-ulp-off cases, specials. Generated from the
/// decode tables so the midpoints are exact.
fn generate_test_values(table: &[(u32, f32)], include_specials: bool) -> Vec<f32> {
    let mut v: Vec<f32> = Vec::new();
    for w in table.windows(2) {
        let a = w[0].1;
        let b = w[1].1;
        if a == 0.0 || b == 0.0 {
            v.push(0.0);
            continue;
        }
        let mid = a + (b - a) / 2.0;
        v.push(mid);
        // One ulp toward each side of the midpoint (round-away cases).
        let up = f32::from_bits(mid.to_bits() + 1);
        let down = f32::from_bits(mid.to_bits() - 1);
        v.push(up);
        v.push(down);
        v.push(a);
        v.push(b);
    }
    if include_specials {
        v.push(0.0);
        v.push(f32::INFINITY);
        v.push(f32::NEG_INFINITY);
        v.push(f32::NAN);
        v.push(f32::MAX);
        v.push(-f32::MAX);
        v.push(f32::MIN_POSITIVE);
        v.push(448.0);
        v.push(464.0);
        v.push(480.0);
        v.push(6.0);
        v.push(7.0);
        v.push(8.0);
    }
    // Negatives of everything (RNE is symmetric).
    let negs: Vec<f32> = v.iter().map(|x| -x).collect();
    v.extend(negs);
    v
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU FP8/FP4 Software Decode Test ===\n");

    let ctx = cuda_core::CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let mut failures = 0usize;
    let mut total = 0usize;

    // ---- Family 1: E4M3 decode (256 patterns, exhaustive) ----
    {
        let mut out = cuda_core::DeviceBuffer::<f32>::zeroed(&stream, 256)?;
        // SAFETY: launch shape covers the 256-element output.
        unsafe {
            module.decode_e4m3(
                (stream).as_ref(),
                cuda_core::simt::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut out,
            )
        }?;
        stream.synchronize()?;
        let got = out.to_host_vec(&stream)?;
        for b in 0u16..256 {
            let want = host_e4m3_to_f32(b as u8);
            let gotv = got[b as usize];
            total += 1;
            if gotv.to_bits() != want.to_bits() {
                failures += 1;
                if failures <= 4 {
                    println!(
                        "E4M3 decode FAIL: pattern {b:#04x}: got {} ({:#x}) want {} ({:#x})",
                        gotv,
                        gotv.to_bits(),
                        want,
                        want.to_bits()
                    );
                }
            }
        }
        println!("E4M3 decode: {} patterns checked", 256);
    }

    // ---- Family 2: E5M2 decode (256 patterns, exhaustive) ----
    {
        let mut out = cuda_core::DeviceBuffer::<f32>::zeroed(&stream, 256)?;
        // SAFETY: launch shape covers the 256-element output.
        unsafe {
            module.decode_e5m2(
                (stream).as_ref(),
                cuda_core::simt::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut out,
            )
        }?;
        stream.synchronize()?;
        let got = out.to_host_vec(&stream)?;
        for b in 0u16..256 {
            let want = host_e5m2_to_f32(b as u8);
            let gotv = got[b as usize];
            total += 1;
            if gotv.to_bits() != want.to_bits() && !(gotv.is_nan() && want.is_nan()) {
                failures += 1;
                if failures <= 8 {
                    println!(
                        "E5M2 decode FAIL: pattern {b:#04x}: got {} ({:#x}) want {} ({:#x})",
                        gotv,
                        gotv.to_bits(),
                        want,
                        want.to_bits()
                    );
                }
            }
        }
        println!("E5M2 decode: {} patterns checked", 256);
    }

    // ---- Family 3: E2M1 decode (16 patterns, exhaustive) ----
    {
        let mut out = cuda_core::DeviceBuffer::<f32>::zeroed(&stream, 16)?;
        // SAFETY: launch shape covers the 16-element output.
        unsafe {
            module.decode_e2m1(
                (stream).as_ref(),
                cuda_core::simt::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (16, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut out,
            )
        }?;
        stream.synchronize()?;
        let got = out.to_host_vec(&stream)?;
        for b in 0u16..16 {
            let want = host_e2m1_to_f32(b as u8);
            let gotv = got[b as usize];
            total += 1;
            if gotv.to_bits() != want.to_bits() {
                failures += 1;
                println!(
                    "E2M1 decode FAIL: pattern {b:#04x}: got {} want {}",
                    gotv, want
                );
            }
        }
        println!("E2M1 decode: {} patterns checked", 16);
    }

    // ---- Families 4-6: encodes over the generated spread ----
    let e4_table = build_e4m3_table();
    let e5_table = build_e5m2_table();
    let e2_table = build_e2m1_table();

    let cases: [(&str, &[(u32, f32)]); 3] = [
        ("E4M3", &e4_table),
        ("E5M2", &e5_table),
        ("E2M1", &e2_table),
    ];

    for (name, mags) in cases {
        let values = generate_test_values(mags, true);
        let n = values.len();
        let values_dev = cuda_core::DeviceBuffer::from_host(&stream, &values)?;
        let mut out = cuda_core::DeviceBuffer::<u8>::zeroed(&stream, n)?;
        let cfg = cuda_core::simt::LaunchConfig {
            grid_dim: (((n as u32) + 1023) / 1024, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: launch shape covers the n-element buffers.
        match name {
            "E4M3" => unsafe { module.encode_e4m3(&stream, cfg, &values_dev, &mut out) }?,
            "E5M2" => unsafe { module.encode_e5m2(&stream, cfg, &values_dev, &mut out) }?,
            _ => unsafe { module.encode_e2m1(&stream, cfg, &values_dev, &mut out) }?,
        }
        stream.synchronize()?;
        let got = out.to_host_vec(&stream)?;

        let (nan_byte, inf_byte, sat_max, sign_shift, inf_threshold) = match name {
            "E4M3" => (0x7Fu8, 0x7Fu8, 448.0f32, 7u32, f32::INFINITY),
            "E5M2" => (0x7Eu8, 0x7Cu8, f32::INFINITY, 7u32, 61440.0f32),
            _ => (0x07u8, 0x07u8, 6.0f32, 3u32, f32::INFINITY),
        };
        let mut fam_fail = 0;
        for (i, x) in values.iter().enumerate() {
            let want = host_encode_reference(
                *x,
                mags,
                sat_max,
                nan_byte,
                inf_byte,
                sign_shift,
                inf_threshold,
            );
            let gotv = got[i];
            total += 1;
            // NaN results compare by NaN-ness (payload free).
            // Both implementations emit fixed bytes for every input
            // (NaN payloads included), so byte-exact comparison is sound.
            let ok = gotv == want;
            if !ok {
                fam_fail += 1;
                failures += 1;
                if fam_fail <= 4 {
                    println!(
                        "{} encode FAIL: {} ({:#x}): got {:#04x} want {:#04x}",
                        name,
                        x,
                        x.to_bits(),
                        gotv,
                        want
                    );
                }
            }
        }
        println!(
            "{} encode: {} values checked, {} failures",
            name, n, fam_fail
        );
    }

    // ---- Packed x2 families ------------------------------------------------
    // Lane convention under test: lane 0 (x) in the low byte (low nibble for
    // FP4), lane 1 (y) in the high byte (high nibble) — the CUDA
    // __nv_fp8x2_* / __nv_fp4x2_* layout.

    /// Host packed-encode reference: per-lane scalar reference, packed.
    /// `hi_shift` is 8 for the FP8 x2 pairs (y in bits [15:8]) and 4 for the
    /// FP4 x2 pairs (y nibble in bits [7:4]).
    fn pack_ref(
        x: f32,
        y: f32,
        mags: &[(u32, f32)],
        sat_max: f32,
        nan_byte: u8,
        inf_byte: u8,
        sign_shift: u32,
        inf_threshold: f32,
        hi_shift: u32,
    ) -> u16 {
        let lo = host_encode_reference(
            x,
            mags,
            sat_max,
            nan_byte,
            inf_byte,
            sign_shift,
            inf_threshold,
        );
        let hi = host_encode_reference(
            y,
            mags,
            sat_max,
            nan_byte,
            inf_byte,
            sign_shift,
            inf_threshold,
        );
        u16::from(lo) | (u16::from(hi) << hi_shift)
    }

    let wide_cfg = cuda_core::simt::LaunchConfig {
        grid_dim: ((65536u32 + 1023) / 1024, 1, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: 0,
    };

    // ---- Family 7-8: E4M3x2/E5M2x2 decode (all 2^16 patterns, exhaustive) --
    for family in ["E4M3x2", "E5M2x2"] {
        let patterns: Vec<u16> = (0u32..65536).map(|v| v as u16).collect();
        let values_dev = cuda_core::DeviceBuffer::from_host(&stream, &patterns)?;
        let mut out_x = cuda_core::DeviceBuffer::<f32>::zeroed(&stream, 65536)?;
        let mut out_y = cuda_core::DeviceBuffer::<f32>::zeroed(&stream, 65536)?;
        // SAFETY: launch shape covers the 65536-element buffers.
        if family == "E4M3x2" {
            unsafe {
                module.decode_e4m3x2(&stream, wide_cfg, &values_dev, &mut out_x, &mut out_y)
            }?;
        } else {
            unsafe {
                module.decode_e5m2x2(&stream, wide_cfg, &values_dev, &mut out_x, &mut out_y)
            }?;
        }
        stream.synchronize()?;
        let gx = out_x.to_host_vec(&stream)?;
        let gy = out_y.to_host_vec(&stream)?;
        let mut fam_fail = 0;
        for v in 0u32..65536 {
            let lo = (v & 0xFF) as u8;
            let hi = (v >> 8) as u8;
            let (want_x, want_y) = if family == "E4M3x2" {
                (host_e4m3_to_f32(lo), host_e4m3_to_f32(hi))
            } else {
                (host_e5m2_to_f32(lo), host_e5m2_to_f32(hi))
            };
            total += 2;
            if gx[v as usize].to_bits() != want_x.to_bits() {
                fam_fail += 1;
                failures += 1;
                if fam_fail <= 4 {
                    println!(
                        "{family} decode FAIL x-lane: {v:#06x}: got {} want {}",
                        gx[v as usize], want_x
                    );
                }
            }
            if gy[v as usize].to_bits() != want_y.to_bits() {
                fam_fail += 1;
                failures += 1;
                if fam_fail <= 4 {
                    println!(
                        "{family} decode FAIL y-lane: {v:#06x}: got {} want {}",
                        gy[v as usize], want_y
                    );
                }
            }
        }
        println!("{family} decode: 65536 patterns x 2 lanes checked, {fam_fail} failures");
    }

    // ---- Family 9: E2M1x2 decode (all 2^8 patterns, exhaustive) ------------
    {
        let patterns: Vec<u8> = (0u16..256).map(|v| v as u8).collect();
        let values_dev = cuda_core::DeviceBuffer::from_host(&stream, &patterns)?;
        let mut out_x = cuda_core::DeviceBuffer::<f32>::zeroed(&stream, 256)?;
        let mut out_y = cuda_core::DeviceBuffer::<f32>::zeroed(&stream, 256)?;
        let cfg = cuda_core::simt::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: launch shape covers the 256-element buffers.
        unsafe { module.decode_e2m1x2(&stream, cfg, &values_dev, &mut out_x, &mut out_y) }?;
        stream.synchronize()?;
        let gx = out_x.to_host_vec(&stream)?;
        let gy = out_y.to_host_vec(&stream)?;
        let mut fam_fail = 0;
        for v in 0u16..256 {
            let want_x = host_e2m1_to_f32((v & 0xF) as u8);
            let want_y = host_e2m1_to_f32((v >> 4) as u8);
            total += 2;
            if gx[v as usize].to_bits() != want_x.to_bits() {
                fam_fail += 1;
                failures += 1;
                println!("{v:#04x} x-lane: got {} want {}", gx[v as usize], want_x);
            }
            if gy[v as usize].to_bits() != want_y.to_bits() {
                fam_fail += 1;
                failures += 1;
                println!("{v:#04x} y-lane: got {} want {}", gy[v as usize], want_y);
            }
        }
        println!("E2M1x2 decode: 256 patterns x 2 lanes checked, {fam_fail} failures");
    }

    // ---- Families 10-12: packed round-trip over the FULL pattern space -----
    // Every pattern decodes and re-encodes; the expectation is the per-lane
    // host reference of the decoded lanes (not identity), which also pins
    // NaN canonicalization (E5M2 payload NaNs -> 0x7E|s).
    for (name, mags, sat_max, nan_byte, inf_byte, sign_shift, inf_threshold, fp4) in [
        (
            "E4M3x2",
            &e4_table,
            448.0f32,
            0x7Fu8,
            0x7Fu8,
            7u32,
            f32::INFINITY,
            false,
        ),
        (
            "E5M2x2",
            &e5_table,
            f32::INFINITY,
            0x7Eu8,
            0x7Cu8,
            7u32,
            61440.0f32,
            false,
        ),
        (
            "E2M1x2",
            &e2_table,
            6.0f32,
            0x07u8,
            0x07u8,
            3u32,
            f32::INFINITY,
            true,
        ),
    ] {
        let is_e4 = name == "E4M3x2";
        let is_e5 = name == "E5M2x2";
        let (n, wide) = if fp4 {
            (256usize, false)
        } else {
            (65536, true)
        };
        let cfg = if wide {
            wide_cfg
        } else {
            cuda_core::simt::LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }
        };
        let mut fam_fail = 0;
        if fp4 {
            let patterns: Vec<u8> = (0u16..256).map(|v| v as u8).collect();
            let values_dev = cuda_core::DeviceBuffer::from_host(&stream, &patterns)?;
            let mut out = cuda_core::DeviceBuffer::<u8>::zeroed(&stream, n)?;
            // SAFETY: launch shape covers the 256-element buffers.
            unsafe { module.roundtrip_e2m1x2(&stream, cfg, &values_dev, &mut out) }?;
            stream.synchronize()?;
            let got = out.to_host_vec(&stream)?;
            for v in 0u16..256 {
                let (x, y) = e2m1x2_to_f32x2(v as u8);
                let want = pack_ref(
                    x,
                    y,
                    mags,
                    sat_max,
                    nan_byte,
                    inf_byte,
                    sign_shift,
                    inf_threshold,
                    4,
                ) as u8;
                total += 1;
                if got[v as usize] != want {
                    fam_fail += 1;
                    failures += 1;
                    if fam_fail <= 4 {
                        println!(
                            "{name} roundtrip FAIL: {v:#04x}: got {:#04x} want {:#04x}",
                            got[v as usize], want
                        );
                    }
                }
            }
        } else {
            let patterns: Vec<u16> = (0u32..65536).map(|v| v as u16).collect();
            let values_dev = cuda_core::DeviceBuffer::from_host(&stream, &patterns)?;
            let mut out = cuda_core::DeviceBuffer::<u16>::zeroed(&stream, n)?;
            // SAFETY: launch shape covers the 65536-element buffers.
            if is_e4 {
                unsafe { module.roundtrip_e4m3x2(&stream, wide_cfg, &values_dev, &mut out) }?;
            } else if is_e5 {
                unsafe { module.roundtrip_e5m2x2(&stream, wide_cfg, &values_dev, &mut out) }?;
            }
            stream.synchronize()?;
            let got = out.to_host_vec(&stream)?;
            for v in 0u32..65536 {
                let bits = v as u16;
                let (x, y) = if is_e4 {
                    e4m3x2_to_f32x2(bits)
                } else {
                    e5m2x2_to_f32x2(bits)
                };
                let want = pack_ref(
                    x,
                    y,
                    mags,
                    sat_max,
                    nan_byte,
                    inf_byte,
                    sign_shift,
                    inf_threshold,
                    8,
                );
                total += 1;
                if got[v as usize] != want {
                    fam_fail += 1;
                    failures += 1;
                    if fam_fail <= 4 {
                        println!(
                            "{name} roundtrip FAIL: {bits:#06x}: got {:#06x} want {:#06x}",
                            got[v as usize], want
                        );
                    }
                }
            }
        }
        let space = if fp4 { 256 } else { 65536 };
        println!("{name} roundtrip: {space} patterns checked, {fam_fail} failures");
    }

    // ---- Families 13-15: packed encode of the RNE spread as lane-mixed pairs
    // x = spread[i] (forward), y = spread[n-1-i] (mirrored): any lane swap or
    // wrong shift breaks the symmetry of the check.
    for (name, mags, sat_max, nan_byte, inf_byte, sign_shift, inf_threshold, fp4) in [
        (
            "E4M3x2",
            &e4_table,
            448.0f32,
            0x7Fu8,
            0x7Fu8,
            7u32,
            f32::INFINITY,
            false,
        ),
        (
            "E5M2x2",
            &e5_table,
            f32::INFINITY,
            0x7Eu8,
            0x7Cu8,
            7u32,
            61440.0f32,
            false,
        ),
        (
            "E2M1x2",
            &e2_table,
            6.0f32,
            0x07u8,
            0x07u8,
            3u32,
            f32::INFINITY,
            true,
        ),
    ] {
        let values = generate_test_values(mags, true);
        let n = values.len();
        let a: Vec<f32> = values.clone();
        let b: Vec<f32> = values.iter().rev().copied().collect();
        let a_dev = cuda_core::DeviceBuffer::from_host(&stream, &a)?;
        let b_dev = cuda_core::DeviceBuffer::from_host(&stream, &b)?;
        let cfg = cuda_core::simt::LaunchConfig {
            grid_dim: (((n as u32) + 1023) / 1024, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut fam_fail = 0;
        if fp4 {
            let mut out = cuda_core::DeviceBuffer::<u8>::zeroed(&stream, n)?;
            // SAFETY: launch shape covers the n-element buffers.
            unsafe { module.encode_pairs_e2m1x2(&stream, cfg, &a_dev, &b_dev, &mut out) }?;
            stream.synchronize()?;
            let got = out.to_host_vec(&stream)?;
            for i in 0..n {
                let want = pack_ref(
                    a[i],
                    b[i],
                    mags,
                    sat_max,
                    nan_byte,
                    inf_byte,
                    sign_shift,
                    inf_threshold,
                    4,
                ) as u8;
                total += 1;
                if got[i] != want {
                    fam_fail += 1;
                    failures += 1;
                    if fam_fail <= 4 {
                        println!(
                            "{name} pair FAIL [{}]: got {:#04x} want {:#04x}",
                            i, got[i], want
                        );
                    }
                }
            }
        } else {
            let mut out = cuda_core::DeviceBuffer::<u16>::zeroed(&stream, n)?;
            // SAFETY: launch shape covers the n-element buffers.
            if name == "E4M3x2" {
                unsafe { module.encode_pairs_e4m3x2(&stream, cfg, &a_dev, &b_dev, &mut out) }?;
            } else {
                unsafe { module.encode_pairs_e5m2x2(&stream, cfg, &a_dev, &b_dev, &mut out) }?;
            }
            stream.synchronize()?;
            let got = out.to_host_vec(&stream)?;
            for i in 0..n {
                let want = pack_ref(
                    a[i],
                    b[i],
                    mags,
                    sat_max,
                    nan_byte,
                    inf_byte,
                    sign_shift,
                    inf_threshold,
                    8,
                );
                total += 1;
                if got[i] != want {
                    fam_fail += 1;
                    failures += 1;
                    if fam_fail <= 4 {
                        println!(
                            "{name} pair FAIL [{}]: got {:#06x} want {:#06x}",
                            i, got[i], want
                        );
                    }
                }
            }
        }
        println!("{name} encode pairs: {n} lane-mixed pairs checked, {fam_fail} failures");
    }

    // ---- Families 16-19: packed x4 (composition verification) --------------
    // The x2 layers are exhaustively verified above; x4 composes two x2
    // halves, so a sampled sweep (2^32 space at a stride, plus the full
    // all-byte-patterns block in the low half with a fixed high half)
    // checks the composition: byte order (x in bits [7:0], w in [31:24]).
    {
        const N4: usize = 4096;
        let mut patterns: Vec<u32> = Vec::with_capacity(N4);
        // Fixed-stride samples across the whole u32 space.
        let stride = ((u32::MAX as u64 + 1) / N4 as u64) as u32;
        for i in 0..N4 {
            patterns.push((i as u64 * stride as u64) as u32);
        }
        // All byte patterns in the low half, fixed high half.
        for b in 0..=255u32 {
            patterns.push(b | (0x37_AB_00_00));
        }
        // Round up to a whole block so every thread's idx is in bounds.
        let n_round = (patterns.len() + 1023) / 1024 * 1024;
        while patterns.len() < n_round {
            patterns.push(0);
        }
        let n = patterns.len();
        let values_dev = cuda_core::DeviceBuffer::from_host(&stream, &patterns)?;
        let mut outs: Vec<cuda_core::DeviceBuffer<f32>> = (0..4)
            .map(|_| cuda_core::DeviceBuffer::<f32>::zeroed(&stream, n))
            .collect::<Result<Vec<_>, _>>()
            .map_err(Box::new)?;
        let cfg = cuda_core::simt::LaunchConfig {
            grid_dim: ((n as u32 + 1023) / 1024, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut fam_fail = 0;
        for family in ["E4M3x4", "E5M2x4"] {
            // SAFETY: launch shape covers the n-element buffers.
            if family == "E4M3x4" {
                let [o0, o1, o2, o3] = outs.as_mut_slice() else {
                    unreachable!("4 buffers");
                };
                // SAFETY: launch shape covers the n-element buffers.
                unsafe { module.decode_e4m3x4(&stream, cfg, &values_dev, o0, o1, o2, o3) }?;
            } else {
                let [o0, o1, o2, o3] = outs.as_mut_slice() else {
                    unreachable!("4 buffers");
                };
                // SAFETY: launch shape covers the n-element buffers.
                unsafe { module.decode_e5m2x4(&stream, cfg, &values_dev, o0, o1, o2, o3) }?;
            }
            stream.synchronize()?;
            let lanes: [Vec<f32>; 4] = [
                outs[0].to_host_vec(&stream)?,
                outs[1].to_host_vec(&stream)?,
                outs[2].to_host_vec(&stream)?,
                outs[3].to_host_vec(&stream)?,
            ];
            for (i, &bits) in patterns.iter().enumerate() {
                let bytes = [
                    (bits & 0xFF) as u8,
                    ((bits >> 8) & 0xFF) as u8,
                    ((bits >> 16) & 0xFF) as u8,
                    ((bits >> 24) & 0xFF) as u8,
                ];
                for lane in 0..4 {
                    let want = if family == "E4M3x4" {
                        host_e4m3_to_f32(bytes[lane])
                    } else {
                        host_e5m2_to_f32(bytes[lane])
                    };
                    total += 1;
                    if lanes[lane][i].to_bits() != want.to_bits() {
                        fam_fail += 1;
                        failures += 1;
                        if fam_fail <= 4 {
                            println!(
                                "{family} decode FAIL lane {lane} [{bits:#010x}]: got {} want {}",
                                lanes[lane][i], want
                            );
                        }
                    }
                }
            }
            println!(
                "{family} x4 decode: {n} sampled patterns x 4 lanes checked, {fam_fail} failures"
            );
        }
    }
    // x4 encode: four independent spreads, checked against the per-lane
    // scalar reference packed at the right shifts.
    for family in ["E4M3x4", "E5M2x4"] {
        let mags = if family == "E4M3x4" {
            &e4_table
        } else {
            &e5_table
        };
        let (sat_max, nan_byte, inf_byte, inf_threshold) = if family == "E4M3x4" {
            (448.0f32, 0x7Fu8, 0x7Fu8, f32::INFINITY)
        } else {
            (f32::INFINITY, 0x7Eu8, 0x7Cu8, 61440.0f32)
        };
        let values = generate_test_values(mags, true);
        let n = values.len();
        let mk = |v: &[f32]| cuda_core::DeviceBuffer::from_host(&stream, v).map_err(Box::new);
        let a_dev = mk(&values)?;
        let b_dev = mk(&values.iter().rev().copied().collect::<Vec<_>>())?;
        let c_dev = mk(&values)?;
        let d_dev = mk(&values.iter().rev().copied().collect::<Vec<_>>())?;
        let mut out = cuda_core::DeviceBuffer::<u32>::zeroed(&stream, n).map_err(Box::new)?;
        let cfg = cuda_core::simt::LaunchConfig {
            grid_dim: (((n as u32) + 1023) / 1024, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: launch shape covers the n-element buffers.
        if family == "E4M3x4" {
            unsafe {
                module.encode_e4m3x4(&stream, cfg, &a_dev, &b_dev, &c_dev, &d_dev, &mut out)
            }?;
        } else {
            unsafe {
                module.encode_e5m2x4(&stream, cfg, &a_dev, &b_dev, &c_dev, &d_dev, &mut out)
            }?;
        }
        stream.synchronize()?;
        let got = out.to_host_vec(&stream)?;
        let mut fam_fail = 0;
        let rev: Vec<f32> = values.iter().rev().copied().collect();
        for (i, &x) in values.iter().enumerate() {
            let y = rev[i];
            let z = values[i];
            let w = rev[i];
            let x2_lo =
                host_encode_reference(x, mags, sat_max, nan_byte, inf_byte, 7, inf_threshold);
            let x2_hi =
                host_encode_reference(y, mags, sat_max, nan_byte, inf_byte, 7, inf_threshold);
            let lo16 = (u16::from(x2_hi) << 8) | u16::from(x2_lo);
            let z2_lo =
                host_encode_reference(z, mags, sat_max, nan_byte, inf_byte, 7, inf_threshold);
            let z2_hi =
                host_encode_reference(w, mags, sat_max, nan_byte, inf_byte, 7, inf_threshold);
            let hi16 = (u16::from(z2_hi) << 8) | u16::from(z2_lo);
            let want = (u32::from(hi16) << 16) | u32::from(lo16);
            total += 1;
            if got[i] != want {
                fam_fail += 1;
                failures += 1;
                if fam_fail <= 4 {
                    println!(
                        "{family} x4 encode FAIL [{i}]: got {:#010x} want {:#010x}",
                        got[i], want
                    );
                }
            }
        }
        println!("{family} x4 encode: {n} quad encodes checked, {fam_fail} failures");
    }

    println!("\nTotal checks: {}, failures: {}", total, failures);
    if failures == 0 {
        println!("=== ALL FP8/FP4 TESTS PASSED ===");
    } else {
        println!("=== FAILURES PRESENT ===");
        std::process::exit(1);
    }
    Ok(())
}
