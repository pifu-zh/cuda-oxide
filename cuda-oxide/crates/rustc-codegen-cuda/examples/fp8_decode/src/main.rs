/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Software FP8/FP4 <-> f32 conversion test (RDNA2 has no fp8/fp4 hardware).
//!
//! The kernel exhaustively decodes every E4M3/E5M2 byte (256 each) and every
//! E2M1 nibble (16), and encodes a host-generated spread of f32 test values
//! (boundaries, exact RNE midpoints, one-ulp-off cases, specials) with the
//! three encoders. The host cross-checks everything against independent
//! reference implementations and reports PASS/FAIL per family.
//!
//! Run: cargo oxide run fp8_decode

use cuda_device::fp8::{e2m1_to_f32, e4m3_to_f32, e5m2_to_f32, f32_to_e2m1_rne_satfinite,
    f32_to_e4m3_rne_satfinite, f32_to_e5m2_rne};
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
    sign_shift: u32,     // 7 for the byte formats, 3 for the E2M1 nibble
    inf_threshold: f32,  // at/above this, RNE rounds to Inf (E5M2); the
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
        unsafe { module.decode_e4m3((stream).as_ref(), cuda_core::simt::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }, &mut out) }?;
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
                        gotv, gotv.to_bits(), want, want.to_bits()
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
        unsafe { module.decode_e5m2((stream).as_ref(), cuda_core::simt::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }, &mut out) }?;
        stream.synchronize()?;
        let got = out.to_host_vec(&stream)?;
        for b in 0u16..256 {
            let want = host_e5m2_to_f32(b as u8);
            let gotv = got[b as usize];
            total += 1;
            if gotv.to_bits() != want.to_bits()
                && !(gotv.is_nan() && want.is_nan())
            {
                failures += 1;
                if failures <= 8 {
                    println!(
                        "E5M2 decode FAIL: pattern {b:#04x}: got {} ({:#x}) want {} ({:#x})",
                        gotv, gotv.to_bits(), want, want.to_bits()
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
        unsafe { module.decode_e2m1((stream).as_ref(), cuda_core::simt::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (16, 1, 1),
            shared_mem_bytes: 0,
        }, &mut out) }?;
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
            "E4M3" => unsafe {
                module.encode_e4m3(&stream, cfg, &values_dev, &mut out)
            }?,
            "E5M2" => unsafe {
                module.encode_e5m2(&stream, cfg, &values_dev, &mut out)
            }?,
            _ => unsafe {
                module.encode_e2m1(&stream, cfg, &values_dev, &mut out)
            }?,
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
            let want = host_encode_reference(*x, mags, sat_max, nan_byte, inf_byte, sign_shift, inf_threshold);
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
                        name, x, x.to_bits(), gotv, want
                    );
                }
            }
        }
        println!(
            "{} encode: {} values checked, {} failures",
            name,
            n,
            fam_fail
        );
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
