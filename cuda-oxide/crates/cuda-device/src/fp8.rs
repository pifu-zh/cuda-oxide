/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Software FP8/FP4 <-> `f32` conversion for RDNA2.
//!
//! gfx1030 has no packed-fp8/fp4 hardware conversion (no `v_cvt_pk_fp8`
//! family, no MFMA fp8 paths), so the conversions here are pure software bit
//! manipulation over `f32` — the same semantics as CUDA's
//! `cvt.rn.satfinite.e4m3x2.f32` / `cvt.rn.e5m2x2.f32` families:
//!
//! - [`e4m3_to_f32`] / [`f32_to_e4m3_rne_satfinite`]: OCP E4M3FN (1-4-3,
//!   bias 7, no infinities; `0x7F`/`0xFF` = NaN; max finite 448). Overflow
//!   saturates to the max finite per `.satfinite`.
//! - [`e5m2_to_f32`] / [`f32_to_e5m2_rne`]: OCP E5M2 (1-5-2, bias 15) with
//!   IEEE-style Inf/NaN; overflow rounds to Inf (no saturation).
//! - [`e2m1_to_f32`] / [`f32_to_e2m1_rne_satfinite`]: OCP E2M1 (1-2-1,
//!   bias 1, max finite 6.0, no Inf/NaN). Overflow saturates.
//!
//! All encodes round to nearest, ties to even (RNE). Decodes are exact:
//! every FP8/FP4 value is representable in `f32`.
//!
//! The packed x2 forms at the bottom (`__nv_fp8x2_*` / `__nv_fp4x2_*`) compose
//! the scalar rules per lane: lane 0 (x) in the low byte/nibble, lane 1 (y)
//! in the high byte/nibble.

/// Decodes one E4M3FN byte (CUDA `__nv_fp8_e4m3` layout) to `f32`, exactly.
#[inline(always)]
pub fn e4m3_to_f32(bits: u8) -> f32 {
    let s = u32::from((bits >> 7) & 1);
    let e = u32::from((bits >> 3) & 0xF);
    let m = u32::from(bits & 0x7);
    let bits32: u32 = if e == 0 {
        // Denormal: m * 2^-9. The f32 exponent field for 2^-9 is 118 and the
        // mantissa must hold m exactly (m <= 7 -> m << 21 within an exponent
        // that steps per m) — the linear form `m << 20` would wrongly enable
        // the implicit leading one, so decode arithmetically instead.
        let mag = (m as f32) * f32::from_bits(118 << 23); // * 2^-9, exact
        f32::to_bits(mag) | (s << 31)
    } else if e == 0xF && m == 0x7 {
        // NaN: E4M3FN reserves 1111.111; no infinities exist.
        (s << 31) | 0x7FC0_0000
    } else {
        // Normal: 2^(e-7) * (1 + m/8) -> f32 exponent field (e-7)+127 = e+120
        // with the mantissa shifted into the top 3 fraction bits (exact).
        (s << 31) | ((e + 120) << 23) | (m << 20)
    };
    f32::from_bits(bits32)
}

/// Decodes one E5M2 byte (CUDA `__nv_fp8_e5m2` layout) to `f32`, exactly.
#[inline(always)]
pub fn e5m2_to_f32(bits: u8) -> f32 {
    let s = u32::from((bits >> 7) & 1);
    let e = u32::from((bits >> 2) & 0x1F);
    let m = u32::from(bits & 0x3);
    let bits32: u32 = if e == 0 {
        // Denormal: m * 2^-16, decoded arithmetically (same implicit-one
        // pitfall as E4M3 denormals).
        let mag = (m as f32) * f32::from_bits(111 << 23); // * 2^-16, exact
        f32::to_bits(mag) | (s << 31)
    } else if e == 0x1F {
        if m != 0 {
            (s << 31) | 0x7FC0_0000 // NaN
        } else {
            (s << 31) | 0x7F80_0000 // Inf: raw f32 infinity bits (e+112 = 143
                                    // would overflow the 8-bit f32 field)
        }
    } else {
        // Normal: exponent field (e-15)+127 = e+112.
        (s << 31) | ((e + 112) << 23) | (m << 21)
    };
    f32::from_bits(bits32)
}

/// The eight E2M1 magnitudes, indexed by `exp << 1 | mantissa`.
const E2M1_TABLE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decodes one E2M1 nibble (low 4 bits, CUDA `__nv_fp4_e2m1` layout) to
/// `f32`, exactly.
#[inline(always)]
pub fn e2m1_to_f32(bits: u8) -> f32 {
    let s = u32::from((bits >> 3) & 1);
    let mag = E2M1_TABLE[usize::from(bits & 0x7)];
    f32::from_bits(f32::to_bits(mag) | (s << 31))
}

/// Round-to-nearest-even shift right. `shift == 0` returns the value;
/// `shift > 31` returns zero.
#[inline(always)]
fn rne_shift(v: u32, shift: u32) -> u32 {
    if shift == 0 {
        return v;
    }
    if shift > 31 {
        return 0;
    }
    let q = v >> shift;
    let rem = v & ((1u32 << shift) - 1);
    let half = 1u32 << (shift - 1);
    if rem > half || (rem == half && (q & 1) == 1) {
        q + 1
    } else {
        q
    }
}

/// Encodes an `f32` to one E4M3FN byte: RNE, overflow saturates to the max
/// finite (448) per `.satfinite`; NaN maps to the NaN byte; Inf maps to the
/// NaN byte (E4M3FN has no infinity).
pub fn f32_to_e4m3_rne_satfinite(x: f32) -> u8 {
    let bits = x.to_bits();
    let s = ((bits >> 31) & 1) as u8;
    let e32 = ((bits >> 23) & 0xFF) as i32;
    let m32 = bits & 0x7F_FFFF;
    if e32 == 0xFF {
        // NaN and Inf both map to the E4M3FN NaN byte (no Inf exists).
        return (s << 7) | 0x7F;
    }
    if e32 == 0 {
        // f32 denormals are below 2^-126, far under the half-of-min-denormal
        // point (2^-10): they all round to zero.
        return s << 7;
    }
    let e = e32 - 127 + 7; // E4M3-biased exponent (value = 1.m * 2^(e-7))
    let mant = m32 | 0x80_0000; // implicit leading one

    if e < 1 {
        // Denormal target (value below the min normal 2^-6): the value in
        // units of the denormal step 2^-9 is mant >> (21 - e), RNE. An RNE
        // overflow to 8 means the value rounds up to the min normal
        // (2^-6 = byte 0x08).
        let m3 = rne_shift(mant, (21 - e) as u32);
        if m3 <= 7 {
            return (s << 7) | (m3 as u8);
        }
        return (s << 7) | 0x08;
    }

    // Normal target: round the 24-bit significand to 3 fraction bits.
    let v = rne_shift(mant, 20); // in [8, 16]; 16 = mantissa overflow
    let (e3, m3) = if v == 16 { (e + 1, 0) } else { (e, (v - 8) as i32) };
    if e3 > 15 || (e3 == 15 && m3 == 7) {
        // Overflow past the max finite (448): the NaN zone (1111.111) is
        // next, so .satfinite clamps to 448.
        return (s << 7) | 0x7E;
    }
    (s << 7) | ((e3 as u8) << 3) | (m3 as u8)
}

/// Encodes an `f32` to one E5M2 byte: RNE; overflow rounds to Inf (IEEE
/// half-style semantics, no saturation); NaN maps to a NaN byte.
pub fn f32_to_e5m2_rne(x: f32) -> u8 {
    let bits = x.to_bits();
    let s = ((bits >> 31) & 1) as u8;
    let e32 = ((bits >> 23) & 0xFF) as i32;
    let m32 = bits & 0x7F_FFFF;
    if e32 == 0xFF {
        if m32 == 0 {
            return (s << 7) | 0x7C; // Inf
        }
        return (s << 7) | 0x7E; // NaN
    }
    if e32 == 0 {
        // f32 denormals: min E5M2 denormal = 2^-16, half point 2^-17; f32
        // denormals are below 2^-126 -> zero. (f32 normals from 2^-126 up
        // round through the E5M2 denormal range via the normal path's shift,
        // which lands in the denormal exponent naturally.)
        return s << 7;
    }
    let e = e32 - 127 + 15; // E5M2-biased exponent
    let mant = m32 | 0x80_0000;

    if e < 1 {
        // Denormal target: value in units of 2^-16 = mant >> (22 - e), RNE.
        // An RNE overflow to 4 means the value rounds up to the min normal
        // (2^-14 = byte 0x08).
        let m2 = rne_shift(mant, (22 - e) as u32) as u8;
        if m2 <= 3 {
            return (s << 7) | m2; // denormal pattern (exp field 0)
        }
        // RNE carried into the min normal 2^-14 = exp field 1, mantissa 0.
        return (s << 7) | 0x04;
    }

    // Normal target: round the 24-bit significand to 2 fraction bits.
    let v = rne_shift(mant, 21); // in [4, 8]; 8 = mantissa overflow
    let (e5, m2) = if v == 8 { (e + 1, 0) } else { (e, (v - 4) as i32) };
    if e5 >= 31 {
        // Overflow: round to Inf (IEEE semantics for E5M2).
        return (s << 7) | 0x7C;
    }
    (s << 7) | ((e5 as u8) << 2) | (m2 as u8)
}

/// Encodes an `f32` to one E2M1 nibble: RNE; overflow saturates to the max
/// finite (6.0) per OCP FP4 satfinite semantics.
pub fn f32_to_e2m1_rne_satfinite(x: f32) -> u8 {
    let bits = x.to_bits();
    let s = ((bits >> 31) & 1) as u8;
    let e32 = ((bits >> 23) & 0xFF) as i32;
    let m32 = bits & 0x7F_FFFF;
    if e32 == 0xFF || e32 == 0 {
        // NaN/Inf saturate to max finite (E2M1 has no NaN/Inf); f32
        // denormals are below the half point of 0.5 (0.25) -> zero.
        return if e32 == 0 { s << 3 } else { (s << 3) | 0b0111 };
    }
    let e = e32 - 127 + 1; // E2M1-biased exponent
    let mant = m32 | 0x80_0000;

    if e < 1 {
        // Values below the min normal 1.0: candidates 0 (even) and 0.5.
        // The value in 0.5 units is mant >> (23 - e), RNE; an overflow to 2
        // means the value rounds up to 1.0 (byte 0b0010).
        let m2 = rne_shift(mant, (23 - e) as u32) as u8;
        return match m2 {
            0 => s << 3,
            1 => (s << 3) | 0b0001,
            _ => (s << 3) | 0b0010,
        };
    }

    // Normal target: round the 24-bit significand to 1 fraction bit.
    let v = rne_shift(mant, 22); // in [2, 4]; 4 = mantissa overflow
    let (e2, m2) = if v == 4 { (e + 1, 0) } else { (e, (v - 2) as i32) };
    if e2 > 3 || (e2 == 3 && m2 == 1) {
        // Overflow past 6.0: the 8.0 slot does not exist in E2M1, so
        // satfinite clamps to the max finite 6.0 (e=3, m=1).
        return (s << 3) | 0b0111;
    }
    (s << 3) | (((e2 as u8) << 1) | (m2 as u8))
}

// ---------------------------------------------------------------------------
// Packed x2 forms — CUDA `__nv_fp8x2_e4m3` / `__nv_fp8x2_e5m2` /
// `__nv_fp4x2_e2m1` semantics: two values per storage unit, lane 0 (x) in the
// LOW byte (bits [7:0], or the low nibble for FP4) and lane 1 (y) in the high
// byte (bits [15:8], high nibble). Each lane converts with the scalar rules
// above; the packing itself is the only added behavior. There is no fp8/fp4
// hardware on gfx1030 to match bit-for-bit, so the convention is pinned to
// the CUDA header layout and verified lane-exhaustively in fp8_decode.
// ---------------------------------------------------------------------------

/// Decodes a packed E4M3 x2 (`__nv_fp8x2_e4m3`) to `(x, y)` exactly.
#[inline(always)]
pub fn e4m3x2_to_f32x2(bits: u16) -> (f32, f32) {
    (e4m3_to_f32(bits as u8), e4m3_to_f32((bits >> 8) as u8))
}

/// Decodes a packed E5M2 x2 (`__nv_fp8x2_e5m2`) to `(x, y)` exactly.
#[inline(always)]
pub fn e5m2x2_to_f32x2(bits: u16) -> (f32, f32) {
    (e5m2_to_f32(bits as u8), e5m2_to_f32((bits >> 8) as u8))
}

/// Decodes a packed E2M1 x2 (`__nv_fp4x2_e2m1`) to `(x, y)` exactly.
#[inline(always)]
pub fn e2m1x2_to_f32x2(bits: u8) -> (f32, f32) {
    (e2m1_to_f32(bits & 0xF), e2m1_to_f32(bits >> 4))
}

/// Packs two `f32` into one E4M3 x2: each lane RNE + satfinite (the scalar
/// rules); `x` in the low byte.
#[inline(always)]
pub fn f32x2_to_e4m3x2_rne_satfinite(x: f32, y: f32) -> u16 {
    u16::from(f32_to_e4m3_rne_satfinite(x)) | (u16::from(f32_to_e4m3_rne_satfinite(y)) << 8)
}

/// Packs two `f32` into one E5M2 x2: each lane RNE with IEEE overflow
/// (scalar rules); `x` in the low byte.
#[inline(always)]
pub fn f32x2_to_e5m2x2_rne(x: f32, y: f32) -> u16 {
    u16::from(f32_to_e5m2_rne(x)) | (u16::from(f32_to_e5m2_rne(y)) << 8)
}

/// Packs two `f32` into one E2M1 x2: each lane RNE + satfinite (scalar
/// rules); `x` in the low nibble.
#[inline(always)]
pub fn f32x2_to_e2m1x2_rne_satfinite(x: f32, y: f32) -> u8 {
    f32_to_e2m1_rne_satfinite(x) | (f32_to_e2m1_rne_satfinite(y) << 4)
}
