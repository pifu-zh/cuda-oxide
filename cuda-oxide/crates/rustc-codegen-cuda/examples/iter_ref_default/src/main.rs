/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Probe for the reference-form iterator default: `data.iter().min().unwrap_or(&0)`
//! (Option<&u32> with a reference default — the matrix's last IMPLEMENTABLE
//! edge; the copied()-value form already works via iter_unwrap_or_probe).

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{cuda_module, kernel, thread};
use cuda_host::cuda_module as _;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn ref_default_probe(data: &[u32], out: &mut [u32]) {
        let tid = thread::index_1d().get();
        if tid == 0 {
            // 形态 1：Some 路径（min/max 命中元素）
            let min: u32 = *data.iter().min().unwrap_or(&0);
            let max: u32 = *data.iter().max().unwrap_or(&0);
            // 形态 2：None 路径（空 slice → 迭代器无元素）——引用 default
            // 实际被消费
            let empty: &[u32] = &[];
            let none_min: u32 = *empty.iter().min().unwrap_or(&7);
            // 形态 3：unwrap_or_else 引用形态（闭包返回 &u32）——空切片
            // 命中 None 路径，闭包结果 &42 真正被消费
            let via_else: u32 = *empty.iter().min().unwrap_or_else(|| &42);
            // 形态 4：map_or 引用形态（iter_unwrap_or_probe 时代被拒的路径）
            let first_missing: u32 = *data.iter().position(|&x| x == 99).map_or(&0, |p| &0);
            out[0] = min;
            out[1] = max;
            out[2] = none_min;
            out[3] = via_else;
            out[4] = first_missing;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let data: Vec<u32> = vec![7, 3, 9, 1];
    let dev = DeviceBuffer::from_host(&stream, &data)?;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, 5)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: launch shape covers the buffers.
    unsafe { module.ref_default_probe((stream).as_ref(), cfg, &dev, &mut out) }?;
    stream.synchronize()?;
    let got = out.to_host_vec(&stream)?;
    let want = [
        *data.iter().min().unwrap_or(&0),
        *data.iter().max().unwrap_or(&0),
        7,  // 空切片 None 路径的 &7
        42, // unwrap_or_else 空切片 None 路径的 &42
        0,  // map_or 的 &0
    ];
    println!("got {got:?}, want {want:?}");
    assert_eq!(got, want, "reference-form default mismatch");
    println!("PASS: reference-form iterator default works (all 4 shapes)");
    Ok(())
}
