//! [PORT gfx1030] Phase 1 探针: vecadd kernel 的 device 产物(.ll)将用于
//! amdgcn 后端验证(llc -march=amdgcn → hsaco → hipModuleLoadData)。

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

fn main() {
    println!("probe host stub");
}
