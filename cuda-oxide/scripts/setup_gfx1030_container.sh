#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Idempotent setup for the gfx1030-dev container (stock ROCm 7.2.4 image).
# Run INSIDE the container (via `gfx1030_container.sh exec bash <this>` or
# docker exec). Everything ROCm/HIP/CUDA-related lives in the container; the
# host keeps only docker, the repo checkout, and the Rust toolchain mounts.
#
# Recreates after a container wipe:
#   gfx1030_container.sh ensure
#   gfx1030_container.sh exec bash /repo/cuda-oxide/scripts/setup_gfx1030_container.sh
#   docker commit gfx1030-dev gfx1030-dev:snapshot   # optional restore point

set -euo pipefail

echo "== apt: python3-pytest (battery runner) =="
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq python3-pytest >/dev/null 2>&1 || echo "pytest install skipped/failed (may already exist)"

echo "== micromamba -> /opt/tools =="
if [ ! -x /opt/tools/bin/micromamba ]; then
  mkdir -p /opt/tools
  cd /tmp
  if command -v curl >/dev/null; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o mm.tar.bz2
  else
    apt-get install -y -qq curl bzip2 >/dev/null 2>&1
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o mm.tar.bz2
  fi
  mkdir -p /opt/tools
  tar -xjf mm.tar.bz2 -C /opt/tools bin/micromamba \
    || python3 -c "import tarfile; tarfile.open('/tmp/mm.tar.bz2').extractall('/opt/tools')"
  chmod +x /opt/tools/bin/micromamba
fi
/opt/tools/bin/micromamba --version

echo "== CUDA 13 toolkit (bindgen/binds for cuda-bindings) -> /opt/cuda13 =="
if [ ! -e /opt/cuda13/targets/x86_64-linux/include/cuda.h ]; then
  MAMBA_ROOT_PREFIX=/opt/tools/mamba-root /opt/tools/bin/micromamba create -y -q \
    -p /opt/cuda13 -c conda-forge 'cuda-nvcc=13.4' 'cuda-cudart-dev=13.4' 'libcurand-dev'
fi
ls /opt/cuda13/targets/x86_64-linux/include/cuda.h

echo "== HOME symlinks: rustup/cargo mounts =="
ln -sfn /host-rustup /home/ubuntu/.rustup
ln -sfn /host-cargo /home/ubuntu/.cargo

echo "== hipinfo probe -> /home/ubuntu/bin =="
mkdir -p /home/ubuntu/bin
cat > /tmp/hipinfo.c <<'EOF'
/* HIP device identity probe: prints "agent N: <gcnArchName> (<name>)" per
 * GPU — the interface tests/test_gfx1030.py parses. rc=0 iff a GPU answers. */
#include <hip/hip_runtime.h>
#include <stdio.h>
int main() {
  if (hipInit(0) != 0) return 1;
  int n = 0;
  if (hipGetDeviceCount(&n) != 0 || n <= 0) return 1;
  for (int i = 0; i < n; i++) {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, i) != 0) return 1;
    printf("agent %d: %s (%s)\n", i, p.gcnArchName, p.name);
  }
  return 0;
}
EOF
/opt/rocm/bin/hipcc -x hip /tmp/hipinfo.c -o /home/ubuntu/bin/hipinfo
/home/ubuntu/bin/hipinfo

echo "== backend dylib (container gcc) =="
cd /repo/cuda-oxide/crates/rustc-codegen-cuda
touch src/lib.rs
cargo build --target x86_64-unknown-linux-gnu -p rustc_codegen_cuda 2>&1 | tail -1

echo "setup complete"
