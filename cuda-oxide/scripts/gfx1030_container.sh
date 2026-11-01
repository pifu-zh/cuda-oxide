#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Container-first gfx1030 workflow: everything ROCm/HIP lives inside the
# stock ROCm 7.2.4 container; the host stays clean (docker + repo + Rust
# toolchain mounts only).
#
# Usage:
#   gfx1030_container.sh ensure        create/start the persistent container
#   gfx1030_container.sh shell         interactive bash inside
#   gfx1030_container.sh exec CMD...   run a command inside
#
# The container keeps: the complete ROCm 7.2.4 userspace (hipcc, comgr,
# oclc/ockl bitcode, rocprofv3), a container-built backend dylib, and the
# hipinfo probe under /home/ubuntu/bin.

set -euo pipefail

IMAGE="docker.m.daocloud.io/rocm/dev-ubuntu-24.04:7.2.4-complete"
NAME="gfx1030-dev"
# Repository root (outer): the SIMT workspace references shared host crates
# via ../cuda-core etc., so the mount must cover both roots.
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd -P)"
REPO_MOUNT="/repo"
WORKDIR="$REPO_MOUNT/cuda-oxide"

ensure() {
  if ! docker container inspect "$NAME" >/dev/null 2>&1; then
    echo "creating $NAME from $IMAGE"
    docker run -d --name "$NAME" \
      -v "$REPO:$REPO_MOUNT" \
      -v "$HOME/.cargo:/host-cargo" \
      -v "$HOME/.rustup:/host-rustup" \
      --device=/dev/kfd --device=/dev/dri \
      --group-add 44 --group-add 999 \
      --security-opt seccomp=unconfined \
      --cap-add PERFMON --cap-add SYS_ADMIN \
      "$IMAGE" sleep infinity >/dev/null
  fi
  docker start "$NAME" >/dev/null
}

# The image's `ubuntu` user matches the host uid (1000), so workspace file
# ownership survives round trips. ROCm paths come from the image itself.
INNER_ENV='export HOME=/home/ubuntu; export PATH=/home/ubuntu/bin:/host-cargo/bin:/opt/rocm/bin:$PATH; export CARGO_HOME=/host-cargo; export RUSTUP_HOME=/host-rustup; export RUSTC_BOOTSTRAP=1; export CUDA_OXIDE_TARGET=gfx1030; export CUDA_TOOLKIT_PATH=/opt/cuda13; export LIBCLANG_PATH=/opt/rocm/lib/llvm/lib; export BINDGEN_EXTRA_CLANG_ARGS="-I/opt/rocm/lib/llvm/lib/clang/22/include"; export LD_LIBRARY_PATH=/repo/host-hip/cuda-hip-shim/shim-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}'

case "${1:-}" in
  ensure)
    ensure
    docker exec "$NAME" bash -c "$INNER_ENV; hipcc --version | head -1; true"
    ;;
  shell)
    ensure
    docker exec -it -w "$WORKDIR" "$NAME" bash -c "$INNER_ENV; exec bash"
    ;;
  exec)
    ensure
    shift
    docker exec -w "$WORKDIR" "$NAME" bash -c "$INNER_ENV; \$0 \"\$@\"" "${1}" "${@:2}"
    ;;
  *)
    sed -n '2,20p' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac
