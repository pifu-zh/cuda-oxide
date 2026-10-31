#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Source this file to set up the gfx1030 toolchain environment (post-2026-10-27
# /opt wipe recovery; see docs/research/2026-10-28-environment-recovery.md).
# Recreates ~/bin shims if missing.

export CUDA_TOOLKIT_PATH="$HOME/opt/cuda13"
export RUSTC_BOOTSTRAP=1
export CARGO_NET_OFFLINE=true
export LD_LIBRARY_PATH="$HOME/opt/hip-lib:${LD_LIBRARY_PATH:-}"
export ROCM_PATH="$HOME/opt/rocm2-shim"
export PATH="$HOME/bin:$HOME/opt/rocm2/bin:$PATH"
export CC="$HOME/bin/gcc"
export CXX="$HOME/bin/g++"
export BINDGEN_EXTRA_CLANG_ARGS="-I$HOME/opt/rocm2/lib/clang/22/include"

mkdir -p "$HOME/bin"
if [ ! -e "$HOME/bin/cc" ]; then
cat > "$HOME/bin/cc" <<'SH'
#!/bin/bash
exec "$HOME/opt/gcc-env/bin/x86_64-conda-linux-gnu-gcc" "$@"
SH
cat > "$HOME/bin/gcc" <<'SH'
#!/bin/bash
exec "$HOME/opt/gcc-env/bin/x86_64-conda-linux-gnu-gcc" "$@"
SH
cat > "$HOME/bin/g++" <<'SH'
#!/bin/bash
exec "$HOME/opt/gcc-env/bin/x86_64-conda-linux-gnu-g++" "$@"
SH
cat > "$HOME/bin/hipcc" <<'SH'
#!/bin/bash
args=()
for a in "$@"; do
  case "$a" in
    --offload-arch=*|-x) ;;
    hip) ;;
    *) args+=("$a") ;;
  esac
done
exec "$HOME/bin/g++" -D__HIP_PLATFORM_AMD__ \
  -I"$HOME/opt/rocm-headers" \
  -L"$HOME/opt/hip-lib" -lamdhip64 -lz -lzstd \
  -Wl,-rpath,"$HOME/opt/hip-lib" \
  "${args[@]}"
SH
cat > "$HOME/bin/git" <<'SH'
#!/bin/bash
exec "$HOME/opt/git-env/bin/git" "$@"
SH
chmod +x "$HOME/bin/cc" "$HOME/bin/gcc" "$HOME/bin/g++" "$HOME/bin/hipcc" "$HOME/bin/git"
fi
echo "gfx1030 env ready: PATH=$HOME/bin rocm2 hip-lib gcc-env"
