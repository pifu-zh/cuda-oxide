#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 2026-11-01: the gfx1030 workflow is container-first — the host keeps no
# ROCm/HIP/CUDA toolchain (docker + repo + Rust toolchain mounts only). This
# script remains for compatibility and forwards to the container workflow:
#
#   ./prepare_gfx1030_env.sh            == gfx1030_container.sh ensure
#   ./prepare_gfx1030_env.sh shell      == interactive container shell
#   ./prepare_gfx1030_env.sh exec CMD   == run CMD inside the container
#
# The historical host-side stack (~/opt/{rocm2,hip-lib,rocm2-shim,gcc-env,
# cuda13}) is gone; see setup_gfx1030_container.sh for the in-container
# equivalent and docs/research/2026-10-28-environment-recovery.md for the
# host-era record.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/gfx1030_container.sh" "$@"
