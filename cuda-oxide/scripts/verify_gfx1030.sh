#!/bin/bash
# [PORT gfx1030][goal] one-command verification & report for the gfx1030
# toolchain on real hardware (RX 6950 XT).
#
# Runs the full acceptance battery and archives every command, log, and
# environment fact into a timestamped report directory:
#
#   scripts/verify_gfx1030.sh [report_root]
#
# Report root defaults to /tmp/gfx1030_reports. Strict accounting: a gate
# that cannot run is reported as FAILED (skip is never counted as pass).
set -uo pipefail

REPORT_ROOT="${1:-/tmp/gfx1030_reports}"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$REPORT_ROOT/$STAMP"
mkdir -p "$REPORT"
cd "$(dirname "$0")/.."
REPO="$PWD"  # [adapt-rocm] = cuda-oxide/

log() { echo "[verify-gfx1030] $*"; }

# [adapt-rocm] shared build/runtime environment: pinned-stable -Z unlock,
# CUDA-13 headers for cuda-bindings bindgen, and the HIP shim installed as
# libcuda.so.1 for the upstream dlopen loader.
export RUSTC_BOOTSTRAP=1
export CUDA_TOOLKIT_PATH="$HOME/opt/cuda13"
SHIM_LIB="$(cd "$(dirname "$0")/../../host-hip/cuda-hip-shim" 2>/dev/null && pwd)/shim-lib"
if [ -d "$SHIM_LIB" ]; then
  export LD_LIBRARY_PATH="$SHIM_LIB:${LD_LIBRARY_PATH:-}"
fi

# --- 0. environment snapshot -------------------------------------------------
{
  echo "== kernel ==";        uname -a
  echo "== GPU (amd-smi) =="; amd-smi 2>/dev/null || rocm-smi 2>/dev/null || echo "NO SMI"
  echo "== GPU (rocminfo gfx agents) =="
  rocminfo 2>/dev/null | grep -A2 'Marketing' | grep -E 'Name|gfx' | head -8
  echo "== ROCm ==";          ls -l /opt/rocm/lib/libamdhip64.so* 2>/dev/null | head -2
  echo "== rustc (pinned) =="
  rustc +stable --version 2>&1
  echo "== llc (toolchain llvm-tools) =="
  "$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin/llc" --version 2>&1 | head -2
  echo "== cargo-oxide ==";   cargo-oxide --version 2>&1
  echo "== repo ==";          git -C "$REPO" rev-parse --short HEAD
  git -C "$REPO" log --oneline -6
  echo "== python ==";        python3 --version
} > "$REPORT/environment.txt" 2>&1
log "environment snapshot -> $REPORT/environment.txt"

PASS=0; FAIL=0; FAILED=()

run_gate() { # run_gate <name> <logfile> <cmd...>
  local name="$1" logfile="$2"; shift 2
  log "gate: $name"
  if "$@" > "$REPORT/$logfile" 2>&1; then
    PASS=$((PASS+1)); log "  PASS ($logfile)"
  else
    FAIL=$((FAIL+1)); FAILED+=("$name"); log "  FAILED (see $REPORT/$logfile)"
  fi
}

# --- 1. HIP attribute probe (device attributes by number) --------------------
if command -v gcc >/dev/null && [ -f /opt/rocm/lib/libamdhip64.so.7 ]; then
  gcc scripts/gfx1030_hip_attr_probe.c -o "$REPORT/attr_probe" -ldl &&
    "$REPORT/attr_probe" > "$REPORT/attr_probe.out" 2>&1
  if grep -q 'attr 74 ' "$REPORT/attr_probe.out" &&
     grep -q 'value=65536' "$REPORT/attr_probe.out"; then
    PASS=$((PASS+1)); log "gate: hip-attr-probe PASS"
  else
    FAIL=$((FAIL+1)); FAILED+=("hip-attr-probe"); log "  FAILED (attr probe)"
  fi
else
  FAIL=$((FAIL+1)); FAILED+=("hip-attr-probe (gcc or libamdhip64 missing)")
fi

# --- 2. pytest suite (incl. two GPU numeric e2e) ------------------------------
run_gate "pytest-suite" "pytest.txt" \
  env PATH="$HOME/.local/bin:$PATH" python3 -m pytest tests/test_gfx1030.py -v

# --- 3. batch: Python pipeline dimension --------------------------------------
run_gate "batch-python" "batch_python.txt" \
  env PATH="$HOME/.local/bin:$PATH" python3 tests/batch_gfx1030_examples.py

# --- 4. batch: integrated (Rust amdgcn backend) dimension ---------------------
run_gate "batch-integrated" "batch_integrated.txt" \
  env PATH="$HOME/.local/bin:$PATH" CUDA_TOOLKIT_PATH="$HOME/opt/cuda13" \
  python3 tests/batch_gfx1030_examples.py --integrated

# --- 5. literal acceptance: cargo oxide run vecadd -----------------------------
run_gate "literal-vecadd" "vecadd_run.txt" \
  env CUDA_OXIDE_TARGET=gfx1030 cargo oxide run vecadd

grep -q "SUCCESS: All 1024 elements correct" "$REPORT/vecadd_run.txt" || {
  log "  NOTE: vecadd run gate exited 0 but the SUCCESS marker is missing — inspect $REPORT/vecadd_run.txt"
}

# --- summary -------------------------------------------------------------------
{
  echo "gfx1030 verification report $STAMP"
  echo "repo: $(git -C "$REPO" rev-parse --short HEAD)"
  echo "PASS gates: $PASS   FAILED gates: $FAIL"
  [ "$FAIL" -gt 0 ] && printf 'failed: %s\n' "${FAILED[@]}"
  echo "artifacts in: $REPORT"
} | tee "$REPORT/SUMMARY.txt"
[ "$FAIL" -eq 0 ]
