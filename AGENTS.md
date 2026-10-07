# AGENTS.md

cuda-oxide: a custom rustc backend that compiles Rust GPU kernels (`#[kernel]`)
to CUDA PTX. Pipeline: Rust MIR → dialect-mir (pliron IR) → LLVM IR → PTX.

**This checkout is the pifu-zh fork of NVlabs/cuda-oxide with a local
gfx1030 (AMD ROCm) port.** Branch context:

- `main` tracks upstream; `replay-develop` (default working branch) carries
  ~40 `[PORT gfx1030]` commits porting the pipeline to AMD `amdgcn`/gfx1030.
- Port layout: `host-hip/cuda-bindings/` (HIP `libamdhip64` dlopen stand-in for
  crates.io `cuda-bindings` 0.3.1, swapped into gfx example workspaces via
  `[patch.crates-io]`; zero CUDA-toolkit build dep), `crates/cuda-oxide-codegen/src/amdgcn.rs`
  (NVVM→AMDGPU prep pass), `tests/translate_amdgcn.py` (IR rewriter),
  `tests/test_gfx1030.py`, `tests/batch_gfx1030_examples.py`.

## Layout

- `crates/` — root (virtual) Cargo workspace: user-facing crates
  (`cuda-device`, `cuda-macros`, `cuda-host`; `cuda-core`/`cuda-async` come
  from crates.io/cutile-rs), compiler crates (`mir-importer`, `mir-lower`,
  `dialect-*`, `llvm-export`, `nvvm-transforms`), tooling (`cargo-oxide`,
  `cuda-intrinsics-gen`).
- `crates/rustc-codegen-cuda/` — **not a workspace member**; its own workspace
  (rustc_private nightly). Its `examples/` holds 190+ examples, each its own
  workspace. Root `cargo` commands cannot reach any of them.
- `intrinsics/` — intrinsic catalog (overlay TOMLs, ABI ledger
  `abi-v1.toml`, `upstream.lock`) consumed by `cuda-intrinsics-gen`.
- `scripts/` — CI guard scripts; `cuda-oxide-book/` — Sphinx docs.

## Commands

`just --list` shows all recipes; every Justfile comment names the CI job it mirrors.

- `just check` — full CI mirror minus examples-compile/book/CodeQL. Needs CUDA
  toolkit 13+, `cargo-deny`, `python3`; no GPU/driver needed.
- `just test` (no-CUDA packages) / `just test-cuda` (toolkit-dependent, still
  no driver; driver tests are `#[ignore]`d).
- `just fmt` / `just clippy` — cover all workspaces incl. the codegen one.
  Bare `cargo fmt`/`cargo clippy` at root does NOT — always use `cargo oxide fmt`.
- `just check-intrinsics [base_ref]` — 3 generated-intrinsics gates (~13s);
  required after touching `crates/cuda-intrinsics-gen` or `intrinsics/`.
- `cargo oxide run <example>` / `pipeline <example>` / `inspect <example>` —
  build/run/dump an example (alias in `.cargo/config.toml`).
- gfx1030: `python3 -m pytest tests/test_gfx1030.py -v` (end-to-end needs the
  ROCm GPU + docker container `zhuo`), `python3 tests/batch_gfx1030_examples.py
  [--integrated]` (integrated = `CUDA_OXIDE_TARGET=gfx1030` through the
  cargo-oxide amdgcn backend).
- `just book` — book build; the only gate needing a Python venv.

## Non-obvious rules

- Toolchain pinned `nightly-2026-08-28` (`rust-toolchain.toml`); `just
  check-toolchain-parity` keeps other pins in step.
- pliron is pinned to a git rev; `crates/rustc-codegen-cuda/Cargo.toml` MUST
  keep the same rev or pliron resolves to two crates and types stop unifying.
- `oxide-artifacts` depends on the crates.io release, not the path (a path
  copy is a second crate to rustc); parity is enforced by
  `scripts/check-oxide-artifacts-parity.sh`.
- Every first-party source file carries the exact NVIDIA SPDX header (wording
  in CONTRIBUTING.md); `scripts/check-spdx-headers.sh` enforces it.
- New dependency: permissive license only, plus a row in
  `dependency-licenses.csv`.
- New example: must print a SUCCESS/PASS/Complete marker and be registered in
  `STATUS.md` and the `scripts/smoketest.sh` example arrays.
- `clippy.toml` bans minting kinded MIR pointers outside
  `mir-importer translator::facts::PointerOrigin`.
- Commits need DCO sign-off: `git commit -s`.
- llc: prefers the toolchain's own `llc`, then `llc-23/22/21` on PATH; pin
  with `CUDA_OXIDE_LLC`. TMA/tcgen05/WGMMA need LLVM 21+.
- gfx1030 invariants: `tests/translate_amdgcn.py` is the single source of
  truth (test + batch scripts import it — never copy it). The amdgcn prep pass
  must fail loudly on unhandled NVVM shapes (`llc: Cannot select`), never
  emit a silently wrong code object. `*.hsaco` / `.amdgcn*.ll` are gitignored
  build artifacts.

## Cross-repo skill library (develop-skill)

The porting methodology for this gfx1030 effort lives in the sibling repo
`pifu-zh/develop-skill` (local clone: `~/workspace/src/develop-skill`),
mirrored into `~/.zcode/skills/` for ZCode skill auto-loading:

- `gpu_lib_port/` — porting discipline (five rules) +
  `references/nv-amdgcn-map.md` — the **NV→AMDGCN mapping registry**
  (intrinsic/primone mappings with evidence grades). Consult it before
  writing any new intrinsic mapping; backfill it after each verified mapping.
- `gpu_lib_port/SKILL.md` §3.5 — replay-verification workflow (per-commit
  byte-diff + triage), probe-first rules, address-space contracts, prep-pass
  ordering.
- `gpu_lib_build/` `gpu_lib_test/` — build/test pipeline discipline
  (COv5, --genco semantics, fake-PASS traps).
- `docs/research/cuda-oxide/` (host, outside this repo) — the full
  research archive this AGENTS.md summarizes.

Update flow: improve the skill in `~/.zcode/skills/` first, then mirror to
`pifu-zh/develop-skill` (which also feeds the Aliyun workspace backup).

## Read before touching

- `Justfile` — each recipe's comment explains its CI job and prerequisites.
- `CONTRIBUTING.md` — DCO, SPDX header wording, dependency and example rules.
- `crates/cuda-oxide-codegen/src/amdgcn.rs` module docs — the gfx1030
  pipeline's division of labor (what is lowered where, and why).
- `host-hip/cuda-bindings/README.md` — HIP backend design contract.
- `cuda-oxide-book/` — architecture deep-dives and API reference.
