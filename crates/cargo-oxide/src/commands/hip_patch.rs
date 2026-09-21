/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! [PORT gfx1030 PhaseA.2] Automatic HIP patch-table injection.
//!
//! AMDGPU (`CUDA_OXIDE_TARGET=gfx…`) example builds need the crates.io
//! `cuda-bindings` (dlopen `libcuda`, build-time CUDA SDK headers) swapped
//! for the HIP-backed stand-in at `<repo>/host-hip/cuda-bindings`. Until
//! now every example carried a handwritten `[patch.crates-io]` table; this
//! module makes `cargo oxide build/run` ensure the table itself whenever a
//! gfx target is active, so new and external examples work without manual
//! wiring (batch device-only variant crates under /tmp being the driving
//! case — their generated manifests can never carry a handwritten table).
//!
//! Idempotent three-state contract (guarded by a reachability precondition:
//! the manifest must depend on `cuda-host`/`cuda-core`/`cuda-macros`, i.e. be
//! able to link cuda-bindings at all — pure device-only crates are skipped):
//! 1. no `[patch.crates-io] cuda-bindings` entry → inject (into an existing
//!    patch table, or as a fresh marked table at the end of the manifest);
//! 2. entry already present WITH this tool's marker → byte-identical no-op;
//! 3. entry present without the marker (handwritten, e.g. the committed
//!    example tables) → byte-identical no-op; the manual table always wins.
//!
//! Editing is deliberately line-based instead of going through the `toml`
//! crate: re-serializing would rewrite the whole manifest (losing comments
//! and formatting of committed files) where the change is one appended
//! block. The edit result staying valid TOML is covered by a test.

use std::path::{Path, PathBuf};

/// Marker comment on the auto-injected table (state 2 detection).
pub(super) const PATCH_MARKER: &str =
    "# [PORT gfx1030] auto-injected by cargo-oxide (ensure_hip_patch)";

/// Ensures the example manifest patches `cuda-bindings` to the HIP stand-in
/// when the active target names a `gfx…` family. No-op otherwise.
///
/// `configured_arch` is the already-resolved target (the same value the
/// callers pass to `apply_output_mode`). Note that `configured_arch`
/// (codegen_env) deliberately answers `None` when only the `CUDA_OXIDE_TARGET`
/// environment variable is set — the backend child reads the inherited env
/// directly — so this gate falls back to that exact variable; a CLI/config
/// resolution always wins over it.
pub(super) fn ensure_hip_patch(example_dir: &Path, configured_arch: Option<&str>) {
    let env_target = std::env::var("CUDA_OXIDE_TARGET").ok();
    if !is_gfx_activation(configured_arch, env_target.as_deref()) {
        return;
    }
    ensure_hip_patch_with(example_dir, &hip_bindings_dir());
}

/// Pure activation gate: a resolved configured target wins; only when
/// nothing was resolved does the inherited `CUDA_OXIDE_TARGET` decide.
fn is_gfx_activation(configured: Option<&str>, env: Option<&str>) -> bool {
    is_gfx_target(configured.or(env))
}

/// Testable core: same contract as [`ensure_hip_patch`] with an explicit
/// stand-in location.
fn ensure_hip_patch_with(example_dir: &Path, stand_in: &Path) {
    let manifest = example_dir.join("Cargo.toml");
    let Ok(existing) = std::fs::read_to_string(&manifest) else {
        return; // no manifest (not a package dir): nothing to patch
    };
    if !pulls_cuda_bindings(&existing) {
        // Pure device-only crates (the ox-amdgcn-probe shape) never link
        // cuda-bindings on the host side; injecting into their committed
        // manifests would only create git noise on every gfx build.
        return;
    }
    let updated = match classify(&existing) {
        PatchState::Patched => return,
        PatchState::TableWithoutKey => {
            insert_into_existing_table(&existing, &patch_path_for(example_dir, stand_in))
        }
        PatchState::Absent => append_marked_table(&existing, &patch_path_for(example_dir, stand_in)),
    };
    if std::fs::write(&manifest, &updated).is_ok() {
        println!(
            "cargo-oxide: injected HIP cuda-bindings patch into {} \
             (gfx target active; delete the marked block to fall back to crates.io cuda-bindings)",
            manifest.display()
        );
    }
}

/// Where the stand-in lives in this checkout: `<repo>/host-hip/cuda-bindings`
/// (`cargo-oxide` is built from `<repo>/crates/cargo-oxide`).
fn hip_bindings_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("cargo-oxide manifest sits two levels below the repo root")
        .join("host-hip")
        .join("cuda-bindings")
}

/// Whether `arch` names an AMDGPU target — the same grammar as the codegen
/// amdgcn parser (`gfx` + digits + optional lowercase product suffix), which
/// is also what `cuda-host::launch::amdgcn_ntid_append_active` keys on.
pub(super) fn is_gfx_target(arch: Option<&str>) -> bool {
    let Some(arch) = arch.map(str::trim) else {
        return false;
    };
    let Some(rest) = arch.strip_prefix("gfx") else {
        return false;
    };
    let split = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    let (digits, suffix) = rest.split_at(split);
    !digits.is_empty() && suffix.bytes().all(|b| b.is_ascii_lowercase())
}

/// The three manifest states the contract distinguishes.
enum PatchState {
    /// `cuda-bindings` is patched (auto or manual): never touch.
    Patched,
    /// A `[patch.crates-io]` table exists without a `cuda-bindings` key:
    /// insert the key into it (a second `[patch.crates-io]` header would be
    /// invalid TOML).
    TableWithoutKey,
    /// No patch table: append a fresh marked one.
    Absent,
}

fn classify(manifest: &str) -> PatchState {
    let mut in_patch_table = false;
    let mut table_seen = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_patch_table = trimmed == "[patch.crates-io]";
            table_seen |= in_patch_table;
            continue;
        }
        if in_patch_table && is_cuda_bindings_key(trimmed) {
            return PatchState::Patched;
        }
    }
    if table_seen {
        PatchState::TableWithoutKey
    } else {
        PatchState::Absent
    }
}

/// `cuda-bindings = ...` key line (whitespace-tolerant).
fn is_cuda_bindings_key(trimmed_line: &str) -> bool {
    trimmed_line
        .strip_prefix("cuda-bindings")
        .is_some_and(|rest| rest.trim_start().starts_with('='))
}

/// Whether the manifest's dependency set can reach `cuda-bindings` at all —
/// everything the host surface generates names `cuda-host`/`cuda-core`
/// directly, and both (plus the proc-macro's host feature) pull it
/// transitively. A key-sniff of the dependency tables; no value parsing.
fn pulls_cuda_bindings(manifest: &str) -> bool {
    let mut in_deps = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_deps = matches!(trimmed, "[dependencies]" | "[dev-dependencies]");
            continue;
        }
        if !in_deps || trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        let key = trimmed.split('=').next().unwrap_or("").trim();
        if matches!(key, "cuda-host" | "cuda-core" | "cuda-macros") {
            return true;
        }
    }
    false
}

/// Renders the stand-in path for the manifest: relative when the example
/// lives in the same repository tree as the stand-in (stable across
/// machines, same shape as the handwritten example tables), absolute
/// otherwise (external projects and batch variant crates under /tmp).
fn patch_path_for(example_dir: &Path, stand_in: &Path) -> String {
    let (Ok(example), Some(root)) = (example_dir.canonicalize(), repo_root(stand_in)) else {
        return absolute(stand_in);
    };
    let Ok(in_example) = example.strip_prefix(&root) else {
        return absolute(stand_in);
    };
    let Ok(in_root) = stand_in.strip_prefix(&root) else {
        return absolute(stand_in);
    };
    // From the example dir: climb out of `in_example`, then descend `in_root`.
    let mut rel = PathBuf::new();
    for _ in 0..in_example.components().count() {
        rel.push("..");
    }
    for part in in_root.components() {
        rel.push(part);
    }
    if rel.as_os_str().is_empty() {
        absolute(stand_in)
    } else {
        rel.to_string_lossy().into_owned()
    }
}

fn absolute(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// The repo root implied by the stand-in location (`<root>/host-hip/…`).
fn repo_root(stand_in: &Path) -> Option<PathBuf> {
    stand_in.parent().and_then(Path::parent).map(Path::to_path_buf)
}

fn table_block(stand_in_text: &str) -> String {
    format!(
        "\n{PATCH_MARKER}\n\
         # Swaps the crates.io cuda-bindings (dlopen libcuda, build-time CUDA\n\
         # SDK headers) for the HIP-backed stand-in; see\n\
         # host-hip/cuda-bindings/README.md.\n\
         [patch.crates-io]\n\
         cuda-bindings = {{ path = \"{stand_in_text}\" }}\n"
    )
}

fn append_marked_table(manifest: &str, stand_in_text: &str) -> String {
    let mut updated = manifest.to_string();
    if !updated.ends_with('\n') {
        updated.push('\n');
    }
    updated.push_str(&table_block(stand_in_text));
    updated
}

fn insert_into_existing_table(manifest: &str, stand_in_text: &str) -> String {
    // Find the line after the `[patch.crates-io]` section (next header or
    // EOF) and splice the key line in before it.
    let mut section_end = None;
    let mut in_patch_table = false;
    for (idx, line) in manifest.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            if in_patch_table {
                section_end = Some(idx);
                break;
            }
            in_patch_table = trimmed == "[patch.crates-io]";
        }
    }
    let key = format!("cuda-bindings = {{ path = \"{stand_in_text}\" }}");
    let marker_line = format!("{PATCH_MARKER} (key added to the existing table)");
    let Some(end) = section_end else {
        let mut updated = manifest.to_string();
        if !updated.ends_with('\n') {
            updated.push('\n');
        }
        updated.push_str(&marker_line);
        updated.push('\n');
        updated.push_str(&key);
        updated.push('\n');
        return updated;
    };
    let mut updated = String::with_capacity(manifest.len() + key.len() + marker_line.len() + 2);
    for (idx, line) in manifest.lines().enumerate() {
        if idx == end {
            updated.push_str(&marker_line);
            updated.push('\n');
            updated.push_str(&key);
            updated.push('\n');
        }
        updated.push_str(line);
        updated.push('\n');
    }
    updated
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDir(PathBuf);
    impl TestDir {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "cargo-oxide-hip-patch-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A fake stand-in tree (only the path matters).
    fn fake_repo(dir: &Path) -> (PathBuf, PathBuf) {
        let root = dir.join("repo");
        let stand_in = root.join("host-hip").join("cuda-bindings");
        std::fs::create_dir_all(&stand_in).unwrap();
        (root, stand_in)
    }

    fn manifest_of(example: &Path) -> String {
        std::fs::read_to_string(example.join("Cargo.toml")).unwrap()
    }

    #[test]
    fn gfx_target_grammar_matches_the_codegen_parser() {
        assert!(is_gfx_target(Some("gfx1030")));
        assert!(is_gfx_target(Some(" gfx90a ")));
        assert!(is_gfx_target(Some("gfx1030a")));
        assert!(!is_gfx_target(Some("gfx")));
        assert!(!is_gfx_target(Some("gfx1030X")));
        assert!(!is_gfx_target(Some("sm_86")));
        assert!(!is_gfx_target(Some("compute_90")));
        assert!(!is_gfx_target(None));
    }

    #[test]
    fn manifest_without_table_gets_a_marked_one() {
        let dir = TestDir::new("absent");
        let (_root, stand_in) = fake_repo(&dir.0);
        let example = dir.0.join("example");
        std::fs::create_dir_all(&example).unwrap();
        std::fs::write(
            example.join("Cargo.toml"),
            "[package]\nname = \"demo\"\n\n[dependencies]\ncuda-core = \"0.3.1\"\n",
        )
        .unwrap();

        ensure_hip_patch_with(&example, &stand_in);

        let text = manifest_of(&example);
        assert_eq!(text.matches("[patch.crates-io]").count(), 1);
        assert!(text.contains("cuda-bindings = { path = \""));
        assert!(text.contains(PATCH_MARKER));
        // Preceding manifest content untouched (append-only).
        assert!(text.starts_with("[package]\nname = \"demo\"\n"));
        assert!(text.contains("[dependencies]\ncuda-core = \"0.3.1\"\n"));
    }

    #[test]
    fn injection_is_idempotent() {
        let dir = TestDir::new("idempotent");
        let (_root, stand_in) = fake_repo(&dir.0);
        let example = dir.0.join("example");
        std::fs::create_dir_all(&example).unwrap();
        std::fs::write(
            example.join("Cargo.toml"),
            "[package]\nname = \"demo\"\n\n[dependencies]\ncuda-core = \"0.3.1\"\n",
        )
        .unwrap();

        ensure_hip_patch_with(&example, &stand_in);
        let once = manifest_of(&example);
        ensure_hip_patch_with(&example, &stand_in);
        let twice = manifest_of(&example);
        assert_eq!(once, twice);
        assert_eq!(twice.matches("[patch.crates-io]").count(), 1);
        assert_eq!(twice.matches("cuda-bindings =").count(), 1);
    }

    #[test]
    fn manual_patch_table_is_respected_verbatim() {
        let dir = TestDir::new("manual");
        let (_root, stand_in) = fake_repo(&dir.0);
        let example = dir.0.join("example");
        std::fs::create_dir_all(&example).unwrap();
        let manual = "[package]\nname = \"demo\"\n\n\
                      [dependencies]\n\
                      cuda-core = \"0.3.1\"\n\n\
                      # handwritten, no marker\n\
                      [patch.crates-io]\n\
                      cuda-bindings = { path = \"../../../../host-hip/cuda-bindings\" }\n";
        std::fs::write(example.join("Cargo.toml"), manual).unwrap();

        ensure_hip_patch_with(&example, &stand_in);

        assert_eq!(manifest_of(&example), manual);
    }

    #[test]
    fn existing_table_gains_the_key_without_a_second_table() {
        let dir = TestDir::new("existing-table");
        let (_root, stand_in) = fake_repo(&dir.0);
        let example = dir.0.join("example");
        std::fs::create_dir_all(&example).unwrap();
        std::fs::write(
            example.join("Cargo.toml"),
            "[package]\nname = \"demo\"\n\n\
             [dependencies]\n\
             cuda-core = \"0.3.1\"\n\n\
             [patch.crates-io]\n\
             other-crate = { path = \"../other\" }\n",
        )
        .unwrap();

        ensure_hip_patch_with(&example, &stand_in);

        let text = manifest_of(&example);
        assert_eq!(text.matches("[patch.crates-io]").count(), 1);
        assert!(text.contains("cuda-bindings = { path = \""));
        // The existing key stays inside the same table, before our insertion.
        let other_idx = text.find("other-crate").unwrap();
        let key_idx = text.find("cuda-bindings =").unwrap();
        assert!(other_idx < key_idx);
        // The edited manifest still parses as TOML with a patch table.
        let value: toml::Table = text.parse().expect("edited manifest must stay valid TOML");
        assert!(value.contains_key("patch"));
    }

    #[test]
    fn non_gfx_targets_never_touch_the_manifest() {
        let dir = TestDir::new("non-gfx");
        let (_root, _stand_in) = fake_repo(&dir.0);
        let example = dir.0.join("example");
        std::fs::create_dir_all(&example).unwrap();
        let original = "[package]\nname = \"demo\"\n";
        std::fs::write(example.join("Cargo.toml"), original).unwrap();

        // The gate must reject every non-gfx resolution — with and without an
        // ambient non-gfx env value — so the injection core is never reached
        // (and the manifest stays byte-identical).
        for arch in [None, Some("sm_86"), Some("compute_90")] {
            assert!(!is_gfx_activation(arch, None), "{arch:?} must not activate");
            assert!(!is_gfx_activation(arch, Some("sm_86")), "{arch:?} must not activate");
        }
        assert_eq!(manifest_of(&example), original);
    }

    #[test]
    fn activation_gate_resolves_configured_over_env() {
        // Configured (CLI/config) resolution wins over the inherited env.
        assert!(!is_gfx_activation(Some("sm_86"), Some("gfx1030")));
        assert!(is_gfx_activation(Some("gfx90a"), Some("sm_86")));
        // Env-only activation — the `configured_arch`-returns-None case the
        // backend child handles by inheriting the variable.
        assert!(is_gfx_activation(None, Some("gfx1030")));
        assert!(!is_gfx_activation(None, Some("sm_86")));
        assert!(!is_gfx_activation(None, None));
    }

    #[test]
    fn device_only_manifests_are_never_patched() {
        let dir = TestDir::new("device-only");
        let (_root, stand_in) = fake_repo(&dir.0);
        let example = dir.0.join("example");
        std::fs::create_dir_all(&example).unwrap();
        // The ox-amdgcn-probe shape: no cuda-host/core/macros edge, so the
        // host side can never link cuda-bindings and the committed manifest
        // must stay untouched on every gfx build.
        let original = "[package]\nname = \"probe\"\n\n[dependencies]\ncuda-device = { path = \"../../../cuda-device\" }\n";
        std::fs::write(example.join("Cargo.toml"), original).unwrap();

        ensure_hip_patch_with(&example, &stand_in);

        assert_eq!(manifest_of(&example), original);
    }

    #[test]
    fn in_repo_examples_get_a_relative_path_and_external_ones_an_absolute() {
        let dir = TestDir::new("relpath");
        let (root, stand_in) = fake_repo(&dir.0);

        // In-repo shape: example nested under the same repo root.
        let example = root.join("crates").join("demo");
        std::fs::create_dir_all(&example).unwrap();
        let text = patch_path_for(&example, &stand_in);
        assert!(
            text.contains("..") && text.ends_with("host-hip/cuda-bindings"),
            "expected a relative climb, got {text}"
        );

        // External shape: outside the repo root → absolute.
        let external = dir.0.join("elsewhere").join("probe");
        std::fs::create_dir_all(&external).unwrap();
        let text = patch_path_for(&external, &stand_in);
        assert!(text.starts_with('/'), "expected absolute, got {text}");
        assert!(text.ends_with("host-hip/cuda-bindings"));
    }

    #[test]
    fn deep_relative_climb_counts_every_component() {
        let dir = TestDir::new("deep");
        let (root, stand_in) = fake_repo(&dir.0);
        let example = root.join("a").join("b").join("c");
        std::fs::create_dir_all(&example).unwrap();
        let text = patch_path_for(&example, &stand_in);
        assert_eq!(text, "../../../host-hip/cuda-bindings");
    }
}
