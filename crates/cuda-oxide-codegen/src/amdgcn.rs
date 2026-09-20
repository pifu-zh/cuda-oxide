/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! [PORT gfx1030] AMDGPU (`amdgcn`) target support.
//!
//! Stage 1 of the gfx1030 integration: target-string parsing plus the NVVM →
//! AMDGPU IR *prep pass* that runs on the exported `.ll` before `llc
//! -march=amdgcn`.
//!
//! # Division of labor (what lives where)
//!
//! Most of the Phase-1 manual rewrite is done *natively* by earlier pipeline
//! stages under [`mir_lower::IntrinsicBackend::Amdgcn`] and
//! `AmdgcnExportConfig`:
//!
//! | Phase-1 step | Where it now happens |
//! |---|---|
//! | 1. triple → `amdgcn-amd-amdhsa` | `AmdgcnExportConfig::target_triple` |
//! | 2. datalayout → AMDGPU | `AmdgcnExportConfig::datalayout` |
//! | 3. `ptx_kernel` → `amdgpu_kernel` | `AmdgcnExportConfig::kernel_callconv_keyword` |
//! | 4. ctaid/tid → `llvm.amdgcn.{workgroup,workitem}.id.*` | mir-lower sreg converters |
//!
//! What remains here is exactly the part that *cannot* be expressed at those
//! levels, Rust-ified from `tests/translate_amdgcn.py` (steps 5-7 of the
//! Phase-1-validated sequence, each step GPU-verified on gfx1030):
//!
//! 5. `ntid.x` reads → a new trailing `i32 %ntid_x` kernel parameter
//!    (`blockDim.x` is host policy on AMDGPU: no LLVM IR intrinsic exists, and
//!    a lowering-time conversion cannot change the kernel signature). Call
//!    sites become `or i32 %ntid_x, 0` — an identity op producing a fresh SSA
//!    value so no use-site renaming is needed. The host must pass the extra
//!    kernarg (mirrors Phase-1 host_v6: `kparams` gains `&ntid`).
//! 6. `ntid.{y,z}` / `nctaid.{y,z}` reads → `or i32 0, 1` (constant 1; the
//!    kernel's multi-dimension guard assumes 1-D launches — same semantics
//!    decision as Phase 1).
//! 7. the `declare`s of the rewritten NVVM sreg intrinsics are deleted;
//!    `llc` auto-declares the AMDGPU target intrinsics, and leftover NVVM
//!    declares would leak into the code object as undefined externs.
//!
//! Anything else NVVM-shaped (e.g. `nctaid.x`, barriers, shuffles) is left
//! untouched on purpose: `llc` rejects it with an explicit `Cannot select`
//! error — the "capability not yet covered" signal, never a silent wrong
//! result (Stage 3 maps those families one by one).
//!
//! # Idempotence
//!
//! Like the Phase-1 script, [`rewrite_ir_for_amdgcn`] is single-pass and
//! idempotent: it always runs on the pristine exporter output and never on
//! its own output (the pipeline writes the result to a separate
//! `<name>.amdgcn.ll` file). Re-running on already-rewritten text is a no-op.

/// Parses an explicit target override into the AMDGPU path.
///
/// Returns the canonicalized `gfx…` spelling (e.g. `gfx1030`) when `arch`
/// names an AMDGPU target: the literal prefix `gfx` followed by ASCII digits.
/// Anything else (including every NVIDIA `sm_…` value and `None`) is `None`,
/// so the default NVPTX pipeline is selected exactly as before — setting the
/// variable to a gfx target is the only way to enter the AMDGPU path.
pub fn amdgcn_target(arch: Option<&str>) -> Option<&str> {
    // `gfx` + digits + optional lowercase product suffix (gfx90a, gfx1101…).
    // Deliberately permissive: the authoritative check is `llc -mcpu=<gfx>`
    // rejecting unknown chips with a clear error.
    let arch = arch?.trim();
    let rest = arch.strip_prefix("gfx")?;
    let split = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    let (digits, suffix) = rest.split_at(split);
    let valid = !digits.is_empty()
        && suffix.bytes().all(|b| b.is_ascii_lowercase());
    valid.then_some(arch)
}

/// Whether a line is a typed-pointer-IR `bitcast <ty>* %name` use (the
/// non-`opt` IR shape where an addrspace(5) alloca would make the bitcast
/// illegal across address spaces).
fn has_typed_pointer_bitcast(ll: &str) -> bool {
    ll.lines().any(|line| {
        let Some(idx) = line.find("bitcast") else {
            return false;
        };
        line[idx..].split(';').next().is_some_and(|code| code.contains("* %"))
    })
}

/// Splits `s` on top-level commas: `<`, `[`, `(`, `{` open a group and `,`
/// inside a group does not split (port of the script's
/// `_split_top_level_commas`; [PORT gfx1030 Stage3-1] `{}` added — struct
/// alloca types like `{ i32, i1, [3 x i8] }` otherwise split at their
/// interior commas and the alloca rewriter silently skipped them, tripping
/// the `alloca on amdgpu must be in addrspace(5)` verifier (batch
/// first_error class: layering — only visible once an example used
/// non-trivial aggregate locals).
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '<' | '[' | '(' | '{' => depth += 1,
            '>' | ']' | ')' | '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(s[start..].trim());
    parts
}

/// Replaces whole-token occurrences of `name` with `repl` (token = not
/// surrounded by [`w.$`]).
fn replace_token(line: &str, name: &str, repl: &str) -> String {
    let is_ident = |c: char| c.is_alphanumeric() || c == '_' || c == '.' || c == '$';
    let mut out = String::with_capacity(line.len() + repl.len());
    let mut rest = line;
    while let Some(pos) = rest.find(name) {
        let before_ok = pos == 0 || !rest[..pos].chars().next_back().is_some_and(is_ident);
        let after = &rest[pos + name.len()..];
        let after_ok = !after.chars().next().is_some_and(is_ident);
        if before_ok && after_ok {
            out.push_str(&rest[..pos]);
            out.push_str(repl);
            rest = after;
        } else {
            out.push_str(&rest[..pos + name.len()]);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Rewrites generic-address-space allocas to `addrspace(5)` inside every
/// function, inserting one `addrspacecast` per alloca right after its
/// definition and renaming uses to the cast (Rust port of the script's
/// `_rewrite_allocas`, gfx1030 batch-verified).
///
/// Correctness (script doc): an entry-block alloca dominates every use, so
/// one cast per alloca suffices; use replacement is a token-level SSA rename.
/// `llvm.lifetime`/`dbg` uses that require the alloca's own address space do
/// not occur in cuda-oxide output (14-example batch grep evidence); if one
/// ever does, the module fails verification explicitly instead of silently.
fn rewrite_allocas(ll: &str) -> String {
    if !ll.contains("= alloca ") || has_typed_pointer_bitcast(ll) {
        return ll.to_string();
    }
    let lines: Vec<&str> = ll.lines().collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0;
    while i < lines.len() {
        if !(lines[i].starts_with("define ") && lines[i].trim_end().ends_with('{')) {
            out.push(lines[i].to_string());
            i += 1;
            continue;
        }
        // Collect the function body (define line .. closing "}" line).
        let func_start = i;
        let mut end = i + 1;
        while end < lines.len() && lines[end] != "}" {
            end += 1;
        }
        let body: Vec<&str> = lines[func_start..=end.min(lines.len() - 1)].to_vec();
        out.extend(rewrite_allocas_in_function(&body));
        i = end + 1;
    }
    let mut text = out.join("\n");
    if ll.ends_with('\n') && !text.is_empty() {
        text.push('\n');
    }
    text
}

/// Single-function alloca rewrite (body = define line .. "}" line).
fn rewrite_allocas_in_function(body: &[&str]) -> Vec<String> {
    struct Alloca<'a> {
        line: usize,
        name: &'a str,
        new_def: String,
    }
    let mut allocas: Vec<Alloca> = Vec::new();
    for (k, line) in body.iter().enumerate() {
        let trimmed = line.trim_start();
        let indent_len = line.len() - trimmed.len();
        let Some(eq) = trimmed.find(" = alloca ") else {
            continue;
        };
        if !trimmed.starts_with('%') {
            continue;
        }
        let name = &trimmed[..eq];
        // `%` prefix + [\w.$] body (script's `%[\w.$]+`).
        if !name.starts_with('%')
            || !name[1..]
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'$'))
        {
            continue;
        }
        let rest = trimmed[eq + " = alloca ".len()..].trim();
        if rest.contains('-') || rest.contains('>') {
            continue;
        }
        // Script order: strip the trailing `, align N` FIRST, then split the
        // remaining `<ty>[, <count>]` on top-level commas.
        let mut align = None;
        let mut head = rest;
        if let Some(idx) = rest.rfind(", align ") {
            let candidate = rest[idx + ", align ".len()..].trim();
            if !candidate.is_empty() && candidate.bytes().all(|b| b.is_ascii_digit()) {
                align = Some(candidate);
                head = rest[..idx].trim();
            }
        }
        let parts = split_top_level_commas(head);
        if parts.len() > 2 || parts.first() == Some(&"void") {
            continue; // unparseable shape: keep as-is, llc verifier fails loudly
        }
        // [PORT gfx1030 Stage3-1] Skip only allocas whose OWN address space is
        // already specified as a trailing parameter. The old whole-string
        // `rest.contains("addrspace(")` guard also false-positived on allocas
        // whose ELEMENT type is an addrspace pointer (e.g.
        // `[2 x ptr addrspace(3)]` — a generic-stack slot holding shared
        // pointers), leaving them generic and tripping the verifier.
        if parts.iter().skip(1).any(|part| {
            part.starts_with("addrspace(") && part.ends_with(')')
        }) {
            continue;
        }
        let ty = parts[0];
        let count = parts.get(1).copied();
        let mut new_def = format!("{}{} = alloca {}", &line[..indent_len], name, ty);
        if let Some(count) = count {
            new_def.push_str(&format!(", {count}"));
        }
        if let Some(align) = align {
            new_def.push_str(&format!(", align {align}"));
        }
        new_def.push_str(", addrspace(5)");
        allocas.push(Alloca {
            line: k,
            name,
            new_def,
        });
    }
    if allocas.is_empty() {
        return body.iter().map(|l| l.to_string()).collect();
    }
    let mut out: Vec<String> = Vec::with_capacity(body.len() + allocas.len());
    for (k, line) in body.iter().enumerate() {
        if let Some(alloca) = allocas.iter().find(|a| a.line == k) {
            // Alloca definition line: new def + the cast right after it (the
            // alloca dominates all uses, so one cast serves the function).
            out.push(alloca.new_def.clone());
            out.push(format!(
                "{}.ac = addrspacecast ptr addrspace(5) {} to ptr",
                alloca.name, alloca.name
            ));
            continue;
        }
        let mut line = line.to_string();
        for alloca in &allocas {
            line = replace_token(&line, alloca.name, &format!("{}.ac", alloca.name));
        }
        out.push(line);
    }
    out
}

/// Removes a `declare` line for one fully-qualified intrinsic name.
fn strip_declares(ll: &str, callee: &str) -> String {
    let needle = format!("@{callee}(");
    let mut text = ll
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !(trimmed.starts_with("declare ") && trimmed.contains(&needle))
        })
        .collect::<Vec<_>>()
        .join("\n");
    if ll.ends_with('\n') && !text.is_empty() {
        text.push('\n');
    }
    text
}

/// Replaces every `(tail )?call i32 @callee() #N` value with `replacement`.
///
/// Only the right-hand side after `= ` changes; the SSA name on the left and
/// every use of it stay untouched (the Phase-1 "same-name replacement"
/// discipline). Returns the text plus whether anything was rewritten.
fn rewrite_calls(ll: &str, callee: &str, replacement: &str) -> (String, bool) {
    let call_prefix = format!("i32 @{callee}()");
    let mut changed = false;
    let mut out = Vec::with_capacity(ll.lines().count().max(1));
    for line in ll.lines() {
        let Some(eq) = line.find(" = ") else {
            out.push(line.to_string());
            continue;
        };
        let rhs = line[eq + 3..].trim_start();
        let body = rhs
            .strip_prefix("tail call ")
            .or_else(|| rhs.strip_prefix("call "));
        let Some(body) = body else {
            out.push(line.to_string());
            continue;
        };
        let Some(attrs) = body.strip_prefix(&call_prefix) else {
            out.push(line.to_string());
            continue;
        };
        // Only bare calls or calls with `#N` attribute suffixes are rewrite
        // targets; anything else (unusual spellings) is left for llc to
        // reject explicitly rather than silently mangled.
        let attrs = attrs.trim_start();
        let is_bare = attrs.is_empty();
        let is_attr_list = attrs.starts_with('#')
            && attrs
                .split_whitespace()
                .all(|token| token.starts_with('#') && token[1..].bytes().all(|b| b.is_ascii_digit()));
        if !(is_bare || is_attr_list) {
            out.push(line.to_string());
            continue;
        }
        let indent = &line[..line.len() - line.trim_start().len()];
        out.push(format!("{indent}{} = {replacement}", &line[..eq]));
        changed = true;
    }
    let mut text = out.join("\n");
    if ll.ends_with('\n') && !text.is_empty() {
        text.push('\n');
    }
    (text, changed)
}

/// Appends a trailing `i32 %ntid_x` parameter to every kernel `define` whose
/// body references `%ntid_x` (Rust port of the Phase-1 script's
/// `_append_ntid_kernarg_param`).
///
/// The parameter list ends at the last `)` before the final `{`; the text
/// between may only be whitespace, attribute group references (`#N`), flags
/// and commas (`re.fullmatch(r"[\s#\w,=]*")` in the script). A `())` inside
/// the parameters (e.g. `captures(none)`) always sits *before* that closing
/// parenthesis, so the heuristic cannot cut inside a nested group.
fn append_ntid_kernarg_param(ll: &str) -> String {
    let lines: Vec<&str> = ll.lines().collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if !(line.starts_with("define ") && line.trim_end().ends_with('{')) {
            out.push(line.to_string());
            i += 1;
            continue;
        }
        let mut block: Vec<String> = vec![line.to_string()];
        i += 1;
        while i < lines.len() && lines[i] != "}" {
            block.push(lines[i].to_string());
            i += 1;
        }
        if i < lines.len() {
            block.push(lines[i].to_string());
            i += 1;
        }
        let define = block[0].clone();
        let body_uses = block.iter().skip(1).any(|l| l.contains("%ntid_x"));
        if body_uses && !define.contains("%ntid_x") {
            if let Some(brace) = define.rfind('{') {
                if let Some(paren) = define[..brace].rfind(')') {
                    let between = &define[paren + 1..brace];
                    let between_ok = between
                        .chars()
                        .all(|c| c.is_whitespace() || c.is_ascii_alphanumeric() || "#,=_".contains(c));
                    if between_ok {
                        // Zero-parameter functions take the parameter without
                        // a leading comma (a case the Phase-1 script's fixed
                        // `", i32 %ntid_x"` splice would have mangled).
                        let open = define[..paren].rfind('(').unwrap_or(paren);
                        let insertion = if define[open + 1..paren].trim().is_empty() {
                            "i32 %ntid_x"
                        } else {
                            ", i32 %ntid_x"
                        };
                        block[0] = format!(
                            "{}{}{}",
                            &define[..paren],
                            insertion,
                            &define[paren..]
                        );
                    }
                }
            }
        }
        out.extend(block.into_iter());
    }
    let mut text = out.join("\n");
    if ll.ends_with('\n') && !text.is_empty() {
        text.push('\n');
    }
    text
}

/// Runs the NVVM → AMDGPU prep pass over exported LLVM IR text.
///
/// See the [module documentation](self) for the step table and why each
/// rewrite lives here instead of in lowering or export. Errors only when the
/// input is not text this pass understands (currently: never — unknown
/// constructs pass through untouched for `llc` to accept or reject).
pub fn rewrite_ir_for_amdgcn(ll: &str) -> Result<String, String> {
    // 5. ntid.x → new kernarg parameter.
    let mut text = rewrite_calls(ll, "llvm.nvvm.read.ptx.sreg.ntid.x", "or i32 %ntid_x, 0").0;
    text = append_ntid_kernarg_param(&text);
    // 6. ntid/nctaid .y/.z → constant 1 (1-D launch semantics, Phase-1 step 6).
    const ONE_D_CONSTANTS: [(&str, &str); 4] = [
        ("llvm.nvvm.read.ptx.sreg.ntid.y", "or i32 0, 1"),
        ("llvm.nvvm.read.ptx.sreg.ntid.z", "or i32 0, 1"),
        ("llvm.nvvm.read.ptx.sreg.nctaid.y", "or i32 0, 1"),
        ("llvm.nvvm.read.ptx.sreg.nctaid.z", "or i32 0, 1"),
    ];
    for (callee, constant) in ONE_D_CONSTANTS {
        text = rewrite_calls(&text, callee, constant).0;
    }
    // 7. drop the declares of everything this pass rewrote (and, defensively,
    //    of the sregs the lowering already remapped natively — a module
    //    compiled under a different backend could still carry them). Unmapped
    //    declares stay, so their calls fail at llc with a legible name.
    for callee in [
        "llvm.nvvm.read.ptx.sreg.ntid.x",
        "llvm.nvvm.read.ptx.sreg.ntid.y",
        "llvm.nvvm.read.ptx.sreg.ntid.z",
        "llvm.nvvm.read.ptx.sreg.nctaid.y",
        "llvm.nvvm.read.ptx.sreg.nctaid.z",
        "llvm.nvvm.read.ptx.sreg.ctaid.x",
        "llvm.nvvm.read.ptx.sreg.tid.x",
    ] {
        text = strip_declares(&text, callee);
    }
    // 8. generic allocas → addrspace(5) + per-alloca addrspacecast at the use
    //    sites (AMDGPU module verifier contract; batch-verified).
    text = rewrite_allocas(&text);
    Ok(text)
}

/// The pre-`opt` half of the prep pass: only the alloca → `addrspace(5)`
/// rewrite (step 8), which `opt` requires just to *accept* an amdgcn-layout
/// module.
///
/// `opt` must run BEFORE the ntid/nctaid rewrite (steps 5-7): `alwaysinline`
/// device helpers that read `ntid` fold into their kernels there, so the
/// kernarg parameter lands on the kernel's own signature exactly like the
/// Phase-1 pipeline (which rewrote post-`opt` output). Appending the
/// parameter to an as-yet-uninlined device function would leave its call
/// sites at the old arity — a silently mismatched ABI.
pub fn rewrite_allocas_for_amdgcn(ll: &str) -> Result<String, String> {
    Ok(rewrite_allocas(ll))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx_targets_parse_and_sm_targets_do_not() {
        assert_eq!(amdgcn_target(Some("gfx1030")), Some("gfx1030"));
        assert_eq!(amdgcn_target(Some(" gfx90a ")), Some("gfx90a"));
        assert_eq!(amdgcn_target(Some("gfx1030a")), Some("gfx1030a"));
        assert_eq!(amdgcn_target(Some("gfx")), None);
        assert_eq!(amdgcn_target(Some("gfx1030X")), None);
        assert_eq!(amdgcn_target(Some("sm_80")), None);
        assert_eq!(amdgcn_target(None), None);
    }

    /// Shape of the exported probe kernel (nvptx-free: lowering already
    /// remapped ctaid/tid) reduced to the constructs this pass touches.
    const PROBE_SHAPED_LL: &str = "\
define amdgpu_kernel void @vecadd(ptr nofree nonnull readonly align 4 captures(none) %v0, i64 %v1, ptr nofree writeonly captures(address_is_null) %v2, i64 %v3) #0 {
entry:
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2
  %v5.i = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %v6.i = zext nneg i32 %v3.i to i64
  br label %bb8

bb8:
  ret void

bb16:
  tail call void @llvm.trap() #2
  unreachable
}

declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #1

declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #1

declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { convergent nounwind }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent }
";

    #[test]
    fn probe_shaped_module_is_rewritten_like_phase1() {
        let out = rewrite_ir_for_amdgcn(PROBE_SHAPED_LL).unwrap();

        // 5. ntid.x call became the identity-or and the signature gained the
        //    kernarg parameter right before the closing paren / `{`.
        assert!(out.contains("  %v3.i = or i32 %ntid_x, 0\n"));
        assert!(out.contains(
            "ptr nofree writeonly captures(address_is_null) %v2, i64 %v3, i32 %ntid_x) #0 {"
        ));
        // 6. ntid/nctaid y|z became constant 1.
        assert!(out.contains("  %v4.i2 = or i32 0, 1\n"));
        assert!(out.contains("  %v5.i = or i32 0, 1\n"));
        // 7. rewritten intrinsics' declares are gone…
        assert!(!out.contains("@llvm.nvvm.read.ptx.sreg.ntid.x()"));
        assert!(!out.contains("@llvm.nvvm.read.ptx.sreg.ntid.y()"));
        assert!(!out.contains("@llvm.nvvm.read.ptx.sreg.nctaid.z()"));
        // …but everything else survives: the amdgcn declare (llc would also
        // auto-declare it), the untouched trap call, and the attrs groups.
        assert!(out.contains("@llvm.amdgcn.workitem.id.x()"));
        assert!(out.contains("tail call void @llvm.trap() #2"));
        assert!(out.contains("attributes #1 = { mustprogress nocallback"));
    }

    #[test]
    fn rewrite_is_idempotent() {
        let once = rewrite_ir_for_amdgcn(PROBE_SHAPED_LL).unwrap();
        let twice = rewrite_ir_for_amdgcn(&once).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn defines_without_ntid_are_untouched() {
        let ll = "\
define amdgpu_kernel void @other(ptr %p) #0 {
entry:
  ret void
}
";
        assert_eq!(rewrite_ir_for_amdgcn(ll).unwrap(), ll);
    }

    #[test]
    fn nested_parens_do_not_confuse_the_signature_append() {
        // `captures(none)` closes before the parameter-list paren; the append
        // must target the parameter-list paren, not the nested one.
        let ll = "\
define amdgpu_kernel void @k(ptr captures(none) %v0) #0 {
entry:
  %v1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1
  ret void
}
";
        let out = rewrite_ir_for_amdgcn(ll).unwrap();
        assert!(out.contains("@k(ptr captures(none) %v0, i32 %ntid_x) #0 {"));
    }

    #[test]
    fn device_functions_gain_the_parameter_like_phase1() {
        // The Phase-1 script appends the kernarg parameter to ANY `define`
        // whose body reads ntid.x — this port keeps that behavior verbatim.
        // Recorded limitation (Stage-1 contract): a device function that
        // reads ntid must only be called by kernels whose own signature was
        // extended consistently; the common case (kernels only) is exact.
        let ll = "\
define hidden i32 @helper() #0 {
entry:
  %v0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1
  ret i32 %v0
}
";
        let out = rewrite_ir_for_amdgcn(ll).unwrap();
        assert!(out.contains("= or i32 %ntid_x, 0"));
        assert!(out.contains("define hidden i32 @helper(i32 %ntid_x) #0 {"));
    }

    #[test]
    fn zst_alloca_moves_to_addrspace5_with_use_renamed() {
        // The exact shape that failed `opt` on the probe example.
        let ll = "\
define amdgpu_kernel void @k(i64 %n) #0 {
entry:
  %v15 = alloca {}, align 1
  %v16 = getelementptr inbounds {}, ptr %v15, i64 0
  ret void
}
";
        let out = rewrite_ir_for_amdgcn(ll).unwrap();
        assert!(out.contains("  %v15 = alloca {}, align 1, addrspace(5)\n"));
        assert!(out.contains("%v15.ac = addrspacecast ptr addrspace(5) %v15 to ptr\n"));
        assert!(out.contains("getelementptr inbounds {}, ptr %v15.ac, i64 0"));
        // no bare uses of the alloca name outside its own definition/cast
        for line in out.lines() {
            if line.contains("%v15")
                && !line.contains("alloca")
                && !line.contains("addrspacecast")
            {
                assert!(line.contains("%v15.ac"), "bare use survived: {line}");
            }
        }
    }

    #[test]
    fn allocas_already_in_addrspace5_are_untouched() {
        let ll = "\
define amdgpu_kernel void @k() #0 {
entry:
  %buf = alloca [16 x i32], align 4, addrspace(5)
  %g = addrspacecast ptr addrspace(5) %buf to ptr
  ret void
}
";
        assert_eq!(rewrite_ir_for_amdgcn(ll).unwrap(), ll);
    }

    #[test]
    fn typed_pointer_bitcast_shapes_are_left_for_an_explicit_failure() {
        let ll = "\
define amdgpu_kernel void @k() #0 {
entry:
  %p = alloca i8, align 1
  %q = bitcast i8* %p to i32*
  ret void
}
";
        // Typed-pointer IR is rejected upstream anyway; the pass must not
        // produce an illegal cross-address-space bitcast.
        assert!(rewrite_ir_for_amdgcn(ll).unwrap().contains("alloca i8, align 1\n"));
    }
}
