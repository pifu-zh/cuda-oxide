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

// ---------------------------------------------------------------------------
// [PORT gfx1030 PhaseB.2] warp shuffle → ds_bpermute (Step 10) + the f64
// PTX-asm shuffle split (Step 10c), ported from tests/translate_amdgcn.py
// (GPU-verified on gfx1030: four modes × f32/i32 + f64, wave64 32 warps).
//
// Semantics (wave64 machines keep CUDA's 32-lane warp semantics via 32-aligned
// segmentation): lane = mbcnt.lo(-1, 0); seg = lane & -32;
//   idx : src = seg + (delta & 31)
//   bfly: src = lane ^ delta            (delta < 32 never crosses a segment)
//   down: t = (lane&31)+delta; src = t <= 31 ? seg+t : lane
//   up  : t = (lane&31)-delta; src = in-segment ? seg+t : lane
//   off = src * 4  ← the core cross-arch difference: ds_bpermute takes a BYTE
//                    offset (lane*4) where NVVM shfl takes a lane number.
// Conservative boundary (port discipline 5): membermask != -1 (partial warp)
// and clamp != 31 (width != 32) stay as the nvvm call so llc fails with an
// explicit `Cannot select` instead of silently mis-shuffling.
// ---------------------------------------------------------------------------

/// One parsed `@llvm.nvvm.shfl.sync.<mode>.<st>` call line.
struct ShflLine<'a> {
    indent: &'a str,
    res: &'a str,
    ty: &'a str,
    mode: &'a str,
    delta: &'a str,
    val: &'a str,
    mask: &'a str,
    clamp: &'a str,
}

fn match_shfl_line(line: &str) -> Option<ShflLine<'_>> {
    let (indent, res, rhs) = split_assign(line)?;
    let call = strip_call_prefix(rhs)?;
    let (ty, rest) = match call.split_once(' ')? {
        ("float", rest) => ("float", rest),
        ("i32", rest) => ("i32", rest),
        _ => return None,
    };
    let callee = rest.strip_prefix("@llvm.nvvm.shfl.sync.")?;
    let (mode, rest) = callee.split_once('.')?;
    if !matches!(mode, "idx" | "bfly" | "up" | "down") {
        return None;
    }
    let (st, args_text) = rest.split_once('(')?;
    if !matches!(st, "f32" | "i32") {
        return None;
    }
    // The script's `[^)]*`: shfl arguments never contain a nested `)`.
    let close = args_text.find(')')?;
    if !is_attrs_tail(&args_text[close + 1..]) {
        return None;
    }
    let args = split_top_level_commas(&args_text[..close]);
    if args.len() != 4 {
        return None;
    }
    // (ty, st) type consistency: float ↔ f32, i32 ↔ i32.
    if (ty == "float") != (st == "f32") {
        return None;
    }
    Some(ShflLine {
        indent,
        res,
        ty,
        mode,
        delta: arg_operand(args[2])?,
        val: arg_operand(args[1])?,
        mask: arg_operand(args[0])?,
        clamp: arg_operand(args[3])?,
    })
}

/// The Step-10 driver: rewrite every shfl call line whose mask/clamp pair is
/// in the supported envelope; other shapes pass through untouched.
fn rewrite_shuffle(ll: &str) -> String {
    if !ll.contains("shfl.sync.") {
        return ll.to_string();
    }
    let mut out = Vec::with_capacity(ll.lines().count());
    for line in ll.lines() {
        let expanded = match_shfl_line(line)
            .filter(|shfl| shfl.mask == "-1" && shfl.clamp == "31")
            .map(expand_shfl);
        match expanded {
            Some(lines) => out.extend(lines),
            None => out.push(line.to_string()),
        }
    }
    let mut text = out.join("\n");
    if ll.ends_with('\n') && !text.is_empty() {
        text.push('\n');
    }
    text
}

/// The straight-line ds_bpermute expansion for one shfl call (script's
/// `_expand_shfl`); the result keeps the original SSA name so every
/// downstream use is untouched.
fn expand_shfl(shfl: ShflLine<'_>) -> Vec<String> {
    let ShflLine {
        indent,
        res,
        ty,
        mode,
        delta,
        val,
        ..
    } = shfl;
    let base = format!("{res}.b_");
    let mut seq = Vec::with_capacity(11);
    let mut def = |suffix: &str, rhs: String| {
        seq.push(format!("{indent}{base}{suffix} = {rhs}"));
    };
    def(
        "ln",
        "call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)".to_string(),
    );
    match mode {
        "bfly" => def("src", format!("xor i32 {base}ln, {delta}")),
        "idx" => {
            def("seg", format!("and i32 {base}ln, -32"));
            def("d", format!("and i32 {delta}, 31"));
            def("src", format!("add i32 {base}seg, {base}d"));
        }
        "down" => {
            def("seg", format!("and i32 {base}ln, -32"));
            def("lo", format!("and i32 {base}ln, 31"));
            def("t", format!("add i32 {base}lo, {delta}"));
            def("in", format!("icmp ule i32 {base}t, 31"));
            def("s2", format!("add i32 {base}seg, {base}t"));
            def(
                "src",
                format!("select i1 {base}in, i32 {base}s2, i32 {base}ln"),
            );
        }
        _ => {
            // "up"
            def("seg", format!("and i32 {base}ln, -32"));
            def("lo", format!("and i32 {base}ln, 31"));
            def("t", format!("sub i32 {base}lo, {delta}"));
            def("in", format!("icmp uge i32 {base}lo, {delta}"));
            def("s2", format!("add i32 {base}seg, {base}t"));
            def(
                "src",
                format!("select i1 {base}in, i32 {base}s2, i32 {base}ln"),
            );
        }
    }
    def("off", format!("shl i32 {base}src, 2"));
    if ty == "float" {
        def("bits", format!("bitcast float {val} to i32"));
        def(
            "got",
            format!("call i32 @llvm.amdgcn.ds.bpermute(i32 {base}off, i32 {base}bits)"),
        );
        seq.push(format!("{indent}{res} = bitcast i32 {base}got to float"));
    } else {
        seq.push(format!(
            "{indent}{res} = call i32 @llvm.amdgcn.ds.bpermute(i32 {base}off, i32 {val})"
        ));
    }
    seq
}

/// Step 10c: cuda-oxide's f64 shuffle emits an embedded PTX asm block
/// (`shfl.sync.bfly.b32` lo/hi split, same lineage as aiter's warp.rs). The
/// AMDGPU backend rejects the PTX asm outright, so the recognized call shape
/// expands into two ds_bpermute round trips plus an i64 recombine.
fn rewrite_shfl_asm(ll: &str) -> String {
    if !ll.contains("shfl.sync.bfly.b32") {
        return ll.to_string();
    }
    let mut out = Vec::with_capacity(ll.lines().count());
    for line in ll.lines() {
        let expanded = match_shfl_asm_line(line).filter(|parts| parts.mask == "-1");
        match expanded {
            Some(parts) => out.extend(expand_shfl_asm(parts)),
            None => out.push(line.to_string()),
        }
    }
    let mut text = out.join("\n");
    if ll.ends_with('\n') && !text.is_empty() {
        text.push('\n');
    }
    text
}

/// One parsed f64 asm shuffle line (script's `_SHFL_ASM_RE`).
struct ShflAsmLine<'a> {
    indent: &'a str,
    res: &'a str,
    val: &'a str,
    delta: &'a str,
    mask: &'a str,
}

fn match_shfl_asm_line(line: &str) -> Option<ShflAsmLine<'_>> {
    let (indent, res, rhs) = split_assign(line)?;
    let call = strip_call_prefix(rhs)?;
    let rest = call.strip_prefix("i64 asm sideeffect \"")?;
    // These generated blobs contain no escaped quotes; the script's `[^"]*`
    // anchor is preserved by finding the first closing quote.
    let close = rest.find('"')?;
    if !rest[..close].contains("shfl.sync.bfly.b32 lo") {
        return None;
    }
    let after = rest[close + 1..].trim_start();
    let after = after.strip_prefix(',')?.trim_start();
    let after = after.strip_prefix("\"=l,l,r,r\"(")?;
    let close_paren = after.rfind(')')?;
    if !is_attrs_tail(&after[close_paren + 1..]) {
        return None;
    }
    let args = split_top_level_commas(&after[..close_paren]);
    if args.len() != 3 {
        return None;
    }
    // (i64 <val>, i32 <delta>, i32 <mask>)
    let val = arg_operand(args[0])?;
    if !args[0].trim().starts_with("i64 ") {
        return None;
    }
    Some(ShflAsmLine {
        indent,
        res,
        val,
        delta: arg_operand(args[1])?,
        mask: arg_operand(args[2])?,
    })
}

fn expand_shfl_asm(parts: ShflAsmLine<'_>) -> Vec<String> {
    let ShflAsmLine {
        indent,
        res,
        val,
        delta,
        ..
    } = parts;
    let base = format!("{res}.b_");
    let mut seq = Vec::with_capacity(12);
    let mut def = |suffix: &str, rhs: String| {
        seq.push(format!("{indent}{base}{suffix} = {rhs}"));
    };
    def(
        "ln",
        "call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)".to_string(),
    );
    def("src", format!("xor i32 {base}ln, {delta}"));
    def("off", format!("shl i32 {base}src, 2"));
    def("lo", format!("trunc i64 {val} to i32"));
    def("hi64", format!("lshr i64 {val}, 32"));
    def("hi", format!("trunc i64 {base}hi64 to i32"));
    def(
        "glo",
        format!("call i32 @llvm.amdgcn.ds.bpermute(i32 {base}off, i32 {base}lo)"),
    );
    def(
        "ghi",
        format!("call i32 @llvm.amdgcn.ds.bpermute(i32 {base}off, i32 {base}hi)"),
    );
    def("ghi64", format!("zext i32 {base}ghi to i64"));
    def("ghiup", format!("shl i64 {base}ghi64, 32"));
    def("glo64", format!("zext i32 {base}glo to i64"));
    seq.push(format!("{indent}{res} = or i64 {base}glo64, {base}ghiup"));
    seq
}

// ---------------------------------------------------------------------------
// [PORT gfx1030 PhaseB.3] atomics family (Step 12/12d), ported from
// tests/translate_amdgcn.py (GPU-verified: is.* classification, generic
// fetch_add + fences, MP litmus release→acquire 200 rounds).
//
// 12  isspacep.local/shared/global → llvm.amdgcn.is.private/shared/global
//     (HIP semantics: NV thread-local scratch = AMD private AS(5)); same-name
//     replacement keeps every SSA use. membar.gl/cta/sys → fence in the
//     agent/workgroup/system domain. NV scope names are not accepted by the
//     AMDGPU backend ("Unsupported atomic synchronization scope") → device→
//     agent, block→workgroup. Layering matters (v3 lesson 4): isspacep must
//     clear before the membar/scope rewrites, or later passes crash into the
//     leftovers.
// 12d Rust core::sync::atomic's embedded PTX-asm fallbacks (fence.acq_rel/,
//     ld.acquire., fence.sc+ld.acquire., st.release.) → LLVM atomic
//     instructions; the seqcst asm pair folds into a single load atomic
//     seq_cst (the stronger single-instruction equivalent).
// ---------------------------------------------------------------------------

/// NV scope → AMD syncscope text ("" = system domain, default scope).
fn amd_scope(scope: &str) -> Option<&'static str> {
    match scope {
        "cta" | "block" => Some("syncscope(\"workgroup\")"),
        // `gl` is membar.gl's grid-level (= device/agent) scope; `device`/
        // `block` are the atomicrmw/cmpxchg spellings.
        "gl" | "gpu" | "device" => Some("syncscope(\"agent\")"),
        "sys" => Some(""),
        _ => None,
    }
}

/// Strips one leading `nonnull ` parameter attribute (the script's capture
/// of the same optional group).
fn strip_nonnull(operand: &str) -> &str {
    operand
        .trim()
        .strip_prefix("nonnull ")
        .map(str::trim_start)
        .unwrap_or_else(|| operand.trim())
}

/// Strips one leading `nonnull `/`noundef` parameter attribute (the script's
/// `_strip_param_attrs`).
fn strip_param_attrs(operand: &str) -> &str {
    let trimmed = operand.trim();
    if let Some(rest) = trimmed.strip_prefix("nonnull ") {
        rest.trim_start()
    } else if let Some(rest) = trimmed.strip_prefix("noundef ") {
        rest.trim_start()
    } else {
        trimmed
    }
}

/// `fence.acq_rel.cta;`-style asm body → its (order, scope) pair.
fn match_ptx_fence_asm(body: &str) -> Option<(&str, &str)> {
    let rest = body.strip_prefix("fence.")?;
    let (ord, scope) = rest.split_once('.')?;
    if !matches!(ord, "acq_rel" | "sc") {
        return None;
    }
    let scope = scope.strip_suffix(';')?;
    amd_scope(scope)?;
    Some((ord, scope))
}

/// Statement-line matcher: `(tail )call void asm sideeffect "<body>",
/// "~{memory}"()` with attribute tail. Returns (indent, body).
fn match_void_asm_line(line: &str) -> Option<(&str, &str)> {
    let indent = &line[..line.len() - line.trim_start().len()];
    let call = strip_call_prefix(line[indent.len()..].trim_start())?;
    let rest = call.strip_prefix("void asm sideeffect \"")?;
    let close = rest.find('"')?;
    let body = &rest[..close];
    let tail = rest[close + 1..].trim_start();
    let tail = tail.strip_prefix(',')?.trim_start();
    let tail = tail.strip_prefix("\"~{memory}\"()")?;
    if !is_attrs_tail(tail) {
        return None;
    }
    Some((indent, body))
}

/// Value-call matcher for `{ty} asm sideeffect "<body>", "<constraint>"(<args>)`.
/// Returns (ty, body, constraint, args-text).
fn match_typed_asm_call<'a>(call: &'a str) -> Option<(&'a str, &'a str, &'a str, &'a str)> {
    let (ty, rest) = call.split_once(' ')?;
    let rest = rest.strip_prefix("asm sideeffect \"")?;
    let close = rest.find('"')?;
    let body = &rest[..close];
    let tail = rest[close + 1..].trim_start();
    let tail = tail.strip_prefix(',')?.trim_start();
    let open = tail.find('(')?;
    let constraint = &tail[..open];
    let constraint = constraint.strip_prefix('"')?.strip_suffix('"')?;
    let args = &tail[open + 1..];
    let close_paren = args.rfind(')')?;
    if !is_attrs_tail(&args[close_paren + 1..]) {
        return None;
    }
    Some((ty, body, constraint, &args[..close_paren]))
}

/// Step 12: address classification, memory fences, scope renames.
fn rewrite_atomics(ll: &str) -> String {
    let mut text = ll.to_string();
    if text.contains("isspacep") {
        text = text.replace("@llvm.nvvm.isspacep.local(", "@llvm.amdgcn.is.private(");
        text = text.replace("@llvm.nvvm.isspacep.shared(", "@llvm.amdgcn.is.shared(");
        text = text.replace("@llvm.nvvm.isspacep.global(", "@llvm.amdgcn.is.global(");
    }
    // membar.gl/cta/sys → fence. Statement lines, so a small line pass.
    if text.contains("membar") {
        let mut out = Vec::with_capacity(text.lines().count());
        for line in text.lines() {
            let replaced = match_membar_line(line);
            match replaced {
                Some(replacement) => out.push(replacement),
                None => out.push(line.to_string()),
            }
        }
        text = out.join("\n");
        if ll.ends_with('\n') && !text.is_empty() {
            text.push('\n');
        }
    }
    // NV and AMD scope names differ (batch residue: device, block).
    text = text.replace("syncscope(\"device\")", "syncscope(\"agent\")");
    text = text.replace("syncscope(\"block\")", "syncscope(\"workgroup\")");
    text
}

/// `(tail )call void @llvm.nvvm.membar.<gl|cta|sys>()` → fence statement.
fn match_membar_line(line: &str) -> Option<String> {
    let indent = &line[..line.len() - line.trim_start().len()];
    let call = strip_call_prefix(line[indent.len()..].trim_start())?;
    let rest = call.strip_prefix("void @llvm.nvvm.membar.")?;
    let (scope, tail) = rest.split_once('(')?;
    let tail = tail.strip_prefix(")")?;
    if !is_attrs_tail(tail) {
        return None;
    }
    amd_scope(scope)?;
    Some(match scope {
        "gl" => format!("{indent}fence syncscope(\"agent\") seq_cst"),
        "cta" => format!("{indent}fence syncscope(\"workgroup\") seq_cst"),
        _ => format!("{indent}fence seq_cst"),
    })
}

/// Step 12d: the PTX-asm atomic family → LLVM atomic instructions.
fn rewrite_ptx_atomic_asm(ll: &str) -> String {
    if !ll.contains("fence.") && !ll.contains("ld.acquire.") && !ll.contains("st.release.") {
        return ll.to_string();
    }
    let mut out = Vec::with_capacity(ll.lines().count());
    for line in ll.lines() {
        // fence.acq_rel.{cta,gpu,sys}; → fence
        if let Some((indent, body)) = match_void_asm_line(line) {
            if let Some((ord, scope)) = match_ptx_fence_asm(body) {
                let syncscope = amd_scope(scope).unwrap_or_default();
                let scope_s = if syncscope.is_empty() {
                    String::new()
                } else {
                    format!("{syncscope} ")
                };
                let order = if ord == "sc" { "seq_cst" } else { "acq_rel" };
                out.push(format!("{indent}fence {scope_s}{order}"));
                continue;
            }
        }
        // ld.acquire.{gpu,sys}.b{32,64} (optionally behind fence.sc.sys) →
        // load atomic acquire/seq_cst
        if let Some(expanded) = expand_ptx_atomic_load(line) {
            out.push(expanded);
            continue;
        }
        // st.release.{gpu,sys}.b{32,64} → store atomic release
        if let Some(expanded) = expand_ptx_atomic_store(line) {
            out.push(expanded);
            continue;
        }
        out.push(line.to_string());
    }
    let mut text = out.join("\n");
    if ll.ends_with('\n') && !text.is_empty() {
        text.push('\n');
    }
    text
}

fn expand_ptx_atomic_load(line: &str) -> Option<String> {
    let (indent, res, rhs) = split_assign(line)?;
    let call = strip_call_prefix(rhs)?;
    let (ty, body, constraint, args) = match_typed_asm_call(call)?;
    // A used result is required; anything else keeps the line (conservative).
    if !matches!(ty, "i32" | "i64" | "ptr") {
        return None;
    }
    let (seqcst, body) = match body.strip_prefix("fence.sc.sys; ") {
        Some(rest) => (true, rest),
        None => (false, body),
    };
    let rest = body.strip_prefix("ld.acquire.")?;
    let (scope, rest) = rest.split_once('.')?;
    if !matches!(scope, "gpu" | "sys") {
        return None;
    }
    let (bits, rest) = match rest.strip_prefix("b32 ") {
        Some(rest) => ("32", rest),
        None => match rest.strip_prefix("b64 ") {
            Some(rest) => ("64", rest),
            None => return None,
        },
    };
    if rest.trim() != "$0, [$1];" {
        return None;
    }
    if constraint != "=r,l,~{memory}" && constraint != "=l,l,~{memory}" {
        return None;
    }
    // (ptr <operand>)
    let operand = args.trim().strip_prefix("ptr ")?;
    if operand.contains(',') {
        return None;
    }
    let ptr = strip_nonnull(operand);
    let syncscope = amd_scope(scope)?;
    let order = if seqcst { "seq_cst" } else { "acquire" };
    let scope_s = if syncscope.is_empty() {
        String::new()
    } else {
        format!(" {syncscope}")
    };
    let align = if bits == "32" { 4 } else { 8 };
    Some(format!(
        "{indent}{res} = load atomic {ty}, ptr {ptr}{scope_s} {order}, align {align}"
    ))
}

fn expand_ptx_atomic_store(line: &str) -> Option<String> {
    let indent = &line[..line.len() - line.trim_start().len()];
    let call = strip_call_prefix(line[indent.len()..].trim_start())?;
    let (_ty, body, constraint, args) = match_typed_asm_call(call)?;
    let rest = body.strip_prefix("st.release.")?;
    let (scope, rest) = rest.split_once('.')?;
    if !matches!(scope, "gpu" | "sys") {
        return None;
    }
    let (bits, rest) = match rest.strip_prefix("b32 ") {
        Some(rest) => ("32", rest),
        None => match rest.strip_prefix("b64 ") {
            Some(rest) => ("64", rest),
            None => return None,
        },
    };
    if rest.trim() != "[$0], $1;" {
        return None;
    }
    if constraint != "l,r,~{memory}" && constraint != "l,l,~{memory}" {
        return None;
    }
    let parts = split_top_level_commas(args);
    if parts.len() != 2 {
        return None;
    }
    let ptr_arg = parts[0].trim().strip_prefix("ptr ")?;
    let ptr = strip_nonnull(ptr_arg);
    let (vty, val_operand) = parts[1].trim().split_once(char::is_whitespace)?;
    if !matches!(vty, "i32" | "i64" | "ptr") {
        return None;
    }
    let val = strip_param_attrs(val_operand);
    let syncscope = amd_scope(scope)?;
    let scope_s = if syncscope.is_empty() {
        String::new()
    } else {
        format!(" {syncscope}")
    };
    let align = if bits == "32" { 4 } else { 8 };
    Some(format!(
        "{indent}store atomic {vty} {val}, ptr {ptr}{scope_s} release, align {align}"
    ))
}

// ---------------------------------------------------------------------------
// [PORT gfx1030 PhaseB.4] counted barrier → software LDS counter + generation
// spin (Step 14), ported from tests/translate_amdgcn.py (GPU-verified:
// producer/consumer + split arrive/sync on a 128-thread wave64 block,
// including bystander warps).
//
// RDNA2's S_BARRIER is the unnumbered whole-workgroup barrier; a direct
// mapping of numbered barriers deadlocks the bystander warps the source
// deliberately tests. Fallback design (Python v3, kept): per-barrier-id
// {count, generation} slots in LDS; arrive increments (the last arriver
// resets the count and bumps the generation to release waiters); sync
// captures the generation, arrives, then volatile-spins until it changes.
// All seq_cst LDS atomics. Using kernels get an entry-block slot-zeroing
// prologue + one whole-group s.barrier (LDS cannot be statically
// initialized; the group barrier makes the zeroing visible before any
// atomic lands). Documented assumption: the entry block is executed
// uniformly by the whole CTA (the CUDA norm). Constant ids/counts only —
// anything else keeps its nvvm call for llc to reject explicitly.
// ---------------------------------------------------------------------------

/// One parsed counted-barrier call line: constant `id` and `n` only.
struct CountedBarrierLine<'a> {
    indent: &'a str,
    kind: &'a str,
    id: i32,
    n: u32,
}

fn match_counted_barrier_line(line: &str) -> Option<CountedBarrierLine<'_>> {
    let indent = &line[..line.len() - line.trim_start().len()];
    let call = strip_call_prefix(line[indent.len()..].trim_start())?;
    let rest = call.strip_prefix("void @llvm.nvvm.barrier.cta.")?;
    let (kind, rest) = rest.split_once('.')?;
    if !matches!(kind, "sync" | "arrive") {
        return None;
    }
    let rest = rest.strip_prefix("count(")?;
    let rest = rest.strip_prefix("i32 ")?;
    let (id_text, rest) = rest.split_once(',')?;
    let id: i32 = id_text.trim().parse().ok()?;
    let rest = rest.trim().strip_prefix("i32 ")?.trim();
    let (n_text, tail) = rest.split_once(')')?;
    let n: u32 = n_text.trim().parse().ok()?;
    if !is_attrs_tail(tail) {
        return None;
    }
    Some(CountedBarrierLine { indent, kind, id, n })
}

/// The helper module text appended when any counted barrier is rewritten.
/// Byte-parity with the Python `_CB_HELPERS` (batch-verified IR).
const COUNTED_BARRIER_HELPERS: &str = r#"

; [PORT gfx1030] 软件 counted barrier 状态槽（LDS 不可静态初始化 → undef，
; 由使用 barrier 的 kernel 入口清零，见各 kernel 的 cb-init 注释块）
@__port_cb_cnt = addrspace(3) global [16 x i32] undef, align 4
@__port_cb_gen = addrspace(3) global [16 x i32] undef, align 4

define internal void @__port_cb_arrive(i32 %id, i32 %n) nounwind {
entry:
  %cnt.p = getelementptr inbounds [16 x i32], ptr addrspace(3) @__port_cb_cnt, i32 0, i32 %id
  %old = atomicrmw add ptr addrspace(3) %cnt.p, i32 1 seq_cst
  %lastn = add i32 %n, -1
  %last = icmp eq i32 %old, %lastn
  br i1 %last, label %release, label %out

release:
  store atomic i32 0, ptr addrspace(3) %cnt.p seq_cst, align 4
  %gen.p = getelementptr inbounds [16 x i32], ptr addrspace(3) @__port_cb_gen, i32 0, i32 %id
  %g = load atomic i32, ptr addrspace(3) %gen.p seq_cst, align 4
  %g1 = add i32 %g, 1
  store atomic i32 %g1, ptr addrspace(3) %gen.p seq_cst, align 4
  ret void

out:
  ret void
}

define internal void @__port_cb_sync(i32 %id, i32 %n) convergent nounwind {
entry:
  %gen.p = getelementptr inbounds [16 x i32], ptr addrspace(3) @__port_cb_gen, i32 0, i32 %id
  %myg = load atomic i32, ptr addrspace(3) %gen.p seq_cst, align 4
  call void @__port_cb_arrive(i32 %id, i32 %n)
  br label %spin

spin:
  %g = load atomic volatile i32, ptr addrspace(3) %gen.p seq_cst, align 4
  %done = icmp ne i32 %g, %myg
  br i1 %done, label %out, label %spin

out:
  ret void
}
"#;

const CB_INIT_MARKER: &str = "; [PORT gfx1030] counted-barrier slot init";

/// Step 14 driver: rewrite counted-barrier call sites, inject the entry
/// prologue into each using function, append the helpers once per module.
fn rewrite_counted_barriers(ll: &str) -> String {
    if !ll.contains("barrier.cta.sync.count") && !ll.contains("barrier.cta.arrive.count") {
        return ll.to_string();
    }
    let lines: Vec<&str> = ll.lines().collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut module_uses_cb = false;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if !(line.starts_with("define ") && line.trim_end().ends_with('{')) {
            out.push(line.to_string());
            i += 1;
            continue;
        }
        let func_end = find_function_end(&lines, i);
        let body = &lines[i..=func_end.min(lines.len() - 1)];
        let (new_body, used) = rewrite_counted_barriers_in_function(body);
        module_uses_cb |= used;
        out.extend(new_body);
        i = func_end + 1;
    }
    let mut text = out.join("\n");
    if module_uses_cb && !text.contains("@__port_cb_cnt = addrspace(3) global") {
        text.push_str(COUNTED_BARRIER_HELPERS);
    }
    if ll.ends_with('\n') && !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// Per-function: replace call sites and inject the slot-zeroing prologue.
/// Returns (new body lines, whether any counted barrier was rewritten).
fn rewrite_counted_barriers_in_function(body: &[&str]) -> (Vec<String>, bool) {
    let mut ids: Vec<i32> = Vec::new();
    let mut changed = false;
    let mut new_body: Vec<String> = Vec::with_capacity(body.len());
    for line in body {
        match match_counted_barrier_line(line) {
            Some(cb) => {
                if !ids.contains(&cb.id) {
                    ids.push(cb.id);
                }
                changed = true;
                new_body.push(format!(
                    "{}call void @__port_cb_{}(i32 {}, i32 {})",
                    cb.indent, cb.kind, cb.id, cb.n
                ));
            }
            None => new_body.push((*line).to_string()),
        }
    }
    if !changed {
        return (new_body, false);
    }
    // Idempotence: the prologue comment block is already in place.
    if new_body.iter().any(|l| l.contains(CB_INIT_MARKER)) {
        return (new_body, true);
    }
    ids.sort_unstable();
    // Injection point: after the entry label line and any entry allocas.
    let mut ins = 1;
    while ins < new_body.len() && !new_body[ins].trim_end().ends_with(':') {
        ins += 1;
    }
    ins += 1; // past the label line
    while ins < new_body.len() && new_body[ins].contains("= alloca ") {
        ins += 1;
    }
    let mut init = vec![format!(
        "{CB_INIT_MARKER} (ids: {})",
        ids.iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )];
    for id in &ids {
        init.push(format!(
            "  %__port_cb_i{id}c = getelementptr inbounds [16 x i32], ptr addrspace(3) @__port_cb_cnt, i32 0, i32 {id}"
        ));
        init.push(format!(
            "  store atomic i32 0, ptr addrspace(3) %__port_cb_i{id}c seq_cst, align 4"
        ));
        init.push(format!(
            "  %__port_cb_i{id}g = getelementptr inbounds [16 x i32], ptr addrspace(3) @__port_cb_gen, i32 0, i32 {id}"
        ));
        init.push(format!(
            "  store atomic i32 0, ptr addrspace(3) %__port_cb_i{id}g seq_cst, align 4"
        ));
    }
    init.push("  call void @llvm.amdgcn.s.barrier()".to_string());
    new_body.splice(ins..ins, init);
    (new_body, true)
}

// ---------------------------------------------------------------------------
// [PORT gfx1030 PhaseB.4] libdevice __nv_* → ocml __ocml_* (Step 15), ported
// from tests/translate_amdgcn.py (llvm-nm-verified exports in the ROCm
// container's ocml.bc; TANPROBE f32/f64 ≤ 2 ULP vs host libm on gfx1030).
//
// NV libdevice names math f32 variants as `__nv_<stem>f(` and f64 as
// `__nv_<stem>(`; ocml spells them `__ocml_<stem>_f32/f64(`. The `(` suffix
// terminates the stem so `__nv_tanf(` can never be mangled by the `tan`
// replacement. The table keeps exactly the stems whose NV and ocml names map
// one-to-one AND that ocml.bc exports; anything else keeps its `__nv_*`
// symbol and fails the code-object link as an explicit, legible signal.
// Linking happens at the code-object link (clang -c pulls ocml, or
// llvm-link with ocml.bc + the oclc_*.bc configuration blobs).
// ---------------------------------------------------------------------------

/// The shared stem set (identical to the Python `_OCML_STEM_SET`).
const OCML_STEMS: &[&str] = &[
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cabs", "cacos", "cacosh",
    "casin", "casinh", "catan", "catanh", "cbrt", "ccos", "ccosh", "ceil", "cexp", "clog",
    "copysign", "cos", "cosh", "csin", "csinh", "csqrt", "ctan", "ctanh", "erf", "erfc",
    "erfcinv", "erfcx", "erfinv", "exp", "exp10", "exp2", "expm1", "fabs", "fdim", "floor",
    "fma", "fmax", "fmin", "fmod", "frexp", "hypot", "i0", "i1", "ilogb", "isfinite", "isinf",
    "isnan", "j0", "j1", "ldexp", "lgamma", "log", "log10", "log1p", "log2", "modf",
    "nearbyint", "nextafter", "pow", "remainder", "remquo", "rint", "round", "rsqrt",
    "scalbn", "signbit", "sin", "sincos", "sinh", "sqrt", "tan", "tanh", "tgamma", "trunc",
    "y0", "y1",
];

/// Step 15 driver: swap the `__nv_` symbol spellings for their ocml ones.
fn rewrite_ocml(ll: &str) -> String {
    if !ll.contains("__nv_") {
        return ll.to_string();
    }
    let mut text = ll.to_string();
    for stem in OCML_STEMS {
        // f32 variant first; the `(` terminator keeps the pairs unambiguous.
        text = text.replace(
            &format!("@__nv_{stem}f("),
            &format!("@__ocml_{stem}_f32("),
        );
        text = text.replace(&format!("@__nv_{stem}("), &format!("@__ocml_{stem}_f64("));
    }
    text
}

/// First column-0 `}` line after `start` (the function body's closing brace;
/// basic blocks never open a column). Falls back to the last line.
fn find_function_end(lines: &[&str], start: usize) -> usize {
    let mut end = start + 1;
    while end < lines.len() && lines[end] != "}" {
        end += 1;
    }
    end
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
        // [PORT gfx1030 PhaseB.2] laneid + the shuffle family
        "llvm.nvvm.read.ptx.sreg.laneid",
        "llvm.nvvm.shfl.sync.idx.f32",
        "llvm.nvvm.shfl.sync.idx.i32",
        "llvm.nvvm.shfl.sync.bfly.f32",
        "llvm.nvvm.shfl.sync.bfly.i32",
        "llvm.nvvm.shfl.sync.up.f32",
        "llvm.nvvm.shfl.sync.up.i32",
        "llvm.nvvm.shfl.sync.down.f32",
        "llvm.nvvm.shfl.sync.down.i32",
        // [PORT gfx1030 PhaseB.3] address classification + memory fences
        "llvm.nvvm.isspacep.local",
        "llvm.nvvm.isspacep.shared",
        "llvm.nvvm.isspacep.global",
        "llvm.nvvm.membar.gl",
        "llvm.nvvm.membar.cta",
        "llvm.nvvm.membar.sys",
        // [PORT gfx1030 PhaseB.4] numbered barriers (software fallback)
        "llvm.nvvm.barrier.cta.sync.count",
        "llvm.nvvm.barrier.cta.arrive.count",
    ] {
        text = strip_declares(&text, callee);
    }
    // 8. generic allocas → addrspace(5) + per-alloca addrspacecast at the use
    //    sites (AMDGPU module verifier contract; batch-verified).
    text = rewrite_allocas(&text);
    // 10. [PORT gfx1030 PhaseB.2] laneid → mbcnt.lo(-1, 0) and the four shfl
    //     modes → segmented ds_bpermute expansions (byte offset = lane * 4),
    //     then 10c: the f64 PTX-asm shuffle → double ds_bpermute. Shapes
    //     outside the supported envelope keep their nvvm call for llc to
    //     reject explicitly.
    text = rewrite_calls(&text, "llvm.nvvm.read.ptx.sreg.laneid", "call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)").0;
    text = rewrite_shuffle(&text);
    text = rewrite_shfl_asm(&text);
    // 12. [PORT gfx1030 PhaseB.3] atomics: isspacep clears FIRST (the v3
    //     layering lesson — membar/scope rewrites crash into leftovers
    //     otherwise), then membar → fence, then NV scope renames, then 12d
    //     the PTX-asm atomic family → LLVM atomic instructions.
    text = rewrite_atomics(&text);
    text = rewrite_ptx_atomic_asm(&text);
    // 14. [PORT gfx1030 PhaseB.4] counted barrier → LDS counter + generation
    //     spin software fallback (a direct S_BARRIER mapping would deadlock
    //     bystander warps), then 15: libdevice __nv_* → ocml __ocml_*.
    text = rewrite_counted_barriers(&text);
    text = rewrite_ocml(&text);
    Ok(text)
}

/// Matches a leading `%[\w.$]+` SSA name and returns `(name, rest)`.
fn parse_ssa_name(s: &str) -> Option<(&str, &str)> {
    if !s.starts_with('%') {
        return None;
    }
    let mut end = 1;
    for (i, c) in s.char_indices().skip(1) {
        if !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '$')) {
            break;
        }
        end = i + c.len_utf8();
    }
    if end == 1 {
        return None;
    }
    Some((&s[..end], &s[end..]))
}

/// Decomposes `%name = ...` into `(indent, name, rest-of-line)`.
fn split_assign(line: &str) -> Option<(&str, &str, &str)> {
    let indent = &line[..line.len() - line.trim_start().len()];
    let rest = &line[indent.len()..];
    let (name, after) = parse_ssa_name(rest)?;
    let after = after.strip_prefix(" = ")?;
    Some((indent, name, after))
}

/// Strips the `(tail )call ` prefix.
fn strip_call_prefix(rhs: &str) -> Option<&str> {
    rhs.strip_prefix("tail call ").or_else(|| rhs.strip_prefix("call "))
}

/// Whether a call's tail after `)` is only whitespace and `#N` attribute
/// groups — the shapes this pass rewrites (same contract as `rewrite_calls`;
/// anything else is left for llc to reject explicitly).
fn is_attrs_tail(tail: &str) -> bool {
    tail.split_whitespace().all(|token| {
        let digits = token.strip_prefix('#').unwrap_or("x");
        !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
    })
}

/// `"i32 16"`/`"float %v19"` → `"16"`/`"%v19"` (strips the type prefix;
/// a missing operand is a conservative parse failure, like the script).
fn arg_operand(arg: &str) -> Option<&str> {
    let mut parts = arg.trim().splitn(2, char::is_whitespace);
    let _ty = parts.next()?;
    parts.next().map(str::trim)
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

    // ---- [PORT gfx1030 PhaseB.2] shuffle (Step 10/10c) ------------------

    /// Shape of the warp_reduce shuffle family in post-`opt` output (the
    /// same fixture shape tests/test_gfx1030.py::test_ir_translate_shuffle
    /// GPU-verified four modes + i32 + f64 on gfx1030).
    const SAMPLE_SHFL_LL: &str = "\
declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32) #3
declare float @llvm.nvvm.shfl.sync.down.f32(i32, float, i32, i32) #3
declare float @llvm.nvvm.shfl.sync.idx.f32(i32, float, i32, i32) #3
declare float @llvm.nvvm.shfl.sync.up.f32(i32, float, i32, i32) #3
declare i32 @llvm.nvvm.read.ptx.sreg.laneid() #2

define amdgpu_kernel void @warp_probe(ptr %v0, i64 %v1) #1 {
entry:
  %lane = tail call i32 @llvm.nvvm.read.ptx.sreg.laneid() #3
  %val = load float, ptr %v0, align 4
  %bf = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %val, i32 1, i32 31) #3
  %dn = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %val, i32 2, i32 31) #3
  %ix = tail call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float %val, i32 0, i32 31) #3
  %up = tail call float @llvm.nvvm.shfl.sync.up.f32(i32 -1, float %val, i32 1, i32 31) #3
  %s = fadd float %bf, %dn
  %s2 = fadd float %s, %ix
  %s3 = fadd float %s2, %up
  store float %s3, ptr %v0, align 4
  ret void
}

define amdgpu_kernel void @f64_probe(ptr %v0, i64 %v1) #1 {
entry:
  %v = load i64, ptr %v0, align 8
  %bf64 = tail call i64 asm sideeffect \"{ .reg .b32 lo; .reg .b32 hi; mov.b64 {lo, hi}, $1; shfl.sync.bfly.b32 lo, lo, $2, 31, $3; shfl.sync.bfly.b32 hi, hi, $2, 31, $3; mov.b64 $0, {lo, hi}; }\", \"=l,l,r,r\"(i64 %v, i32 1, i32 -1) #3
  store i64 %bf64, ptr %v0, align 8
  ret void
}
";

    #[test]
    fn shfl_expands_to_segmented_ds_bpermute() {
        let out = rewrite_ir_for_amdgcn(SAMPLE_SHFL_LL).unwrap();
        // no nvvm residue at all (declares and calls)
        assert!(!out.contains("llvm.nvvm"), "shfl/laneid residue");
        // laneid → mbcnt.lo with the SSA name kept
        assert!(out.contains("%lane = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)"));
        // four f32 shuffles → four bitcast round trips; f64 split → two
        // bpermutes per call
        assert_eq!(out.matches("@llvm.amdgcn.ds.bpermute").count(), 4 + 2);
        // the NV lane number becomes a byte offset (shl src, 2) per call
        assert_eq!(out.matches("= shl i32 ").count(), 4 + 1);
        // down's out-of-segment self-fallback: select + icmp ule
        assert!(out.contains("icmp ule i32"));
        assert!(out.contains("select i1"));
        // f64: lo/hi channels + recombine
        assert!(out.contains("trunc i64 %v to i32"));
        assert!(out.contains("or i64"));
        // results keep their SSA names and downstream uses are untouched
        for name in ["%bf", "%dn", "%ix", "%up", "%bf64"] {
            assert!(out.contains(&format!("{name} = ")), "{name} result missing");
        }
        assert!(out.contains("fadd float %bf, %dn"));
        assert!(out.contains("fadd float %s, %ix"));
        assert!(out.contains("fadd float %s2, %up"));
        assert!(out.contains("store i64 %bf64, ptr %v0, align 8"));
    }

    #[test]
    fn shfl_rewrite_is_idempotent() {
        let once = rewrite_ir_for_amdgcn(SAMPLE_SHFL_LL).unwrap();
        let twice = rewrite_ir_for_amdgcn(&once).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn partial_warp_and_nonstandard_width_shuffles_stay_for_llc_to_reject() {
        let ll = "\
define amdgpu_kernel void @k(ptr %v0) #1 {
entry:
  %a = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 7, float %m, i32 1, i32 31) #3
  %b = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %m, i32 1, i32 15) #3
  store float %a, ptr %v0, align 4
  store float %b, ptr %v0, align 4
  ret void
}
";
        // mask 7 (partial warp) and clamp 15 (width 32≠) are out of envelope
        assert!(rewrite_ir_for_amdgcn(ll).is_ok());
        let out = rewrite_ir_for_amdgcn(ll).unwrap();
        assert_eq!(out.matches("llvm.nvvm.shfl.sync.bfly.f32(").count(), 2);
        assert!(!out.contains("ds.bpermute"));
    }

    #[test]
    fn f64_asm_with_non_full_mask_is_left_alone() {
        let ll = "\
define amdgpu_kernel void @k(ptr %v0) #1 {
entry:
  %bf64 = tail call i64 asm sideeffect \"{ shfl.sync.bfly.b32 lo, lo, $2, 31, $3; }\", \"=l,l,r,r\"(i64 %v, i32 1, i32 5) #3
  store i64 %bf64, ptr %v0, align 8
  ret void
}
";
        assert!(rewrite_ir_for_amdgcn(ll)
            .unwrap()
            .contains("shfl.sync.bfly.b32"));
    }

    // ---- [PORT gfx1030 PhaseB.3] atomics (Step 12/12d) -------------------

    /// Shape of the atomics example in post-`opt` output (the fixture shape
    /// tests/test_gfx1030.py::test_ir_translate_atomics GPU-verified:
    /// is.* classification, generic fetch_add + fence, MP litmus).
    const SAMPLE_ATOMICS_LL: &str = "\
declare i1 @llvm.nvvm.isspacep.local(ptr captures(none)) #0
declare void @llvm.nvvm.membar.gl() #2
declare void @llvm.nvvm.membar.sys() #2

define amdgpu_kernel void @atom_probe(ptr %buf, ptr %out) #1 {
entry:
  %isp = tail call i1 @llvm.nvvm.isspacep.local(ptr %buf) #8
  br i1 %isp, label %priv, label %glob

priv:
  %pp = addrspacecast ptr %buf to ptr addrspace(5)
  br label %done

glob:
  %r = atomicrmw add ptr %buf, i32 1 syncscope(\"device\") monotonic, align 4
  tail call void @llvm.nvvm.membar.gl() #8
  tail call void @llvm.nvvm.membar.sys() #8
  br label %done

done:
  ret void
}

define amdgpu_kernel void @atom_asm_probe(ptr %buf, ptr %out) #1 {
entry:
  ; Rust core::sync::atomic's PTX-asm fallback shapes (real output)
  tail call void asm sideeffect \"fence.acq_rel.cta;\", \"~{memory}\"() #9
  tail call void asm sideeffect \"fence.acq_rel.gpu;\", \"~{memory}\"() #9
  tail call void asm sideeffect \"fence.acq_rel.sys;\", \"~{memory}\"() #9
  %la = tail call i32 asm sideeffect \"ld.acquire.gpu.b32 $0, [$1];\", \"=r,l,~{memory}\"(ptr %buf) #9
  %lb = tail call i64 asm sideeffect \"ld.acquire.sys.b64 $0, [$1];\", \"=l,l,~{memory}\"(ptr %buf) #9
  %lc = call ptr asm sideeffect \"fence.sc.sys; ld.acquire.sys.b64 $0, [$1];\", \"=l,l,~{memory}\"(ptr %buf) #9
  tail call void asm sideeffect \"st.release.gpu.b32 [$0], $1;\", \"l,r,~{memory}\"(ptr %buf, i32 %la) #9
  tail call void asm sideeffect \"st.release.sys.b64 [$0], $1;\", \"l,l,~{memory}\"(ptr %buf, ptr %out) #9
  ret void
}
";

    #[test]
    fn atomics_map_to_amdgcn_is_fences_and_scopes() {
        let out = rewrite_ir_for_amdgcn(SAMPLE_ATOMICS_LL).unwrap();
        assert!(!out.contains("llvm.nvvm"), "isspacep/membar residue");
        assert!(out.contains("@llvm.amdgcn.is.private(ptr %buf)"));
        assert!(out.contains("fence syncscope(\"agent\") seq_cst"));
        assert!(out.contains("\n  fence seq_cst"));
        assert!(out.contains(
            "atomicrmw add ptr %buf, i32 1 syncscope(\"agent\") monotonic, align 4"
        ));
        assert!(!out.contains("syncscope(\"device\")"));
        assert!(!out.contains("syncscope(\"block\")"));
    }

    #[test]
    fn ptx_atomic_asm_maps_to_llvm_atomic_instructions() {
        let out = rewrite_ir_for_amdgcn(SAMPLE_ATOMICS_LL).unwrap();
        assert!(!out.contains("asm sideeffect"), "PTX asm residue");
        assert!(out.contains("fence syncscope(\"workgroup\") acq_rel"));
        assert!(out.contains("fence syncscope(\"agent\") acq_rel"));
        assert!(out.contains("\n  fence acq_rel"));
        assert!(out.contains(
            "%la = load atomic i32, ptr %buf syncscope(\"agent\") acquire, align 4"
        ));
        assert!(out.contains("%lb = load atomic i64, ptr %buf acquire, align 8"));
        // the fence.sc + ld.acquire asm pair folds into one seq_cst load
        assert!(out.contains("%lc = load atomic ptr, ptr %buf seq_cst, align 8"));
        assert!(out.contains(
            "store atomic i32 %la, ptr %buf syncscope(\"agent\") release, align 4"
        ));
        assert!(out.contains("store atomic ptr %out, ptr %buf release, align 8"));
    }

    #[test]
    fn atomics_rewrite_is_idempotent() {
        let once = rewrite_ir_for_amdgcn(SAMPLE_ATOMICS_LL).unwrap();
        let twice = rewrite_ir_for_amdgcn(&once).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn membar_cta_maps_to_workgroup_fence() {
        let ll = "\
define amdgpu_kernel void @k() #1 {
entry:
  tail call void @llvm.nvvm.membar.cta() #8
  ret void
}
";
        assert!(rewrite_ir_for_amdgcn(ll)
            .unwrap()
            .contains("fence syncscope(\"workgroup\") seq_cst"));
    }

    // ---- [PORT gfx1030 PhaseB.4] counted barrier + ocml (Step 14/15) ------

    /// Shape of the counted_barrier example in post-`opt` output (the fixture
    /// shape tests/test_gfx1030.py::test_ir_translate_counted_barrier
    /// GPU-verified: producer/consumer + split arrive/sync, 128 threads,
    /// wave64, including bystander warps).
    const SAMPLE_COUNTED_BARRIER_LL: &str = "\
declare void @llvm.nvvm.barrier.cta.sync.count(i32, i32) #1
declare void @llvm.nvvm.barrier.cta.arrive.count(i32, i32) #1

define amdgpu_kernel void @cb_probe(ptr %out, i64 %nout) #1 {
entry:
  tail call void @llvm.nvvm.barrier.cta.sync.count(i32 1, i32 64) #3
  tail call void @llvm.nvvm.barrier.cta.arrive.count(i32 2, i32 64) #3
  ret void
}
";

    #[test]
    fn counted_barriers_map_to_software_fallback() {
        let out = rewrite_ir_for_amdgcn(SAMPLE_COUNTED_BARRIER_LL).unwrap();
        assert!(!out.contains("llvm.nvvm.barrier"), "barrier residue");
        assert!(out.contains("call void @__port_cb_sync(i32 1, i32 64)"));
        assert!(out.contains("call void @__port_cb_arrive(i32 2, i32 64)"));
        // slot state + helper definitions
        assert!(out.contains("@__port_cb_cnt = addrspace(3) global [16 x i32] undef, align 4"));
        assert!(out.contains("@__port_cb_gen = addrspace(3) global [16 x i32] undef, align 4"));
        assert!(out.contains("define internal void @__port_cb_arrive(i32 %id, i32 %n)"));
        assert!(out.contains("define internal void @__port_cb_sync(i32 %id, i32 %n) convergent"));
        // entry init: both slots zeroed + one whole-group barrier
        assert!(out.contains("counted-barrier slot init (ids: 1,2)"));
        assert_eq!(
            out.matches("store atomic i32 0, ptr addrspace(3) %__port_cb_i").count(),
            4
        );
        assert!(out.contains("call void @llvm.amdgcn.s.barrier()"));
        // generation spin must be volatile to survive the optimizer
        assert!(out.contains("load atomic volatile i32"));
    }

    #[test]
    fn counted_barrier_rewrite_is_idempotent() {
        let once = rewrite_ir_for_amdgcn(SAMPLE_COUNTED_BARRIER_LL).unwrap();
        let twice = rewrite_ir_for_amdgcn(&once).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn non_constant_counted_barriers_stay_for_llc_to_reject() {
        let ll = "\
define amdgpu_kernel void @k(i32 %bid) #1 {
entry:
  tail call void @llvm.nvvm.barrier.cta.sync.count(i32 %bid, i32 64) #3
  ret void
}
";
        let out = rewrite_ir_for_amdgcn(ll).unwrap();
        // dynamic id: no rewrite, no helpers, llc fails with a legible name
        assert!(out.contains("llvm.nvvm.barrier.cta.sync.count"));
        assert!(!out.contains("__port_cb_cnt"));
    }

    #[test]
    fn ocml_swaps_known_stems_and_keeps_unknown_ones() {
        let ll = "\
declare double @__nv_tan(double)
declare float @__nv_tanf(float)
declare double @__nv_nonstandard_weird(double)

define amdgpu_kernel void @tan_probe(ptr %out, double %x, float %y) #1 {
entry:
  %t64 = call double @__nv_tan(double %x) #0
  %t32 = call float @__nv_tanf(float %y) #0
  %w = call double @__nv_nonstandard_weird(double %x) #0
  %s = fadd double %t64, %w
  store float %t32, ptr %out, align 4
  ret void
}
";
        let out = rewrite_ir_for_amdgcn(ll).unwrap();
        // call sites AND declares swap; the `(` terminator protects the stem
        assert!(out.contains("@__ocml_tan_f64(double %x)"));
        assert!(out.contains("@__ocml_tan_f32(float %y)"));
        assert!(out.contains("declare double @__ocml_tan_f64(double)"));
        // stems outside the table keep failing loudly at the link
        assert!(out.contains("@__nv_nonstandard_weird("));
        assert!(!out.contains("@__nv_tan"));
    }

    #[test]
    fn ocml_rewrite_is_idempotent() {
        let ll = "\
declare float @__nv_tanf(float)

define amdgpu_kernel void @k(float %y) #1 {
entry:
  %t32 = call float @__nv_tanf(float %y) #0
  ret void
}
";
        let once = rewrite_ir_for_amdgcn(ll).unwrap();
        let twice = rewrite_ir_for_amdgcn(&once).unwrap();
        assert_eq!(once, twice);
    }
}
