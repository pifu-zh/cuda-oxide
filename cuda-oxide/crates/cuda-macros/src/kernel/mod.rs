/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[kernel]` argument parsing, validation, and expansion dispatch.

pub(crate) mod codegen;
pub(crate) mod scope;

use crate::common::{
    attr_path_ends_with, grid_constant_pointee, impl_trait_parameter_error, reject_reserved_name,
    track_codegen_environment,
};
use crate::cuda_module::launchers::has_codegen_generics;
use crate::kernel::codegen::{
    generate_generic_kernel, generate_generic_kernel_no_instantiation, generate_simple_kernel,
};
use crate::launch_attrs::{add_launch_bounds_evaluatability_from_attrs, rewrite_loop_unroll_attrs};
use proc_macro::TokenStream;
use syn::{
    Expr, FnArg, GenericParam, Ident, ItemFn, ReturnType, Stmt, Token, Type,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
};

/// Attribute arguments for `#[kernel(...)]`.
///
/// Legacy explicit-instantiation types, the optional launch-context binding,
/// and the bare `unchecked_indexing` flag may appear in any order:
///
/// ```ignore
/// #[kernel(launch_context = launch_context)]
/// #[kernel(f32, f64, launch_context = launch_context)]
/// #[kernel(unchecked_indexing)]
/// #[kernel(f32, unchecked_indexing)]
/// #[kernel(launch_context = launch_context, unchecked_indexing)]
/// ```
pub(crate) struct KernelArgs {
    /// Types to instantiate generic kernels for
    pub(crate) instantiate_types: Vec<Type>,
    /// User-selected name for the entry's typed launch context.
    pub(crate) launch_context: Option<Ident>,
    /// Elide slice/array bounds checks in this kernel's body (UB contract).
    pub(crate) unchecked_indexing: bool,
}

/// Returns true when the next attribute argument is exactly the bare flag
/// word `flag` (an identifier followed by `,` or end of input, never `=`).
///
/// A bare identifier also parses as a `Type`, so this peek must run before
/// the legacy instantiation-type fallback to keep flag words from being
/// swallowed as type arguments.
fn peek_bare_kernel_flag(input: ParseStream, flag: &str) -> bool {
    if !input.peek(Ident) || input.peek2(Token![=]) {
        return false;
    }
    let fork = input.fork();
    match fork.parse::<Ident>() {
        Ok(ident) => ident == flag && (fork.is_empty() || fork.peek(Token![,])),
        Err(_) => false,
    }
}

impl Parse for KernelArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut instantiate_types = Vec::new();
        let mut launch_context = None;
        let mut unchecked_indexing = false;

        while !input.is_empty() {
            if input.peek(Ident) && input.peek2(Token![=]) {
                let name: Ident = input.parse()?;
                input.parse::<Token![=]>()?;
                if name != "launch_context" {
                    return Err(syn::Error::new(
                        name.span(),
                        format!(
                            "unknown #[kernel] named argument `{name}`; expected `launch_context = IDENT`"
                        ),
                    ));
                }
                let value: Ident = input.parse().map_err(|_| {
                    syn::Error::new(
                        input.span(),
                        "`launch_context` must be a single Rust identifier",
                    )
                })?;
                if launch_context.replace(value).is_some() {
                    return Err(syn::Error::new(
                        name.span(),
                        "duplicate `launch_context` argument in #[kernel]",
                    ));
                }
            } else if peek_bare_kernel_flag(input, "unchecked_indexing") {
                let flag: Ident = input.parse()?;
                if unchecked_indexing {
                    return Err(syn::Error::new(
                        flag.span(),
                        "duplicate `unchecked_indexing` argument in #[kernel]",
                    ));
                }
                unchecked_indexing = true;
            } else {
                instantiate_types.push(input.parse::<Type>()?);
            }

            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }

        Ok(KernelArgs {
            instantiate_types,
            launch_context,
            unchecked_indexing,
        })
    }
}

fn scope_parameter_collision(input: &ItemFn, scope: &Ident) -> Option<Ident> {
    struct Finder<'a> {
        scope: &'a Ident,
        found: Option<Ident>,
    }

    impl<'ast> syn::visit::Visit<'ast> for Finder<'_> {
        fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
            if self.found.is_none() && pattern.ident == *self.scope {
                self.found = Some(pattern.ident.clone());
            }
            syn::visit::visit_pat_ident(self, pattern);
        }
    }

    let mut finder = Finder { scope, found: None };
    for argument in &input.sig.inputs {
        if let FnArg::Typed(argument) = argument {
            syn::visit::Visit::visit_pat(&mut finder, &argument.pat);
        }
    }
    finder.found
}

/// Rewrites explicit `panic!("format", args...)` calls inside the kernel body
/// into a formatted panic report — `panic: <rendered>` through the ockl
/// printf transport, arguments packed exactly like `gpu_printf!` — followed
/// by a clean wave exit.
///
/// Scope:
/// - Unit-return kernels only. The exit is a bare `return`; a non-unit
///   kernel cannot exit its wave this way, so those keep the MIR-level
///   plain trap (matching the importer's kernel-only report gate).
/// - The format string must be a literal with at least one argument.
///   Single-literal panics keep the MIR path, which additionally supports
///   the deterministic global readback.
/// - Panic machinery that never appears as a source `panic!` (slice bounds
///   checks, `unwrap`, `assert!` expansion inside core) keeps the plain
///   trap.
/// - `panic!` inside a closure keeps its original form: a `return` there
///   would exit the closure, not the wave.
///
/// The rewrite is type-safe in every expression position: the replacement
/// block ends in `return`, so its type is `!` and coerces wherever the
/// diverging `panic!` was accepted.
pub(crate) fn rewrite_format_arg_panics(input: &mut ItemFn) {
    if !matches!(input.sig.output, ReturnType::Default) {
        return;
    }
    use syn::visit_mut::VisitMut as _;
    PanicRewriter.visit_block_mut(&mut input.block);
}

struct PanicRewriter;

/// The argument list of a `panic!(fmt, args...)` invocation.
struct PanicArgs(Vec<Expr>);

impl Parse for PanicArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = Vec::new();
        while !input.is_empty() {
            args.push(input.parse::<Expr>()?);
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(PanicArgs(args))
    }
}

impl syn::visit_mut::VisitMut for PanicRewriter {
    fn visit_stmt_mut(&mut self, stmt: &mut Stmt) {
        // A statement-level `panic!(...);` parses as `Stmt::Macro`, not as
        // an expression statement holding an `Expr::Macro`.
        if let Stmt::Macro(stmt_macro) = stmt {
            let mut replacement = None;
            if self.rewrite_panic_macro(&stmt_macro.mac, &mut replacement) {
                *stmt = Stmt::Expr(
                    replacement.expect("rewrite sets the replacement"),
                    Some(<Token![;]>::default()),
                );
                return;
            }
        }
        syn::visit_mut::visit_stmt_mut(self, stmt);
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // Closures keep their original panic form (see the fn doc).
        if matches!(expr, Expr::Closure(_)) {
            return;
        }
        if let Expr::Macro(macro_call) = expr {
            let mut replacement = None;
            if self.rewrite_panic_macro(&macro_call.mac, &mut replacement) {
                *expr = replacement.expect("rewrite sets the replacement");
                return;
            }
        }
        syn::visit_mut::visit_expr_mut(self, expr);
    }
}

impl PanicRewriter {
    /// Matches `panic!("literal", args...)` and writes the report-and-exit
    /// block into `out`. Returns false (and leaves `out` untouched) when the
    /// call is out of the rewrite's scope.
    fn rewrite_panic_macro(&mut self, mac: &syn::Macro, out: &mut Option<Expr>) -> bool {
        let path = &mac.path;
        let is_panic =
            path.is_ident("panic") || path.is_ident("core::panic") || path.is_ident("std::panic");
        if !is_panic {
            return false;
        }
        let Ok(PanicArgs(args)) = syn::parse2::<PanicArgs>(mac.tokens.clone()) else {
            return false;
        };
        let mut args = args;
        let fmt_is_literal = matches!(
            args.first(),
            Some(Expr::Lit(lit)) if matches!(lit.lit, syn::Lit::Str(_))
        );
        if args.len() < 2 || !fmt_is_literal {
            return false;
        }
        let first = args.remove(0);
        // The first argument was just matched as a string literal.
        let fmt = match first {
            Expr::Lit(lit) => match lit.lit {
                syn::Lit::Str(s) => s,
                _ => unreachable!("checked above"),
            },
            _ => unreachable!("checked above"),
        };
        let report = crate::printf::panic_report_expr(&fmt, args);
        *out = Some(parse_quote!({
            #report;
            return;
        }));
        true
    }
}

/// Consume parameter-local `#[grid_constant]` declarations and plant one
/// source-index marker per parameter for the MIR importer.
///
/// An immutable reference is the source-level representation because LLVM's
/// grid-constant contract is a pointer parameter with `byval(T)`: device code
/// must keep the parameter-space address rather than receive a copied `T`.
pub(crate) fn inject_grid_constant_markers(input: &mut ItemFn) -> syn::Result<()> {
    let mut markers = Vec::new();
    for (index, argument) in input.sig.inputs.iter_mut().enumerate() {
        let FnArg::Typed(argument) = argument else {
            continue;
        };
        if grid_constant_pointee(argument)?.is_none() {
            continue;
        }
        argument
            .attrs
            .retain(|attribute| !attr_path_ends_with(attribute, "grid_constant"));
        markers.push(parse_quote! {
            // SAFETY: this marker is emitted only on the kernel entry whose
            // parameter declaration also determines host by-value marshalling.
            unsafe { ::cuda_device::thread::__grid_constant_config::<#index>(); }
        });
    }
    input.block.stmts.splice(0..0, markers);
    Ok(())
}

pub(crate) fn kernel_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    track_codegen_environment();
    let args = parse_macro_input!(attr as KernelArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    let KernelArgs {
        instantiate_types,
        launch_context,
        unchecked_indexing,
    } = args;

    if let Some(err) = reject_reserved_name(&input.sig.ident) {
        return err;
    }
    if let Some(err) = impl_trait_parameter_error(&input, "kernel") {
        return err.to_compile_error().into();
    }
    if let Err(error) = inject_grid_constant_markers(&mut input) {
        return error.to_compile_error().into();
    }
    if let Some(launch_context) = &launch_context
        && let Some(parameter) = scope_parameter_collision(&input, launch_context)
    {
        return syn::Error::new_spanned(
            parameter,
            format!(
                "kernel launch-context binding `{launch_context}` conflicts with a function parameter; choose a distinct name in `#[kernel(launch_context = ...)]`"
            ),
        )
        .to_compile_error()
        .into();
    }

    // Consume any `#[unroll]` / `#[unroll(N)]` attributes written directly on
    // loops inside the kernel body. We strip the attribute from the loop
    // expression (so rustc never sees an expression attribute, keeping us off
    // nightly `stmt_expr_attributes`) and inject an
    // `__unroll_config::<FACTOR>()` marker into the loop body. If the visitor
    // hits a malformed attribute it records the error so we can surface it as a
    // compile error below.
    if let Err(err) = rewrite_loop_unroll_attrs(&mut input) {
        return err.to_compile_error().into();
    }
    if let Err(err) = add_launch_bounds_evaluatability_from_attrs(&mut input) {
        return err.to_compile_error().into();
    }

    // Insert the unchecked-indexing compiler marker at the top of the kernel
    // body. The MIR importer detects the call, elides bounds-check asserts in
    // the translated body, and strips the call before code generation. For
    // simple kernels the function itself becomes the entry, so the marker
    // stays put. Generic kernel generation strips the marker from the
    // re-emitted user-named implementation helper and confines it to the
    // generated entry wrapper (forwarded via
    // `top_level_kernel_configuration_markers`) plus a hidden unchecked twin
    // of the implementation that only the entry calls; see
    // `strip_unchecked_indexing_config_marker` and
    // `unchecked_indexing_impl_clone`.
    if unchecked_indexing {
        let marker_call: syn::Stmt = syn::parse_quote! {
            ::cuda_device::thread::__unchecked_indexing_config::<true>();
        };
        input.block.stmts.insert(0, marker_call);
    }

    // Format-arg `panic!(fmt, args...)` calls report through the ockl printf
    // transport and exit the wave cleanly (see the fn doc for the scope).
    rewrite_format_arg_panics(&mut input);

    // Only type and const parameters create distinct codegen instances.
    // Lifetimes are erased before monomorphization.
    let has_generics = has_codegen_generics(&input.sig.generics);

    if has_generics && !instantiate_types.is_empty() {
        let type_param_count = input
            .sig
            .generics
            .params
            .iter()
            .filter(|param| matches!(param, GenericParam::Type(_)))
            .count();
        let has_const_params = input
            .sig
            .generics
            .params
            .iter()
            .any(|param| matches!(param, GenericParam::Const(_)));
        let has_lifetime_params = input
            .sig
            .generics
            .params
            .iter()
            .any(|param| matches!(param, GenericParam::Lifetime(_)));
        if type_param_count != 1 || has_const_params || has_lifetime_params {
            return syn::Error::new_spanned(
                &input.sig.generics,
                "legacy #[kernel(Type, ...)] instantiation supports exactly one type parameter and no lifetime or const parameters; use #[kernel] and instantiate the kernel with a normal turbofish at the launch site",
            )
            .to_compile_error()
            .into();
        }
    }

    if has_generics && instantiate_types.is_empty() {
        // Generic kernel without explicit types - allow it!
        // Instantiation will happen from call sites (nvcc-style)
        return generate_generic_kernel_no_instantiation(input, launch_context);
    }

    if !has_generics && !instantiate_types.is_empty() {
        // Non-generic kernel with instantiation types - error
        return syn::Error::new_spanned(
            &input.sig.ident,
            "Instantiation types only apply to generic kernels",
        )
        .to_compile_error()
        .into();
    }

    if has_generics {
        // Generate wrapper kernels for each instantiation type
        generate_generic_kernel(input, instantiate_types, launch_context)
    } else {
        // Simple non-generic kernel
        generate_simple_kernel(input, launch_context)
    }
}
