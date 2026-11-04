/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device panic-message reporting at dropped-panic blocks.
//!
//! The collector records each `panic!("literal")` message keyed by the
//! function export name and the basic block index (both sides walk the same
//! monomorphized bodies, so the pair is stable without span reasoning) and
//! enqueues the `__cuda_panic_str` device helper so its `mir.func` lands
//! in-module. This module materializes the message bytes as an immutable
//! device global and emits the helper call; the kernel trap itself is
//! emitted by the existing dropped-panic path right after the call.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue::coerce::cast_to_declared_rust_pointer_type_if_needed;
use crate::translator::terminator::helpers::emit_raw_call;
use dialect_mir::attributes::MirPointerKindAuthorityAttr;
use dialect_mir::ops::MirConstantOp;
use dialect_mir::ops::MirGlobalAllocOp;
use dialect_mir::types::{MirArrayType, MirPointerKind, MirPtrType, MirTupleType};
use pliron::builtin::attributes::{StringAttr, TypeAttr};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::Location;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

/// The device helper the panic arm calls. The collector enqueues its body
/// under this exact export name.
const PANIC_HELPER_SYMBOL: &str = "__cuda_panic_str";

/// Emit `__cuda_panic_str(msg_ptr, len)` for a recorded panic message.
///
/// Returns the operation to chain the wave-exit after, or `None` when the
/// call could not be emitted (the caller falls back to the plain trap).
pub(crate) fn emit_panic_message_report(
    ctx: &mut Context,
    bytes: &[u8],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> Option<Ptr<Operation>> {
    match emit_panic_message_report_inner(ctx, bytes, block_ptr, prev_op, loc) {
        Ok(last) => Some(last),
        Err(error) => {
            eprintln!(
                "[panic-report] message report emission failed, falling back to plain trap: {error:?}"
            );
            None
        }
    }
}

fn emit_panic_message_report_inner(
    ctx: &mut Context,
    bytes: &[u8],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::op::Op;

    if bytes.is_empty() {
        return Err(pliron::input_error_noloc!(TranslationErr::unsupported(
            "empty panic message",
        )));
    }

    // 1. The message global: an immutable `[u8; N]` in CONSTANT memory (as4)
    //    with a tagged rust-static key. The as4 naming branch exports it under
    //    the embedded symbol name, so the host resolves the message via
    //    cuModuleGetGlobal after a panic (the ockl hostcall buffer cannot be
    //    used for this: its drain happens at kernel completion and a trap
    //    aborts the queue first; here the panic wave exits cleanly instead).
    let u8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let array_ty: TypeHandle = MirArrayType::get(ctx, u8_ty, bytes.len() as u64).into();
    let symbol = format!("__cuda_panic_msg_{:08x}", fnv1a(bytes));
    let global_key = format!(
        "cuda-oxide:rust-static:v1:{}:{}",
        symbol.len(),
        symbol
    );
    let global_ptr_ty: TypeHandle = MirPtrType::get_with_kind(
        ctx,
        array_ty,
        false,
        dialect_mir::types::address_space::CONSTANT,
        MirPointerKind::Erased,
    )
    .into();
    let global_op = Operation::new(
        ctx,
        dialect_mir::ops::MirGlobalAllocOp::get_concrete_op_info(),
        vec![global_ptr_ty],
        vec![],
        vec![],
        0,
    );
    global_op.deref_mut(ctx).set_loc(loc.clone());
    let global_alloc = dialect_mir::ops::MirGlobalAllocOp::new(global_op);
    global_alloc.set_attr_global_type(ctx, TypeAttr::new(array_ty));
    global_alloc.set_attr_global_key(ctx, StringAttr::new(global_key.into()));
    let hex = bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    {
        // The promoted helper's hex-attr setter is pub(super) in statics.rs;
        // inline the two-line attribute write here (same attribute key).
        let key = pliron::identifier::Identifier::try_new(
            "global_initializer_hex".to_string(),
        )
        .expect("valid identifier");
        global_alloc
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, StringAttr::new(hex.clone().into()));
    }
    global_alloc.set_alignment_value(ctx, 1);
    global_alloc.mark_immutable(ctx);
    match prev_op {
        Some(prev) => global_alloc.get_operation().insert_after(ctx, prev),
        None => global_alloc.get_operation().insert_at_front(block_ptr, ctx),
    }
    let mut last_op = Some(global_alloc.get_operation());
    let global_ptr = global_alloc.get_operation().deref(ctx).get_result(0);

    // 2. Establish the raw pointer the helper takes, in two sanctioned steps.
    //    (a) The StaticAddress-authority boundary cast turns the Erased
    //        storage address into the declared `*const [u8; N]` (same pointee;
    //        this is exactly how promoted &str constants establish their
    //        declared pointer type).
    //    (b) The ArrayToPointer decay (raw source kind, same address space,
    //        element == target pointee) yields the `*const u8` parameter.
    let declared_array_ptr_ty: TypeHandle = MirPtrType::get_with_kind(
        ctx,
        array_ty,
        false,
        dialect_mir::types::address_space::GENERIC,
        MirPointerKind::RawConst,
    )
    .into();
    let msg_param_ty: TypeHandle = MirPtrType::get_with_kind(
        ctx,
        u8_ty,
        false,
        dialect_mir::types::address_space::GENERIC,
        MirPointerKind::RawConst,
    )
    .into();
    let len_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let unit_ty: TypeHandle = MirTupleType::get(ctx, vec![]).into();

    let (declared_ptr, establish_last) = cast_to_declared_rust_pointer_type_if_needed(
        ctx,
        global_ptr,
        declared_array_ptr_ty,
        block_ptr,
        last_op,
        loc.clone(),
        MirPointerKindAuthorityAttr::StaticAddress,
    );
    last_op = establish_last;
    let (mut arg_values, decay_last) =
        decay_to_byte_pointer(ctx, declared_ptr, msg_param_ty, block_ptr, last_op, &loc)?;
    last_op = decay_last;

    // 4. The length constant.
    let len_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![len_ty],
        vec![],
        vec![],
        0,
    );
    len_op.deref_mut(ctx).set_loc(loc.clone());
    {
        use pliron::op::Op as _;
        let const_op = MirConstantOp::new(len_op);
        const_op.set_attr_value(
            ctx,
            pliron::builtin::attributes::IntegerAttr::new(
                IntegerType::get(ctx, 64, Signedness::Unsigned),
                APInt::from_u128(bytes.len() as u128, NonZeroUsize::new(64).unwrap()),
            ),
        );
        let op_ptr = const_op.get_operation();
        match last_op {
            Some(prev) => op_ptr.insert_after(ctx, prev),
            None => op_ptr.insert_at_front(block_ptr, ctx),
        }
        last_op = Some(op_ptr);
        arg_values.push(op_ptr.deref(ctx).get_result(0));
    }

    // 5. The call. The helper is a real in-module function (enqueued by the
    //    collector), so the ordinary signature verification applies and no
    //    external signature is attached.
    let call_ptr = emit_raw_call(
        ctx,
        PANIC_HELPER_SYMBOL,
        arg_values,
        unit_ty,
        block_ptr,
        last_op,
        loc,
    );
    Ok(call_ptr)
}

/// FNV-1a over the message bytes; names the per-site message global so the
/// host can resolve it deterministically without a rustc StaticDef.
fn fnv1a(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for b in bytes {
        hash ^= u32::from(*b);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// Decay the message-global pointer (`ptr<as1, [u8; N]>`) to the helper's
/// `*const u8` parameter via the sanctioned ArrayToPointer cast.
fn decay_to_byte_pointer(
    ctx: &mut Context,
    global_ptr: Value,
    param_ty: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: &Location,
) -> TranslationResult<(Vec<Value>, Option<Ptr<Operation>>)> {
    use dialect_mir::ops::MirCastOp;
    use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
    use pliron::op::Op;

    if global_ptr.get_type(ctx) == param_ty {
        return Ok((vec![global_ptr], prev_op));
    }
    let op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![param_ty],
        vec![global_ptr],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    let cast_op = MirCastOp::new(op);
    cast_op.set_attr_cast_kind(ctx, MirCastKindAttr::PointerCoercionArrayToPointer);
    let op_ptr = cast_op.get_operation();
    match prev_op {
        Some(prev) => op_ptr.insert_after(ctx, prev),
        None => op_ptr.insert_at_front(block_ptr, ctx),
    }
    Ok((vec![op_ptr.deref(ctx).get_result(0)], Some(op_ptr)))
}
