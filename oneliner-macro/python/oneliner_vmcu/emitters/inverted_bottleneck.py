"""Emitter for the paper-faithful K²+2 inverted-bottleneck schedule."""

from __future__ import annotations

from iree.compiler import ir
from iree.compiler.dialects import arith, flow, linalg, scf, tensor

from ..ir_utils import erase_dead_operation, owner_operation, replace_all_uses
from ..model import InvertedBottleneckMatch, PatternMatch
from ..schedules import InvertedBottleneckSegmentSchedule
from .common import constant
from .convolution import (
    emit_requantize_i8_expanded,
    nested_reduction,
    nested_tensor_loop,
)
from .segments import emit_static_segment_buffer


def _and(*conditions: ir.Value) -> ir.Value:
    """Combines a non-empty sequence of scalar i1 conditions."""
    result = conditions[0]
    for condition in conditions[1:]:
        result = arith.AndIOp(result, condition).result
    return result


def _in_range(index: ir.Value, lower: int, extent: int) -> ir.Value:
    """Checks ``lower <= index < lower + extent`` in index arithmetic."""
    index_type = ir.IndexType.get()
    lower_value = constant(index_type, lower)
    upper_value = constant(index_type, lower + extent)
    return _and(
        arith.CmpIOp(arith.CmpIPredicate.uge, index, lower_value).result,
        arith.CmpIOp(arith.CmpIPredicate.ult, index, upper_value).result,
    )


def _i8_residual_add(
    projection: ir.Value,
    residual: ir.Value,
    zero_point: int,
) -> ir.Value:
    """Reconstructs the proven same-quantization residual scalar semantics."""
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    zp = constant(i32, zero_point)
    projection_centered = arith.SubIOp(arith.ExtSIOp(i32, projection).result, zp).result
    residual_centered = arith.SubIOp(arith.ExtSIOp(i32, residual).result, zp).result
    summed = arith.AddIOp(projection_centered, residual_centered).result
    shifted = arith.AddIOp(summed, zp).result
    lower = arith.MaxSIOp(shifted, constant(i32, -128)).result
    upper = arith.MinSIOp(lower, constant(i32, 127)).result
    return arith.TruncIOp(i8, upper).result


def emit_inverted_bottleneck(match: PatternMatch) -> None:
    """Fuses IBN layers using K² B, 1 C, and 1 D segment buffers.

    B caches the complete Kh×Kw expansion patch for one expanded-channel chunk;
    C caches each depthwise result once; D carries all projection accumulators
    across chunks.  No output channel recursively recomputes B or C.
    """
    if not isinstance(match, InvertedBottleneckMatch):
        raise TypeError(f"IBN emitter received {type(match).__name__}")
    candidate = match
    expansion = candidate.expansion
    depthwise = candidate.depthwise
    projection = candidate.projection
    kernel_height, kernel_width = depthwise.weight_shape[:2]
    schedule = InvertedBottleneckSegmentSchedule(
        expansion.input_shape[3],
        expansion.output_shape[3],
        projection.output_shape[3],
        kernel_height,
        kernel_width,
    )
    patch_segments = kernel_height * kernel_width
    segment_lanes = schedule.segment_lanes
    expanded_channels = expansion.output_shape[3]
    output_channels = projection.output_shape[3]
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    index_type = ir.IndexType.get()

    # A split IREE residual path keeps its elementwise generics in place.  We
    # replace only the projection value that enters that already-validated
    # chain, preserving its exact scalar rounding sequence.  The compact
    # fused form is rebuilt inside the dispatch as before.
    split_residual = candidate.residual is not None and candidate.residual.mode == "split"
    final_operation = (
        projection.rescale
        if split_residual
        else (candidate.residual.final_operation if candidate.residual is not None else projection.rescale)
    )
    final_initial = final_operation.opview.outputs[0]
    with ir.InsertionPoint(final_operation):
        # A top-level scf loop would remain host-side VM control flow. The
        # explicit region is the dispatchability boundary that lowers the
        # complete fixed schedule into one native IREE executable entry point.
        workload = constant(index_type, 1)
        dispatch = flow.DispatchRegionOp(
            [final_initial.type], [], [workload]
        )

        def emit_pixel(indices: tuple[ir.Value, ...], output: ir.Value) -> ir.Value:
            """Emits one spatial IBN step while carrying the fixed buffers."""
            n, oh, ow = indices
            # D is indexed by projection output channel and therefore holds
            # all Cout lanes; B/C retain the min(Cin, Cout) shared lanes.
            d_initial = projection.bias
            lower = constant(index_type, 0)
            upper = constant(index_type, expanded_channels)
            step = constant(index_type, segment_lanes)
            chunk_loop = scf.ForOp(lower, upper, step, [d_initial])
            with ir.InsertionPoint(chunk_loop.body):
                chunk, d_state = chunk_loop.body.arguments
                expansion_padding = constant(
                    i8, expansion.output_quantization.zero_point_at()
                )
                b_initial = emit_static_segment_buffer(
                    patch_segments, segment_lanes, i8, expansion_padding
                )

                def fill_b(b_indices: tuple[ir.Value, ...], state: ir.Value) -> ir.Value:
                    """Fills one lane of one expansion patch segment."""
                    patch, lane = b_indices
                    kernel_width_value = constant(index_type, kernel_width)
                    kh = arith.DivUIOp(patch, kernel_width_value).result
                    kw = arith.RemUIOp(patch, kernel_width_value).result
                    expanded_channel = arith.AddIOp(chunk, lane).result
                    padded_h = arith.AddIOp(
                        arith.MulIOp(oh, constant(index_type, depthwise.strides[0])).result,
                        arith.MulIOp(kh, constant(index_type, depthwise.dilations[0])).result,
                    ).result
                    padded_w = arith.AddIOp(
                        arith.MulIOp(ow, constant(index_type, depthwise.strides[1])).result,
                        arith.MulIOp(kw, constant(index_type, depthwise.dilations[1])).result,
                    ).result
                    valid = _and(
                        arith.CmpIOp(
                            arith.CmpIPredicate.ult,
                            expanded_channel,
                            constant(index_type, expanded_channels),
                        ).result,
                        _in_range(
                            padded_h,
                            depthwise.padding_low[1],
                            expansion.output_shape[1],
                        ),
                        _in_range(
                            padded_w,
                            depthwise.padding_low[2],
                            expansion.output_shape[2],
                        ),
                    )
                    value_branch = scf.IfOp(valid, [i8], has_else=True)
                    with ir.InsertionPoint(value_branch.then_block):
                        source_h = arith.SubIOp(
                            padded_h, constant(index_type, depthwise.padding_low[1])
                        ).result
                        source_w = arith.SubIOp(
                            padded_w, constant(index_type, depthwise.padding_low[2])
                        ).result
                        bias = tensor.ExtractOp(
                            expansion.bias, [expanded_channel]
                        ).result
                        input_zp = constant(
                            i32, expansion.input_quantization.zero_point_at()
                        )
                        weight_zp = constant(
                            i32, expansion.weight_quantization.zero_point_at()
                        )

                        def expansion_product(reduction, accumulator):
                            """Accumulates one centered 1x1 expansion product."""
                            input_channel = reduction[0]
                            input_value = tensor.ExtractOp(
                                expansion.input,
                                [n, source_h, source_w, input_channel],
                            ).result
                            weight_value = tensor.ExtractOp(
                                expansion.weight,
                                [
                                    constant(index_type, 0),
                                    constant(index_type, 0),
                                    input_channel,
                                    expanded_channel,
                                ],
                            ).result
                            lhs = arith.SubIOp(
                                arith.ExtSIOp(i32, input_value).result, input_zp
                            ).result
                            rhs = arith.SubIOp(
                                arith.ExtSIOp(i32, weight_value).result, weight_zp
                            ).result
                            return arith.AddIOp(
                                accumulator, arith.MulIOp(lhs, rhs).result
                            ).result

                        accumulator = nested_reduction(
                            (expansion.input_shape[3],), bias, expansion_product
                        )
                        expanded = emit_requantize_i8_expanded(
                            accumulator,
                            expanded_channel,
                            expansion.multiplier,
                            expansion.shift,
                            expansion.output_quantization.zero_point_at(),
                        )
                        scf.YieldOp([expanded])
                    with ir.InsertionPoint(value_branch.else_block):
                        scf.YieldOp([expansion_padding])
                    return tensor.InsertOp(
                        value_branch.results[0], state, [patch, lane]
                    ).result

                b_buffer = nested_tensor_loop(
                    (patch_segments, segment_lanes), b_initial, fill_b
                )
                depthwise_padding = constant(
                    i8, depthwise.output_quantization.zero_point_at()
                )
                # A rank-1 tensor is still one logical C segment. Keeping the
                # singleton segment dimension out avoids an over-conservative
                # stream range analysis in stock IREE for tensor.insert.
                c_type = ir.RankedTensorType.get([segment_lanes], i8)
                c_initial = tensor.SplatOp(c_type, depthwise_padding, []).result

                def fill_c(c_indices: tuple[ir.Value, ...], state: ir.Value) -> ir.Value:
                    """Computes one depthwise result exactly once into C."""
                    lane = c_indices[0]
                    expanded_channel = arith.AddIOp(chunk, lane).result
                    valid = arith.CmpIOp(
                        arith.CmpIPredicate.ult,
                        expanded_channel,
                        constant(index_type, expanded_channels),
                    ).result
                    value_branch = scf.IfOp(valid, [i8], has_else=True)
                    with ir.InsertionPoint(value_branch.then_block):
                        bias = tensor.ExtractOp(
                            depthwise.bias, [expanded_channel]
                        ).result
                        input_zp = constant(
                            i32, depthwise.input_quantization.zero_point_at()
                        )
                        weight_zp = constant(
                            i32, depthwise.weight_quantization.zero_point_at()
                        )

                        def depthwise_product(reduction, accumulator):
                            """Consumes the retained Kh×Kw B patch for one C lane."""
                            patch = reduction[0]
                            kernel_width_value = constant(index_type, kernel_width)
                            kh = arith.DivUIOp(patch, kernel_width_value).result
                            kw = arith.RemUIOp(patch, kernel_width_value).result
                            b_value = tensor.ExtractOp(
                                b_buffer, [patch, lane]
                            ).result
                            weight_value = tensor.ExtractOp(
                                depthwise.weight,
                                [kh, kw, expanded_channel, constant(index_type, 0)],
                            ).result
                            lhs = arith.SubIOp(
                                arith.ExtSIOp(i32, b_value).result, input_zp
                            ).result
                            rhs = arith.SubIOp(
                                arith.ExtSIOp(i32, weight_value).result, weight_zp
                            ).result
                            return arith.AddIOp(
                                accumulator, arith.MulIOp(lhs, rhs).result
                            ).result

                        accumulator = nested_reduction(
                            (patch_segments,), bias, depthwise_product
                        )
                        depthwise_value = emit_requantize_i8_expanded(
                            accumulator,
                            expanded_channel,
                            depthwise.multiplier,
                            depthwise.shift,
                            depthwise.output_quantization.zero_point_at(),
                        )
                        scf.YieldOp([depthwise_value])
                    with ir.InsertionPoint(value_branch.else_block):
                        scf.YieldOp([depthwise_padding])
                    return tensor.InsertOp(value_branch.results[0], state, [lane]).result

                c_buffer = nested_tensor_loop((segment_lanes,), c_initial, fill_c)
                projection_input_zp = constant(
                    i32, projection.input_quantization.zero_point_at()
                )
                projection_weight_zp = constant(
                    i32, projection.weight_quantization.zero_point_at()
                )

                def update_d(d_indices: tuple[ir.Value, ...], state: ir.Value) -> ir.Value:
                    """Applies one retained C lane to one D accumulator lane."""
                    output_channel, lane = d_indices
                    expanded_channel = arith.AddIOp(chunk, lane).result
                    valid = arith.CmpIOp(
                        arith.CmpIPredicate.ult,
                        expanded_channel,
                        constant(index_type, expanded_channels),
                    ).result
                    branch = scf.IfOp(valid, [state.type], has_else=True)
                    with ir.InsertionPoint(branch.then_block):
                        c_value = tensor.ExtractOp(c_buffer, [lane]).result
                        weight_value = tensor.ExtractOp(
                            projection.weight,
                            [
                                constant(index_type, 0),
                                constant(index_type, 0),
                                expanded_channel,
                                output_channel,
                            ],
                        ).result
                        accumulator = tensor.ExtractOp(
                            state, [output_channel]
                        ).result
                        lhs = arith.SubIOp(
                            arith.ExtSIOp(i32, c_value).result, projection_input_zp
                        ).result
                        rhs = arith.SubIOp(
                            arith.ExtSIOp(i32, weight_value).result,
                            projection_weight_zp,
                        ).result
                        updated = arith.AddIOp(
                            accumulator, arith.MulIOp(lhs, rhs).result
                        ).result
                        scf.YieldOp(
                            [tensor.InsertOp(updated, state, [output_channel]).result]
                        )
                    with ir.InsertionPoint(branch.else_block):
                        scf.YieldOp([state])
                    return branch.results[0]

                d_updated = nested_tensor_loop(
                    (output_channels, segment_lanes), d_state, update_d
                )
                scf.YieldOp([d_updated])
            d_final = chunk_loop.results[0]

            def store_output_lane(
                output_indices: tuple[ir.Value, ...], state: ir.Value
            ) -> ir.Value:
                """Requantizes one D lane and optionally applies the skip add."""
                output_channel = output_indices[0]
                accumulator = tensor.ExtractOp(d_final, [output_channel]).result
                value = emit_requantize_i8_expanded(
                    accumulator,
                    output_channel,
                    projection.multiplier,
                    projection.shift,
                    projection.output_quantization.zero_point_at(),
                )
                if candidate.residual_input is not None and not split_residual:
                    residual = tensor.ExtractOp(
                        candidate.residual_input, [n, oh, ow, output_channel]
                    ).result
                    value = _i8_residual_add(
                        value,
                        residual,
                        projection.output_quantization.zero_point_at(),
                    )
                return tensor.InsertOp(value, state, [output_channel]).result

            pixel_type = ir.RankedTensorType.get([output_channels], i8)
            pixel_initial = tensor.SplatOp(
                pixel_type,
                constant(i8, projection.output_quantization.zero_point_at()),
                [],
            ).result
            pixel = nested_tensor_loop(
                (output_channels,), pixel_initial, store_output_lane
            )
            # Store the complete D/output segment in one rank-reduced slice.
            # This also gives stock IREE an exact static access length instead
            # of four independently dynamic tensor.insert coordinates.
            dynamic = ir.ShapedType.get_dynamic_size()
            return tensor.InsertSliceOp(
                pixel,
                output,
                [n, oh, ow],
                [],
                [],
                [dynamic, dynamic, dynamic, 0],
                [1, 1, 1, output_channels],
                [1, 1, 1, 1],
            ).result

        dispatch_body = ir.Block.create_at_start(dispatch.body, [])
        with ir.InsertionPoint(dispatch_body):
            dispatch_result = nested_tensor_loop(
                projection.output_shape[:3], final_initial, emit_pixel
            )
            flow.ReturnOp([dispatch_result])
        count_body = ir.Block.create_at_start(
            dispatch.workgroup_count, [index_type]
        )
        with ir.InsertionPoint(count_body):
            count = count_body.arguments[0]
            flow.ReturnOp([count, count, count])
        generated = dispatch.results[0]

    replace_all_uses(final_operation.results[0], generated)
    # Deduplicate before any erase: preprocessing CSE can make several DPS ops
    # share one tensor.empty, and dereferencing a second stale Python handle is
    # unsafe in the MLIR bindings.
    cleanup_empty: set[ir.Operation] = set()
    for operation in (
        expansion.bias_initializer,
        expansion.rescale,
        depthwise.accumulator_initializer,
        depthwise.bias_add,
        depthwise.rescale,
        projection.bias_initializer,
        projection.rescale,
    ):
        try:
            empty = owner_operation(operation.operands[-1])
            if empty is not None and empty.name == "tensor.empty":
                cleanup_empty.add(empty)
        except (IndexError, AttributeError):
            pass
    if candidate.residual is not None and candidate.residual.mode == "fused":
        candidate.residual.final_operation.erase()
    projection.rescale.erase()
    projection.conv.erase()
    erase_dead_operation(projection.bias_initializer)
    depthwise.rescale.erase()
    depthwise.bias_add.erase()
    depthwise.collapse.erase()
    depthwise.conv.erase()
    erase_dead_operation(depthwise.accumulator_initializer)
    erase_dead_operation(candidate.depthwise_padding)
    expansion.rescale.erase()
    expansion.conv.erase()
    erase_dead_operation(expansion.bias_initializer)
    for empty in cleanup_empty:
        erase_dead_operation(empty)
