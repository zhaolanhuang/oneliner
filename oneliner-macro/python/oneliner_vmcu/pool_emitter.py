"""Emits byte-addressed tied-pool dispatches for a complete compact graph.

Paper correspondence:
* vMCU §3 (PDF p.4), Figure 2: coordinates memory management, kernel
  execution, and compiler-generated code around one compact activation pool.
* vMCU §4 (PDF pp.4–5): implements the circular pool, row-major logical
  addresses, ``b_in``/``b_out`` offsets, and modulo physical addressing.
* vMCU §5.1–§5.2 (PDF pp.5–7), Figures 4–6: lowers FC, convolution, and
  inverted-bottleneck schedules into load/compute/store operations.

Engineering adaptation: the paper's compiler emits C++ and vector intrinsics
(§6 and §6.1, PDF p.7). This backend instead emits IREE Flow dispatches with
one tied read/write pool operand. The present RAMLoad/RAMStore implementation is
scalar; it does not claim to implement the paper's vectorized memcpy fast path.
"""

from __future__ import annotations

from math import prod

from iree.compiler import ir
from iree.compiler.dialects import arith, flow, iree_tensor_ext, scf, tensor

from .compact_analysis import CompactAnalysis, CompactBindings, MaterializedBoundary
from .ir_utils import dense_ints, generic_io, operation, owner_operation, replace_all_uses
from .model import (
    Conv2DMatch,
    DepthwiseConv2DMatch,
    FullyConnectedMatch,
    InvertedBottleneckMatch,
    PatternMatch,
)
from .emitters.common import constant
from .emitters.convolution import emit_requantize_i8_expanded, nested_reduction, nested_tensor_loop
from .emitters.inverted_bottleneck import _i8_residual_add, _in_range
from .emitters.segments import emit_static_segment_buffer


_DYNAMIC = ir.ShapedType.get_dynamic_size()


def _tied_operands(*indices: int) -> ir.ArrayAttr:
    index_type = ir.IndexType.get()
    return ir.ArrayAttr.get(
        [ir.IntegerAttr.get(index_type, value) for value in indices]
    )


def _direct_operations(function: ir.Operation) -> list[ir.Operation]:
    return [operation(item) for item in function.regions[0].blocks[0].operations]


def _find_abi(module: ir.Module) -> tuple[ir.Operation, ir.Operation, ir.Operation]:
    functions: list[ir.Operation] = []

    def collect(candidate: ir.Operation) -> ir.WalkResult:
        if candidate.name in ("util.func", "func.func"):
            functions.append(candidate)
            return ir.WalkResult.SKIP
        return ir.WalkResult.ADVANCE

    module.operation.walk(collect)
    if len(functions) != 1:
        raise ValueError("compact pool ABI requires exactly one function")
    function = functions[0]
    imports = [item for item in _direct_operations(function) if item.name == "hal.tensor.import"]
    exports = [item for item in _direct_operations(function) if item.name == "hal.tensor.export"]
    if len(imports) != 1 or len(exports) != 1:
        raise ValueError("compact pool ABI requires one tensor import and one tensor export")
    return function, imports[0], exports[0]


def _index(value: int) -> ir.Value:
    return constant(ir.IndexType.get(), value)


def _flatten(indices: tuple[ir.Value, ...], shape: tuple[int, ...]) -> ir.Value:
    """Maps tensor coordinates to ``Laddr`` in row-major order.

    This is the linearized-address mapping used by vMCU §4 (PDF p.4), before
    the tensor-specific base offset and circular-pool modulo are applied.
    """
    result = indices[0]
    for extent, index in zip(shape[1:], indices[1:], strict=True):
        result = arith.AddIOp(arith.MulIOp(result, _index(extent)).result, index).result
    return result


def _physical(logical: ir.Value, base: int, capacity: int | None) -> ir.Value:
    """Applies the paper's base offset and circular-pool address equation.

    vMCU §4 (PDF p.4) defines ``Pool[addr]`` as
    ``Pool[addr % (MemCap / Seg)]``. This implementation measures both address
    and capacity in bytes, so the equivalent divisor is simply ``capacity``.
    ``capacity=None`` is an engineering fast path for a proven non-wrapping
    external/materialized tensor, not a separate paper schedule.
    """
    offset = arith.AddIOp(logical, _index(base)).result
    return offset if capacity is None else arith.RemUIOp(offset, _index(capacity)).result


def _pool_load(pool: ir.Value, logical: ir.Value, base: int, capacity: int | None) -> ir.Value:
    """Emits scalar RAMLoad from the compact pool.

    Corresponds to RAMLoad in vMCU §5.1, Figures 4–5 (PDF pp.5–6), including
    the modulo boundary check described on PDF p.6. Unlike §6.1's vectorized
    RAMLoad, this operation loads one i8 element.
    """
    offset = _physical(logical, base, capacity)
    i8 = ir.IntegerType.get_signless(8)
    loaded = iree_tensor_ext.DispatchTensorLoadOp(
        ir.RankedTensorType.get([1], i8),
        pool,
        [],
        [offset],
        [],
        [],
        [_DYNAMIC],
        [1],
        [1],
    ).result
    return tensor.ExtractOp(loaded, [_index(0)]).result


def _pool_store(
    pool: ir.Value, value: ir.Value, logical: ir.Value, base: int, capacity: int | None
) -> None:
    """Emits scalar RAMStore at the planned output base.

    Corresponds to RAMStore in vMCU §5.1, Figures 4–5 (PDF pp.5–6). The
    overwrite is safe only because compact_memory.py has already enforced §4
    Equation (1), generalized to graph edges by §5.2 Equation (2).
    """
    offset = _physical(logical, base, capacity)
    one = tensor.FromElementsOp(ir.RankedTensorType.get([1], value.type), [value]).result
    iree_tensor_ext.DispatchTensorStoreOp(
        one,
        pool,
        [],
        [offset],
        [],
        [],
        [_DYNAMIC],
        [1],
        [1],
    )


def _load_dispatch_tensor(source: ir.Value, tensor_type: ir.RankedTensorType) -> ir.Value:
    rank = tensor_type.rank
    return iree_tensor_ext.DispatchTensorLoadOp(
        tensor_type,
        source,
        [],
        [],
        [],
        [],
        [0] * rank,
        list(tensor_type.shape),
        [1] * rank,
    ).result


def _dispatch_type(access: str, tensor_type: ir.Type) -> ir.Type:
    return ir.Type.parse(f"!iree_tensor_ext.dispatch.tensor<{access}:{tensor_type}>")


def _emit_dispatch(
    anchor: ir.Operation,
    pool: ir.Value,
    constants: tuple[ir.Value, ...],
    body_builder,
) -> ir.Value:
    """Wraps one scheduled kernel in a tied read/write IREE dispatch.

    The generated-kernel role corresponds to vMCU §3 Figure 2 and §6
    (PDF pp.4, 7). A single tied IREE operand/result is an ABI adaptation that
    serializes destructive pool updates; it is not an interface specified by
    the paper.
    """
    pool_type = ir.RankedTensorType(pool.type)
    with ir.InsertionPoint(anchor):
        workload = _index(1)
        dispatch = flow.DispatchWorkgroupsOp(
            [pool_type],
            [workload],
            [pool, *constants],
            [],
            [],
            tied_operands=_tied_operands(0),
        )
    block_types = [_dispatch_type("readwrite", pool_type)] + [
        _dispatch_type("readonly", value.type) for value in constants
    ]
    body = ir.Block.create_at_start(dispatch.workgroup_body, block_types)
    with ir.InsertionPoint(body):
        loaded = tuple(
            _load_dispatch_tensor(argument, ir.RankedTensorType(value.type))
            for argument, value in zip(body.arguments[1:], constants, strict=True)
        )
        body_builder(body.arguments[0], loaded)
        flow.ReturnOp([])
    count = ir.Block.create_at_start(dispatch.workgroup_count, [ir.IndexType.get()])
    with ir.InsertionPoint(count):
        one = _index(1)
        flow.ReturnOp([one, one, one])
    return dispatch.results[0]


def _loops(extents: tuple[int, ...], body, prefix: tuple[ir.Value, ...] = ()) -> None:
    """Builds the loop nests that realize the scheduled activation traversal.

    This is the structural counterpart of the two-level tiling loops in vMCU
    §5.1, Figures 4–5 (PDF pp.5–6). Hardware-vector inner tiles from §6.1 are
    not represented by this generic scalar helper.
    """
    if not extents:
        body(prefix)
        return
    loop = scf.ForOp(_index(0), _index(extents[0]), _index(1), [])
    with ir.InsertionPoint(loop.body):
        _loops(extents[1:], body, prefix + (loop.body.arguments[0],))
        scf.YieldOp([])


def _input_value(
    pool: ir.Value,
    indices: tuple[ir.Value, ir.Value, ir.Value, ir.Value],
    shape: tuple[int, int, int, int],
    base: int,
    capacity: int | None,
) -> ir.Value:
    return _pool_load(pool, _flatten(indices, shape), base, capacity)


def _emit_conv(
    candidate: Conv2DMatch,
    pool: ir.Value,
    loaded: tuple[ir.Value, ...],
    input_shape: tuple[int, int, int, int],
    input_base: int,
    output_base: int,
    input_capacity: int | None,
    output_capacity: int | None,
) -> None:
    """Lowers a quantized convolution to compact-pool accesses.

    Follows vMCU §5.1 Figure 5 (PDF p.6): read an input tile/segment, reduce
    with Flash-resident weights, quantize, and store directly at ``b_out``.
    Returning the input zero-point for out-of-bounds padding is this compiler's
    non-materializing implementation of the paper's boundary-check step.
    """
    weight, bias_values, multiplier, shift = loaded
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    input_zp = constant(i32, candidate.input_quantization.zero_point_at())
    weight_zp = constant(i32, candidate.weight_quantization.zero_point_at())

    def output(indices: tuple[ir.Value, ...]) -> None:
        n, oh, ow, oc = indices
        bias = tensor.ExtractOp(bias_values, [oc]).result

        def product(indices2, accumulator):
            kh, kw, ic = indices2
            padded_h = arith.AddIOp(
                arith.MulIOp(oh, _index(candidate.strides[0])).result,
                arith.MulIOp(kh, _index(candidate.dilations[0])).result,
            ).result
            padded_w = arith.AddIOp(
                arith.MulIOp(ow, _index(candidate.strides[1])).result,
                arith.MulIOp(kw, _index(candidate.dilations[1])).result,
            ).result
            valid = arith.AndIOp(
                _in_range(padded_h, candidate.padding_low[1], input_shape[1]),
                _in_range(padded_w, candidate.padding_low[2], input_shape[2]),
            ).result
            branch = scf.IfOp(valid, [i8], has_else=True)
            with ir.InsertionPoint(branch.then_block):
                ih = arith.SubIOp(padded_h, _index(candidate.padding_low[1])).result
                iw = arith.SubIOp(padded_w, _index(candidate.padding_low[2])).result
                scf.YieldOp([
                    _input_value(
                        pool, (n, ih, iw, ic), input_shape, input_base, input_capacity
                    )
                ])
            with ir.InsertionPoint(branch.else_block):
                scf.YieldOp([constant(i8, candidate.input_quantization.zero_point_at())])
            weight_value = tensor.ExtractOp(weight, [kh, kw, ic, oc]).result
            lhs = arith.SubIOp(arith.ExtSIOp(i32, branch.results[0]).result, input_zp).result
            rhs = arith.SubIOp(arith.ExtSIOp(i32, weight_value).result, weight_zp).result
            return arith.AddIOp(accumulator, arith.MulIOp(lhs, rhs).result).result

        accumulator = nested_reduction(candidate.weight_shape[:3], bias, product)
        value = emit_requantize_i8_expanded(
            accumulator,
            oc,
            multiplier,
            shift,
            candidate.output_quantization.zero_point_at(),
        )
        _pool_store(
            pool,
            value,
            _flatten(indices, candidate.output_shape),
            output_base,
            output_capacity,
        )

    _loops(candidate.output_shape, output)


def _emit_depthwise(
    candidate: DepthwiseConv2DMatch,
    pool: ir.Value,
    loaded: tuple[ir.Value, ...],
    input_shape: tuple[int, int, int, int],
    input_base: int,
    output_base: int,
    input_capacity: int | None,
    output_capacity: int | None,
) -> None:
    """Lowers standalone depthwise convolution to compact-pool accesses.

    The load/compute/store discipline is from vMCU §5.1 Figure 5 (PDF p.6).
    Standalone arbitrary-K depthwise support is an engineering extension; the
    paper discusses depthwise primarily inside the IBN schedule in §5.2.
    """
    weight, bias_values, multiplier, shift = loaded
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    input_zp = constant(i32, candidate.input_quantization.zero_point_at())
    weight_zp = constant(i32, candidate.weight_quantization.zero_point_at())

    def output(indices: tuple[ir.Value, ...]) -> None:
        n, oh, ow, channel = indices
        bias = tensor.ExtractOp(bias_values, [channel]).result

        def product(indices2, accumulator):
            kh, kw = indices2
            padded_h = arith.AddIOp(
                arith.MulIOp(oh, _index(candidate.strides[0])).result,
                arith.MulIOp(kh, _index(candidate.dilations[0])).result,
            ).result
            padded_w = arith.AddIOp(
                arith.MulIOp(ow, _index(candidate.strides[1])).result,
                arith.MulIOp(kw, _index(candidate.dilations[1])).result,
            ).result
            valid = arith.AndIOp(
                _in_range(padded_h, candidate.padding_low[1], input_shape[1]),
                _in_range(padded_w, candidate.padding_low[2], input_shape[2]),
            ).result
            branch = scf.IfOp(valid, [i8], has_else=True)
            with ir.InsertionPoint(branch.then_block):
                ih = arith.SubIOp(padded_h, _index(candidate.padding_low[1])).result
                iw = arith.SubIOp(padded_w, _index(candidate.padding_low[2])).result
                scf.YieldOp([
                    _input_value(
                        pool,
                        (n, ih, iw, channel),
                        input_shape,
                        input_base,
                        input_capacity,
                    )
                ])
            with ir.InsertionPoint(branch.else_block):
                scf.YieldOp([constant(i8, candidate.input_quantization.zero_point_at())])
            weight_value = tensor.ExtractOp(weight, [kh, kw, channel, _index(0)]).result
            lhs = arith.SubIOp(arith.ExtSIOp(i32, branch.results[0]).result, input_zp).result
            rhs = arith.SubIOp(arith.ExtSIOp(i32, weight_value).result, weight_zp).result
            return arith.AddIOp(accumulator, arith.MulIOp(lhs, rhs).result).result

        accumulator = nested_reduction(candidate.weight_shape[:2], bias, product)
        value = emit_requantize_i8_expanded(
            accumulator,
            channel,
            multiplier,
            shift,
            candidate.output_quantization.zero_point_at(),
        )
        _pool_store(
            pool,
            value,
            _flatten(indices, candidate.output_shape),
            output_base,
            output_capacity,
        )

    _loops(candidate.output_shape, output)


def _emit_fc(
    candidate: FullyConnectedMatch,
    pool: ir.Value,
    loaded: tuple[ir.Value, ...],
    input_shape: tuple[int, ...],
    input_base: int,
    output_base: int,
    input_capacity: int | None,
    output_capacity: int | None,
) -> None:
    """Lowers fully connected/GEMM traversal into the circular pool.

    Corresponds to vMCU §2.4 Figure 1(c), §4 Figure 3, and §5.1 Figure 4
    (PDF pp.3–5): consume one input row segment, accumulate an output segment,
    and write it where the planner proves the input has become dead.
    """
    weight, bias_values, multiplier, shift = loaded
    i32 = ir.IntegerType.get_signless(32)
    input_zp = constant(i32, candidate.input_zero_point)
    weight_zp = constant(i32, candidate.weight_zero_point)

    def output(indices: tuple[ir.Value, ...]) -> None:
        flat_output = _flatten(indices, candidate.output_shape)
        output_channel = indices[-1]
        row = arith.DivUIOp(flat_output, _index(candidate.output_channels)).result
        bias = tensor.ExtractOp(bias_values, [output_channel]).result

        def product(reduction, accumulator):
            input_channel = reduction[0]
            logical = arith.AddIOp(
                arith.MulIOp(row, _index(candidate.input_channels)).result,
                input_channel,
            ).result
            input_value = _pool_load(pool, logical, input_base, input_capacity)
            weight_value = tensor.ExtractOp(weight, [output_channel, input_channel]).result
            lhs = arith.SubIOp(arith.ExtSIOp(i32, input_value).result, input_zp).result
            rhs = arith.SubIOp(arith.ExtSIOp(i32, weight_value).result, weight_zp).result
            return arith.AddIOp(accumulator, arith.MulIOp(lhs, rhs).result).result

        accumulator = nested_reduction((candidate.input_channels,), bias, product)
        value = emit_requantize_i8_expanded(
            accumulator, output_channel, multiplier, shift, candidate.output_zero_point
        )
        _pool_store(pool, value, flat_output, output_base, output_capacity)

    _loops(candidate.output_shape, output)


def _emit_ibn(
    candidate: InvertedBottleneckMatch,
    pool: ir.Value,
    loaded: tuple[ir.Value, ...],
    input_shape: tuple[int, int, int, int],
    input_base: int,
    output_base: int,
    input_capacity: int | None,
    output_capacity: int | None,
) -> None:
    """Fuses expansion, depthwise, projection, and residual for one IBN.

    vMCU §5.2 Figure 6 (PDF pp.6–7) names the local states B, C, D, and E and
    derives the 3×3 workspace as ``9 + 1 + 1 = 11`` segments. Here that schedule
    is generalized to ``Kh*Kw + 2`` segments; arbitrary kernel sizes and the
    exact IREE tensor representation are engineering extensions.
    """
    (
        expansion_weight,
        expansion_bias,
        expansion_multiplier,
        expansion_shift,
        depthwise_weight,
        depthwise_bias,
        depthwise_multiplier,
        depthwise_shift,
        projection_weight,
        projection_bias,
        projection_multiplier,
        projection_shift,
    ) = loaded
    expansion = candidate.expansion
    depthwise = candidate.depthwise
    projection = candidate.projection
    kernel_h, kernel_w = depthwise.weight_shape[:2]
    patch_segments = kernel_h * kernel_w
    segment_lanes = min(input_shape[3], projection.output_shape[3])
    expanded_channels = expansion.output_shape[3]
    output_channels = projection.output_shape[3]
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)

    def pixel(pixel_indices: tuple[ir.Value, ...]) -> None:
        n, oh, ow = pixel_indices
        chunk_loop = scf.ForOp(_index(0), _index(expanded_channels), _index(segment_lanes), [projection_bias])
        with ir.InsertionPoint(chunk_loop.body):
            chunk, d_state = chunk_loop.body.arguments
            expansion_padding = constant(i8, expansion.output_quantization.zero_point_at())
            # vMCU §5.2 Figure 6 (PDF p.7): B holds one expansion segment for
            # every depthwise kernel point (nine segments for a 3×3 kernel).
            b_initial = emit_static_segment_buffer(
                patch_segments, segment_lanes, i8, expansion_padding
            )

            def fill_b(indices: tuple[ir.Value, ...], state: ir.Value) -> ir.Value:
                patch, lane = indices
                kh = arith.DivUIOp(patch, _index(kernel_w)).result
                kw = arith.RemUIOp(patch, _index(kernel_w)).result
                expanded_channel = arith.AddIOp(chunk, lane).result
                padded_h = arith.AddIOp(
                    arith.MulIOp(oh, _index(depthwise.strides[0])).result,
                    arith.MulIOp(kh, _index(depthwise.dilations[0])).result,
                ).result
                padded_w = arith.AddIOp(
                    arith.MulIOp(ow, _index(depthwise.strides[1])).result,
                    arith.MulIOp(kw, _index(depthwise.dilations[1])).result,
                ).result
                valid = arith.AndIOp(
                    arith.CmpIOp(
                        arith.CmpIPredicate.ult, expanded_channel, _index(expanded_channels)
                    ).result,
                    arith.AndIOp(
                        _in_range(padded_h, depthwise.padding_low[1], input_shape[1]),
                        _in_range(padded_w, depthwise.padding_low[2], input_shape[2]),
                    ).result,
                ).result
                branch = scf.IfOp(valid, [i8], has_else=True)
                with ir.InsertionPoint(branch.then_block):
                    ih = arith.SubIOp(padded_h, _index(depthwise.padding_low[1])).result
                    iw = arith.SubIOp(padded_w, _index(depthwise.padding_low[2])).result
                    bias = tensor.ExtractOp(expansion_bias, [expanded_channel]).result
                    expansion_input_zp = constant(
                        i32, expansion.input_quantization.zero_point_at()
                    )
                    expansion_weight_zp = constant(
                        i32, expansion.weight_quantization.zero_point_at()
                    )

                    def expansion_product(reduction, accumulator):
                        input_channel = reduction[0]
                        input_value = _input_value(
                            pool,
                            (n, ih, iw, input_channel),
                            input_shape,
                            input_base,
                            input_capacity,
                        )
                        weight_value = tensor.ExtractOp(
                            expansion_weight,
                            [_index(0), _index(0), input_channel, expanded_channel],
                        ).result
                        lhs = arith.SubIOp(
                            arith.ExtSIOp(i32, input_value).result, expansion_input_zp
                        ).result
                        rhs = arith.SubIOp(
                            arith.ExtSIOp(i32, weight_value).result, expansion_weight_zp
                        ).result
                        return arith.AddIOp(
                            accumulator, arith.MulIOp(lhs, rhs).result
                        ).result

                    accumulator = nested_reduction(
                        (input_shape[3],), bias, expansion_product
                    )
                    expanded = emit_requantize_i8_expanded(
                        accumulator,
                        expanded_channel,
                        expansion_multiplier,
                        expansion_shift,
                        expansion.output_quantization.zero_point_at(),
                    )
                    scf.YieldOp([expanded])
                with ir.InsertionPoint(branch.else_block):
                    scf.YieldOp([expansion_padding])
                return tensor.InsertOp(branch.results[0], state, [patch, lane]).result

            b_buffer = nested_tensor_loop(
                (patch_segments, segment_lanes), b_initial, fill_b
            )
            depthwise_padding = constant(
                i8, depthwise.output_quantization.zero_point_at()
            )
            # vMCU §5.2 Figure 6: C is the single post-depthwise i8 segment.
            c_type = ir.RankedTensorType.get([segment_lanes], i8)
            c_initial = tensor.SplatOp(c_type, depthwise_padding, []).result

            def fill_c(indices: tuple[ir.Value, ...], state: ir.Value) -> ir.Value:
                lane = indices[0]
                expanded_channel = arith.AddIOp(chunk, lane).result
                valid = arith.CmpIOp(
                    arith.CmpIPredicate.ult, expanded_channel, _index(expanded_channels)
                ).result
                branch = scf.IfOp(valid, [i8], has_else=True)
                with ir.InsertionPoint(branch.then_block):
                    bias = tensor.ExtractOp(depthwise_bias, [expanded_channel]).result
                    depthwise_input_zp = constant(
                        i32, depthwise.input_quantization.zero_point_at()
                    )
                    depthwise_weight_zp = constant(
                        i32, depthwise.weight_quantization.zero_point_at()
                    )

                    def depthwise_product(reduction, accumulator):
                        patch = reduction[0]
                        kh = arith.DivUIOp(patch, _index(kernel_w)).result
                        kw = arith.RemUIOp(patch, _index(kernel_w)).result
                        b_value = tensor.ExtractOp(b_buffer, [patch, lane]).result
                        weight_value = tensor.ExtractOp(
                            depthwise_weight,
                            [kh, kw, expanded_channel, _index(0)],
                        ).result
                        lhs = arith.SubIOp(
                            arith.ExtSIOp(i32, b_value).result, depthwise_input_zp
                        ).result
                        rhs = arith.SubIOp(
                            arith.ExtSIOp(i32, weight_value).result, depthwise_weight_zp
                        ).result
                        return arith.AddIOp(
                            accumulator, arith.MulIOp(lhs, rhs).result
                        ).result

                    accumulator = nested_reduction(
                        (patch_segments,), bias, depthwise_product
                    )
                    value = emit_requantize_i8_expanded(
                        accumulator,
                        expanded_channel,
                        depthwise_multiplier,
                        depthwise_shift,
                        depthwise.output_quantization.zero_point_at(),
                    )
                    scf.YieldOp([value])
                with ir.InsertionPoint(branch.else_block):
                    scf.YieldOp([depthwise_padding])
                return tensor.InsertOp(branch.results[0], state, [lane]).result

            c_buffer = nested_tensor_loop((segment_lanes,), c_initial, fill_c)
            projection_input_zp = constant(
                i32, projection.input_quantization.zero_point_at()
            )
            projection_weight_zp = constant(
                i32, projection.weight_quantization.zero_point_at()
            )

            # vMCU §5.2 Figure 6: D is the projection accumulator segment.
            # This implementation uses Cout i32 lanes so Cin != Cout remains
            # valid; that lane-shape generalization is repository-specific.
            def update_d(indices: tuple[ir.Value, ...], state: ir.Value) -> ir.Value:
                output_channel, lane = indices
                expanded_channel = arith.AddIOp(chunk, lane).result
                valid = arith.CmpIOp(
                    arith.CmpIPredicate.ult, expanded_channel, _index(expanded_channels)
                ).result
                branch = scf.IfOp(valid, [state.type], has_else=True)
                with ir.InsertionPoint(branch.then_block):
                    c_value = tensor.ExtractOp(c_buffer, [lane]).result
                    weight_value = tensor.ExtractOp(
                        projection_weight,
                        [_index(0), _index(0), expanded_channel, output_channel],
                    ).result
                    accumulator = tensor.ExtractOp(state, [output_channel]).result
                    lhs = arith.SubIOp(
                        arith.ExtSIOp(i32, c_value).result, projection_input_zp
                    ).result
                    rhs = arith.SubIOp(
                        arith.ExtSIOp(i32, weight_value).result, projection_weight_zp
                    ).result
                    updated = arith.AddIOp(
                        accumulator, arith.MulIOp(lhs, rhs).result
                    ).result
                    scf.YieldOp([
                        tensor.InsertOp(updated, state, [output_channel]).result
                    ])
                with ir.InsertionPoint(branch.else_block):
                    scf.YieldOp([state])
                return branch.results[0]

            d_updated = nested_tensor_loop(
                (output_channels, segment_lanes), d_state, update_d
            )
            scf.YieldOp([d_updated])
        d_final = chunk_loop.results[0]

        def output_lane(indices: tuple[ir.Value, ...]) -> None:
            output_channel = indices[0]
            accumulator = tensor.ExtractOp(d_final, [output_channel]).result
            value = emit_requantize_i8_expanded(
                accumulator,
                output_channel,
                projection_multiplier,
                projection_shift,
                projection.output_quantization.zero_point_at(),
            )
            if candidate.residual_input is not None:
                residual = _input_value(
                    pool,
                    (n, oh, ow, output_channel),
                    input_shape,
                    input_base,
                    input_capacity,
                )
                value = _i8_residual_add(
                    value, residual, projection.output_quantization.zero_point_at()
                )
            logical = _flatten((n, oh, ow, output_channel), candidate.output_shape)
            _pool_store(pool, value, logical, output_base, output_capacity)

        _loops((output_channels,), output_lane)

    _loops(candidate.output_shape[:3], pixel)


def _clone_scalar_value(value: ir.Value, mapping: dict[ir.Value, ir.Value]) -> ir.Value:
    """Clones a side-effect-free scalar expression into the active dispatch."""
    if value in mapping:
        return mapping[value]
    owner = owner_operation(value)
    if owner is None or owner.name != "arith.constant":
        raise ValueError("compact boundary scalar expression has an external value")
    cloned = ir.Operation.create(
        owner.name,
        results=[result.type for result in owner.results],
        attributes={name: owner.attributes[name] for name in owner.attributes},
    )
    for original, result in zip(owner.results, cloned.results, strict=True):
        mapping[original] = result
    return mapping[value]


def _evaluate_generic(
    generic: ir.Operation,
    indices: tuple[ir.Value, ...],
    evaluate_tensor,
) -> ir.Value:
    """Evaluates one validated parallel identity linalg.generic scalar region."""
    inputs, outputs = generic_io(generic)
    block = generic.regions[0].blocks[0]
    mapping = {
        argument: evaluate_tensor(value, indices)
        for argument, value in zip(block.arguments, inputs, strict=False)
    }
    output_argument = block.arguments[len(inputs)]
    mapping[output_argument] = constant(output_argument.type, 0)
    for item in block.operations:
        scalar = operation(item)
        if scalar.name == "linalg.yield":
            return _clone_scalar_value(scalar.operands[0], mapping)
        operands = [_clone_scalar_value(value, mapping) for value in scalar.operands]
        cloned = ir.Operation.create(
            scalar.name,
            results=[result.type for result in scalar.results],
            operands=operands,
            attributes={name: scalar.attributes[name] for name in scalar.attributes},
        )
        for original, result in zip(scalar.results, cloned.results, strict=True):
            mapping[original] = result
    raise ValueError("compact boundary linalg.generic has no yield")


def _evaluate_pooling_sum(
    pooling: ir.Operation,
    indices: tuple[ir.Value, ...],
    evaluate_tensor,
) -> ir.Value:
    """Evaluates one static NHWC sum-pooling result without a tensor arena."""
    inputs = list(pooling.opview.inputs)
    input_value, kernel = inputs
    kernel_shape = tuple(int(item) for item in ir.RankedTensorType(kernel.type).shape)
    strides = dense_ints(pooling.attributes["strides"])
    dilations = dense_ints(pooling.attributes["dilations"])
    result_type = ir.RankedTensorType(pooling.results[0].type).element_type
    n, oh, ow, channel = indices

    def accumulate(reduction: tuple[ir.Value, ...], accumulator: ir.Value) -> ir.Value:
        kh, kw = reduction
        ih = arith.AddIOp(
            arith.MulIOp(oh, _index(strides[0])).result,
            arith.MulIOp(kh, _index(dilations[0])).result,
        ).result
        iw = arith.AddIOp(
            arith.MulIOp(ow, _index(strides[1])).result,
            arith.MulIOp(kw, _index(dilations[1])).result,
        ).result
        value = evaluate_tensor(input_value, (n, ih, iw, channel))
        if value.type != result_type:
            value = arith.ExtSIOp(result_type, value).result
        return arith.AddIOp(accumulator, value).result

    return nested_reduction(kernel_shape, constant(result_type, 0), accumulate)


def _emit_direct_boundary(
    anchor: ir.Operation,
    pool: ir.Value,
    boundary: MaterializedBoundary,
    source_layouts: dict[ir.Value, tuple[tuple[int, ...], int, int | None]],
    output_base: int,
    output_capacity: int | None,
) -> ir.Value:
    """Emits a validated residual/pooling expression directly over the pool.

    Residual lifetime preservation follows vMCU §5.2 Equation (2) and the E
    edge in Figure 6 (PDF pp.6–7). Generic identity/view and pooling evaluation
    are engineering extensions used to keep graph boundaries allocation-free.
    """
    if boundary.direct_kind is None:
        raise ValueError("materialized boundary cannot use direct lowering")

    def body(pool_argument: ir.Value, _loaded: tuple[ir.Value, ...]) -> None:
        def output(indices: tuple[ir.Value, ...]) -> None:
            def evaluate_tensor(
                value: ir.Value, value_indices: tuple[ir.Value, ...]
            ) -> ir.Value:
                layout = source_layouts.get(value)
                if layout is not None:
                    shape, base, source_capacity = layout
                    return _pool_load(
                        pool_argument,
                        _flatten(value_indices, shape),
                        base,
                        source_capacity,
                    )
                owner = owner_operation(value)
                if owner is None:
                    raise ValueError("compact boundary tensor has no defining operation")
                if owner.name == "linalg.generic":
                    return _evaluate_generic(owner, value_indices, evaluate_tensor)
                if owner.name == "linalg.pooling_nhwc_sum":
                    return _evaluate_pooling_sum(owner, value_indices, evaluate_tensor)
                if owner.name in (
                    "tensor.cast",
                    "tensor.collapse_shape",
                    "tensor.expand_shape",
                    "tensor.reshape",
                    "flow.tensor.reshape",
                ):
                    source_shape = tuple(
                        int(item) for item in ir.RankedTensorType(owner.operands[0].type).shape
                    )
                    logical = _flatten(value_indices, boundary.output_shape)
                    remapped: list[ir.Value] = []
                    remaining = logical
                    for extent in reversed(source_shape[1:]):
                        remapped.append(arith.RemUIOp(remaining, _index(extent)).result)
                        remaining = arith.DivUIOp(remaining, _index(extent)).result
                    return evaluate_tensor(
                        owner.operands[0], tuple([remaining, *reversed(remapped)])
                    )
                raise ValueError(
                    f"unsupported direct compact boundary operation: {owner.name}"
                )

            result = evaluate_tensor(boundary.target_value, indices)
            _pool_store(
                pool_argument,
                result,
                _flatten(indices, boundary.output_shape),
                output_base,
                output_capacity,
            )

        _loops(boundary.output_shape, output)

    return _emit_dispatch(anchor, pool, (), body)


def _emit_unpack(
    anchor: ir.Operation,
    pool: ir.Value,
    shape: tuple[int, ...],
    base: int,
    capacity: int,
) -> ir.Value:
    pool_type = ir.RankedTensorType(pool.type)
    output_type = ir.RankedTensorType.get(shape, ir.IntegerType.get_signless(8))
    with ir.InsertionPoint(anchor):
        workload = _index(1)
        dispatch = flow.DispatchWorkgroupsOp(
            [output_type],
            [workload],
            [pool],
            [],
            [],
            tied_operands=_tied_operands(-1),
        )
    body = ir.Block.create_at_start(
        dispatch.workgroup_body,
        [
            _dispatch_type("readonly", pool_type),
            _dispatch_type("writeonly", output_type),
        ],
    )
    with ir.InsertionPoint(body):
        initial = tensor.EmptyOp(shape, ir.IntegerType.get_signless(8)).result

        def load(indices: tuple[ir.Value, ...], state: ir.Value) -> ir.Value:
            value = _pool_load(body.arguments[0], _flatten(indices, shape), base, capacity)
            return tensor.InsertOp(value, state, list(indices)).result

        materialized = nested_tensor_loop(shape, initial, load)
        iree_tensor_ext.DispatchTensorStoreOp(
            materialized,
            body.arguments[1],
            [],
            [],
            [],
            [],
            [0] * len(shape),
            list(shape),
            [1] * len(shape),
        )
        flow.ReturnOp([])
    count = ir.Block.create_at_start(dispatch.workgroup_count, [ir.IndexType.get()])
    with ir.InsertionPoint(count):
        one = _index(1)
        flow.ReturnOp([one, one, one])
    return dispatch.results[0]


def _emit_pack(
    anchor: ir.Operation,
    pool: ir.Value,
    source: ir.Value,
    shape: tuple[int, ...],
    base: int,
    capacity: int,
) -> ir.Value:
    def body(pool_argument: ir.Value, loaded: tuple[ir.Value, ...]) -> None:
        source_tensor = loaded[0]

        def store(indices: tuple[ir.Value, ...]) -> None:
            value = tensor.ExtractOp(source_tensor, list(indices)).result
            _pool_store(
                pool_argument,
                value,
                _flatten(indices, shape),
                base,
                capacity,
            )

        _loops(shape, store)

    return _emit_dispatch(anchor, pool, (source,), body)


def _candidate_constants(candidate: PatternMatch) -> tuple[ir.Value, ...]:
    if isinstance(candidate, (Conv2DMatch, DepthwiseConv2DMatch)):
        return (candidate.weight, candidate.bias, candidate.multiplier, candidate.shift)
    if isinstance(candidate, FullyConnectedMatch):
        return (
            candidate.output_major_weight,
            candidate.bias,
            candidate.multiplier,
            candidate.shift,
        )
    if isinstance(candidate, InvertedBottleneckMatch):
        return tuple(
            value
            for layer in (candidate.expansion, candidate.depthwise, candidate.projection)
            for value in (layer.weight, layer.bias, layer.multiplier, layer.shift)
        )
    raise ValueError(f"unsupported compact candidate {candidate.kind!r}")


def _candidate_anchor(candidate: PatternMatch) -> ir.Operation:
    if isinstance(candidate, FullyConnectedMatch):
        return candidate.matmul
    if isinstance(candidate, InvertedBottleneckMatch):
        # Projection weights/scales are defined after expansion and depthwise;
        # anchoring at the terminal requantization makes every captured Flash
        # tensor dominate the fused dispatch.
        return candidate.projection.rescale
    if isinstance(candidate, (Conv2DMatch, DepthwiseConv2DMatch)):
        return candidate.conv
    raise ValueError(f"unsupported compact candidate {candidate.kind!r}")


def _next_operation(operation_: ir.Operation) -> ir.Operation:
    operations = [operation(item) for item in operation_.block.operations]
    index = operations.index(operation_)
    if index + 1 >= len(operations):
        raise ValueError("materialized boundary has no following insertion anchor")
    return operations[index + 1]


def _first_source_user(
    source: ir.Value, excluded_operations: set[ir.Operation]
) -> ir.Operation:
    source_owner = owner_operation(source)
    if source_owner is None:
        raise ValueError("materialized boundary source has no defining operation")
    operations = [operation(item) for item in source_owner.block.operations]
    positions = {item: index for index, item in enumerate(operations)}
    users = [
        operation(use.owner)
        for use in source.uses
        if operation(use.owner) in positions
        and operation(use.owner) not in excluded_operations
    ]
    if not users:
        raise ValueError("materialized boundary source has no direct user")
    return min(users, key=positions.__getitem__)


def _replace_uses_except(value: ir.Value, replacement: ir.Value, excluded: set[ir.Operation]) -> None:
    for use in list(value.uses):
        owner = operation(use.owner)
        if owner not in excluded:
            use.owner.operands[use.operand_number] = replacement


def _rewrite_abi(
    module: ir.Module, compact: CompactAnalysis
) -> tuple[ir.Operation, ir.Operation, ir.Operation, ir.Value]:
    """Changes the public entry point to one in-place circular-pool tensor.

    The single pool realizes vMCU §4's shared input/output memory model. IREE
    reflection metadata and tied operands are repository-specific mechanisms
    for expressing the destructive update safely through the compiler ABI.
    """
    function, old_import, old_export = _find_abi(module)
    pool_type = ir.RankedTensorType.get(
        [compact.plan.allocated_pool_bytes], ir.IntegerType.get_signless(8)
    )
    import_attributes = {name: old_import.attributes[name] for name in old_import.attributes}
    import_attributes["target_encoding"] = ir.TypeAttr.get(pool_type)
    with ir.InsertionPoint(old_import):
        new_import = ir.Operation.create(
            "hal.tensor.import",
            results=[pool_type],
            operands=list(old_import.operands),
            attributes=import_attributes,
        )
    symbol_name = str(function.attributes["sym_name"]).strip('"')
    declaration = (
        f"sync func @{symbol_name}(%io_pool: {pool_type}) -> (%io_pool: {pool_type})"
    )
    function.attributes["iree.reflection"] = ir.DictAttr.get(
        {"iree.abi.declaration": ir.StringAttr.get(declaration)}
    )
    function.attributes["tied_operands"] = _tied_operands(0)
    return function, old_import, old_export, new_import.results[0]


def emit_compact_graph(
    module: ir.Module,
    candidates: tuple[PatternMatch, ...],
    compact: CompactAnalysis,
    bindings: CompactBindings,
) -> None:
    """Atomically replaces all supported activations with one tied byte pool.

    This is the integration point for vMCU §3 Figure 2 and §6 (PDF pp.4, 7):
    the memory plan controls every generated kernel's physical accesses. The
    atomic graph rewrite, tied SSA chain, and unsupported-boundary pack/unpack
    path are IREE engineering extensions, not algorithms stated in the paper.
    """
    candidate_by_id = {item.root.identifier: item for item in candidates}
    if set(candidate_by_id) != set(compact.candidate_order):
        raise ValueError("compact emitter candidate identities drifted after reparse")
    candidates = tuple(candidate_by_id[item] for item in compact.candidate_order)
    function, old_import, old_export, pool = _rewrite_abi(module, compact)
    candidate_by_kernel = {
        f"kernel_{index}": candidate for index, candidate in enumerate(candidates)
    }
    tensor_by_name = {item.name: item for item in compact.plan.tensors}
    boundary_by_kernel = {
        f"kernel_{item.name}": item for item in bindings.boundaries
    }
    source_candidate_operations = set().union(
        *(item.claimed_operations for item in candidates)
    )
    capacity = compact.plan.logical_pool_bytes
    unpacked: dict[str, ir.Value] = {}

    for execution in compact.plan.execution:
        kernel = next(item for item in compact.plan.kernels if item.name == execution.kernel)
        output_placement = compact.plan.placement_for(kernel.output)
        if execution.kernel in boundary_by_kernel:
            boundary = boundary_by_kernel[execution.kernel]
            target_owner = owner_operation(boundary.target_value)
            if target_owner is None:
                raise ValueError("compact boundary target has no defining operation")
            if boundary.direct_kind is not None:
                source_layouts = {}
                for source_name, source_value in zip(
                    boundary.source_tensors, boundary.source_values, strict=True
                ):
                    source_tensor = tensor_by_name[source_name]
                    source_placement = compact.plan.placement_for(source_name)
                    source_layouts[source_value] = (
                        source_tensor.shape,
                        source_placement.base,
                        capacity if source_placement.wraps else None,
                    )
                pool = _emit_direct_boundary(
                    _next_operation(target_owner),
                    pool,
                    boundary,
                    source_layouts,
                    output_placement.base,
                    capacity if output_placement.wraps else None,
                )
                continue
            excluded: set[ir.Operation] = set()
            for source_name, source_value in zip(
                boundary.source_tensors, boundary.source_values, strict=True
            ):
                if source_name not in unpacked:
                    source_tensor = tensor_by_name[source_name]
                    source_placement = compact.plan.placement_for(source_name)
                    anchor = _first_source_user(
                        source_value, source_candidate_operations
                    )
                    unpacked[source_name] = _emit_unpack(
                        anchor,
                        pool,
                        source_tensor.shape,
                        source_placement.base,
                        capacity,
                    )
                _replace_uses_except(source_value, unpacked[source_name], excluded)
            pool = _emit_pack(
                _next_operation(target_owner),
                pool,
                boundary.target_value,
                boundary.output_shape,
                output_placement.base,
                capacity,
            )
            continue

        candidate = candidate_by_kernel[execution.kernel]
        input_name = kernel.inputs[0]
        input_tensor = tensor_by_name[input_name]
        input_placement = compact.plan.placement_for(input_name)
        input_capacity = capacity if input_placement.wraps else None
        output_capacity = capacity if output_placement.wraps else None

        def body(pool_argument: ir.Value, loaded: tuple[ir.Value, ...]) -> None:
            if isinstance(candidate, Conv2DMatch):
                _emit_conv(
                    candidate,
                    pool_argument,
                    loaded,
                    input_tensor.shape,
                    input_placement.base,
                    output_placement.base,
                    input_capacity,
                    output_capacity,
                )
            elif isinstance(candidate, DepthwiseConv2DMatch):
                _emit_depthwise(
                    candidate,
                    pool_argument,
                    loaded,
                    input_tensor.shape,
                    input_placement.base,
                    output_placement.base,
                    input_capacity,
                    output_capacity,
                )
            elif isinstance(candidate, FullyConnectedMatch):
                _emit_fc(
                    candidate,
                    pool_argument,
                    loaded,
                    input_tensor.shape,
                    input_placement.base,
                    output_placement.base,
                    input_capacity,
                    output_capacity,
                )
            elif isinstance(candidate, InvertedBottleneckMatch):
                _emit_ibn(
                    candidate,
                    pool_argument,
                    loaded,
                    input_tensor.shape,
                    input_placement.base,
                    output_placement.base,
                    input_capacity,
                    output_capacity,
                )
            else:
                raise ValueError(f"unsupported compact candidate {candidate.kind!r}")

        pool = _emit_dispatch(
            _candidate_anchor(candidate), pool, _candidate_constants(candidate), body
        )

    export_attributes = {name: old_export.attributes[name] for name in old_export.attributes}
    export_attributes["source_encoding"] = ir.TypeAttr.get(pool.type)
    with ir.InsertionPoint(old_export):
        new_export = ir.Operation.create(
            "hal.tensor.export",
            results=[result.type for result in old_export.results],
            operands=[pool],
            attributes=export_attributes,
        )
    replace_all_uses(old_export.results[0], new_export.results[0])
    old_export.erase()

    # Remove dead tensor/linalg scaffolding one operation per fresh walk. This
    # deliberately avoids retaining any Python MLIR handle across an erase;
    # shared DPS initializers otherwise reproduce the original stale-handle
    # SIGSEGV in the bindings.
    removable_prefixes = ("tensor.", "linalg.")
    changed = True
    while changed:
        changed = False
        for item in reversed(_direct_operations(function)):
            if item.name.startswith(removable_prefixes) and all(
                not list(result.uses) for result in item.results
            ):
                item.erase()
                changed = True
                break
    if not list(old_import.results[0].uses):
        old_import.erase()
