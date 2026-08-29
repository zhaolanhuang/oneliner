"""Shared scalar reduction construction for Conv2D and depthwise emitters."""

from __future__ import annotations

from collections.abc import Callable

from iree.compiler import ir
from iree.compiler.dialects import arith, linalg, scf, tensor, tosa

from ..ir_utils import expected_map
from .common import constant


def index_multiply_add(base: ir.Value, scale: int, offset: ir.Value, dilation: int) -> ir.Value:
    """Builds ``base * scale + offset * dilation`` in index arithmetic."""
    index_type = ir.IndexType.get()
    scaled_base = arith.MulIOp(base, constant(index_type, scale)).result
    scaled_offset = arith.MulIOp(offset, constant(index_type, dilation)).result
    return arith.AddIOp(scaled_base, scaled_offset).result


def nested_reduction(
    bounds: tuple[int, ...],
    initial: ir.Value,
    body: Callable[[tuple[ir.Value, ...], ir.Value], ir.Value],
) -> ir.Value:
    """Builds perfectly nested scf.for reductions with one i32 iter_arg."""
    index_type = ir.IndexType.get()
    lower = constant(index_type, 0)
    step = constant(index_type, 1)

    def build(level: int, indices: tuple[ir.Value, ...], accumulator: ir.Value) -> ir.Value:
        """Recursively emits one reduction dimension and returns its result."""
        if level == len(bounds):
            return body(indices, accumulator)
        upper = constant(index_type, bounds[level])
        loop = scf.ForOp(lower, upper, step, [accumulator])
        with ir.InsertionPoint(loop.body):
            updated = build(
                level + 1,
                indices + (loop.body.arguments[0],),
                loop.body.arguments[1],
            )
            scf.YieldOp([updated])
        return loop.results[0]

    return build(0, (), initial)


def nested_tensor_loop(
    bounds: tuple[int, ...],
    initial: ir.Value,
    body: Callable[[tuple[ir.Value, ...], ir.Value], ir.Value],
) -> ir.Value:
    """Builds nested loops carrying one statically shaped tensor state."""
    index_type = ir.IndexType.get()
    lower = constant(index_type, 0)
    step = constant(index_type, 1)

    def build(level: int, indices: tuple[ir.Value, ...], state: ir.Value) -> ir.Value:
        """Recursively emits one tensor-state dimension and returns its result."""
        if level == len(bounds):
            return body(indices, state)
        upper = constant(index_type, bounds[level])
        loop = scf.ForOp(lower, upper, step, [state])
        with ir.InsertionPoint(loop.body):
            updated = build(
                level + 1,
                indices + (loop.body.arguments[0],),
                loop.body.arguments[1],
            )
            scf.YieldOp([updated])
        return loop.results[0]

    return build(0, (), initial)


def create_output_generic(rescale: ir.Operation, output_shape: tuple[int, ...]) -> ir.Operation:
    """Creates a parallel output shell using the original destination tensor."""
    identity = expected_map(
        rescale.context, len(output_shape), tuple(range(len(output_shape)))
    )
    parallel = ir.Attribute.parse(
        "#linalg.iterator_type<parallel>", context=rescale.context
    )
    with ir.InsertionPoint(rescale):
        return linalg.GenericOp(
            [rescale.results[0].type],
            [],
            list(rescale.opview.outputs),
            ir.ArrayAttr.get([identity]),
            ir.ArrayAttr.get([parallel] * len(output_shape)),
        ).operation


def emit_requantize_i8(
    accumulator: ir.Value,
    channel: ir.Value,
    multiplier_tensor: ir.Value,
    shift_tensor: ir.Value,
    output_zero_point: int,
    context: ir.Context,
) -> ir.Value:
    """Emits exact per-channel DOUBLE_ROUND and signed-int8 saturation."""
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    multiplier = tensor.ExtractOp(multiplier_tensor, [channel]).result
    shift = tensor.ExtractOp(shift_tensor, [channel]).result
    rounding = ir.Attribute.parse("#tosa.rounding_mode<DOUBLE_ROUND>", context=context)
    scaled = tosa.ApplyScaleOp(i32, accumulator, multiplier, shift, rounding).result
    shifted = arith.AddIOp(scaled, constant(i32, output_zero_point)).result
    lower = arith.MaxSIOp(shifted, constant(i32, -128)).result
    upper = arith.MinSIOp(lower, constant(i32, 127)).result
    return arith.TruncIOp(i8, upper).result


def emit_requantize_i8_expanded(
    accumulator: ir.Value,
    channel: ir.Value,
    multiplier_tensor: ir.Value,
    shift_tensor: ir.Value,
    output_zero_point: int,
) -> ir.Value:
    """Emits the authoritative TOSA DOUBLE_ROUND operation in scalar schedules."""
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    multiplier = tensor.ExtractOp(multiplier_tensor, [channel]).result
    shift = tensor.ExtractOp(shift_tensor, [channel]).result
    rounding = ir.Attribute.parse(
        "#tosa.rounding_mode<DOUBLE_ROUND>", context=accumulator.context
    )
    scaled = tosa.ApplyScaleOp(i32, accumulator, multiplier, shift, rounding).result
    offset = arith.AddIOp(scaled, constant(i32, output_zero_point)).result
    lower = arith.MaxSIOp(offset, constant(i32, -128)).result
    upper = arith.MinSIOp(lower, constant(i32, 127)).result
    return arith.TruncIOp(i8, upper).result
