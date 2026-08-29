"""Standard-MLIR emitter for a proven quantized fully-connected match."""

from __future__ import annotations

from iree.compiler import ir
from iree.compiler.dialects import arith, linalg, scf, tensor, tosa

from ..ir_utils import expected_map, replace_all_uses
from ..model import FullyConnectedMatch, PatternMatch
from .common import constant, flatten_prefix


def emit_fully_connected(match: PatternMatch) -> None:
    """Eliminates a full i32 accumulator tensor using scalar reductions."""
    if not isinstance(match, FullyConnectedMatch):
        raise TypeError(f"fully-connected emitter received {type(match).__name__}")
    candidate = match
    output_type = candidate.rescale.results[0].type
    index_type = ir.IndexType.get()
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    with ir.InsertionPoint(candidate.rescale):
        lower = constant(index_type, 0)
        upper = constant(index_type, candidate.input_channels)
        step = constant(index_type, 1)
        input_zero_point = constant(i32, candidate.input_zero_point)
        weight_zero_point = constant(i32, candidate.weight_zero_point)
        output_zero_point = constant(i32, candidate.output_zero_point)
        clamp_minimum = constant(i32, -128)
        clamp_maximum = constant(i32, 127)
        identity = expected_map(
            candidate.matmul.context,
            len(candidate.output_shape),
            tuple(range(len(candidate.output_shape))),
        )
        parallel = ir.Attribute.parse(
            "#linalg.iterator_type<parallel>", context=candidate.matmul.context
        )
        generated = linalg.GenericOp(
            [output_type],
            [],
            list(candidate.rescale.opview.outputs),
            ir.ArrayAttr.get([identity]),
            ir.ArrayAttr.get([parallel] * len(candidate.output_shape)),
        )

    body = ir.Block.create_at_start(generated.regions[0], [i8])
    with ir.InsertionPoint(body):
        output_indices = [
            linalg.IndexOp(dimension).result
            for dimension in range(len(candidate.output_shape))
        ]
        output_channel = output_indices[-1]
        row = flatten_prefix(output_indices, candidate.output_shape)
        bias = tensor.ExtractOp(candidate.bias, [output_channel]).result
        loop = scf.ForOp(lower, upper, step, [bias])
        with ir.InsertionPoint(loop.body):
            reduction_index, accumulator = loop.body.arguments
            input_value = tensor.ExtractOp(
                candidate.input, [row, reduction_index]
            ).result
            weight_value = tensor.ExtractOp(
                candidate.output_major_weight,
                [output_channel, reduction_index],
            ).result
            input_i32 = arith.ExtSIOp(i32, input_value).result
            weight_i32 = arith.ExtSIOp(i32, weight_value).result
            centered_input = arith.SubIOp(input_i32, input_zero_point).result
            centered_weight = arith.SubIOp(weight_i32, weight_zero_point).result
            product = arith.MulIOp(centered_input, centered_weight).result
            updated = arith.AddIOp(accumulator, product).result
            scf.YieldOp([updated])
        multiplier = tensor.ExtractOp(candidate.multiplier, [output_channel]).result
        shift = tensor.ExtractOp(candidate.shift, [output_channel]).result
        rounding = ir.Attribute.parse(
            "#tosa.rounding_mode<DOUBLE_ROUND>", context=candidate.matmul.context
        )
        scaled = tosa.ApplyScaleOp(
            i32, loop.results[0], multiplier, shift, rounding
        ).result
        shifted = arith.AddIOp(scaled, output_zero_point).result
        clamped_lower = arith.MaxSIOp(shifted, clamp_minimum).result
        clamped = arith.MinSIOp(clamped_lower, clamp_maximum).result
        result = arith.TruncIOp(i8, clamped).result
        linalg.YieldOp([result])

    replace_all_uses(candidate.rescale.results[0], generated.results[0])
    candidate.rescale.erase()
    if candidate.expand is not None:
        candidate.expand.erase()
    candidate.matmul.erase()
