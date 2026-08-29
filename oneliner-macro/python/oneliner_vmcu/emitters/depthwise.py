"""Standard-MLIR emitter for multiplier-one quantized depthwise Conv2D."""

from __future__ import annotations

from iree.compiler import ir
from iree.compiler.dialects import arith, linalg, tensor

from ..ir_utils import erase_dead_operation, owner_operation, replace_all_uses
from ..model import DepthwiseConv2DMatch, PatternMatch
from .common import constant
from .convolution import (
    create_output_generic,
    emit_requantize_i8,
    index_multiply_add,
    nested_reduction,
)


def emit_depthwise(match: PatternMatch) -> None:
    """Computes each NHWC output once and eliminates rank-5 i32 materialization."""
    if not isinstance(match, DepthwiseConv2DMatch):
        raise TypeError(f"depthwise emitter received {type(match).__name__}")
    candidate = match
    generated = create_output_generic(candidate.rescale, candidate.output_shape)
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    body = ir.Block.create_at_start(generated.regions[0], [i8])
    with ir.InsertionPoint(body):
        n, oh, ow, channel = [linalg.IndexOp(index).result for index in range(4)]
        bias = tensor.ExtractOp(candidate.bias, [channel]).result
        input_zp = constant(i32, candidate.input_quantization.zero_point_at())
        weight_zp = constant(i32, candidate.weight_quantization.zero_point_at())
        multiplier_index = constant(ir.IndexType.get(), 0)

        def product(indices, accumulator):
            """Accumulates one centered spatial depthwise product."""
            kh, kw = indices
            ih = index_multiply_add(oh, candidate.strides[0], kh, candidate.dilations[0])
            iw = index_multiply_add(ow, candidate.strides[1], kw, candidate.dilations[1])
            input_value = tensor.ExtractOp(candidate.input, [n, ih, iw, channel]).result
            weight_value = tensor.ExtractOp(
                candidate.weight, [kh, kw, channel, multiplier_index]
            ).result
            lhs = arith.SubIOp(arith.ExtSIOp(i32, input_value).result, input_zp).result
            rhs = arith.SubIOp(arith.ExtSIOp(i32, weight_value).result, weight_zp).result
            return arith.AddIOp(accumulator, arith.MulIOp(lhs, rhs).result).result

        accumulator = nested_reduction(candidate.weight_shape[:2], bias, product)
        result = emit_requantize_i8(
            accumulator,
            channel,
            candidate.multiplier,
            candidate.shift,
            candidate.output_quantization.zero_point_at(),
            candidate.conv.context,
        )
        linalg.YieldOp([result])
    replace_all_uses(candidate.rescale.results[0], generated.results[0])
    bias_empty = owner_operation(candidate.bias_add.operands[-1])
    accumulator_empty = owner_operation(candidate.accumulator_initializer.operands[-1])
    candidate.rescale.erase()
    candidate.bias_add.erase()
    candidate.collapse.erase()
    candidate.conv.erase()
    erase_dead_operation(candidate.accumulator_initializer)
    if bias_empty is not None and bias_empty.name == "tensor.empty":
        erase_dead_operation(bias_empty)
    if accumulator_empty is not None and accumulator_empty.name == "tensor.empty":
        erase_dead_operation(accumulator_empty)
