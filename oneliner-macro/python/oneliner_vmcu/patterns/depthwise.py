"""Model-independent matcher for quantized NHWC/HWCM depthwise Conv2D."""

from __future__ import annotations

from iree.compiler import ir

from ..graph import FunctionGraph
from ..ir_utils import owner_operation, require_scalar, require_zero_point, tensor_shape, unique_user
from ..model import Analysis, DepthwiseConv2DMatch, RejectedCandidate
from ..quantization import AffineQuantization
from .convolution import (
    convolution_geometry,
    validate_depthwise_bias_add,
    validate_quantized_padding,
    validate_weight_source,
)
from .fully_connected import _validate_rescale


def _validate_zero_fill(value: ir.Value) -> ir.Operation:
    """Requires the depthwise accumulator initializer to be an i32 zero fill."""
    fill = owner_operation(value)
    if fill is None or fill.name != "linalg.fill" or not fill.operands:
        raise ValueError("depthwise accumulator must start from linalg.fill")
    if require_scalar(fill.operands[0], "depthwise accumulator fill") != 0:
        raise ValueError("depthwise accumulator fill must be zero before bias add")
    return fill


def _match(root: ir.Operation, key) -> DepthwiseConv2DMatch:
    """Proves depthwise→collapse→bias→requantize with multiplier one."""
    if len(root.operands) != 5 or len(root.results) != 1:
        raise ValueError("quantized depthwise conv must have five operands")
    input_shape = tensor_shape(root.operands[0], "i8", 4)
    weight_shape = tensor_shape(root.operands[1], "i8", 4)
    accumulator_shape = tensor_shape(root.results[0], "i32", 5)
    if weight_shape[2] != input_shape[3] or weight_shape[3] != 1:
        raise ValueError("depthwise fixed schedule requires channel multiplier 1")
    if any(dimension <= 0 for dimension in weight_shape[:2]):
        raise ValueError("depthwise kernel dimensions must be positive")
    expected_output = accumulator_shape[:3] + (accumulator_shape[3],)
    if accumulator_shape[3:] != (input_shape[3], 1):
        raise ValueError("depthwise accumulator channel dimensions disagree")
    validate_weight_source(root.operands[1], weight_shape)
    input_zp = require_zero_point(root.operands[2], "depthwise input zero point")
    weight_zp = require_zero_point(root.operands[3], "depthwise weight zero point")
    padding_low, padding_high = validate_quantized_padding(root.operands[0], input_zp)
    strides, dilations = convolution_geometry(
        root, input_shape, weight_shape, expected_output
    )
    accumulator_initializer = _validate_zero_fill(root.operands[4])
    collapse = unique_user(root.results[0], "tensor.collapse_shape", "depthwise result")
    collapsed_shape = tensor_shape(collapse.results[0], "i32", 4)
    if collapsed_shape != expected_output:
        raise ValueError("depthwise collapse must fold only channel multiplier")
    bias_add = unique_user(collapse.results[0], "linalg.generic", "collapsed depthwise")
    bias = validate_depthwise_bias_add(bias_add, expected_output[3], collapse.results[0])
    rescale = unique_user(bias_add.results[0], "linalg.generic", "biased depthwise")
    multiplier, shift, output_zp, output_shape = _validate_rescale(
        rescale, expected_output[3], bias_add.results[0]
    )
    if output_shape != expected_output:
        raise ValueError("depthwise requantization shape changed")
    return DepthwiseConv2DMatch(
        root=key,
        conv=root,
        accumulator_initializer=accumulator_initializer,
        collapse=collapse,
        bias_add=bias_add,
        rescale=rescale,
        input=root.operands[0],
        weight=root.operands[1],
        bias=bias,
        multiplier=multiplier,
        shift=shift,
        input_quantization=AffineQuantization("i8", None, input_zp, None),
        weight_quantization=AffineQuantization("i8", None, weight_zp, None),
        output_quantization=AffineQuantization("i8", None, output_zp, None),
        input_shape=input_shape,
        weight_shape=weight_shape,
        output_shape=output_shape,
        strides=strides,
        dilations=dilations,
        padding_low=padding_low,
        padding_high=padding_high,
    )


def analyze_depthwise(
    graphs: list[FunctionGraph], occupied: set[ir.Operation]
) -> Analysis:
    """Collects deterministic depthwise matches and rejection reasons."""
    matches = []
    rejected = []
    for graph in graphs:
        for node in graph.nodes:
            if node.operation.name != "linalg.depthwise_conv_2d_nhwc_hwcm_q":
                continue
            try:
                candidate = _match(node.operation, node.key)
                if occupied.intersection(candidate.claimed_operations):
                    # An accepted IBN claims this standalone depthwise root.
                    # Selection overlap is expected and is not a rejection.
                    continue
                occupied.update(candidate.claimed_operations)
                matches.append(candidate)
            except (ValueError, IndexError, KeyError) as error:
                rejected.append(
                    RejectedCandidate(
                        node.key,
                        DepthwiseConv2DMatch.kind,
                        str(error),
                        str(node.operation.location),
                    )
                )
    return Analysis(matches, rejected)
