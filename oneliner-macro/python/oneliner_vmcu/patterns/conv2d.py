"""Model-independent matcher for quantized NHWC/HWCF Conv2D."""

from __future__ import annotations

from iree.compiler import ir

from ..graph import FunctionGraph
from ..ir_utils import owner_operation, require_zero_point, tensor_shape, unique_user
from ..model import Analysis, Conv2DMatch, RejectedCandidate
from ..quantization import AffineQuantization
from .convolution import (
    convolution_geometry,
    validate_bias_initializer,
    validate_quantized_padding,
    validate_weight_source,
)
from .fully_connected import _validate_rescale


def _match(root: ir.Operation, key) -> Conv2DMatch:
    """Proves one complete quantized Conv2D→requantization subgraph."""
    quantized_op = root.name == "linalg.conv_2d_nhwc_hwcf_q"
    if root.name not in (
        "linalg.conv_2d_nhwc_hwcf_q",
        "linalg.conv_2d_nhwc_hwcf",
    ) or len(root.operands) != (5 if quantized_op else 3) or len(root.results) != 1:
        raise ValueError("int8 conv2d has an unexpected operand/result signature")
    input_shape = tensor_shape(root.operands[0], "i8", 4)
    weight_shape = tensor_shape(root.operands[1], "i8", 4)
    output_shape = tensor_shape(root.results[0], "i32", 4)
    if weight_shape[2] != input_shape[3] or output_shape[3] != weight_shape[3]:
        raise ValueError("conv2d input/filter/output channel dimensions disagree")
    if any(dimension <= 0 for dimension in weight_shape[:2]):
        raise ValueError("conv2d kernel dimensions must be positive")
    validate_weight_source(root.operands[1], weight_shape)
    input_zp = (
        require_zero_point(root.operands[2], "conv2d input zero point")
        if quantized_op
        else 0
    )
    weight_zp = (
        require_zero_point(root.operands[3], "conv2d weight zero point")
        if quantized_op
        else 0
    )
    padding_low, padding_high = validate_quantized_padding(root.operands[0], input_zp)
    strides, dilations = convolution_geometry(root, input_shape, weight_shape, output_shape)
    bias_initializer = owner_operation(root.operands[4 if quantized_op else 2])
    if bias_initializer is None or bias_initializer.name != "linalg.generic":
        raise ValueError("conv2d accumulator must use a bias broadcast")
    bias = validate_bias_initializer(
        bias_initializer, output_shape[3], output_rank=4, channel_dimension=3
    )
    if unique_user(
        bias_initializer.results[0], root.name, "conv2d bias initializer"
    ) != root:
        raise ValueError("conv2d bias initializer must exclusively feed convolution")
    rescale = unique_user(root.results[0], "linalg.generic", "conv2d accumulator")
    multiplier, shift, output_zp, requant_shape = _validate_rescale(
        rescale, output_shape[3], root.results[0]
    )
    if requant_shape != output_shape:
        raise ValueError("conv2d requantization shape changed")
    return Conv2DMatch(
        root=key,
        conv=root,
        bias_initializer=bias_initializer,
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


def analyze_conv2d(
    graphs: list[FunctionGraph], occupied: set[ir.Operation]
) -> Analysis:
    """Collects deterministic Conv2D matches and capability rejections."""
    matches = []
    rejected = []
    for graph in graphs:
        for node in graph.nodes:
            if node.operation.name not in (
                "linalg.conv_2d_nhwc_hwcf_q",
                "linalg.conv_2d_nhwc_hwcf",
            ):
                continue
            try:
                candidate = _match(node.operation, node.key)
                if occupied.intersection(candidate.claimed_operations):
                    # The fixed IBN matcher runs first and owns all three layer
                    # boundaries as one transaction. Do not emit overlap noise.
                    continue
                occupied.update(candidate.claimed_operations)
                matches.append(candidate)
            except (ValueError, IndexError, KeyError) as error:
                rejected.append(
                    RejectedCandidate(
                        node.key, Conv2DMatch.kind, str(error), str(node.operation.location)
                    )
                )
    return Analysis(matches, rejected)
