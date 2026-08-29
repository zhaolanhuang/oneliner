"""Shared semantic validation for quantized NHWC convolution patterns."""

from __future__ import annotations

from iree.compiler import ir

from ..ir_utils import (
    body_operations,
    dense_ints,
    generic_io,
    owner_operation,
    require_constant_tensor,
    require_scalar,
    tensor_shape,
    validate_maps,
)


def convolution_geometry(
    candidate: ir.Operation,
    input_shape: tuple[int, int, int, int],
    kernel_shape: tuple[int, int, int, int],
    output_shape: tuple[int, ...],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Validates static stride/dilation geometry and returns both pairs."""
    try:
        strides = dense_ints(candidate.attributes["strides"])
        dilations = dense_ints(candidate.attributes["dilations"])
    except KeyError as error:
        raise ValueError("convolution requires explicit strides and dilations") from error
    if len(strides) != 2 or len(dilations) != 2:
        raise ValueError("convolution strides/dilations must have two dimensions")
    if any(value <= 0 for value in (*strides, *dilations)):
        raise ValueError("convolution strides/dilations must be positive")
    effective_h = dilations[0] * (kernel_shape[0] - 1) + 1
    effective_w = dilations[1] * (kernel_shape[1] - 1) + 1
    expected_h = (input_shape[1] - effective_h) // strides[0] + 1
    expected_w = (input_shape[2] - effective_w) // strides[1] + 1
    if expected_h <= 0 or expected_w <= 0:
        raise ValueError("convolution receptive field exceeds padded input")
    if tuple(output_shape[:3]) != (input_shape[0], expected_h, expected_w):
        raise ValueError("convolution output geometry is inconsistent")
    return (strides[0], strides[1]), (dilations[0], dilations[1])


def validate_quantized_padding(
    value: ir.Value, input_zero_point: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Proves tensor.pad uses the affine input zero-point on every boundary."""
    rank = len(tensor_shape(value, "i8"))
    pad = owner_operation(value)
    if pad is None or pad.name != "tensor.pad":
        return (0,) * rank, (0,) * rank
    static_low = pad.attributes.get("static_low")
    static_high = pad.attributes.get("static_high")
    if static_low is not None or static_high is not None:
        if static_low is None or static_high is None:
            raise ValueError("tensor.pad must provide both static low/high indices")
        low_values = dense_ints(static_low)
        high_values = dense_ints(static_high)
        if len(low_values) != rank or len(high_values) != rank:
            raise ValueError("tensor.pad static low/high rank does not match input")
        dynamic_size = ir.ShapedType.get_dynamic_size()
        segment_sizes = pad.attributes.get("operandSegmentSizes")
        if segment_sizes is not None:
            segments = dense_ints(segment_sizes)
            if len(segments) != 3 or segments[0] != 1:
                raise ValueError("tensor.pad has invalid operand segment sizes")
            if len(pad.operands) != sum(segments):
                raise ValueError("tensor.pad operands do not match operand segments")
            low_dynamic, high_dynamic = segments[1:]
        else:
            low_dynamic = sum(extent == dynamic_size for extent in low_values)
            high_dynamic = sum(extent == dynamic_size for extent in high_values)
            if len(pad.operands) != 1 + low_dynamic + high_dynamic:
                raise ValueError("tensor.pad operands do not match static low/high indices")
        dynamic_operands = list(pad.operands[1:])
        if len(dynamic_operands) != low_dynamic + high_dynamic:
            raise ValueError("tensor.pad dynamic operand count is inconsistent")
        dynamic_cursor = 0

        def materialize(extents: tuple[int, ...], count: int, label: str) -> tuple[int, ...]:
            nonlocal dynamic_cursor
            actual = []
            dynamic_seen = 0
            for extent in extents:
                if extent != dynamic_size:
                    actual.append(extent)
                    continue
                if dynamic_cursor >= len(dynamic_operands):
                    raise ValueError(f"tensor.pad {label} index is missing")
                # Older IREE textual forms print dynamic syntax while still
                # supplying scalar arith.constants.  Accept those constants,
                # but reject genuinely runtime-dependent extents.
                actual.append(require_scalar(dynamic_operands[dynamic_cursor], f"padding {label} index"))
                dynamic_cursor += 1
                dynamic_seen += 1
            if dynamic_seen != count:
                raise ValueError(f"tensor.pad {label} dynamic index count is inconsistent")
            return tuple(actual)

        low = materialize(low_values, low_dynamic, "low")
        high = materialize(high_values, high_dynamic, "high")
        if dynamic_cursor != len(dynamic_operands):
            raise ValueError("tensor.pad has unused dynamic padding operands")
    else:
        # Compatibility with older textual IREE forms that materialized every
        # static extent as a scalar operand.
        if len(pad.operands) != 1 + 2 * rank:
            raise ValueError("tensor.pad must provide static low/high indices")
        low = tuple(
            require_scalar(pad.operands[1 + index], "padding low index")
            for index in range(rank)
        )
        high = tuple(
            require_scalar(pad.operands[1 + rank + index], "padding high index")
            for index in range(rank)
        )
    block, operations = body_operations(pad, "quantized padding")
    if (
        len(block.arguments) != rank
        or len(operations) != 1
        or operations[0].name != "tensor.yield"
        or require_scalar(operations[0].operands[0], "padding value")
        != input_zero_point
    ):
        raise ValueError("padding value must equal the affine input zero point")
    if any(value < 0 for value in (*low, *high)):
        raise ValueError("padding extents cannot be negative")
    return low, high


def validate_weight_source(value: ir.Value, shape: tuple[int, ...]) -> None:
    """Requires a Flash-resident constant or a constant permutation thereof."""
    owner = owner_operation(value)
    if owner is None:
        raise ValueError("convolution weight must be statically defined")
    if owner.name == "arith.constant":
        require_constant_tensor(value, "i8", shape, "convolution weight")
        return
    if owner.name != "linalg.transpose" or not owner.operands:
        raise ValueError("convolution weight must be constant or constant transpose")
    source = owner.operands[0]
    source_owner = owner_operation(source)
    if source_owner is None or source_owner.name != "arith.constant":
        raise ValueError("transposed convolution weight must originate in a constant")
    tensor_shape(source, "i8", len(shape))


def validate_bias_initializer(
    candidate: ir.Operation,
    channels: int,
    output_rank: int,
    channel_dimension: int,
) -> ir.Value:
    """Proves a pure one-dimensional bias broadcast into an accumulator."""
    inputs, outputs = generic_io(candidate)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("bias initializer must have one input and one output")
    require_constant_tensor(inputs[0], "i32", (channels,), "bias")
    validate_maps(
        candidate,
        ((channel_dimension,), tuple(range(output_rank))),
        "bias initializer",
    )
    block, operations = body_operations(candidate, "bias initializer")
    if (
        len(block.arguments) != 2
        or len(operations) != 1
        or operations[0].name != "linalg.yield"
        or operations[0].operands[0] != block.arguments[0]
    ):
        raise ValueError("bias initializer has modified scalar semantics")
    return inputs[0]


def validate_depthwise_bias_add(
    candidate: ir.Operation, channels: int, source: ir.Value
) -> ir.Value:
    """Proves the canonical post-collapse ``bias + depthwise`` dataflow."""
    inputs, outputs = generic_io(candidate)
    if len(inputs) != 2 or len(outputs) != 1 or inputs[1] != source:
        raise ValueError("depthwise bias add must consume bias and collapsed result")
    require_constant_tensor(inputs[0], "i32", (channels,), "depthwise bias")
    validate_maps(
        candidate,
        ((3,), (0, 1, 2, 3), (0, 1, 2, 3)),
        "depthwise bias add",
    )
    block, operations = body_operations(candidate, "depthwise bias add")
    if (
        len(block.arguments) != 3
        or tuple(item.name for item in operations) != ("arith.addi", "linalg.yield")
        or list(operations[0].operands) != list(block.arguments[:2])
        or operations[1].operands[0] != operations[0].results[0]
    ):
        raise ValueError("depthwise bias add has modified scalar semantics")
    return inputs[0]
