#!/usr/bin/env python3
"""Fuses vMCU-compatible subgraphs into segment-buffer ukernels.

Matches every subgraph in any straight-line model that satisfies the vMCU
patterns (inverted bottleneck, pointwise pair, single 2D convolution, fully
connected) and leaves all other operations to IREE codegen.
"""

from __future__ import annotations

import argparse
import json
import sys
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path

from iree.compiler import ir
from iree.compiler.dialects import arith, tensor


DEPTHWISE = "linalg.depthwise_conv_2d_nhwc_hwcm_q"
CONV = "linalg.conv_2d_nhwc_hwcf_q"
MATMUL = "linalg.quantized_matmul"
UKERNEL_NAME = "oneliner_vmcu_ibn_s8"
BITCODE_PATH = "oneliner_vmcu_mcunet.bc"
CONV2D_KERNEL_NAME = "oneliner_vmcu_conv2d_s8"
FC_KERNEL_NAME = "oneliner_vmcu_fc_s8"
PAIR_KERNEL_NAME = "oneliner_vmcu_pointwise_pair_s8"
GENERIC_BITCODE_PATH = "oneliner_vmcu_generic.bc"
CONFIG_VERSION = 1
CONFIG_MAGIC = 0x564D4355
GENERIC_KERNEL_NAMES = (UKERNEL_NAME, CONV2D_KERNEL_NAME, FC_KERNEL_NAME, PAIR_KERNEL_NAME)


@dataclass(frozen=True)
class McunetPlan:
    mode: str
    block_count: int
    residual_block_count: int
    in_place_block_count: int
    standard_peak_intermediate_bytes: int
    max_segment_bytes: int
    total_segment_bytes: int
    full_intermediate_bytes: int
    segment_bytes: int
    saved_intermediate_bytes: int


@dataclass
class Stage:
    contraction: ir.Operation
    collapse_input: ir.Operation
    transpose: ir.Operation
    weight: ir.Value
    bias_init: ir.Operation
    bias: ir.Value
    expand: ir.Operation
    rescale: ir.Operation
    multiplier: ir.Value
    shift: ir.Value
    input_zp: int
    output_zp: int
    operations: set[ir.Operation]
    rank: int = 4


@dataclass
class Residual:
    operations: set[ir.Operation]
    output: ir.Value
    final_zp: int
    config: tuple[int, ...]


@dataclass
class InvertedBottleneck:
    number: int
    input: ir.Value
    output: ir.Value
    expansion: Stage
    depthwise: ir.Operation
    depthwise_weight: ir.Value
    depthwise_bias: ir.Value
    depthwise_multiplier: ir.Value
    depthwise_shift: ir.Value
    depthwise_zp: int
    projection: Stage
    residual: Residual | None
    geometry: tuple[int, ...]
    scratch_bytes: int
    erase_operations: tuple[ir.Operation, ...]
    semantic_operations: set[ir.Operation]


@dataclass
class ConvStage:
    conv: ir.Operation
    pad: ir.Operation | None
    transpose: ir.Operation
    weight: ir.Value
    bias_init: ir.Operation
    bias: ir.Value
    rescale: ir.Operation
    multiplier: ir.Value
    shift: ir.Value
    input_zp: int
    output_zp: int
    geometry: tuple[int, ...]
    scratch_bytes: int
    operations: set[ir.Operation]


@dataclass
class GenericModule:
    kind: str
    input: ir.Value
    output: ir.Value
    kernel_name: str
    config: tuple[int, ...]
    input_operands: tuple[ir.Value, ...]
    output_type: ir.Type
    scratch_bytes: int
    erase_operations: tuple[ir.Operation, ...]
    standard_bytes: int
    residual: bool = False
    variant: str | None = None


@dataclass
class McunetMatch:
    context: ir.Context
    module: ir.Module
    blocks: list[InvertedBottleneck]
    plan: McunetPlan
    modules: list[GenericModule] = field(default_factory=list)


def operation(value) -> ir.Operation:
    return value.operation if hasattr(value, "operation") else value


def owner_operation(value: ir.Value) -> ir.Operation | None:
    if not isinstance(value, ir.OpResult):
        return None
    return operation(value.owner)


def operations_named(root, name: str) -> list[ir.Operation]:
    found: list[ir.Operation] = []

    def collect(candidate: ir.Operation) -> ir.WalkResult:
        if candidate.name == name:
            found.append(candidate)
        return ir.WalkResult.ADVANCE

    root.operation.walk(collect)
    return found


def verify(candidate: ir.Operation, label: str) -> None:
    try:
        candidate.verify()
    except ir.MLIRError as error:
        raise ValueError(f"{label} failed verification: {error}") from error


def tensor_shape(value: ir.Value, element_type: str, rank: int) -> tuple[int, ...]:
    try:
        value_type = ir.RankedTensorType(value.type)
    except ValueError as error:
        raise ValueError(f"expected a rank-{rank} {element_type} tensor, got {value.type}") from error
    shape = tuple(value_type.shape)
    if value_type.rank != rank or str(value_type.element_type) != element_type or any(d < 0 for d in shape):
        raise ValueError(f"expected a static rank-{rank} {element_type} tensor, got {value.type}")
    return shape


def scalar_integer(value: ir.Value) -> int | None:
    owner = owner_operation(value)
    if owner is None or owner.name != "arith.constant" or "value" not in owner.attributes:
        return None
    try:
        return ir.IntegerAttr(owner.attributes["value"]).value
    except ValueError:
        return None


def require_scalar(value: ir.Value, label: str) -> int:
    result = scalar_integer(value)
    if result is None:
        raise ValueError(f"{label} must be a scalar arith.constant")
    return result


def require_shift(value: ir.Value, label: str) -> int:
    result = require_scalar(value, label)
    if result < 1 or result > 62:
        raise ValueError(f"{label} must be in [1, 62]")
    return result


def require_zero_point(value: ir.Value, label: str) -> int:
    result = require_scalar(value, label)
    if result < -128 or result > 127:
        raise ValueError(f"{label} must be in [-128, 127]")
    return result


def pad_low_high(pad: ir.Operation, label: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dynamic = -(1 << 63)
    if "static_low" in pad.attributes and "static_high" in pad.attributes:
        low = tuple(int(item) for item in pad.attributes["static_low"])
        high = tuple(int(item) for item in pad.attributes["static_high"])
        if all(item != dynamic for item in low) and all(item != dynamic for item in high):
            if len(low) != 4 or len(high) != 4:
                raise ValueError(f"{label} padding must be rank-4")
            return low, high
    low = tuple(require_scalar(value, f"{label} low padding") for value in pad.operands[1:5])
    high = tuple(require_scalar(value, f"{label} high padding") for value in pad.operands[5:9])
    if len(low) != 4 or len(high) != 4:
        raise ValueError(f"{label} padding must be rank-4")
    return low, high


def require_constant_tensor(value: ir.Value, element_type: str, shape: tuple[int, ...], label: str) -> None:
    owner = owner_operation(value)
    if owner is None or owner.name != "arith.constant" or tensor_shape(value, element_type, len(shape)) != shape:
        raise ValueError(f"{label} must be an arith.constant tensor<{ 'x'.join(map(str, shape)) }x{element_type}>")


def dense_ints(attribute: ir.Attribute) -> tuple[int, ...]:
    return tuple(int(item) for item in attribute)


def direct_operations(function: ir.Operation) -> list[ir.Operation]:
    if len(function.regions) != 1 or len(function.regions[0].blocks) != 1:
        raise ValueError("MCUNet rewrite requires one straight-line function block")
    return [operation(item) for item in function.regions[0].blocks[0].operations]


def unique_user(value: ir.Value, expected_name: str, label: str) -> ir.Operation:
    users = [operation(use.owner) for use in value.uses]
    matches = [candidate for candidate in users if candidate.name == expected_name]
    if len(users) != 1 or len(matches) != 1:
        raise ValueError(f"{label} must have exactly one {expected_name} user")
    return matches[0]


def generic_io(candidate: ir.Operation) -> tuple[list[ir.Value], list[ir.Value]]:
    if candidate.name != "linalg.generic":
        raise ValueError(f"expected linalg.generic, got {candidate.name}")
    return list(candidate.opview.inputs), list(candidate.opview.outputs)


def body(candidate: ir.Operation, expected_names: tuple[str, ...], label: str):
    if len(candidate.regions) != 1 or len(candidate.regions[0].blocks) != 1:
        raise ValueError(f"{label} must have one scalar region")
    block = candidate.regions[0].blocks[0]
    operations = [operation(item) for item in block.operations]
    if tuple(item.name for item in operations) != expected_names:
        raise ValueError(f"{label} has modified scalar semantics")
    return block, operations


def expected_map(context: ir.Context, dimensions: int, results: tuple[int, ...]) -> ir.AffineMapAttr:
    return ir.AffineMapAttr.get(
        ir.AffineMap.get(
            dimensions,
            0,
            [ir.AffineDimExpr.get(index, context=context) for index in results],
            context=context,
        )
    )


def validate_maps(candidate: ir.Operation, results: tuple[tuple[int, ...], ...], label: str) -> None:
    maps = list(candidate.opview.indexing_maps)
    expected = [expected_map(candidate.context, len(candidate.opview.iterator_types), item) for item in results]
    if maps != expected:
        raise ValueError(f"{label} has modified indexing maps")
    parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>", context=candidate.context)
    if any(item != parallel for item in candidate.opview.iterator_types):
        raise ValueError(f"{label} must use parallel iterators")


def validate_bias_init(candidate: ir.Operation, channels: int, label: str, rank: int | None = None) -> ir.Value:
    inputs, outputs = generic_io(candidate)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"{label} bias initializer has modified operands")
    require_constant_tensor(inputs[0], "i32", (channels,), f"{label} bias")
    dimensions = len(candidate.opview.iterator_types)
    if dimensions == 2:
        results = ((1,), (0, 1))
    elif dimensions == 4:
        results = ((3,), (0, 1, 2, 3))
    else:
        raise ValueError(f"{label} bias initializer has unexpected rank {dimensions}")
    validate_maps(candidate, results, f"{label} bias initializer")
    scalar_block, operations = body(candidate, ("linalg.yield",), f"{label} bias initializer")
    if len(scalar_block.arguments) != 2 or list(operations[0].operands) != [scalar_block.arguments[0]]:
        raise ValueError(f"{label} bias initializer has modified scalar semantics")
    return inputs[0]


def validate_rescale(candidate: ir.Operation, channels: int, label: str, rank: int = 4) -> tuple[ir.Value, ir.Value, int]:
    inputs, outputs = generic_io(candidate)
    if len(outputs) != 1:
        raise ValueError(f"{label} rescale has modified operands")
    scalar_block = candidate.regions[0].blocks[0]
    names = tuple(operation(item).name for item in scalar_block.operations)
    folded = (
        "arith.extsi", "arith.muli", "arith.addi", "arith.cmpi",
        "arith.select", "arith.addi", "arith.shrsi", "arith.trunci",
        "arith.maxsi", "arith.minsi", "arith.trunci", "linalg.yield",
    )
    folded_with_zp = (
        "arith.extsi", "arith.muli", "arith.addi", "arith.cmpi",
        "arith.select", "arith.addi", "arith.shrsi", "arith.trunci",
        "arith.addi", "arith.maxsi", "arith.minsi", "arith.trunci", "linalg.yield",
    )
    if len(inputs) == 1 and names in (folded, folded_with_zp):
        return validate_folded_rescale(candidate, channels, label, names == folded_with_zp)
    if len(inputs) != 3:
        raise ValueError(f"{label} rescale has modified operands")
    multiplier, shift = inputs[1], inputs[2]
    require_constant_tensor(multiplier, "i32", (channels,), f"{label} multiplier")
    require_constant_tensor(shift, "i8", (channels,), f"{label} shift")
    multiplier_values = dense_ints(owner_operation(multiplier).attributes["value"])
    shift_values = dense_ints(owner_operation(shift).attributes["value"])
    for value in multiplier_values:
        if value < 0 or value >= (1 << 31):
            raise ValueError(f"{label} multiplier must be in [0, 2^31)")
    for value in shift_values:
        if value < 1 or value > 62:
            raise ValueError(f"{label} shift must be in [1, 62]")
    dimensions = len(candidate.opview.iterator_types)
    if dimensions == 2:
        results = ((0, 1), (1,), (1,), (0, 1))
    elif dimensions == 4:
        results = ((0, 1, 2, 3), (3,), (3,), (0, 1, 2, 3))
    else:
        raise ValueError(f"{label} rescale has unexpected rank {dimensions}")
    validate_maps(candidate, results, f"{label} rescale")
    names = tuple(operation(item).name for item in scalar_block.operations)
    expected_without_zp = ("tosa.apply_scale", "arith.maxsi", "arith.minsi", "arith.trunci", "linalg.yield")
    expected_with_zp = ("tosa.apply_scale", "arith.addi", "arith.maxsi", "arith.minsi", "arith.trunci", "linalg.yield")
    arith_without_zp = (
        "arith.extui", "arith.extsi", "arith.extsi", "arith.muli",
        "arith.extui", "arith.shli", "arith.shrui", "arith.addi",
        "arith.cmpi", "arith.select", "arith.addi", "arith.cmpi",
        "arith.select", "arith.shrsi", "arith.trunci",
        "arith.maxsi", "arith.minsi", "arith.trunci", "linalg.yield",
    )
    arith_with_zp = (
        "arith.extui", "arith.extsi", "arith.extsi", "arith.muli",
        "arith.extui", "arith.shli", "arith.shrui", "arith.addi",
        "arith.cmpi", "arith.select", "arith.addi", "arith.cmpi",
        "arith.select", "arith.shrsi", "arith.trunci",
        "arith.addi", "arith.maxsi", "arith.minsi", "arith.trunci", "linalg.yield",
    )
    if names not in (expected_without_zp, expected_with_zp, arith_without_zp, arith_with_zp):
        raise ValueError(f"{label} rescale has modified scalar semantics")
    operations = [operation(item) for item in scalar_block.operations]
    if names in (arith_without_zp, arith_with_zp):
        output_zp = validate_arith_rescale(operations, scalar_block, label, names == arith_with_zp)
        return multiplier, shift, output_zp
    scale = operations[0]
    double_round = ir.Attribute.parse("#tosa.rounding_mode<DOUBLE_ROUND>", context=candidate.context)
    if list(scale.operands) != list(scalar_block.arguments[:3]) or scale.attributes.get("rounding_mode") != double_round:
        raise ValueError(f"{label} rescale must use DOUBLE_ROUND per-channel scaling")
    cursor = scale.results[0]
    output_zp = 0
    offset = 1
    if names == expected_with_zp:
        addition = operations[1]
        if addition.operands[0] != cursor:
            raise ValueError(f"{label} rescale has modified scalar dataflow")
        output_zp = require_zero_point(addition.operands[1], f"{label} output zero point")
        cursor = addition.results[0]
        offset = 2
    lower, upper, truncation, yield_operation = operations[offset:]
    if (
        lower.operands[0] != cursor
        or require_scalar(lower.operands[1], f"{label} clamp minimum") != -128
        or upper.operands[0] != lower.results[0]
        or require_scalar(upper.operands[1], f"{label} clamp maximum") != 127
        or truncation.operands[0] != upper.results[0]
        or yield_operation.operands[0] != truncation.results[0]
    ):
        raise ValueError(f"{label} rescale is not the canonical int8 clamp")
    return multiplier, shift, output_zp


def validate_folded_rescale(candidate: ir.Operation, channels: int, label: str, with_zp: bool) -> tuple[ir.Value, ir.Value, int]:
    """Validates a constant-folded DOUBLE_ROUND apply_scale.

    Per-tensor quantized models fold the uniform multiplier and shift into the
    scale body; the surrounding structure is exactly the sign-dependent
    DOUBLE_ROUND semantics the C kernel implements. Broadcast per-channel
    constant tensors are created so the ukernel ABI stays uniform.
    """
    inputs, outputs = generic_io(candidate)
    if len(inputs) != 1 or len(outputs) != 1 or inputs[0] not in list(candidate.opview.inputs):
        raise ValueError(f"{label} folded rescale has modified operands")
    scalar_block = candidate.regions[0].blocks[0]
    operations = [operation(item) for item in scalar_block.operations]
    ext_value, multiply, base_add, cmp_pos, sel_dir, dir_add, shift_right, trunc_scale = operations[:8]
    value_arg = scalar_block.arguments[0]
    if (
        ext_value.operands[0] != value_arg
        or multiply.operands[0] != ext_value.results[0]
        or base_add.operands[0] != multiply.results[0]
        or cmp_pos.operands[0] != value_arg
        or int(ir.IntegerAttr(cmp_pos.attributes["predicate"]).value) != 5
        or require_scalar(cmp_pos.operands[1], f"{label} zero") != 0
        or sel_dir.operands[0] != cmp_pos.results[0]
        or require_scalar(sel_dir.operands[1], f"{label} round up") != 1073741824
        or require_scalar(sel_dir.operands[2], f"{label} round down") != -1073741824
        or dir_add.operands[0] != sel_dir.results[0]
        or dir_add.operands[1] != base_add.results[0]
        or shift_right.operands[0] != dir_add.results[0]
        or trunc_scale.operands[0] != shift_right.results[0]
    ):
        raise ValueError(f"{label} folded rescale has modified scalar dataflow")
    multiplier_value = require_scalar(multiply.operands[1], f"{label} folded multiplier")
    shift_value = require_scalar(shift_right.operands[1], f"{label} folded shift")
    if multiplier_value < 0 or multiplier_value >= (1 << 31) or shift_value < 1 or shift_value > 62:
        raise ValueError(f"{label} folded scale is out of range")
    if require_scalar(base_add.operands[1], f"{label} folded rounding") != (1 << (shift_value - 1)):
        raise ValueError(f"{label} folded rounding does not match the shift")
    cursor = trunc_scale.results[0]
    offset = 8
    output_zp = 0
    if with_zp:
        addition = operations[8]
        if addition.operands[0] != cursor:
            raise ValueError(f"{label} folded rescale has modified scalar dataflow")
        output_zp = require_zero_point(addition.operands[1], f"{label} output zero point")
        cursor = addition.results[0]
        offset = 9
    lower, upper, truncation, yield_operation = operations[offset:]
    if (
        lower.operands[0] != cursor
        or require_scalar(lower.operands[1], f"{label} clamp minimum") != -128
        or upper.operands[0] != lower.results[0]
        or require_scalar(upper.operands[1], f"{label} clamp maximum") != 127
        or truncation.operands[0] != upper.results[0]
        or yield_operation.operands[0] != truncation.results[0]
    ):
        raise ValueError(f"{label} folded rescale is not the canonical int8 clamp")
    with candidate.context, ir.Location.unknown():
        with ir.InsertionPoint(candidate):
            i32 = ir.IntegerType.get_signless(32)
            i8 = ir.IntegerType.get_signless(8)
            multiplier = arith.ConstantOp(
                ir.RankedTensorType.get([channels], i32),
                ir.DenseIntElementsAttr.get(
                    array("i", [multiplier_value] * channels),
                    type=ir.RankedTensorType.get([channels], i32),
                ),
            ).result
            shift = arith.ConstantOp(
                ir.RankedTensorType.get([channels], i8),
                ir.DenseIntElementsAttr.get(
                    array("b", [shift_value] * channels),
                    type=ir.RankedTensorType.get([channels], i8),
                ),
            ).result
    return multiplier, shift, output_zp


def validate_arith_rescale(operations: list[ir.Operation], scalar_block, label: str, with_zp: bool) -> int:
    """Validates the ApplyScaleGenericOpConverter expanded form.

    IREE lowers tosa.apply_scale with DOUBLE_ROUND into this exact 64-bit
    arith sequence; the C kernel implements the same semantics. The sequence
    (indices 0..14) is the canonical scale; optional index 15 adds the output
    zero point; the trailing clamp/truncate/yield complete the int8 requant.
    """
    extui_shift, ext_value, ext_mult, multiply, ext_shift, shl, shr, base_add, cmp_pos, sel_dir, dir_add, cmp_shift, sel_round, shift_right, trunc_scale = operations[:15]
    value_arg, multiplier_arg, shift_arg = scalar_block.arguments[:3]
    if (
        extui_shift.operands[0] != shift_arg
        or ext_value.operands[0] != value_arg
        or ext_mult.operands[0] != multiplier_arg
        or multiply.operands[0] != ext_value.results[0]
        or multiply.operands[1] != ext_mult.results[0]
        or ext_shift.operands[0] != shift_arg
        or require_scalar(shl.operands[0], f"{label} scale base") != 1
        or shl.operands[1] != ext_shift.results[0]
        or shr.operands[0] != shl.results[0]
        or require_scalar(shr.operands[1], f"{label} scale halving") != 1
        or base_add.operands[0] != multiply.results[0]
        or base_add.operands[1] != shr.results[0]
        or cmp_pos.operands[0] != value_arg
        or int(ir.IntegerAttr(cmp_pos.attributes["predicate"]).value) != 5
        or require_scalar(cmp_pos.operands[1], f"{label} zero") != 0
    ):
        raise ValueError(f"{label} rescale has modified scalar dataflow")
    if (
        require_scalar(sel_dir.operands[1], f"{label} round up") != 1073741824
        or require_scalar(sel_dir.operands[2], f"{label} round down") != -1073741824
        or sel_dir.operands[0] != cmp_pos.results[0]
        or dir_add.operands[0] != sel_dir.results[0]
        or dir_add.operands[1] != base_add.results[0]
    ):
        raise ValueError(f"{label} rescale must use DOUBLE_ROUND rounding")
    if (
        int(ir.IntegerAttr(cmp_shift.attributes["predicate"]).value) != 4
        or require_scalar(cmp_shift.operands[1], f"{label} shift bound") != 31
    ):
        raise ValueError(f"{label} rescale has modified rounding semantics")
    if (
        sel_round.operands[0] != cmp_shift.results[0]
        or sel_round.operands[1] != dir_add.results[0]
        or sel_round.operands[2] != base_add.results[0]
        or shift_right.operands[0] != sel_round.results[0]
        or shift_right.operands[1] != ext_shift.results[0]
        or trunc_scale.operands[0] != shift_right.results[0]
    ):
        raise ValueError(f"{label} rescale has modified scalar dataflow")
    cursor = trunc_scale.results[0]
    offset = 15
    output_zp = 0
    if with_zp:
        addition = operations[15]
        if addition.operands[0] != cursor:
            raise ValueError(f"{label} rescale has modified scalar dataflow")
        output_zp = require_zero_point(addition.operands[1], f"{label} output zero point")
        cursor = addition.results[0]
        offset = 16
    lower, upper, truncation, yield_operation = operations[offset:]
    if (
        lower.operands[0] != cursor
        or require_scalar(lower.operands[1], f"{label} clamp minimum") != -128
        or upper.operands[0] != lower.results[0]
        or require_scalar(upper.operands[1], f"{label} clamp maximum") != 127
        or truncation.operands[0] != upper.results[0]
        or yield_operation.operands[0] != truncation.results[0]
    ):
        raise ValueError(f"{label} rescale is not the canonical int8 clamp")
    return output_zp


def match_pointwise(contraction: ir.Operation, label: str, rank: int = 4) -> Stage:
    if len(contraction.operands) != 5 or len(contraction.results) != 1:
        raise ValueError(f"{label} matmul has modified operands")
    collapse = owner_operation(contraction.operands[0])
    transpose = owner_operation(contraction.operands[1])
    bias_init = owner_operation(contraction.operands[4])
    if rank == 4:
        if collapse is None or collapse.name != "tensor.collapse_shape":
            raise ValueError(f"{label} input must be a canonical collapse_shape")
    else:
        tensor_shape(contraction.operands[0], "i8", 2)
        collapse = contraction
    if transpose is None or transpose.name != "linalg.transpose" or dense_ints(transpose.attributes["permutation"]) != (1, 0):
        raise ValueError(f"{label} weight must use a [1, 0] linalg.transpose")
    if bias_init is None or bias_init.name != "linalg.generic":
        raise ValueError(f"{label} accumulator must be initialized from its bias vector")
    matrix_shape = tensor_shape(contraction.operands[0], "i8", 2)
    transposed_shape = tensor_shape(contraction.operands[1], "i8", 2)
    accumulator_shape = tensor_shape(contraction.results[0], "i32", 2)
    rows, input_channels = matrix_shape
    if transposed_shape[0] != input_channels or accumulator_shape != (rows, transposed_shape[1]):
        raise ValueError(f"{label} tensor dimensions are inconsistent")
    channels = transposed_shape[1]
    weight = transpose.operands[0]
    require_constant_tensor(weight, "i8", (channels, input_channels), f"{label} output-major weight")
    if scalar_integer(contraction.operands[3]) != 0:
        raise ValueError(f"{label} weight zero point must be zero")
    bias = validate_bias_init(bias_init, channels, label, rank)
    if rank == 4:
        expand = unique_user(contraction.results[0], "tensor.expand_shape", label)
        rescale = unique_user(expand.results[0], "linalg.generic", label)
    else:
        expand = contraction
        rescale = unique_user(contraction.results[0], "linalg.generic", label)
    multiplier, shift, output_zp = validate_rescale(rescale, channels, label, rank)
    return Stage(
        contraction=contraction,
        collapse_input=collapse,
        transpose=transpose,
        weight=weight,
        bias_init=bias_init,
        bias=bias,
        expand=expand,
        rescale=rescale,
        multiplier=multiplier,
        shift=shift,
        input_zp=require_zero_point(contraction.operands[2], f"{label} input zero point"),
        output_zp=output_zp,
        operations={collapse, transpose, bias_init, contraction, expand, rescale},
        rank=rank,
    )


def validate_fill(candidate: ir.Operation, label: str) -> None:
    if candidate.name != "linalg.fill" or len(candidate.operands) != 2 or require_scalar(candidate.operands[0], label) != 0:
        raise ValueError(f"{label} must be initialized by a zero linalg.fill")


def validate_depthwise_bias(candidate: ir.Operation, channels: int) -> ir.Value:
    inputs, outputs = generic_io(candidate)
    if len(inputs) != 2 or len(outputs) != 1:
        raise ValueError("depthwise bias add has modified operands")
    require_constant_tensor(inputs[0], "i32", (channels,), "depthwise bias")
    validate_maps(candidate, ((3,), (0, 1, 2, 3), (0, 1, 2, 3)), "depthwise bias add")
    scalar_block, operations = body(candidate, ("arith.addi", "linalg.yield"), "depthwise bias add")
    addition, yield_operation = operations
    if list(addition.operands) != list(scalar_block.arguments[:2]) or yield_operation.operands[0] != addition.results[0]:
        raise ValueError("depthwise bias add has modified scalar semantics")
    return inputs[0]


def parse_scale_pair(candidate: ir.Operation, rounding: str, label: str) -> tuple[int, int]:
    inputs, outputs = generic_io(candidate)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"{label} has modified operands")
    validate_maps(candidate, ((0, 1, 2, 3), (0, 1, 2, 3)), label)
    scalar_block, operations = body(candidate, ("tosa.apply_scale", "linalg.yield"), label)
    scale, yield_operation = operations
    expected = ir.Attribute.parse(f"#tosa.rounding_mode<{rounding}>", context=candidate.context)
    if (
        len(scalar_block.arguments) != 2
        or scale.operands[0] != scalar_block.arguments[0]
        or scale.attributes.get("rounding_mode") != expected
        or yield_operation.operands[0] != scale.results[0]
    ):
        raise ValueError(f"{label} has modified scalar semantics")
    return require_scalar(scale.operands[1], f"{label} multiplier"), require_shift(scale.operands[2], f"{label} shift")


def parse_single_branch(candidate: ir.Operation, source: ir.Value, zero_point: int, label: str) -> tuple[int, int]:
    inputs, outputs = generic_io(candidate)
    if len(inputs) != 1 or len(outputs) != 1 or inputs[0] != source:
        raise ValueError(f"{label} has modified operands")
    validate_maps(candidate, ((0, 1, 2, 3), (0, 1, 2, 3)), label)
    scalar_block, operations = body(candidate, ("arith.extsi", "arith.subi", "tosa.apply_scale", "linalg.yield"), label)
    extension, subtraction, scale, yield_operation = operations
    single_round = ir.Attribute.parse("#tosa.rounding_mode<SINGLE_ROUND>", context=candidate.context)
    if (
        extension.operands[0] != scalar_block.arguments[0]
        or subtraction.operands[0] != extension.results[0]
        or require_scalar(subtraction.operands[1], f"{label} zero point") != zero_point
        or scale.operands[0] != subtraction.results[0]
        or scale.attributes.get("rounding_mode") != single_round
        or yield_operation.operands[0] != scale.results[0]
    ):
        raise ValueError(f"{label} has modified scalar semantics")
    return require_scalar(scale.operands[1], f"{label} multiplier"), require_shift(scale.operands[2], f"{label} shift")


def parse_final_residual(candidate: ir.Operation, source: ir.Value) -> tuple[int, int, int]:
    inputs, outputs = generic_io(candidate)
    if len(inputs) != 1 or len(outputs) != 1 or inputs[0] != source:
        raise ValueError("residual final rescale has modified operands")
    validate_maps(candidate, ((0, 1, 2, 3), (0, 1, 2, 3)), "residual final rescale")
    scalar_block = candidate.regions[0].blocks[0]
    names = tuple(operation(item).name for item in scalar_block.operations)
    expected = ("tosa.apply_scale", "arith.maxsi", "arith.minsi", "arith.trunci", "linalg.yield")
    expected_zp = ("tosa.apply_scale", "arith.addi", "arith.maxsi", "arith.minsi", "arith.trunci", "linalg.yield")
    if names not in (expected, expected_zp):
        raise ValueError("residual final rescale has modified scalar semantics")
    operations = [operation(item) for item in scalar_block.operations]
    scale = operations[0]
    double_round = ir.Attribute.parse("#tosa.rounding_mode<DOUBLE_ROUND>", context=candidate.context)
    if scale.operands[0] != scalar_block.arguments[0] or scale.attributes.get("rounding_mode") != double_round:
        raise ValueError("residual final rescale has modified scalar semantics")
    multiplier = require_scalar(scale.operands[1], "residual final multiplier")
    shift = require_shift(scale.operands[2], "residual final shift")
    cursor = scale.results[0]
    output_zp = 0
    offset = 1
    if names == expected_zp:
        addition = operations[1]
        if addition.operands[0] != cursor:
            raise ValueError("residual final rescale has modified scalar dataflow")
        output_zp = require_zero_point(addition.operands[1], "residual final output zero point")
        cursor = addition.results[0]
        offset = 2
    lower, upper, truncation, yield_operation = operations[offset:]
    if (
        lower.operands[0] != cursor
        or require_scalar(lower.operands[1], "residual clamp minimum") != -128
        or upper.operands[0] != lower.results[0]
        or require_scalar(upper.operands[1], "residual clamp maximum") != 127
        or truncation.operands[0] != upper.results[0]
        or yield_operation.operands[0] != truncation.results[0]
    ):
        raise ValueError("residual final rescale is not the canonical int8 clamp")
    return multiplier, shift, output_zp


def match_residual(
    tail: list[ir.Operation], projection_output: ir.Value, semantic_input: ir.Value, projection_zp: int, input_zp: int
) -> Residual:
    generics = [candidate for candidate in tail if candidate.name == "linalg.generic"]
    if len(generics) != 5 or any(candidate.name not in ("linalg.generic", "tensor.empty") for candidate in tail):
        raise ValueError("residual tail is not the exact canonical five-stage sequence")
    first, second, third, fourth, final = generics
    new_single = parse_single_branch(first, projection_output, projection_zp, "residual new branch")
    new_double = (0, 0)
    skip_double = (0, 0)
    if tuple(operation(item).name for item in second.regions[0].blocks[0].operations) == ("tosa.apply_scale", "linalg.yield"):
        inputs, _ = generic_io(second)
        if inputs != [first.results[0]]:
            raise ValueError("residual new double scale has modified dataflow")
        new_double = parse_scale_pair(second, "DOUBLE_ROUND", "residual new double scale")
        skip_single_op = third
        add_op = fourth
        new_value = second.results[0]
        skip_value_source = third.results[0]
    else:
        skip_single_op = second
        inputs, _ = generic_io(third)
        if inputs != [second.results[0]]:
            raise ValueError("residual skip double scale has modified dataflow")
        skip_double = parse_scale_pair(third, "DOUBLE_ROUND", "residual skip double scale")
        add_op = fourth
        new_value = first.results[0]
        skip_value_source = third.results[0]
    skip_single = parse_single_branch(skip_single_op, semantic_input, input_zp, "residual skip branch")
    add_inputs, add_outputs = generic_io(add_op)
    if len(add_inputs) != 2 or len(add_outputs) != 1 or add_inputs != [new_value, skip_value_source]:
        raise ValueError("residual add has modified operands")
    zero = ir.AffineConstantExpr.get(0, context=add_op.context)
    dimensions = [ir.AffineDimExpr.get(index, context=add_op.context) for index in range(4)]
    zero_batch = ir.AffineMapAttr.get(
        ir.AffineMap.get(4, 0, [zero, *dimensions[1:]], context=add_op.context)
    )
    identity = expected_map(add_op.context, 4, (0, 1, 2, 3))
    if list(add_op.opview.indexing_maps) != [zero_batch, zero_batch, identity]:
        raise ValueError("residual add has modified indexing maps")
    parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>", context=add_op.context)
    if list(add_op.opview.iterator_types) != [parallel] * 4:
        raise ValueError("residual add must use parallel iterators")
    scalar_block, add_body = body(add_op, ("arith.addi", "linalg.yield"), "residual add")
    if list(add_body[0].operands) != list(scalar_block.arguments[:2]) or add_body[1].operands[0] != add_body[0].results[0]:
        raise ValueError("residual add has modified scalar semantics")
    final_multiplier, final_shift, final_zp = parse_final_residual(final, add_op.results[0])
    return Residual(
        operations=set(tail),
        output=final.results[0],
        final_zp=final_zp,
        config=(
            new_single[0], new_single[1], new_double[0], new_double[1],
            skip_single[0], skip_single[1], skip_double[0], skip_double[1],
            final_multiplier, final_shift,
        ),
    )

def match_conv_pointwise(candidate: ir.Operation, label: str) -> Stage:
    """Matches a 1x1 pointwise convolution as an inverted-bottleneck stage.

    TFLite-imported models (MLPerf Tiny) lower pointwise convolutions to
    linalg.conv_2d_nhwc_hwcf_q with kernel 1x1 instead of quantized matmul.
    The transposed weight's pre-transpose constant is [f,h,w,c] =
    [cout,1,1,cin], whose flattened values are exactly the output-major
    [cout,cin] layout the inverted bottleneck ukernel expects; a 2D
    constant is synthesized from it.
    """
    stage = match_conv(candidate, label)
    n, ih, iw, cin, oh, ow, cout, kh, kw, sh, sw, dh, dw, pt, pl, pb, pr, izp, ozp, scratch = stage.geometry
    if kh != 1 or kw != 1 or sh != 1 or sw != 1 or dh != 1 or dw != 1:
        raise ValueError(f"{label} pointwise conv must have a 1x1 kernel and unit strides")
    if pt or pl or pb or pr:
        raise ValueError(f"{label} pointwise conv must not pad")
    transpose = stage.transpose
    if dense_ints(transpose.attributes["permutation"]) != (1, 2, 3, 0):
        raise ValueError(f"{label} pointwise conv weight must use a [1, 2, 3, 0] transpose")
    source = transpose.operands[0]
    require_constant_tensor(source, "i8", (cout, kh, kw, cin), f"{label} pointwise weight")
    values = [int(item) for item in owner_operation(source).attributes["value"]]
    with candidate.context, ir.Location.unknown():
        with ir.InsertionPoint(candidate):
            i8 = ir.IntegerType.get_signless(8)
            weight = arith.ConstantOp(
                ir.RankedTensorType.get([cout, cin], i8),
                ir.DenseIntElementsAttr.get(
                    array("b", values),
                    type=ir.RankedTensorType.get([cout, cin], i8),
                ),
            ).result
    operations = set(stage.operations) | {transpose}
    return Stage(
        contraction=candidate,
        collapse_input=candidate,
        transpose=transpose,
        weight=weight,
        bias_init=stage.bias_init,
        bias=stage.bias,
        expand=candidate,
        rescale=stage.rescale,
        multiplier=stage.multiplier,
        shift=stage.shift,
        input_zp=izp,
        output_zp=ozp,
        operations=operations,
        rank=4,
    )


def match_stage(candidate: ir.Operation, label: str) -> Stage:
    if candidate.name == MATMUL:
        return match_pointwise(candidate, label, 4)
    if candidate.name == CONV:
        return match_conv_pointwise(candidate, label)
    raise ValueError(f"{label} is not a pointwise stage")


def pointwise_candidates(direct: list[ir.Operation]) -> list[ir.Operation]:
    candidates = [candidate for candidate in direct if candidate.name == MATMUL]
    for candidate in direct:
        if candidate.name != CONV:
            continue
        try:
            match_conv_pointwise(candidate, "pointwise filter")
        except ValueError:
            continue
        candidates.append(candidate)
    return candidates


def match_conv(candidate: ir.Operation, label: str) -> ConvStage:
    if len(candidate.operands) != 5 or len(candidate.results) != 1:
        raise ValueError(f"{label} conv has modified operands")
    if scalar_integer(candidate.operands[3]) != 0:
        raise ValueError(f"{label} weight zero point must be zero")
    input_zp = require_zero_point(candidate.operands[2], f"{label} input zero point")
    pad = owner_operation(candidate.operands[0])
    input_value = candidate.operands[0]
    if pad is not None and pad.name == "tensor.pad":
        input_value = pad.operands[0]
        low, high = pad_low_high(pad, label)
        if low[0] or low[3] or high[0] or high[3]:
            raise ValueError(f"{label} may only pad spatial dimensions")
        _, pad_body = body(pad, ("tensor.yield",), f"{label} pad")
        if len(pad_body[0].operands) != 1 or require_scalar(pad_body[0].operands[0], f"{label} pad value") != input_zp:
            raise ValueError(f"{label} pad value must equal the input zero point")
    else:
        pad = None
        low = high = (0, 0, 0, 0)
    transpose = owner_operation(candidate.operands[1])
    if transpose is None or transpose.name != "linalg.transpose" or dense_ints(transpose.attributes["permutation"]) != (1, 2, 3, 0):
        raise ValueError(f"{label} weight must use a [1, 2, 3, 0] linalg.transpose")
    bias_init = owner_operation(candidate.operands[4])
    if bias_init is None or bias_init.name != "linalg.generic":
        raise ValueError(f"{label} accumulator must be initialized from its bias vector")
    input_shape = tensor_shape(input_value, "i8", 4)
    n, ih, iw, cin = input_shape
    weight_shape = tensor_shape(candidate.operands[1], "i8", 4)
    kh, kw, weight_channels, cout = weight_shape
    if weight_channels != cin:
        raise ValueError(f"{label} conv weight channels are inconsistent")
    require_constant_tensor(transpose.operands[0], "i8", (cout, kh, kw, cin), f"{label} conv weight")
    strides = dense_ints(candidate.attributes["strides"])
    dilations = dense_ints(candidate.attributes["dilations"])
    if len(strides) != 2 or len(dilations) != 2:
        raise ValueError(f"{label} conv geometry is malformed")
    pt, pl, pb, pr = low[1], low[2], high[1], high[2]
    oh = (ih + pt + pb - dilations[0] * (kh - 1) - 1) // strides[0] + 1
    ow = (iw + pl + pr - dilations[1] * (kw - 1) - 1) // strides[1] + 1
    if tensor_shape(candidate.results[0], "i32", 4) != (n, oh, ow, cout):
        raise ValueError(f"{label} conv output geometry is inconsistent")
    bias = validate_bias_init(bias_init, cout, label, 4)
    rescale = unique_user(candidate.results[0], "linalg.generic", label)
    multiplier, shift, output_zp = validate_rescale(rescale, cout, label, 4)
    scratch_bytes = kh * iw * cin
    geometry = (
        n, ih, iw, cin, oh, ow, cout, kh, kw,
        strides[0], strides[1], dilations[0], dilations[1],
        pt, pl, pb, pr, input_zp, output_zp, scratch_bytes,
    )
    return ConvStage(
        conv=candidate,
        pad=pad,
        transpose=transpose,
        weight=candidate.operands[1],
        bias_init=bias_init,
        bias=bias,
        rescale=rescale,
        multiplier=multiplier,
        shift=shift,
        input_zp=input_zp,
        output_zp=output_zp,
        geometry=geometry,
        scratch_bytes=scratch_bytes,
        operations={candidate, transpose, bias_init, rescale} | ({pad} if pad is not None else set()),
    )


def match_pair_chain(first: ir.Operation, second: ir.Operation, direct: list[ir.Operation], index: dict) -> tuple[tuple[int, ...], tuple[ir.Operation, ...], int]:
    if len(first.operands) != 5 or len(second.operands) != 5 or len(first.results) != 1 or len(second.results) != 1:
        raise ValueError("pointwise pair has an unexpected operand layout")
    if scalar_integer(first.operands[2]) != 0 or scalar_integer(first.operands[3]) != 0 or scalar_integer(second.operands[2]) != 0 or scalar_integer(second.operands[3]) != 0:
        raise ValueError("pointwise pair must use zero zero-points")
    first_index = index[first]
    expected_names = [
        "linalg.quantized_matmul",
        "tensor.empty",
        "linalg.generic",
        "tensor.empty",
        "linalg.fill",
        "linalg.quantized_matmul",
        "tensor.empty",
        "linalg.generic",
    ]
    if first_index + len(expected_names) > len(direct) or [item.name for item in direct[first_index : first_index + len(expected_names)]] != expected_names or direct[first_index + 5] != second:
        raise ValueError("pointwise pair must be one contiguous canonical chain")
    _, middle_empty, first_clamp, _, second_fill, _, _, second_clamp = direct[first_index : first_index + len(expected_names)]
    fill0 = owner_operation(first.operands[4])
    fill1 = second_fill
    if fill0 is None or fill0.name != "linalg.fill" or len(fill0.operands) != 2 or scalar_integer(fill0.operands[0]) != 0:
        raise ValueError("pointwise accumulators must be initialized by zero fills")
    if fill1 is None or fill1.name != "linalg.fill" or len(fill1.operands) != 2 or scalar_integer(fill1.operands[0]) != 0:
        raise ValueError("pointwise accumulators must be initialized by zero fills")
    if second.operands[0] != first_clamp.results[0]:
        raise ValueError("pointwise matmuls are not an adjacent producer-consumer pair")
    for matmul, label in ((first, "first"), (second, "second")):
        owner = owner_operation(matmul.operands[1])
        if owner is None or owner.name not in ("arith.constant", "linalg.transpose"):
            raise ValueError(f"pointwise pair {label} weight must be a compile-time constant")
    input_shape = tensor_shape(first.operands[0], "i8", 2)
    first_weight_source = first.operands[1] if owner_operation(first.operands[1]).name == "arith.constant" else owner_operation(first.operands[1]).operands[0]
    first_weight_shape = tensor_shape(first_weight_source, "i8", 2)
    intermediate_shape = tensor_shape(first_clamp.results[0], "i8", 2)
    second_weight_source = second.operands[1] if owner_operation(second.operands[1]).name == "arith.constant" else owner_operation(second.operands[1]).operands[0]
    second_weight_shape = tensor_shape(second_weight_source, "i8", 2)
    output_shape = tensor_shape(second_clamp.results[0], "i8", 2)
    rows, input_channels = input_shape
    first_weight_rows, intermediate_channels = first_weight_shape
    intermediate_rows, intermediate_width = intermediate_shape
    second_weight_rows, output_channels = second_weight_shape
    output_rows, output_width = output_shape
    if (
        first_weight_rows != input_channels
        or intermediate_rows != rows
        or intermediate_width != intermediate_channels
        or second_weight_rows != intermediate_channels
        or output_rows != rows
        or output_width != output_channels
    ):
        raise ValueError("pointwise pair tensor dimensions are inconsistent")
    config = (rows, input_channels, intermediate_channels, output_channels)
    erase_operations = tuple(direct[first_index : index[second_clamp] + 1])
    return config, erase_operations, intermediate_channels


def match_ib_block(dw: ir.Operation, expansion_op: ir.Operation, projection_op: ir.Operation, direct: list[ir.Operation], index: dict, number: int) -> InvertedBottleneck:
    expansion = match_stage(expansion_op, f"block {number} expansion")
    projection = match_stage(projection_op, f"block {number} projection")
    semantic_input = expansion.collapse_input.operands[0]
    input_shape = tensor_shape(semantic_input, "i8", 4)
    n, ih, iw, cin = input_shape
    exp_shape = tensor_shape(expansion.rescale.results[0], "i8", 4)
    cexp = exp_shape[3]
    if exp_shape != (n, ih, iw, cexp) or tensor_shape(expansion.weight, "i8", 2) != (cexp, cin):
        raise ValueError(f"block {number} expansion dimensions are inconsistent")
    if expansion.collapse_input.operands[0] != semantic_input:
        raise ValueError(f"block {number} expansion input is disconnected")

    pad = owner_operation(dw.operands[0])
    fill = owner_operation(dw.operands[4])
    if pad is None or pad.name != "tensor.pad" or pad.operands[0] != expansion.rescale.results[0]:
        raise ValueError(f"block {number} depthwise input must be the padded expansion result")
    if fill is None:
        raise ValueError(f"block {number} depthwise accumulator is missing")
    validate_fill(fill, f"block {number} depthwise accumulator")
    if len(dw.operands) != 5 or scalar_integer(dw.operands[3]) != 0:
        raise ValueError(f"block {number} depthwise contraction has modified zero points")
    depthwise_input_zp = require_zero_point(dw.operands[2], f"block {number} depthwise input zero point")
    if depthwise_input_zp != expansion.output_zp:
        raise ValueError(f"block {number} expansion/depthwise zero points are inconsistent")
    weight_shape = tensor_shape(dw.operands[1], "i8", 4)
    kh, kw, weight_channels, depth_multiplier = weight_shape
    require_constant_tensor(dw.operands[1], "i8", weight_shape, f"block {number} depthwise weight")
    if weight_channels != cexp or depth_multiplier != 1:
        raise ValueError(f"block {number} depthwise weight dimensions are inconsistent")
    strides = dense_ints(dw.attributes["strides"])
    dilations = dense_ints(dw.attributes["dilations"])
    if len(strides) != 2 or len(dilations) != 2:
        raise ValueError(f"block {number} depthwise geometry is malformed")
    low, high = pad_low_high(pad, f"block {number}")
    if low[0] or low[3] or high[0] or high[3]:
        raise ValueError(f"block {number} may only pad spatial dimensions")
    _, pad_body = body(pad, ("tensor.yield",), f"block {number} pad")
    if len(pad_body[0].operands) != 1 or require_scalar(pad_body[0].operands[0], f"block {number} pad value") != expansion.output_zp:
        raise ValueError(f"block {number} pad value must equal the expansion zero point")
    oh = (ih + low[1] + high[1] - dilations[0] * (kh - 1) - 1) // strides[0] + 1
    ow = (iw + low[2] + high[2] - dilations[1] * (kw - 1) - 1) // strides[1] + 1
    if tensor_shape(dw.results[0], "i32", 5) != (n, oh, ow, cexp, 1):
        raise ValueError(f"block {number} depthwise output geometry is inconsistent")
    dw_collapse = unique_user(dw.results[0], "tensor.collapse_shape", f"block {number} depthwise")
    dw_bias_add = unique_user(dw_collapse.results[0], "linalg.generic", f"block {number} depthwise collapse")
    dw_bias = validate_depthwise_bias(dw_bias_add, cexp)
    dw_rescale = unique_user(dw_bias_add.results[0], "linalg.generic", f"block {number} depthwise bias")
    dw_multiplier, dw_shift, dw_zp = validate_rescale(dw_rescale, cexp, f"block {number} depthwise")
    if projection.collapse_input.operands[0] != dw_rescale.results[0]:
        raise ValueError(f"block {number} depthwise/projection chain is disconnected")
    if projection.contraction.name == CONV:
        if tensor_shape(projection.collapse_input.operands[0], "i8", 4) != (n, oh, ow, cexp):
            raise ValueError(f"block {number} projection input dimensions are inconsistent")
    else:
        projection_matrix = tensor_shape(projection.collapse_input.results[0], "i8", 2)
        if projection_matrix != (n * oh * ow, cexp):
            raise ValueError(f"block {number} projection collapse dimensions are inconsistent")
    cout = tensor_shape(projection.weight, "i8", 2)[0]
    projection_shape = tensor_shape(projection.rescale.results[0], "i8", 4)
    if projection_shape != (n, oh, ow, cout):
        raise ValueError(f"block {number} projection dimensions are inconsistent")
    if projection.input_zp != dw_zp:
        raise ValueError(f"block {number} depthwise/projection zero points are inconsistent")

    start = index[expansion.collapse_input]
    semantic_output = projection.rescale.results[0]
    internal = set(projection.operations)
    internal.update(expansion.operations)
    cursor = projection.rescale.results[0]
    while True:
        users = [
            operation(use.owner)
            for use in cursor.uses
            if operation(use.owner) not in internal
        ]
        if len(users) == 0:
            break
        if len(users) > 1 or users[0].name != "linalg.generic" or len(users[0].results) != 1:
            break
        internal.add(users[0])
        cursor = users[0].results[0]
        semantic_output = cursor
    end = index[owner_operation(semantic_output)]
    if end < index[projection.rescale]:
        raise ValueError(f"block {number} output precedes its projection")
    erase_operations = erase_slice(direct, start, end)
    tail = direct[index[projection.rescale] + 1 : end + 1]
    residual = None
    if semantic_output != projection.rescale.results[0]:
        residual = match_residual(tail, projection.rescale.results[0], semantic_input, projection.output_zp, expansion.input_zp)
        if residual.output != semantic_output or tensor_shape(semantic_output, "i8", 4) != input_shape:
            raise ValueError(f"block {number} residual must have the exact input shape")
    elif tail:
        raise ValueError(f"block {number} has operations after a non-residual projection")
    final_zp = residual.final_zp if residual else projection.output_zp
    residual_config = residual.config if residual else (0,) * 10
    scratch_bytes = kh * iw * cexp + ow * cexp
    if residual:
        scratch_bytes += (high[1] + 1) * ow * cout
    geometry = (
        n, ih, iw, cin, oh, ow, cexp, cout, kh, kw,
        strides[0], strides[1], dilations[0], dilations[1],
        low[1], low[2], high[1], high[2],
        expansion.input_zp, expansion.output_zp, dw_zp, projection.output_zp, final_zp,
        *residual_config, scratch_bytes,
    )
    semantic_operations = (
        expansion.operations
        | projection.operations
        | {pad, fill, dw, dw_collapse, dw_bias_add, dw_rescale}
        | (residual.operations if residual else set())
    )
    for candidate in erase_operations:
        if candidate not in semantic_operations and candidate.name != "tensor.empty":
            raise ValueError(f"block {number} slice contains unexpected operation {candidate.name}")
    return InvertedBottleneck(
        number=number,
        input=semantic_input,
        output=semantic_output,
        expansion=expansion,
        depthwise=dw,
        depthwise_weight=dw.operands[1],
        depthwise_bias=dw_bias,
        depthwise_multiplier=dw_multiplier,
        depthwise_shift=dw_shift,
        depthwise_zp=dw_zp,
        projection=projection,
        residual=residual,
        geometry=geometry,
        scratch_bytes=scratch_bytes,
        erase_operations=erase_operations,
        semantic_operations=semantic_operations,
    )


def match_depthwise_standalone(dw: ir.Operation, dw_collapse: ir.Operation, bias_add: ir.Operation, rescale: ir.Operation) -> ConvStage:
    if len(dw.operands) != 5 or len(dw.results) != 1 or scalar_integer(dw.operands[3]) != 0:
        raise ValueError("depthwise contraction has modified zero points")
    input_zp = require_zero_point(dw.operands[2], "depthwise input zero point")
    pad = owner_operation(dw.operands[0])
    input_value = dw.operands[0]
    if pad is not None and pad.name == "tensor.pad":
        input_value = pad.operands[0]
        low, high = pad_low_high(pad, "depthwise")
        if low[0] or low[3] or high[0] or high[3]:
            raise ValueError("depthwise may only pad spatial dimensions")
        _, pad_body = body(pad, ("tensor.yield",), "depthwise pad")
        if len(pad_body[0].operands) != 1 or require_scalar(pad_body[0].operands[0], "depthwise pad value") != input_zp:
            raise ValueError("depthwise pad value must equal the input zero point")
    else:
        pad = None
        low = high = (0, 0, 0, 0)
    input_shape = tensor_shape(input_value, "i8", 4)
    n, ih, iw, cin = input_shape
    weight_shape = tensor_shape(dw.operands[1], "i8", 4)
    kh, kw, weight_channels, depth_multiplier = weight_shape
    require_constant_tensor(dw.operands[1], "i8", weight_shape, "depthwise weight")
    if weight_channels != cin or depth_multiplier != 1:
        raise ValueError("depthwise weight dimensions are inconsistent")
    strides = dense_ints(dw.attributes["strides"])
    dilations = dense_ints(dw.attributes["dilations"])
    if len(strides) != 2 or len(dilations) != 2:
        raise ValueError("depthwise geometry is malformed")
    pt, pl, pb, pr = low[1], low[2], high[1], high[2]
    oh = (ih + pt + pb - dilations[0] * (kh - 1) - 1) // strides[0] + 1
    ow = (iw + pl + pr - dilations[1] * (kw - 1) - 1) // strides[1] + 1
    if tensor_shape(dw.results[0], "i32", 5) != (n, oh, ow, cin, 1):
        raise ValueError("depthwise output geometry is inconsistent")
    if dw_collapse.name != "tensor.collapse_shape" or list(dw_collapse.operands) != [dw.results[0]]:
        raise ValueError("depthwise collapse is disconnected")
    if bias_add.name != "linalg.generic" or list(bias_add.opview.inputs)[1] != dw_collapse.results[0]:
        raise ValueError("depthwise bias add is disconnected")
    bias = validate_depthwise_bias(bias_add, cin)
    if rescale.name != "linalg.generic" or list(rescale.opview.inputs)[0] != bias_add.results[0]:
        raise ValueError("depthwise rescale is disconnected")
    multiplier, shift, output_zp = validate_rescale(rescale, cin, "depthwise")
    scratch_bytes = kh * iw * cin
    geometry = (
        n, ih, iw, cin, oh, ow, cin, kh, kw,
        strides[0], strides[1], dilations[0], dilations[1],
        pt, pl, pb, pr, input_zp, output_zp, scratch_bytes,
    )
    return ConvStage(
        conv=dw,
        pad=pad,
        transpose=dw,
        weight=dw.operands[1],
        bias_init=bias_add,
        bias=bias,
        rescale=rescale,
        multiplier=multiplier,
        shift=shift,
        input_zp=input_zp,
        output_zp=output_zp,
        geometry=geometry,
        scratch_bytes=scratch_bytes,
        operations={dw, dw_collapse, bias_add, rescale} | ({pad} if pad is not None else set()),
    )


def ibn_config(block: InvertedBottleneck) -> tuple[int, ...]:
    n, ih, iw, cin, oh, ow, cexp, cout, kh, kw, sh, sw, dh, dw, pt, pl, pb, pr, input_zp, exp_zp, dw_zp, proj_zp, final_zp, *tail = block.geometry
    residual_config = tail[:10]
    scratch_bytes = tail[10]
    flags = 3 if block.residual else 0
    return (
        CONFIG_VERSION, flags, n, ih, iw, cin, oh, ow, cexp, cout, kh, kw,
        sh, sw, dh, dw, pt, pl, pb, pr, input_zp, exp_zp, dw_zp, proj_zp,
        final_zp, *residual_config, scratch_bytes, CONFIG_MAGIC,
    )


def ibn_input_operands(block: InvertedBottleneck) -> list[ir.Value]:
    return [
        block.input,
        block.expansion.weight,
        block.depthwise_weight,
        block.projection.weight,
        block.expansion.bias,
        block.depthwise_bias,
        block.projection.bias,
        block.expansion.multiplier,
        block.depthwise_multiplier,
        block.projection.multiplier,
        block.expansion.shift,
        block.depthwise_shift,
        block.projection.shift,
    ]


def ibn_standard_bytes(block: InvertedBottleneck) -> int:
    n, ih, iw, cin, oh, ow, cexp, cout = block.geometry[:8]
    return max(n * ih * iw * cin, n * ih * iw * cexp, n * oh * ow * cexp, n * oh * ow * cout)


def conv_stage_to_module(stage: ConvStage, erase: tuple[ir.Operation, ...], direct: list[ir.Operation], index: dict) -> GenericModule:
    n, ih, iw, cin, oh, ow, cout, kh, kw, sh, sw, dh, dw, pt, pl, pb, pr, izp, ozp, scratch = stage.geometry
    if len(stage.geometry) == 20:
        config = (
            CONFIG_VERSION, 0, n, ih, iw, cin, oh, ow, cout, kh, kw,
            sh, sw, dh, dw, pt, pl, pb, pr, izp, ozp,
            *(0,) * 14, scratch, CONFIG_MAGIC,
        )
    else:
        raise ValueError("conv stage geometry is malformed")
    input_value = stage.pad.operands[0] if stage.pad is not None else stage.conv.operands[0]
    return GenericModule(
        kind="conv2d",
        input=input_value,
        output=stage.rescale.results[0],
        kernel_name=CONV2D_KERNEL_NAME,
        config=config,
        input_operands=(input_value, stage.weight, stage.bias, stage.multiplier, stage.shift),
        output_type=stage.rescale.results[0].type,
        scratch_bytes=scratch,
        erase_operations=erase,
        standard_bytes=max(n * ih * iw * cin, n * oh * ow * cout),
    )


def fc_stage_to_module(stage: Stage, erase: tuple[ir.Operation, ...], rank: int, direct: list[ir.Operation], index: dict) -> GenericModule:
    if rank == 4:
        n, ih, iw, cin = tensor_shape(stage.collapse_input.operands[0], "i8", 4)
        rows = n * ih * iw
        input_channels = cin
    else:
        rows, input_channels = tensor_shape(stage.contraction.operands[0], "i8", 2)
    output_channels = tensor_shape(stage.weight, "i8", 2)[0]
    config = (
        CONFIG_VERSION, 0, rows, input_channels, output_channels,
        *(0,) * 15, stage.input_zp, stage.output_zp,
        *(0,) * 13, output_channels, CONFIG_MAGIC,
    )
    if len(config) != 37:
        raise AssertionError("internal fc config length mismatch")
    return GenericModule(
        kind="fc",
        input=stage.collapse_input.operands[0] if rank == 4 else stage.contraction.operands[0],
        output=stage.rescale.results[0],
        kernel_name=FC_KERNEL_NAME,
        config=config,
        input_operands=(stage.contraction.operands[0], stage.weight, stage.bias, stage.multiplier, stage.shift),
        output_type=stage.rescale.results[0].type,
        scratch_bytes=output_channels,
        erase_operations=erase,
        standard_bytes=max(rows * input_channels, rows * output_channels),
    )


def erase_slice(direct: list[ir.Operation], start: int, end: int, exclude: set[ir.Operation] | None = None) -> tuple[ir.Operation, ...]:
    slice_ops = direct[start : end + 1]
    erase = set(slice_ops)
    if exclude is not None:
        erase -= exclude
    for candidate in slice_ops:
        if candidate.name == "arith.constant":
            erase.discard(candidate)
    for candidate in slice_ops:
        if candidate.name != "tensor.empty":
            continue
        for result in candidate.results:
            for use in result.uses:
                if operation(use.owner) not in erase:
                    erase.discard(candidate)
                    break
            else:
                continue
            break
    return tuple(item for item in slice_ops if item in erase)


def match_generic(text: str) -> McunetMatch:
    context = ir.Context()
    try:
        module = ir.Module.parse(text, context=context)
    except ir.MLIRError as error:
        raise ValueError(f"invalid vMCU MLIR: {error}") from error
    verify(module.operation, "vMCU module")
    return _match_generic_in_context(context, module, text)


def _match_generic_in_context(context: ir.Context, module: ir.Module, text: str) -> McunetMatch:
    functions = operations_named(module, "util.func")
    if len(functions) != 1:
        raise ValueError(f"expected one util.func, found {len(functions)}")
    direct = direct_operations(functions[0])
    index = {candidate: position for position, candidate in enumerate(direct)}
    consumed: set[ir.Operation] = set()
    modules: list[GenericModule] = []
    depthwise = [candidate for candidate in direct if candidate.name == DEPTHWISE]
    convs = [candidate for candidate in direct if candidate.name == CONV]
    matmuls = [candidate for candidate in direct if candidate.name == MATMUL]

    pointwise_ops = pointwise_candidates(direct)
    for dw in depthwise:
        dw_index = index[dw]
        before = [candidate for candidate in pointwise_ops if index[candidate] < dw_index and candidate not in consumed]
        after = [candidate for candidate in pointwise_ops if index[candidate] > dw_index and candidate not in consumed]
        if not before or not after:
            continue
        try:
            block = match_ib_block(dw, before[-1], after[0], direct, index, len(modules) + 1)
        except ValueError:
            continue
        consumed.update(block.erase_operations)
        modules.append(
            GenericModule(
                kind="ibn",
                input=block.input,
                output=block.output,
                kernel_name=UKERNEL_NAME,
                config=ibn_config(block),
                input_operands=tuple(ibn_input_operands(block)),
                output_type=block.output.type,
                scratch_bytes=block.scratch_bytes,
                erase_operations=block.erase_operations,
                standard_bytes=ibn_standard_bytes(block),
                residual=block.residual is not None,
            )
        )

    for dw in depthwise:
        if dw in consumed:
            continue
        dw_index = index[dw]
        tail = direct[dw_index : dw_index + 6]
        names = [item.name for item in tail]
        if names != ["linalg.depthwise_conv_2d_nhwc_hwcm_q", "tensor.collapse_shape", "linalg.generic", "linalg.generic", "linalg.generic", "linalg.generic"]:
            continue
        dw_op, dw_collapse, bias_add, rescale, _, _ = tail
        try:
            stage = match_depthwise_standalone(dw_op, dw_collapse, bias_add, rescale)
        except ValueError:
            continue
        erase_start = index[stage.pad] if stage.pad is not None else index[dw_op]
        erase = erase_slice(direct, erase_start, index[rescale])
        if consumed & set(erase):
            raise ValueError("overlapping vMCU module matches")
        consumed.update(erase)
        modules.append(conv_stage_to_module(stage, erase, direct, index))

    for cv in convs:
        if cv in consumed:
            continue
        try:
            stage = match_conv(cv, "conv2d")
        except ValueError:
            continue
        erase_start = index[stage.pad] if stage.pad is not None else index[stage.transpose]
        exclude = {stage.transpose}
        weight_constant = owner_operation(stage.transpose.operands[0])
        if weight_constant is not None:
            exclude.add(weight_constant)
        transpose_empty = owner_operation(stage.transpose.operands[1])
        if transpose_empty is not None:
            exclude.add(transpose_empty)
        erase = erase_slice(direct, erase_start, index[stage.rescale], exclude)
        if consumed & set(erase):
            raise ValueError("overlapping vMCU module matches")
        consumed.update(erase)
        modules.append(conv_stage_to_module(stage, erase, direct, index))

    for matmul in matmuls:
        if matmul in consumed:
            continue
        position = index[matmul]
        if position + 7 < len(direct):
            names = [item.name for item in direct[position : position + 8]]
            if names == ["linalg.quantized_matmul", "tensor.empty", "linalg.generic", "tensor.empty", "linalg.fill", "linalg.quantized_matmul", "tensor.empty", "linalg.generic"] and direct[position + 5] not in consumed:
                try:
                    config, erase, segment_bytes = match_pair_chain(direct[position], direct[position + 5], direct, index)
                except ValueError:
                    config = None
                if config is not None:
                    if consumed & set(erase):
                        raise ValueError("overlapping vMCU module matches")
                    consumed.update(erase)
                    modules.append(
                        GenericModule(
                            kind="pair",
                            input=direct[position].operands[0],
                            output=direct[position + 7].results[0],
                            kernel_name=PAIR_KERNEL_NAME,
                            config=config,
                            input_operands=(direct[position].operands[0], direct[position].operands[1], direct[position + 5].operands[1]),
                            output_type=direct[position + 7].results[0].type,
                            scratch_bytes=segment_bytes,
                            erase_operations=erase,
                            standard_bytes=config[0] * config[2],
                        )
                    )
                    continue
        try:
            rank = 2 if tensor_shape(matmul.operands[0], "i8", 2) is not None else 4
        except ValueError:
            rank = 4
        try:
            stage = match_pointwise(matmul, "fc", rank)
        except ValueError:
            continue
        start = index[stage.collapse_input] if rank == 4 else index[stage.transpose]
        erase = erase_slice(direct, start, index[stage.rescale])
        if consumed & set(erase):
            raise ValueError("overlapping vMCU module matches")
        consumed.update(erase)
        modules.append(fc_stage_to_module(stage, erase, rank, direct, index))

    for candidate_module in modules:
        internal = set(candidate_module.erase_operations)
        for candidate in candidate_module.erase_operations:
            for result in list(candidate.results):
                for use in result.uses:
                    user = operation(use.owner)
                    if user not in internal and user not in consumed and result != candidate_module.output:
                        raise ValueError(f"{candidate_module.kind} slice is not closed over SSA uses")

    if not modules:
        raise ValueError("no vMCU-compatible subgraphs found")
    max_segment = max(item.scratch_bytes for item in modules)
    standard_peak = max(item.standard_bytes for item in modules)
    plan = McunetPlan(
        mode="auto",
        block_count=len(modules),
        residual_block_count=sum(item.residual for item in modules),
        in_place_block_count=sum(item.residual for item in modules),
        standard_peak_intermediate_bytes=standard_peak,
        max_segment_bytes=max_segment,
        total_segment_bytes=sum(module.scratch_bytes for module in modules),
        full_intermediate_bytes=standard_peak,
        segment_bytes=max_segment,
        saved_intermediate_bytes=standard_peak - max_segment,
    )
    return McunetMatch(context=context, module=module, blocks=[], plan=plan, modules=modules)


def plan_generic(text: str) -> McunetPlan:
    return match_generic(text).plan


def custom_maps() -> ir.ArrayAttr:
    symbols = [ir.AffineSymbolExpr.get(index) for index in range(13)]
    results = (
        (0, 1, 2, 3), (6, 3), (4, 5, 6, 7), (10, 6),
        (6,), (6,), (10,), (6,), (6,), (10,), (6,), (6,), (10,),
        (11,), (0, 8, 9, 10), (12,),
    )
    return ir.ArrayAttr.get(
        [ir.AffineMapAttr.get(ir.AffineMap.get(0, 13, [symbols[index] for index in item])) for item in results]
    )


def create_custom_op(block: InvertedBottleneck, kernel_name: str = UKERNEL_NAME, bitcode_path: str = BITCODE_PATH) -> ir.Operation:
    config_values = list(ibn_config(block))
    if len(config_values) != 37:
        raise AssertionError("internal MCUNet config length mismatch")
    output_type = block.output.type
    with ir.InsertionPoint(block.erase_operations[0]):
        i8 = ir.IntegerType.get_signless(8)
        i32 = ir.IntegerType.get_signless(32)
        config = arith.ConstantOp(ir.RankedTensorType.get([37], i32), array("i", config_values)).result
        scratch = tensor.EmptyOp([block.scratch_bytes], i8).result
        output = block.input if block.residual else tensor.EmptyOp(list(ir.RankedTensorType(output_type).shape), i8).result
        inputs = [*ibn_input_operands(block), config]
        custom = ir.Operation.create(
            "iree_linalg_ext.custom_op",
            results=[output.type, scratch.type],
            operands=[*inputs, output, scratch],
            attributes={
                "indexing_maps": custom_maps(),
                "iterator_types": ir.ArrayAttr.get([]),
                "operandSegmentSizes": ir.DenseI32ArrayAttr.get([14, 2]),
                "iree_codegen.ukernel": ir.Attribute.parse(f'#iree_codegen.ukernel_descriptor<"{kernel_name}", bitcode>'),
                "hal.executable.objects": ir.ArrayAttr.get(
                    [ir.Attribute.parse(f'#hal.executable.object<{{path = "{bitcode_path}"}}>')]
                ),
            },
            regions=1,
        )
    dynamic_types = [
        ir.Type.parse("tensor<?x?x?x?xi8>"),
        ir.Type.parse("tensor<?x?xi8>"),
        ir.Type.parse("tensor<?x?x?x?xi8>"),
        ir.Type.parse("tensor<?x?xi8>"),
        ir.Type.parse("tensor<?xi32>"), ir.Type.parse("tensor<?xi32>"), ir.Type.parse("tensor<?xi32>"),
        ir.Type.parse("tensor<?xi32>"), ir.Type.parse("tensor<?xi32>"), ir.Type.parse("tensor<?xi32>"),
        ir.Type.parse("tensor<?xi8>"), ir.Type.parse("tensor<?xi8>"), ir.Type.parse("tensor<?xi8>"),
        ir.Type.parse("tensor<?xi32>"),
        ir.Type.parse("tensor<?x?x?x?xi8>"),
        ir.Type.parse("tensor<?xi8>"),
    ]
    custom_body = ir.Block.create_at_start(custom.regions[0], dynamic_types)
    with ir.InsertionPoint(custom_body):
        ir.Operation.create("iree_linalg_ext.yield", operands=[custom_body.arguments[14], custom_body.arguments[15]])
    verify(custom, f"generated block {block.number} custom op")
    return custom


def create_generic_custom_op(module: GenericModule, position: int) -> ir.Operation:
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    specialized = module.variant is not None
    if module.kind in ("conv2d", "fc"):
        with ir.InsertionPoint(module.erase_operations[-1]):
            if specialized:
                config_operands: tuple = ()
                segment_sizes = [5, 2]
            else:
                config = arith.ConstantOp(ir.RankedTensorType.get([37], i32), array("i", list(module.config))).result
                config_operands = (config,)
                segment_sizes = [6, 2]
            scratch = tensor.EmptyOp([module.scratch_bytes], i8).result
            output = tensor.EmptyOp(list(ir.RankedTensorType(module.output_type).shape), i8).result
            inputs = [*module.input_operands, *config_operands]
            kernel_name = module.variant if specialized else module.kernel_name
            bitcode_path = f"{module.variant}.bc" if specialized else GENERIC_BITCODE_PATH
            custom = ir.Operation.create(
                "iree_linalg_ext.custom_op",
                results=[output.type, scratch.type],
                operands=[*inputs, output, scratch],
                attributes={
                    "indexing_maps": conv_fc_maps(module.kind, specialized),
                    "iterator_types": ir.ArrayAttr.get([]),
                    "operandSegmentSizes": ir.DenseI32ArrayAttr.get(segment_sizes),
                    "iree_codegen.ukernel": ir.Attribute.parse(f'#iree_codegen.ukernel_descriptor<"{kernel_name}", bitcode>'),
                    "hal.executable.objects": ir.ArrayAttr.get(
                        [ir.Attribute.parse(f'#hal.executable.object<{{path = "{bitcode_path}"}}>')]
                    ),
                },
                regions=1,
            )
        dynamic_types = [
            ir.Type.parse("tensor<?x?x?x?xi8>") if module.kind == "conv2d" else ir.Type.parse("tensor<?x?xi8>"),
            ir.Type.parse("tensor<?x?x?x?xi8>") if module.kind == "conv2d" else ir.Type.parse("tensor<?x?xi8>"),
            ir.Type.parse("tensor<?xi32>"),
            ir.Type.parse("tensor<?xi32>"),
            ir.Type.parse("tensor<?xi8>"),
        ]
        if not specialized:
            dynamic_types.append(ir.Type.parse("tensor<?xi32>"))
        dynamic_types += [
            ir.Type.parse("tensor<?x?x?x?xi8>") if module.kind == "conv2d" else ir.Type.parse("tensor<?x?xi8>"),
            ir.Type.parse("tensor<?xi8>"),
        ]
    elif module.kind == "pair":
        with ir.InsertionPoint(module.erase_operations[-1]):
            config = arith.ConstantOp(ir.RankedTensorType.get([4], i32), array("i", list(module.config))).result
            segment = tensor.EmptyOp([module.scratch_bytes], i8).result
            output = tensor.EmptyOp(list(ir.RankedTensorType(module.output_type).shape), i8).result
            custom = ir.Operation.create(
                "iree_linalg_ext.custom_op",
                results=[output.type, segment.type],
                operands=[*module.input_operands, config, output, segment],
                attributes={
                    "indexing_maps": pair_maps(),
                    "iterator_types": ir.ArrayAttr.get([]),
                    "operandSegmentSizes": ir.DenseI32ArrayAttr.get([4, 2]),
                    "iree_codegen.ukernel": ir.Attribute.parse(f'#iree_codegen.ukernel_descriptor<"{module.kernel_name}", bitcode>'),
                    "hal.executable.objects": ir.ArrayAttr.get(
                        [ir.Attribute.parse(f'#hal.executable.object<{{path = "{GENERIC_BITCODE_PATH}"}}>')]
                    ),
                },
                regions=1,
            )
        dynamic_types = [
            ir.Type.parse("tensor<?x?xi8>"),
            ir.Type.parse("tensor<?x?xi8>"),
            ir.Type.parse("tensor<?x?xi8>"),
            ir.Type.parse("tensor<?xi32>"),
            ir.Type.parse("tensor<?x?xi8>"),
            ir.Type.parse("tensor<?xi8>"),
        ]
    else:
        with ir.InsertionPoint(module.erase_operations[-1]):
            config = arith.ConstantOp(ir.RankedTensorType.get([37], i32), array("i", list(module.config))).result
            scratch = tensor.EmptyOp([module.scratch_bytes], i8).result
            output = module.input if module.residual else tensor.EmptyOp(list(ir.RankedTensorType(module.output_type).shape), i8).result
            custom = ir.Operation.create(
                "iree_linalg_ext.custom_op",
                results=[output.type, scratch.type],
                operands=[*module.input_operands, config, output, scratch],
                attributes={
                    "indexing_maps": custom_maps(),
                    "iterator_types": ir.ArrayAttr.get([]),
                    "operandSegmentSizes": ir.DenseI32ArrayAttr.get([14, 2]),
                    "iree_codegen.ukernel": ir.Attribute.parse(f'#iree_codegen.ukernel_descriptor<"{module.kernel_name}", bitcode>'),
                    "hal.executable.objects": ir.ArrayAttr.get(
                        [ir.Attribute.parse(f'#hal.executable.object<{{path = "{GENERIC_BITCODE_PATH}"}}>')]
                    ),
                },
                regions=1,
            )
        dynamic_types = [
            ir.Type.parse("tensor<?x?x?x?xi8>"),
            ir.Type.parse("tensor<?x?xi8>"),
            ir.Type.parse("tensor<?x?x?x?xi8>"),
            ir.Type.parse("tensor<?x?xi8>"),
            ir.Type.parse("tensor<?xi32>"), ir.Type.parse("tensor<?xi32>"), ir.Type.parse("tensor<?xi32>"),
            ir.Type.parse("tensor<?xi32>"), ir.Type.parse("tensor<?xi32>"), ir.Type.parse("tensor<?xi32>"),
            ir.Type.parse("tensor<?xi8>"), ir.Type.parse("tensor<?xi8>"), ir.Type.parse("tensor<?xi8>"),
            ir.Type.parse("tensor<?xi32>"),
            ir.Type.parse("tensor<?x?x?x?xi8>"),
            ir.Type.parse("tensor<?xi8>"),
        ]
    custom_body = ir.Block.create_at_start(custom.regions[0], dynamic_types)
    with ir.InsertionPoint(custom_body):
        ir.Operation.create(
            "iree_linalg_ext.yield",
            operands=[custom_body.arguments[len(dynamic_types) - 2], custom_body.arguments[len(dynamic_types) - 1]],
        )
    verify(custom, f"generated {module.kind} custom op {position}")
    return custom


def conv_fc_maps(kind: str, specialized: bool = False) -> ir.ArrayAttr:
    if kind == "conv2d":
        symbols = [ir.AffineSymbolExpr.get(index) for index in range(11)]
        results = (
            (0, 1, 2, 3), (7, 8, 3, 6), (6,), (6,), (6,), (9,),
            (0, 4, 5, 6), (10,),
        ) if not specialized else (
            (0, 1, 2, 3), (7, 8, 3, 6), (6,), (6,), (6,),
            (0, 4, 5, 6), (10,),
        )
    else:
        symbols = [ir.AffineSymbolExpr.get(index) for index in range(5)]
        results = (
            (0, 1), (2, 1), (2,), (2,), (2,), (3,),
            (0, 2), (4,),
        ) if not specialized else (
            (0, 1), (2, 1), (2,), (2,), (2,),
            (0, 2), (4,),
        )
    return ir.ArrayAttr.get(
        [ir.AffineMapAttr.get(ir.AffineMap.get(0, len(symbols), [symbols[index] for index in item])) for item in results]
    )


def pair_maps() -> ir.ArrayAttr:
    symbols = [ir.AffineSymbolExpr.get(index) for index in range(5)]
    return ir.ArrayAttr.get(
        [
            ir.AffineMapAttr.get(ir.AffineMap.get(0, 5, results))
            for results in (
                [symbols[0], symbols[1]],
                [symbols[1], symbols[2]],
                [symbols[2], symbols[3]],
                [symbols[4]],
                [symbols[0], symbols[3]],
                [symbols[2]],
            )
        ]
    )


SPECIALIZATION_FILENAME = "vmcu-generic.specializations.json"


def conv2d_shape_key(config: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(config[2:21])


def fc_shape_key(config: tuple[int, ...]) -> tuple[int, ...]:
    return (config[2], config[3], config[4], config[20], config[21])


def shape_key(module: GenericModule) -> tuple[int, ...]:
    if module.kind == "conv2d":
        return conv2d_shape_key(module.config)
    if module.kind == "fc":
        return fc_shape_key(module.config)
    return ()


def specialized_macros(module: GenericModule) -> dict[str, int]:
    if module.kind == "conv2d":
        n, ih, iw, cin, oh, ow, cout, kh, kw, sh, sw, dh, dw, pt, pl, _pb, _pr, izp, ozp = conv2d_shape_key(module.config)
        return {
            "N": n, "IH": ih, "IW": iw, "CIN": cin, "OH": oh, "OW": ow,
            "COUT": cout, "KH": kh, "KW": kw, "SH": sh, "SW": sw,
            "DH": dh, "DW": dw, "PT": pt, "PL": pl,
            "INPUT_ZP": izp, "OUTPUT_ZP": ozp,
        }
    if module.kind == "fc":
        rows, cin, cout, izp, ozp = fc_shape_key(module.config)
        return {"ROWS": rows, "CIN": cin, "COUT": cout, "INPUT_ZP": izp, "OUTPUT_ZP": ozp}
    return {}


def assign_specializations(modules: list[GenericModule], artifact_dir: Path) -> dict[str, dict]:
    """Assigns a unique specialized variant to every conv2d/fc module (sharing
    variants across modules with identical shapes), emits the wrapper C source
    per variant and the specializations manifest consumed by the toolchain."""
    variants: dict[tuple[str, tuple[int, ...]], str] = {}
    variant_meta: dict[str, dict] = {}
    for module in modules:
        if module.kind not in ("conv2d", "fc"):
            continue
        key = (module.kind, shape_key(module))
        if key not in variants:
            index = len(variants) + 1
            name = f"{module.kernel_name}_v{index}"
            macros = specialized_macros(module)
            source = f"{name}.c"
            if module.kind == "conv2d":
                entry = "VMCU_C2_ENTRY"
            else:
                entry = "VMCU_FC_ENTRY"
            lines = ["/* generated by oneliner rewriter: shape-specialized vMCU kernel */",
                     "#define VMCU_SPECIALIZED"]
            lines += [f"#define VMCU_{field} {value}" for field, value in macros.items()]
            lines.append(f"#define {entry} {name}")
            lines.append(f'#include "oneliner_vmcu_{module.kind}.c"')
            (artifact_dir / source).write_text("\n".join(lines) + "\n", encoding="utf-8")
            variants[key] = name
            variant_meta[name] = {
                "kind": module.kind,
                "source": source,
                "bitcode": f"{name}.bc",
                "shapes": {field: value for field, value in macros.items()},
            }
        module.variant = variants[key]
    if variants:
        (artifact_dir / SPECIALIZATION_FILENAME).write_text(
            json.dumps({"schema_version": 2, "variants": variant_meta}, indent=2) + "\n",
            encoding="utf-8",
        )
    return variant_meta


def rewrite_generic(text: str, artifact_dir: Path | None = None) -> tuple[str, McunetPlan]:
    matched = match_generic(text)
    if artifact_dir is not None:
        assign_specializations(matched.modules, artifact_dir)
    with matched.context, ir.Location.unknown():
        replacements = [create_generic_custom_op(module, position) for position, module in enumerate(matched.modules)]
        for module, custom in zip(matched.modules, replacements, strict=True):
            module.output.replace_all_uses_with(custom.results[0])
        erased = []
        seen = set()
        for module in reversed(matched.modules):
            for candidate in reversed(module.erase_operations):
                if candidate in seen:
                    continue
                seen.add(candidate)
                erased.append(candidate)
        for candidate in erased:
            candidate.erase()
        verify(matched.module.operation, "rewritten vMCU module (erased)")
        verify(matched.module.operation, "rewritten vMCU module")
        output = matched.module.operation.get_asm(assume_verified=True)
        reparsed = ir.Module.parse(output)
        verify(reparsed.operation, "reparsed rewritten vMCU module")
    return output, matched.plan


def finalize_generic(text: str) -> tuple[str, int]:
    context = ir.Context()
    try:
        module = ir.Module.parse(text, context=context)
    except ir.MLIRError as error:
        raise ValueError(f"invalid configured vMCU MLIR: {error}") from error
    verify(module.operation, "configured vMCU module")
    with context, ir.Location.unknown():
        matches = []
        for candidate in operations_named(module, "iree_codegen.ukernel.generic"):
            if "u_kernel_fn_name" not in candidate.attributes:
                continue
            name = ir.StringAttr(candidate.attributes["u_kernel_fn_name"]).value
            if name not in GENERIC_KERNEL_NAMES and not any(
                name.startswith(base + "_") for base in GENERIC_KERNEL_NAMES
            ):
                continue
            matches.append(candidate)
        for candidate in matches:
            del candidate.attributes["iree_codegen.ukernel"]
            fn_attributes = {}
            if "fn_def_attrs" in candidate.attributes:
                fn_attributes.update({named.name: named.attr for named in ir.DictAttr(candidate.attributes["fn_def_attrs"])})
            fn_attributes["hal.import.bitcode"] = ir.BoolAttr.get(True)
            candidate.attributes["fn_def_attrs"] = ir.DictAttr.get(fn_attributes)
        verify(module.operation, "finalized configured vMCU module")
        output = module.operation.get_asm(assume_verified=True)
        reparsed = ir.Module.parse(output)
        verify(reparsed.operation, "reparsed finalized vMCU module")
    return output, len(matches)


def plan_json(plan_value: McunetPlan) -> dict:
    return {"schema_version": 2, **asdict(plan_value)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-configured", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    text = sys.stdin.read()
    try:
        if args.finalize_configured:
            output, count = finalize_generic(text)
            if count < 1:
                raise ValueError(f"expected at least one configured vMCU ukernel, found {count}")
        else:
            output, plan_value = rewrite_generic(text, artifact_dir=args.artifact_dir)
            if args.plan_output:
                args.plan_output.write_text(json.dumps(plan_json(plan_value), indent=2) + "\n", encoding="utf-8")
    except ValueError as error:
        print(f"vMCU rewrite failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
