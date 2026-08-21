#!/usr/bin/env python3
"""Fuses one supported int8 pointwise pair into a segment-buffer ukernel."""

from __future__ import annotations

import argparse
import json
import sys
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path

from iree.compiler import ir
from iree.compiler.dialects import arith, func, hal, iree_codegen, linalg, tensor, util


BITCODE_PATH = "oneliner_vmcu_pointwise.bc"
UKERNEL_NAME = "oneliner_vmcu_pointwise_pair_s8"


@dataclass(frozen=True)
class PointwisePairPlan:
    rows: int
    input_channels: int
    intermediate_channels: int
    output_channels: int
    full_intermediate_bytes: int
    segment_bytes: int
    saved_intermediate_bytes: int


@dataclass
class PointwisePairMatch:
    context: ir.Context
    module: ir.Module
    plan: PointwisePairPlan
    first: ir.Operation
    first_clamp: ir.Operation
    second: ir.Operation
    second_clamp: ir.Operation
    erase_operations: tuple[ir.Operation, ...]


def operation(value) -> ir.Operation:
    return value.operation if hasattr(value, "operation") else value


def owner_operation(value: ir.Value) -> ir.Operation | None:
    if not isinstance(value, ir.OpResult):
        return None
    return operation(value.owner)


def operations_named(module: ir.Module, name: str) -> list[ir.Operation]:
    matches: list[ir.Operation] = []

    def collect(candidate: ir.Operation) -> ir.WalkResult:
        if candidate.name == name:
            matches.append(candidate)
        return ir.WalkResult.ADVANCE

    module.operation.walk(collect)
    return matches


def verify(operation_to_check: ir.Operation, label: str) -> None:
    try:
        operation_to_check.verify()
    except ir.MLIRError as error:
        raise ValueError(f"{label} failed verification: {error}") from error


def only_use(value: ir.Value, expected_owner: ir.Operation, operand_number: int) -> bool:
    uses = list(value.uses)
    return (
        len(uses) == 1
        and operation(uses[0].owner) == expected_owner
        and uses[0].operand_number == operand_number
    )


def scalar_integer(value: ir.Value) -> int | None:
    owner = owner_operation(value)
    if owner is None or owner.name != "arith.constant" or "value" not in owner.attributes:
        return None
    try:
        return ir.IntegerAttr(owner.attributes["value"]).value
    except ValueError:
        return None


def ranked_tensor(value: ir.Value, element_type: str) -> tuple[int, ...] | None:
    try:
        tensor_type = ir.RankedTensorType(value.type)
    except ValueError:
        return None
    shape = tuple(tensor_type.shape)
    if tensor_type.rank != 2 or any(dimension < 0 for dimension in shape):
        return None
    if str(tensor_type.element_type) != element_type:
        return None
    return shape


def constant_tensor(value: ir.Value) -> bool:
    owner = owner_operation(value)
    return owner is not None and owner.name == "arith.constant"


def validate_zero_fill(
    matmul: ir.Operation, block_operations: list[ir.Operation]
) -> tuple[ir.Operation, ir.Operation]:
    if len(matmul.operands) != 5 or len(matmul.results) != 1:
        raise ValueError("pointwise matmul has an unexpected operand layout")
    initializer = matmul.operands[4]
    fill = owner_operation(initializer)
    if fill is None or fill.name != "linalg.fill":
        raise ValueError("pointwise accumulators must be initialized by zero fills")
    matmul_index = block_operations.index(matmul)
    if block_operations[matmul_index - 1] != fill:
        raise ValueError("pointwise pair must be one contiguous canonical chain")
    if (
        len(fill.operands) != 2
        or len(fill.results) != 1
        or fill.results[0] != initializer
        or scalar_integer(fill.operands[0]) != 0
        or not only_use(fill.results[0], matmul, 4)
    ):
        raise ValueError("pointwise accumulators must be initialized by zero fills")
    empty = owner_operation(fill.operands[1])
    if (
        empty is None
        or empty.name != "tensor.empty"
        or block_operations[matmul_index - 2] != empty
        or len(empty.results) != 1
        or not only_use(empty.results[0], fill, 1)
    ):
        raise ValueError("pointwise accumulator fill must use a fresh empty tensor")
    return empty, fill


def validate_clamp(
    clamp: ir.Operation,
    source: ir.Value,
    block_operations: list[ir.Operation],
) -> ir.Operation:
    if clamp.name != "linalg.generic" or len(clamp.results) != 1:
        raise ValueError("both pointwise matmuls must be followed by int8 saturation")
    view = clamp.opview
    inputs = list(view.inputs)
    outputs = list(view.outputs)
    if (
        len(inputs) != 1
        or len(outputs) != 1
        or inputs[0] != source
        or not only_use(source, clamp, 0)
    ):
        raise ValueError("both pointwise matmuls must be followed by int8 saturation")

    with clamp.context:
        identity = ir.AffineMapAttr.get(
            ir.AffineMap.get(
                2,
                0,
                [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)],
            )
        )
        parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>")
    if list(view.indexing_maps) != [identity, identity] or list(
        view.iterator_types
    ) != [parallel, parallel]:
        raise ValueError("both pointwise matmuls must use identity saturation maps")

    if len(clamp.regions) != 1 or len(clamp.regions[0].blocks) != 1:
        raise ValueError("pointwise saturation must have one scalar body")
    body = clamp.regions[0].blocks[0]
    arguments = list(body.arguments)
    body_operations = [operation(item) for item in body.operations]
    if (
        len(arguments) != 2
        or str(arguments[0].type) != "i32"
        or str(arguments[1].type) != "i8"
        or list(arguments[1].uses)
        or [item.name for item in body_operations]
        != ["arith.maxsi", "arith.minsi", "arith.trunci", "linalg.yield"]
    ):
        raise ValueError("pointwise saturation body is not the canonical int8 clamp")

    lower, upper, truncation, yield_operation = body_operations
    if (
        len(lower.operands) != 2
        or len(lower.results) != 1
        or lower.operands[0] != arguments[0]
        or scalar_integer(lower.operands[1]) != -128
        or not only_use(lower.results[0], upper, 0)
        or len(upper.operands) != 2
        or len(upper.results) != 1
        or upper.operands[0] != lower.results[0]
        or scalar_integer(upper.operands[1]) != 127
        or not only_use(upper.results[0], truncation, 0)
        or len(truncation.operands) != 1
        or len(truncation.results) != 1
        or truncation.operands[0] != upper.results[0]
        or str(truncation.results[0].type) != "i8"
        or not only_use(truncation.results[0], yield_operation, 0)
        or len(yield_operation.operands) != 1
        or yield_operation.operands[0] != truncation.results[0]
    ):
        raise ValueError("pointwise saturation body is not the canonical int8 clamp")

    initializer = outputs[0]
    empty = owner_operation(initializer)
    clamp_index = block_operations.index(clamp)
    if (
        empty is None
        or empty.name != "tensor.empty"
        or block_operations[clamp_index - 1] != empty
        or not only_use(initializer, clamp, 1)
    ):
        raise ValueError("pointwise saturation must write to a fresh empty tensor")
    return empty


def parse_pointwise_pair(text: str) -> PointwisePairMatch:
    context = ir.Context()
    try:
        module = ir.Module.parse(text, context=context)
    except ir.MLIRError as error:
        raise ValueError(f"invalid IREE preprocessing MLIR: {error}") from error
    verify(module.operation, "IREE preprocessing module")

    functions = operations_named(module, "util.func")
    if len(functions) != 1:
        raise ValueError("pointwise-pair MVP requires exactly one function")
    function = functions[0]
    if len(function.regions) != 1 or len(function.regions[0].blocks) != 1:
        raise ValueError("pointwise-pair MVP requires one straight-line function block")
    block = function.regions[0].blocks[0]
    block_operations = [operation(item) for item in block.operations]

    all_matmuls = operations_named(module, "linalg.quantized_matmul")
    matmuls = [item for item in block_operations if item.name == "linalg.quantized_matmul"]
    if len(all_matmuls) != 2 or len(matmuls) != 2:
        raise ValueError(
            f"expected exactly two static int8 quantized matmuls, found {len(all_matmuls)}"
        )
    first, second = matmuls
    first_index = block_operations.index(first)
    if first_index < 2 or first_index + 7 >= len(block_operations):
        raise ValueError("pointwise pair must be one contiguous canonical chain")
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
    chain = block_operations[first_index : first_index + len(expected_names)]
    if [item.name for item in chain] != expected_names or chain[5] != second:
        raise ValueError("pointwise pair must be one contiguous canonical chain")
    _, middle_empty, first_clamp, second_accumulator_empty, second_fill, _, output_empty, second_clamp = chain

    for matmul, label in ((first, "first"), (second, "second")):
        if scalar_integer(matmul.operands[2]) != 0 or scalar_integer(
            matmul.operands[3]
        ) != 0:
            raise ValueError(f"the {label} pointwise matmul must use zero zero-points")

    first_accumulator_empty, first_fill = validate_zero_fill(first, block_operations)
    validated_second_empty, validated_second_fill = validate_zero_fill(
        second, block_operations
    )
    if validated_second_empty != second_accumulator_empty or validated_second_fill != second_fill:
        raise ValueError("pointwise pair must be one contiguous canonical chain")

    first_clamp_empty = validate_clamp(first_clamp, first.results[0], block_operations)
    second_clamp_empty = validate_clamp(second_clamp, second.results[0], block_operations)
    if first_clamp_empty != middle_empty or second_clamp_empty != output_empty:
        raise ValueError("pointwise pair must be one contiguous canonical chain")
    if second.operands[0] != first_clamp.results[0] or not only_use(
        first_clamp.results[0], second, 0
    ):
        raise ValueError("pointwise matmuls are not an adjacent producer-consumer pair")

    if not constant_tensor(first.operands[1]) or not constant_tensor(second.operands[1]):
        raise ValueError("pointwise weights must be compile-time constants")

    input_shape = ranked_tensor(first.operands[0], "i8")
    first_weight_shape = ranked_tensor(first.operands[1], "i8")
    intermediate_shape = ranked_tensor(first_clamp.results[0], "i8")
    second_weight_shape = ranked_tensor(second.operands[1], "i8")
    output_shape = ranked_tensor(second_clamp.results[0], "i8")
    first_accumulator_shape = ranked_tensor(first.results[0], "i32")
    second_accumulator_shape = ranked_tensor(second.results[0], "i32")
    if None in (
        input_shape,
        first_weight_shape,
        intermediate_shape,
        second_weight_shape,
        output_shape,
        first_accumulator_shape,
        second_accumulator_shape,
    ):
        raise ValueError("pointwise pair requires static rank-2 int8 tensors")

    rows, input_channels = input_shape
    first_weight_rows, intermediate_channels = first_weight_shape
    intermediate_rows, intermediate_width = intermediate_shape
    second_weight_rows, output_channels = second_weight_shape
    output_rows, output_width = output_shape
    if (
        first_weight_rows != input_channels
        or intermediate_rows != rows
        or intermediate_width != intermediate_channels
        or first_accumulator_shape != intermediate_shape
        or second_weight_rows != intermediate_channels
        or output_rows != rows
        or output_width != output_channels
        or second_accumulator_shape != output_shape
    ):
        raise ValueError("pointwise pair tensor dimensions are inconsistent")

    max_i32_reduction = ((1 << 31) - 1) // (128 * 128)
    if input_channels > max_i32_reduction or intermediate_channels > max_i32_reduction:
        raise ValueError("pointwise reduction can overflow an i32 accumulator")
    if any(
        dimension > (1 << 31) - 1
        for dimension in (rows, input_channels, intermediate_channels, output_channels)
    ):
        raise ValueError("pointwise dimension cannot be represented by the i32 ukernel config")
    for left, right in (
        (rows, input_channels),
        (rows, intermediate_channels),
        (rows, output_channels),
        (input_channels, intermediate_channels),
        (intermediate_channels, output_channels),
    ):
        if left * right > (1 << 32) - 1:
            raise ValueError("pointwise tensor exceeds the Cortex-M4 address space")

    plan = PointwisePairPlan(
        rows=rows,
        input_channels=input_channels,
        intermediate_channels=intermediate_channels,
        output_channels=output_channels,
        full_intermediate_bytes=rows * intermediate_channels,
        segment_bytes=intermediate_channels,
        saved_intermediate_bytes=(rows - 1) * intermediate_channels,
    )
    erase_operations = tuple(
        block_operations[
            block_operations.index(first_accumulator_empty) : block_operations.index(second_clamp)
            + 1
        ]
    )
    return PointwisePairMatch(
        context=context,
        module=module,
        plan=plan,
        first=first,
        first_clamp=first_clamp,
        second=second,
        second_clamp=second_clamp,
        erase_operations=erase_operations,
    )


def plan_pointwise_pair(text: str) -> PointwisePairPlan:
    return parse_pointwise_pair(text).plan


def create_custom_op(matched: PointwisePairMatch) -> ir.Operation:
    plan = matched.plan
    first = matched.first
    second = matched.second
    with ir.InsertionPoint(matched.erase_operations[0]):
        i8 = ir.IntegerType.get_signless(8)
        i32 = ir.IntegerType.get_signless(32)
        config_type = ir.RankedTensorType.get([4], i32)
        config = arith.ConstantOp(
            config_type,
            array(
                "i",
                [
                    plan.rows,
                    plan.input_channels,
                    plan.intermediate_channels,
                    plan.output_channels,
                ],
            ),
        ).result
        segment = tensor.EmptyOp([plan.segment_bytes], i8).result
        output = tensor.EmptyOp(
            list(ir.RankedTensorType(matched.second_clamp.results[0].type).shape), i8
        ).result

        symbols = [ir.AffineSymbolExpr.get(index) for index in range(5)]
        maps = ir.ArrayAttr.get(
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
        custom = ir.Operation.create(
            "iree_linalg_ext.custom_op",
            results=[output.type, segment.type],
            operands=[
                first.operands[0],
                first.operands[1],
                second.operands[1],
                config,
                output,
                segment,
            ],
            attributes={
                "indexing_maps": maps,
                "iterator_types": ir.ArrayAttr.get([]),
                "operandSegmentSizes": ir.DenseI32ArrayAttr.get([4, 2]),
                "iree_codegen.ukernel": ir.Attribute.parse(
                    f'#iree_codegen.ukernel_descriptor<"{UKERNEL_NAME}", bitcode>'
                ),
                "hal.executable.objects": ir.ArrayAttr.get(
                    [
                        ir.Attribute.parse(
                            f'#hal.executable.object<{{path = "{BITCODE_PATH}"}}>'
                        )
                    ]
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
    body = ir.Block.create_at_start(custom.regions[0], dynamic_types)
    with ir.InsertionPoint(body):
        ir.Operation.create(
            "iree_linalg_ext.yield",
            operands=[body.arguments[4], body.arguments[5]],
        )
    verify(custom, "generated vMCU custom op")
    return custom


def rewrite(text: str) -> tuple[str, PointwisePairPlan]:
    matched = parse_pointwise_pair(text)
    with matched.context, ir.Location.unknown():
        custom = create_custom_op(matched)
        matched.second_clamp.results[0].replace_all_uses_with(custom.results[0])
        for candidate in reversed(matched.erase_operations):
            candidate.erase()
        verify(matched.module.operation, "rewritten vMCU module")
        output = matched.module.operation.get_asm(assume_verified=True)
    return output, matched.plan


def finalize_configured(text: str) -> tuple[str, int]:
    context = ir.Context()
    try:
        module = ir.Module.parse(text, context=context)
    except ir.MLIRError as error:
        raise ValueError(f"invalid configured IREE MLIR: {error}") from error
    verify(module.operation, "configured vMCU module")
    with context, ir.Location.unknown():
        descriptor = ir.Attribute.parse(
            f'#iree_codegen.ukernel_descriptor<"{UKERNEL_NAME}", bitcode>'
        )
        matches = []
        for candidate in operations_named(module, "iree_codegen.ukernel.generic"):
            if (
                "iree_codegen.ukernel" in candidate.attributes
                and candidate.attributes["iree_codegen.ukernel"] == descriptor
                and "u_kernel_fn_name" in candidate.attributes
                and ir.StringAttr(candidate.attributes["u_kernel_fn_name"]).value
                == UKERNEL_NAME
            ):
                matches.append(candidate)
        for candidate in matches:
            del candidate.attributes["iree_codegen.ukernel"]
            fn_attributes = {}
            if "fn_def_attrs" in candidate.attributes:
                fn_attributes.update(
                    {
                        named.name: named.attr
                        for named in ir.DictAttr(candidate.attributes["fn_def_attrs"])
                    }
                )
            fn_attributes["hal.import.bitcode"] = ir.BoolAttr.get(True)
            candidate.attributes["fn_def_attrs"] = ir.DictAttr.get(
                fn_attributes
            )
        if matches:
            verify(module.operation, "finalized vMCU module")
        output = module.operation.get_asm(assume_verified=bool(matches))
    return output, len(matches)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-configured", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    args = parser.parse_args()
    text = sys.stdin.read()
    try:
        if args.finalize_configured:
            output, count = finalize_configured(text)
            if count != 1:
                raise ValueError(f"expected one configured vMCU ukernel, found {count}")
        else:
            output, plan = rewrite(text)
            if args.plan_output:
                args.plan_output.write_text(
                    json.dumps({"schema_version": 1, **asdict(plan)}, indent=2) + "\n",
                    encoding="utf-8",
                )
    except ValueError as error:
        print(f"vMCU pointwise-pair rewrite failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
