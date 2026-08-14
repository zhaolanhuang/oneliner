#!/usr/bin/env python3
"""Rewrites supported quantized Conv2D dispatches to a CMSIS-NN ukernel."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass


SSA = r"%[A-Za-z0-9_.$-]+"


@dataclass(frozen=True)
class ConvMatch:
    result: str
    input_value: str
    filter_value: str
    input_zero_point: int
    init_value: str
    input_type: str
    filter_type: str
    accumulator_type: str
    stride: tuple[int, int]
    dilation: tuple[int, int]


def parse_shape(tensor_type: str) -> tuple[int, ...] | None:
    match = re.fullmatch(r"tensor<([0-9x]+)x(?:i8|i32)>", tensor_type)
    if not match:
        return None
    return tuple(int(value) for value in match.group(1).split("x"))


def parse_dense_ints(text: str) -> list[int] | None:
    match = re.search(r"dense<\[([^]]*)\]>", text)
    if not match:
        return None
    return [int(value.strip()) for value in match.group(1).split(",")]


def block_end(lines: list[str], start: int) -> int:
    depth = 0
    seen_open = False
    for index in range(start, len(lines)):
        depth += lines[index].count("{")
        if "{" in lines[index]:
            seen_open = True
        depth -= lines[index].count("}")
        if seen_open and depth == 0:
            return index + 1
    raise ValueError(f"unterminated MLIR operation starting on line {start + 1}")


def scalar_constants(lines: list[str]) -> dict[str, int]:
    constants: dict[str, int] = {}
    pattern = re.compile(rf"^\s*({SSA})\s*=\s*arith\.constant\s+(-?[0-9]+)\s*:\s*i(?:8|16|32|64|ndex)")
    for line in lines:
        match = pattern.match(line)
        if match:
            constants[match.group(1)] = int(match.group(2))
    return constants


def resolve_scalar(token: str, constants: dict[str, int]) -> int | None:
    if token in constants:
        return constants[token]
    literal = re.fullmatch(r"-?[0-9]+", token)
    return int(token) if literal else None


def parse_pair(attribute_text: str, name: str) -> tuple[int, int] | None:
    match = re.search(rf"{name}\s*=\s*dense<([^>]*)>", attribute_text)
    if not match:
        return None
    values = [int(value) for value in re.findall(r"-?[0-9]+", match.group(1))]
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    return None


def find_conv(line: str, constants: dict[str, int]) -> ConvMatch | None:
    pattern = re.compile(
        rf"^\s*({SSA})\s*=\s*linalg\.conv_2d_nhwc_hwcf_q\s*"
        rf"(?P<attrs>\{{.*?\}})\s*ins\("
        rf"({SSA}),\s*({SSA}),\s*({SSA}),\s*({SSA})\s*:\s*"
        rf"(tensor<[^>]+>),\s*(tensor<[^>]+>),\s*i32,\s*i32\)\s*"
        rf"outs\(({SSA})\s*:\s*(tensor<[^>]+>)\)"
    )
    match = pattern.match(line)
    if not match:
        return None
    input_zp = resolve_scalar(match.group(5), constants)
    filter_zp = resolve_scalar(match.group(6), constants)
    stride = parse_pair(match.group("attrs"), "strides")
    dilation = parse_pair(match.group("attrs"), "dilations")
    if input_zp is None or filter_zp != 0 or stride is None or dilation is None:
        return None
    return ConvMatch(
        result=match.group(1),
        input_value=match.group(3),
        filter_value=match.group(4),
        input_zero_point=input_zp,
        init_value=match.group(9),
        input_type=match.group(7),
        filter_type=match.group(8),
        accumulator_type=match.group(10),
        stride=stride,
        dilation=dilation,
    )


def defining_op(lines: list[str], value: str, before: int) -> int | None:
    pattern = re.compile(rf"^\s*{re.escape(value)}\s*=")
    for index in range(before - 1, -1, -1):
        if pattern.match(lines[index]):
            return index
    return None


def original_filter(lines: list[str], conv: ConvMatch, before: int) -> tuple[str, str] | None:
    producer = defining_op(lines, conv.filter_value, before)
    if producer is None:
        return None
    match = re.search(
        rf"linalg\.transpose\s+ins\(({SSA})\s*:\s*(tensor<[^>]+>)\).*permutation\s*=\s*\[1,\s*2,\s*3,\s*0\]",
        lines[producer],
    )
    if not match:
        return None
    return match.group(1), match.group(2)


def broadcast_bias(lines: list[str], conv: ConvMatch, before: int) -> tuple[str, str] | None:
    producer = defining_op(lines, conv.init_value, before)
    if producer is None or "linalg.generic" not in lines[producer]:
        return None
    match = re.search(rf"ins\(({SSA})\s*:\s*(tensor<[^>]+>)\)", lines[producer])
    if not match:
        return None
    shape = parse_shape(match.group(2))
    if shape is None or len(shape) != 1:
        return None
    return match.group(1), match.group(2)


def requant_match(
    lines: list[str], conv: ConvMatch, start: int, constants: dict[str, int]
) -> tuple[int, int, str, str, str, str, str, str, str, int, int, int] | None:
    use_pattern = re.compile(
        rf"^\s*({SSA})\s*=\s*linalg\.generic\s+.*ins\("
        rf"{re.escape(conv.result)},\s*({SSA}),\s*({SSA})\s*:\s*"
        rf"{re.escape(conv.accumulator_type)},\s*(tensor<[^>]+>),\s*(tensor<[^>]+>)\)\s*"
        rf"outs\(({SSA})\s*:\s*(tensor<[^>]+>)\)"
    )
    for index in range(start + 1, len(lines)):
        match = use_pattern.match(lines[index])
        if not match:
            continue
        end = block_end(lines, index)
        body = "\n".join(lines[index:end])
        offset_match = re.search(
            rf"({SSA})\s*=\s*arith\.addi\s+{SSA},\s*({SSA})\s*:\s*i32\s*\n\s*"
            rf"({SSA})\s*=\s*arith\.maxsi\s+\1,\s*({SSA})\s*:\s*i32\s*\n\s*"
            rf"{SSA}\s*=\s*arith\.minsi\s+\3,\s*({SSA})\s*:\s*i32",
            body,
        )
        if not offset_match:
            continue
        output_offset = resolve_scalar(offset_match.group(2), constants)
        activation_min = resolve_scalar(offset_match.group(4), constants)
        activation_max = resolve_scalar(offset_match.group(5), constants)
        if None in (output_offset, activation_min, activation_max):
            continue
        return (
            index,
            end,
            match.group(1),
            match.group(2),
            match.group(4),
            match.group(3),
            match.group(5),
            match.group(6),
            match.group(7),
            int(output_offset),
            int(activation_min),
            int(activation_max),
        )
    return None


def dense_definition(lines: list[str], value: str, before: int) -> tuple[list[int], str] | None:
    producer = defining_op(lines, value, before)
    if producer is None:
        return None
    values = parse_dense_ints(lines[producer])
    type_match = re.search(r":\s*(tensor<[^>]+>)\s*$", lines[producer])
    if values is None or not type_match:
        return None
    return values, type_match.group(1)


def rewrite(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    constants = scalar_constants(lines)
    replacements: list[tuple[int, int, list[str]]] = []
    rewritten = 0

    for conv_index, line in enumerate(lines):
        conv = find_conv(line, constants)
        if conv is None:
            continue
        input_shape = parse_shape(conv.input_type)
        filter_shape = parse_shape(conv.filter_type)
        accumulator_shape = parse_shape(conv.accumulator_type)
        filter_source = original_filter(lines, conv, conv_index)
        bias = broadcast_bias(lines, conv, conv_index)
        requant = requant_match(lines, conv, conv_index, constants)
        if (
            input_shape is None or len(input_shape) != 4
            or filter_shape is None or len(filter_shape) != 4
            or accumulator_shape is None or len(accumulator_shape) != 4
            or input_shape[0] != 1 or filter_source is None or bias is None
            or requant is None
        ):
            continue

        (
            req_start, req_end, result, multiplier, multiplier_type, shift,
            shift_type, output_init, output_type, output_offset, activation_min,
            activation_max,
        ) = requant
        shift_def = dense_definition(lines, shift, req_start)
        multiplier_shape = parse_shape(multiplier_type)
        if shift_def is None or multiplier_shape is None:
            continue
        output_shape = parse_shape(output_type)
        if output_shape is None or len(output_shape) != 4:
            continue
        shift_values, parsed_shift_type = shift_def
        if parsed_shift_type != shift_type:
            continue
        if len(shift_values) != output_shape[3] or multiplier_shape != (output_shape[3],):
            continue

        source_filter_value, source_filter_type = filter_source
        source_filter_shape = parse_shape(source_filter_type)
        bias_value, bias_type = bias
        if source_filter_shape != (output_shape[3], filter_shape[0], filter_shape[1], input_shape[3]):
            continue

        stride_h, stride_w = conv.stride
        dilation_h, dilation_w = conv.dilation
        effective_h = (filter_shape[0] - 1) * dilation_h + 1
        effective_w = (filter_shape[1] - 1) * dilation_w + 1
        total_pad_h = max(0, (output_shape[1] - 1) * stride_h + effective_h - input_shape[1])
        total_pad_w = max(0, (output_shape[2] - 1) * stride_w + effective_w - input_shape[2])
        if total_pad_h % 2 or total_pad_w % 2:
            continue
        pad_h, pad_w = total_pad_h // 2, total_pad_w // 2

        rhs_cols = filter_shape[0] * filter_shape[1] * input_shape[3]
        is_1x1 = filter_shape[0:2] == (1, 1) and pad_h == 0 and pad_w == 0
        scratch_size = 0 if is_1x1 else 4 * ((rhs_cols + 3) // 4 * 4)
        scratch_len = max(1, scratch_size)
        cmsis_shifts = [31 - value for value in shift_values]
        prefix = f"cmsis_nn_{rewritten}"
        scalar_values = [
            input_shape[0], input_shape[1], input_shape[2], input_shape[3],
            output_shape[1], output_shape[2], output_shape[3], filter_shape[0],
            filter_shape[1], stride_h, stride_w, dilation_h, dilation_w, pad_h,
            pad_w, -conv.input_zero_point, output_offset, activation_min,
            activation_max, scratch_size,
        ]
        indent = re.match(r"\s*", lines[req_start]).group(0)
        generated = [
            f"{indent}%{prefix}_shift = arith.constant dense<{cmsis_shifts}> : tensor<{len(cmsis_shifts)}xi32>",
            f"{indent}%{prefix}_config = arith.constant dense<{scalar_values}> : tensor<20xi32>",
            f"{indent}%{prefix}_scratch = tensor.empty() : tensor<{scratch_len}xi8>",
        ]
        tensor_values = [
            conv.input_value, source_filter_value, bias_value, multiplier,
            f"%{prefix}_shift", f"%{prefix}_scratch", f"%{prefix}_config",
        ]
        tensor_types = [
            conv.input_type, source_filter_type, bias_type, multiplier_type,
            f"tensor<{len(cmsis_shifts)}xi32>", f"tensor<{scratch_len}xi8>",
            "tensor<20xi32>",
        ]
        symbols = ", ".join(f"s{index}" for index in range(14))
        maps = [
            f"affine_map<()[{symbols}] -> (s4, s5, s6, s7)>",
            f"affine_map<()[{symbols}] -> (s8, s9, s10, s11)>",
            *[f"affine_map<()[{symbols}] -> (s3)>" for _ in range(3)],
            f"affine_map<()[{symbols}] -> (s12)>",
            f"affine_map<()[{symbols}] -> (s13)>",
            f"affine_map<()[{symbols}] -> (s0, s1, s2, s3)>",
        ]
        region_types = [
            "tensor<?x?x?x?xi8>", "tensor<?x?x?x?xi8>",
            *["tensor<?xi32>" for _ in range(3)], "tensor<?xi8>",
            "tensor<?xi32>", "tensor<?x?x?x?xi8>",
        ]
        generated.extend([
            f'{indent}{result} = iree_linalg_ext.custom_op {{indexing_maps = [{", ".join(maps)}], '
            f'iterator_types = []}} attributes {{'
            f'iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"oneliner_cmsis_nn_conv_s8", bitcode>, '
            f'hal.executable.objects = [#hal.executable.object<{{path = "oneliner_cmsis_nn_conv_s8.bc"}}>]}} '
            f'ins({", ".join(tensor_values)} : {", ".join(tensor_types)}) '
            f'outs({output_init} : {output_type}) {{',
            f'{indent}^bb0(%input: {region_types[0]}, %filter: {region_types[1]}, '
            f'%bias: {region_types[2]}, %multiplier: {region_types[3]}, '
            f'%shift: {region_types[4]}, %scratch: {region_types[5]}, '
            f'%config: {region_types[6]}, %out: {region_types[7]}):',
            f"{indent}  iree_linalg_ext.yield %out : {region_types[7]}",
            f"{indent}}} -> {output_type}",
        ])
        replacements.append((req_start, req_end, generated))
        rewritten += 1

    for start, end, generated in reversed(replacements):
        lines[start:end] = generated
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), rewritten


def finalize_configured(text: str) -> tuple[str, int]:
    descriptor = (
        '{iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<'
        '"oneliner_cmsis_nn_conv_s8", bitcode>} '
    )
    finalized = 0
    lines = []
    for line in text.splitlines():
        if "iree_codegen.ukernel.generic" in line and descriptor in line:
            line = line.replace(descriptor, "", 1)
            line = line.replace(
                " strided_dims(",
                " fn_def_attrs {hal.import.bitcode = true} strided_dims(",
                1,
            )
            finalized += 1
        lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), finalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-match", action="store_true")
    parser.add_argument("--finalize-configured", action="store_true")
    args = parser.parse_args()
    operation = finalize_configured if args.finalize_configured else rewrite
    output, count = operation(sys.stdin.read())
    if args.require_match and count == 0:
        print("no supported CMSIS-NN int8 Conv2D found", file=sys.stderr)
        return 1
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
