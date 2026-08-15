#!/usr/bin/env python3
"""Rewrites supported quantized operations to CMSIS-NN ukernels."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass


SSA = r"%[A-Za-z0-9_.$-]+"
BITCODE_PATH = "oneliner_cmsis_nn.bc"
UKERNEL_NAMES = (
    "oneliner_cmsis_nn_conv_s8",
    "oneliner_cmsis_nn_max_pool_s8",
    "oneliner_cmsis_nn_fully_connected_s8",
    "oneliner_cmsis_nn_depthwise_conv_s8",
    "oneliner_cmsis_nn_avg_pool_s8",
)
# CMSIS-NN CH_IN_BLOCK_MVE: number of channels processed per block by the MVE
# depthwise conv opt kernel. Matches arm_nnsupportfunctions.h.
CH_IN_BLOCK_MVE = 124


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


@dataclass(frozen=True)
class PoolMatch:
    result: str
    input_value: str
    init_value: str
    input_type: str
    window_type: str
    output_type: str
    stride: tuple[int, int]
    dilation: tuple[int, int]


@dataclass(frozen=True)
class DepthwiseMatch:
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


@dataclass(frozen=True)
class AvgPoolMatch:
    result: str
    input_value: str
    input_type: str
    window_type: str
    accumulator_type: str
    output_init: str
    output_type: str
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


def find_pool(line: str) -> PoolMatch | None:
    pattern = re.compile(
        rf"^\s*({SSA})\s*=\s*linalg\.pooling_nhwc_max\s*"
        rf"(?P<attrs>\{{.*?\}})\s*ins\("
        rf"({SSA}),\s*({SSA})\s*:\s*(tensor<[^>]+>),\s*(tensor<[^>]+>)\)\s*"
        rf"outs\(({SSA})\s*:\s*(tensor<[^>]+>)\)"
    )
    match = pattern.match(line)
    if not match:
        return None
    stride = parse_pair(match.group("attrs"), "strides")
    dilation = parse_pair(match.group("attrs"), "dilations")
    if stride is None or dilation is None:
        return None
    return PoolMatch(
        result=match.group(1),
        input_value=match.group(3),
        init_value=match.group(7),
        input_type=match.group(5),
        window_type=match.group(6),
        output_type=match.group(8),
        stride=stride,
        dilation=dilation,
    )


def find_avgpool(line: str) -> AvgPoolMatch | None:
    pattern = re.compile(
        rf"^\s*({SSA})\s*=\s*linalg\.pooling_nhwc_sum\s*"
        rf"(?P<attrs>\{{.*?\}})\s*ins\("
        rf"({SSA}),\s*({SSA})\s*:\s*(tensor<[^>]+>),\s*(tensor<[^>]+>)\)\s*"
        rf"outs\(({SSA})\s*:\s*(tensor<[^>]+>)\)"
    )
    match = pattern.match(line)
    if not match:
        return None
    stride = parse_pair(match.group("attrs"), "strides")
    dilation = parse_pair(match.group("attrs"), "dilations")
    if stride is None or dilation is None:
        return None
    return AvgPoolMatch(
        result=match.group(1),
        input_value=match.group(3),
        input_type=match.group(5),
        window_type=match.group(6),
        accumulator_type=match.group(8),
        output_init=match.group(7),
        output_type=match.group(8),
        stride=stride,
        dilation=dilation,
    )


def avgpool_requant_match(
    lines: list[str], pool: AvgPoolMatch, start: int, constants: dict[str, int]
) -> tuple[int, int, str, str, str, int, int, int] | None:
    use_pattern = re.compile(
        rf"^\s*({SSA})\s*=\s*linalg\.generic\s+.*ins\("
        rf"{re.escape(pool.result)}\s*:\s*{re.escape(pool.accumulator_type)}\)\s*"
        rf"outs\(({SSA})\s*:\s*(tensor<[^>]+>)\)"
    )
    for index in range(start + 1, len(lines)):
        match = use_pattern.match(lines[index])
        if not match:
            continue
        end = block_end(lines, index)
        body = "\n".join(lines[index:end])
        multiplier_match = re.search(
            rf"arith\.muli\s+{SSA},\s*({SSA})\s*:\s*i64", body
        )
        rounding_match = re.search(
            rf"arith\.addi\s+{SSA},\s*({SSA})\s*:\s*i64", body
        )
        shift_match = re.search(
            rf"arith\.shrui\s+{SSA},\s*({SSA})\s*:\s*i64", body
        )
        clamp_match = re.search(
            rf"({SSA})\s*=\s*arith\.maxsi\s+{SSA},\s*({SSA})\s*:\s*i32\s*\n\s*"
            rf"({SSA})\s*=\s*arith\.minsi\s+\1,\s*({SSA})\s*:\s*i32",
            body,
        )
        if not (multiplier_match and rounding_match and shift_match and clamp_match):
            continue
        multiplier = resolve_scalar(multiplier_match.group(1), constants)
        rounding = resolve_scalar(rounding_match.group(1), constants)
        shift = resolve_scalar(shift_match.group(1), constants)
        activation_min = resolve_scalar(clamp_match.group(2), constants)
        activation_max = resolve_scalar(clamp_match.group(4), constants)
        if None in (multiplier, rounding, shift, activation_min, activation_max):
            continue
        if multiplier < 0 or multiplier >= (1 << 31) or rounding != (1 << 31) or shift != 32:
            continue
        return (
            index,
            end,
            match.group(1),
            match.group(2),
            match.group(3),
            int(multiplier),
            int(activation_min),
            int(activation_max),
        )
    return None


def find_depthwise(
    line: str, constants: dict[str, int]
) -> DepthwiseMatch | None:
    pattern = re.compile(
        rf"^\s*({SSA})\s*=\s*linalg\.depthwise_conv_2d_nhwc_hwcm_q\s*"
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
    return DepthwiseMatch(
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
    line = lines[producer]
    type_match = re.search(r":\s*(tensor<[^>]+>)\s*$", line)
    if not type_match:
        return None
    values = parse_dense_ints(line)
    if values is None:
        tensor_type = type_match.group(1)
        shape = parse_shape(tensor_type)
        element_match = re.search(r"x(i8|i32)>", tensor_type)
        hex_match = re.search(r'dense<"0x([0-9A-Fa-f]*)">', line)
        if shape is None or not element_match or not hex_match:
            return None
        count = 1
        for dimension in shape:
            count *= dimension
        width = 1 if element_match.group(1) == "i8" else 4
        data = bytes.fromhex(hex_match.group(1))
        if len(data) != count * width:
            return None
        values = [
            int.from_bytes(data[index:index + width], "little", signed=True)
            for index in range(0, len(data), width)
        ]
    return values, type_match.group(1)


def dense_bytes_definition(lines: list[str], value: str, before: int) -> bytes | None:
    producer = defining_op(lines, value, before)
    if producer is None:
        return None
    line = lines[producer]
    type_match = re.search(r":\s*(tensor<[^>]+>)\s*$", line)
    if not type_match:
        return None
    tensor_type = type_match.group(1)
    shape = parse_shape(tensor_type)
    element_match = re.search(r"x(i8|i32)>", tensor_type)
    if shape is None or not element_match:
        return None
    count = 1
    for dimension in shape:
        count *= dimension
    hex_match = re.search(r'dense<"0x([0-9A-Fa-f]*)">', line)
    if hex_match:
        data = bytes.fromhex(hex_match.group(1))
    else:
        values = parse_dense_ints(line)
        if values is None:
            return None
        width = 1 if element_match.group(1) == "i8" else 4
        data = b"".join(
            (value & ((1 << (width * 8)) - 1)).to_bytes(width, "little")
            for value in values
        )
    expected_size = count * (1 if element_match.group(1) == "i8" else 4)
    return data if len(data) == expected_size else None


def fill_scalar(
    lines: list[str], value: str, before: int, constants: dict[str, int]
) -> int | None:
    producer = defining_op(lines, value, before)
    if producer is None:
        return None
    match = re.search(
        rf"linalg\.fill\s+ins\(({SSA})\s*:\s*i(?:8|32)\)", lines[producer]
    )
    return resolve_scalar(match.group(1), constants) if match else None


def scalar_requant_match(
    lines: list[str], conv: ConvMatch, start: int, constants: dict[str, int]
) -> tuple[int, int, str, str, str, int, int, int, int, int] | None:
    use_pattern = re.compile(
        rf"^\s*({SSA})\s*=\s*linalg\.generic\s+.*ins\("
        rf"{re.escape(conv.result)}\s*:\s*{re.escape(conv.accumulator_type)}\)\s*"
        rf"outs\(({SSA})\s*:\s*(tensor<[^>]+>)\)"
    )
    for index in range(start + 1, len(lines)):
        match = use_pattern.match(lines[index])
        if not match:
            continue
        end = block_end(lines, index)
        body = "\n".join(lines[index:end])
        multiplier_match = re.search(
            rf"arith\.muli\s+{SSA},\s*({SSA})\s*:\s*i64", body
        )
        shift_match = re.search(
            rf"arith\.shrsi\s+{SSA},\s*({SSA})\s*:\s*i64", body
        )
        offset_match = re.search(
            rf"({SSA})\s*=\s*arith\.addi\s+{SSA},\s*({SSA})\s*:\s*i32\s*\n\s*"
            rf"({SSA})\s*=\s*arith\.maxsi\s+\1,\s*({SSA})\s*:\s*i32\s*\n\s*"
            rf"{SSA}\s*=\s*arith\.minsi\s+\3,\s*({SSA})\s*:\s*i32",
            body,
        )
        if not multiplier_match or not shift_match or not offset_match:
            continue
        multiplier = resolve_scalar(multiplier_match.group(1), constants)
        shift = resolve_scalar(shift_match.group(1), constants)
        output_offset = resolve_scalar(offset_match.group(2), constants)
        activation_min = resolve_scalar(offset_match.group(4), constants)
        activation_max = resolve_scalar(offset_match.group(5), constants)
        if None in (multiplier, shift, output_offset, activation_min, activation_max):
            continue
        return (
            index,
            end,
            match.group(1),
            match.group(2),
            match.group(3),
            int(multiplier),
            int(shift),
            int(output_offset),
            int(activation_min),
            int(activation_max),
        )
    return None


def depthwise_requant_match(
    lines: list[str], depthwise: DepthwiseMatch, start: int, constants: dict[str, int]
) -> tuple[str, str, tuple[int, int, str, str, str, str, str, str, str, int, int, int]] | None:
    collapse_pattern = re.compile(
        rf"^\s*({SSA})\s*=\s*tensor\.collapse_shape\s+"
        rf"{re.escape(depthwise.result)}\s+.*:\s*"
        rf"{re.escape(depthwise.accumulator_type)}\s+into\s+(tensor<[^>]+>)"
    )
    for collapse_index in range(start + 1, len(lines)):
        collapse_match = collapse_pattern.match(lines[collapse_index])
        if not collapse_match:
            continue
        collapsed_value = collapse_match.group(1)
        collapsed_type = collapse_match.group(2)
        bias_pattern = re.compile(
            rf"^\s*({SSA})\s*=\s*linalg\.generic\s+.*ins\("
            rf"({SSA}),\s*{re.escape(collapsed_value)}\s*:\s*"
            rf"(tensor<[^>]+>),\s*{re.escape(collapsed_type)}\)\s*"
            rf"outs\(({SSA})\s*:\s*{re.escape(collapsed_type)}\)"
        )
        for bias_index in range(collapse_index + 1, len(lines)):
            bias_match = bias_pattern.match(lines[bias_index])
            if not bias_match:
                continue
            bias_end = block_end(lines, bias_index)
            if "arith.addi" not in "\n".join(lines[bias_index:bias_end]):
                continue
            requant_source = ConvMatch(
                result=bias_match.group(1),
                input_value="",
                filter_value="",
                input_zero_point=0,
                init_value="",
                input_type="",
                filter_type="",
                accumulator_type=collapsed_type,
                stride=(1, 1),
                dilation=(1, 1),
            )
            requant = requant_match(lines, requant_source, bias_index, constants)
            if requant is not None:
                return bias_match.group(2), bias_match.group(3), requant
            break
        break
    return None


def rewrite(text: str, kernel_class: str = "dsp") -> tuple[str, int]:
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
        if (
            input_shape is None or len(input_shape) != 4
            or filter_shape is None or len(filter_shape) != 4
            or accumulator_shape is None or len(accumulator_shape) != 4
            or input_shape[0] != 1 or filter_source is None or bias is None
        ):
            continue

        source_filter_value, source_filter_type = filter_source
        source_filter_shape = parse_shape(source_filter_type)
        bias_value, bias_type = bias
        bias_shape = parse_shape(bias_type)
        source_filter_bytes = dense_bytes_definition(
            lines, source_filter_value, conv_index
        )
        bias_bytes = dense_bytes_definition(lines, bias_value, conv_index)
        scalar_requant = scalar_requant_match(lines, conv, conv_index, constants)
        if scalar_requant is not None:
            (
                req_start, req_end, result, output_init, output_type,
                multiplier, shift, output_offset, activation_min, activation_max,
            ) = scalar_requant
            output_shape = parse_shape(output_type)
            if (
                input_shape[1:3] == (1, 1)
                and filter_shape[0:2] == (1, 1)
                and accumulator_shape == output_shape
                and output_shape is not None
                and len(output_shape) == 4
                and output_shape[1:3] == (1, 1)
                and source_filter_shape
                == (output_shape[3], 1, 1, input_shape[3])
                and bias_shape == (output_shape[3],)
                and conv.stride == (1, 1)
                and conv.dilation == (1, 1)
                and source_filter_bytes is not None
                and bias_bytes is not None
            ):
                prefix = f"cmsis_nn_{rewritten}"
                # MVE fully connected needs an output-depth int32 buffer for
                # the bias-accumulator init (kernel_sum); DSP needs none.
                fc_scratch_size = (
                    output_shape[3] * 4 if kernel_class == "mve" else 0
                )
                fc_scratch_len = max(1, fc_scratch_size)
                config_values = [
                    input_shape[0], input_shape[3], output_shape[3],
                    -conv.input_zero_point, 0, output_offset, multiplier,
                    31 - shift, activation_min, activation_max,
                    fc_scratch_size,
                ]
                indent = re.match(r"\s*", lines[req_start]).group(0)
                generated = [
                    f"{indent}%{prefix}_config = arith.constant dense<{config_values}> : tensor<11xi32>",
                    f"{indent}%{prefix}_scratch = tensor.empty() : tensor<{fc_scratch_len}xi8>",
                ]
                tensor_values = [
                    conv.input_value,
                    source_filter_value,
                    bias_value,
                    f"%{prefix}_scratch",
                    f"%{prefix}_config",
                ]
                tensor_types = [
                    conv.input_type,
                    source_filter_type,
                    bias_type,
                    f"tensor<{fc_scratch_len}xi8>",
                    "tensor<11xi32>",
                ]
                symbols = ", ".join(f"s{index}" for index in range(14))
                maps = [
                    f"affine_map<()[{symbols}] -> (s0, s4, s5, s6)>",
                    f"affine_map<()[{symbols}] -> (s7, s8, s9, s10)>",
                    f"affine_map<()[{symbols}] -> (s11)>",
                    f"affine_map<()[{symbols}] -> (s12)>",
                    f"affine_map<()[{symbols}] -> (s13)>",
                    f"affine_map<()[{symbols}] -> (s0, s1, s2, s3)>",
                ]
                region_types = [
                    "tensor<?x?x?x?xi8>",
                    "tensor<?x?x?x?xi8>",
                    "tensor<?xi32>",
                    "tensor<?xi8>",
                    "tensor<?xi32>",
                    "tensor<?x?x?x?xi8>",
                ]
                generated.extend([
                    f'{indent}{result} = iree_linalg_ext.custom_op {{indexing_maps = [{", ".join(maps)}], '
                    f'iterator_types = []}} attributes {{'
                    f'iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"oneliner_cmsis_nn_fully_connected_s8", bitcode>, '
                    f'hal.executable.objects = [#hal.executable.object<{{path = "{BITCODE_PATH}"}}>]}} '
                    f'ins({", ".join(tensor_values)} : {", ".join(tensor_types)}) '
                    f'outs({output_init} : {output_type}) {{',
                    f'{indent}^bb0(%input: {region_types[0]}, %filter: {region_types[1]}, '
                    f'%bias: {region_types[2]}, %scratch: {region_types[3]}, '
                    f'%config: {region_types[4]}, %out: {region_types[5]}):',
                    f"{indent}  iree_linalg_ext.yield %out : {region_types[5]}",
                    f"{indent}}} -> {output_type}",
                ])
                replacements.append((req_start, req_end, generated))
                rewritten += 1
                continue

        requant = requant_match(lines, conv, conv_index, constants)
        if requant is None:
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
        # The CMSIS-NN wrapper only takes the buffer-free 1x1 fast paths when
        # dilation is 1; a dilated 1x1 conv otherwise falls into the generic
        # arm_convolve_s8, which returns ARM_CMSIS_NN_ARG_ERROR without a
        # scratch buffer. Allocate the generic buffer in that case.
        is_1x1 = (
            filter_shape[0:2] == (1, 1)
            and pad_h == 0
            and pad_w == 0
            and dilation_h == 1
            and dilation_w == 1
        )
        if kernel_class == "mve":
            # arm_convolve_s8 MVE im2col buffer: 4 * ceil(rhs_cols/16) * 16.
            scratch_size = 0 if is_1x1 else 4 * ((rhs_cols + 15) // 16 * 16)
        else:
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
            f'hal.executable.objects = [#hal.executable.object<{{path = "oneliner_cmsis_nn.bc"}}>]}} '
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

    for depthwise_index, line in enumerate(lines):
        depthwise = find_depthwise(line, constants)
        if depthwise is None:
            continue
        input_shape = parse_shape(depthwise.input_type)
        filter_shape = parse_shape(depthwise.filter_type)
        accumulator_shape = parse_shape(depthwise.accumulator_type)
        chain = depthwise_requant_match(
            lines, depthwise, depthwise_index, constants
        )
        if (
            input_shape is None
            or len(input_shape) != 4
            or filter_shape is None
            or len(filter_shape) != 4
            or accumulator_shape is None
            or len(accumulator_shape) != 5
            or input_shape[0] != 1
            or filter_shape[2] != input_shape[3]
            or filter_shape[3] != 1
            or accumulator_shape[0] != 1
            or accumulator_shape[3:] != (input_shape[3], 1)
            or depthwise.dilation != (1, 1)
            or fill_scalar(lines, depthwise.init_value, depthwise_index, constants) != 0
            or chain is None
        ):
            continue
        bias_value, bias_type, requant = chain
        (
            req_start, req_end, result, multiplier, multiplier_type, shift,
            shift_type, output_init, output_type, output_offset, activation_min,
            activation_max,
        ) = requant
        output_shape = parse_shape(output_type)
        bias_shape = parse_shape(bias_type)
        multiplier_shape = parse_shape(multiplier_type)
        shift_def = dense_definition(lines, shift, req_start)
        if (
            output_shape != accumulator_shape[:4]
            or bias_shape != (output_shape[3],)
            or multiplier_shape != (output_shape[3],)
            or shift_def is None
        ):
            continue
        shift_values, parsed_shift_type = shift_def
        if parsed_shift_type != shift_type or len(shift_values) != output_shape[3]:
            continue
        filter_bytes = dense_bytes_definition(
            lines, depthwise.filter_value, depthwise_index
        )
        bias_bytes = dense_bytes_definition(lines, bias_value, req_start)
        multiplier_bytes = dense_bytes_definition(lines, multiplier, req_start)
        if filter_bytes is None or bias_bytes is None or multiplier_bytes is None:
            continue
        cmsis_shifts = [31 - value for value in shift_values]
        stride_h, stride_w = depthwise.stride
        filter_h, filter_w = filter_shape[:2]
        total_pad_h = max(
            0,
            (output_shape[1] - 1) * stride_h + filter_h - input_shape[1],
        )
        total_pad_w = max(
            0,
            (output_shape[2] - 1) * stride_w + filter_w - input_shape[2],
        )
        if total_pad_h % 2 or total_pad_w % 2:
            continue
        pad_h, pad_w = total_pad_h // 2, total_pad_w // 2
        if kernel_class == "mve":
            # MVE never uses the 3x3 kernel; all depthwise go through
            # arm_depthwise_conv_s8_opt with a CH_IN_BLOCK_MVE (=124) buffer.
            scratch_size = 4 * CH_IN_BLOCK_MVE * filter_h * filter_w
        else:
            uses_3x3 = filter_shape[:2] == (3, 3) and pad_h <= 1 and pad_w <= 1
            scratch_size = 0 if uses_3x3 else 2 * input_shape[3] * filter_h * filter_w
        scratch_len = max(1, scratch_size)
        config_values = [
            input_shape[0], input_shape[1], input_shape[2], input_shape[3],
            output_shape[1], output_shape[2], output_shape[3], filter_h,
            filter_w, stride_h, stride_w, 1, 1, pad_h, pad_w,
            -depthwise.input_zero_point, output_offset, activation_min,
            activation_max, 1, scratch_size,
        ]
        prefix = f"cmsis_nn_{rewritten}"
        indent = re.match(r"\s*", lines[req_start]).group(0)
        symbols = ", ".join(f"s{index}" for index in range(14))
        maps = [
            f"affine_map<()[{symbols}] -> (s4, s5, s6, s7)>",
            f"affine_map<()[{symbols}] -> (s8, s9, s10, s11)>",
            *[f"affine_map<()[{symbols}] -> (s3)>" for _ in range(3)],
            f"affine_map<()[{symbols}] -> (s12)>",
            f"affine_map<()[{symbols}] -> (s13)>",
            f"affine_map<()[{symbols}] -> (s0, s1, s2, s3)>",
        ]
        generated = [
            f"{indent}%{prefix}_shift = arith.constant dense<{cmsis_shifts}> : tensor<{len(cmsis_shifts)}xi32>",
            f"{indent}%{prefix}_config = arith.constant dense<{config_values}> : tensor<21xi32>",
            f"{indent}%{prefix}_scratch = tensor.empty() : tensor<{scratch_len}xi8>",
            f'{indent}{result} = iree_linalg_ext.custom_op {{indexing_maps = [{", ".join(maps)}], '
            f'iterator_types = []}} attributes {{'
            f'iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"oneliner_cmsis_nn_depthwise_conv_s8", bitcode>, '
            f'hal.executable.objects = [#hal.executable.object<{{path = "{BITCODE_PATH}"}}>]}} '
            f'ins({depthwise.input_value}, {depthwise.filter_value}, {bias_value}, {multiplier}, '
            f'%{prefix}_shift, %{prefix}_scratch, %{prefix}_config : '
            f'{depthwise.input_type}, {depthwise.filter_type}, {bias_type}, {multiplier_type}, '
            f'tensor<{len(cmsis_shifts)}xi32>, tensor<{scratch_len}xi8>, tensor<21xi32>) '
            f'outs({output_init} : {output_type}) {{',
            f'{indent}^bb0(%input: tensor<?x?x?x?xi8>, %filter: tensor<?x?x?x?xi8>, '
            f'%bias: tensor<?xi32>, %multiplier: tensor<?xi32>, %shift: tensor<?xi32>, '
            f'%scratch: tensor<?xi8>, %config: tensor<?xi32>, '
            f'%out: tensor<?x?x?x?xi8>):',
            f"{indent}  iree_linalg_ext.yield %out : tensor<?x?x?x?xi8>",
            f"{indent}}} -> {output_type}",
        ]
        replacements.append((req_start, req_end, generated))
        rewritten += 1

    for pool_index, line in enumerate(lines):
        pool = find_pool(line)
        if pool is None:
            continue
        input_shape = parse_shape(pool.input_type)
        window_shape = parse_shape(pool.window_type)
        output_shape = parse_shape(pool.output_type)
        if (
            input_shape is None
            or len(input_shape) != 4
            or window_shape is None
            or len(window_shape) != 2
            or output_shape is None
            or len(output_shape) != 4
            or input_shape[0] != output_shape[0]
            or input_shape[3] != output_shape[3]
            or pool.dilation != (1, 1)
            or fill_scalar(lines, pool.init_value, pool_index, constants) != -128
        ):
            continue
        stride_h, stride_w = pool.stride
        window_h, window_w = window_shape
        total_pad_h = max(
            0, (output_shape[1] - 1) * stride_h + window_h - input_shape[1]
        )
        total_pad_w = max(
            0, (output_shape[2] - 1) * stride_w + window_w - input_shape[2]
        )
        if total_pad_h % 2 or total_pad_w % 2:
            continue
        pad_h, pad_w = total_pad_h // 2, total_pad_w // 2
        prefix = f"cmsis_nn_{rewritten}"
        config_values = [
            input_shape[0], input_shape[1], input_shape[2], input_shape[3],
            output_shape[1], output_shape[2], window_h, window_w,
            stride_h, stride_w, pad_h, pad_w, -128, 127,
        ]
        indent = re.match(r"\s*", line).group(0)
        symbols = ", ".join(f"s{index}" for index in range(7))
        maps = [
            f"affine_map<()[{symbols}] -> (s0, s4, s5, s3)>",
            f"affine_map<()[{symbols}] -> (s6)>",
            f"affine_map<()[{symbols}] -> (s0, s1, s2, s3)>",
        ]
        generated = [
            f"{indent}%{prefix}_config = arith.constant dense<{config_values}> : tensor<14xi32>",
            f'{indent}{pool.result} = iree_linalg_ext.custom_op {{indexing_maps = [{", ".join(maps)}], '
            f'iterator_types = []}} attributes {{'
            f'iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"oneliner_cmsis_nn_max_pool_s8", bitcode>, '
            f'hal.executable.objects = [#hal.executable.object<{{path = "{BITCODE_PATH}"}}>]}} '
            f'ins({pool.input_value}, %{prefix}_config : {pool.input_type}, tensor<14xi32>) '
            f'outs({pool.init_value} : {pool.output_type}) {{',
            f'{indent}^bb0(%input: tensor<?x?x?x?xi8>, %config: tensor<?xi32>, '
            f'%out: tensor<?x?x?x?xi8>):',
            f"{indent}  iree_linalg_ext.yield %out : tensor<?x?x?x?xi8>",
            f"{indent}}} -> {pool.output_type}",
        ]
        replacements.append((pool_index, pool_index + 1, generated))
        rewritten += 1

    for pool_index, line in enumerate(lines):
        pool = find_avgpool(line)
        if pool is None:
            continue
        input_shape = parse_shape(pool.input_type)
        window_shape = parse_shape(pool.window_type)
        output_shape = parse_shape(pool.output_type)
        accumulator_shape = parse_shape(pool.accumulator_type)
        requant = avgpool_requant_match(
            lines, pool, pool_index, constants
        )
        if (
            input_shape is None
            or len(input_shape) != 4
            or window_shape is None
            or len(window_shape) != 2
            or output_shape is None
            or len(output_shape) != 4
            or accumulator_shape is None
            or len(accumulator_shape) != 4
            or input_shape[0] != 1
            or output_shape[0] != 1
            or input_shape[3] != output_shape[3]
            or accumulator_shape != output_shape
            or pool.dilation != (1, 1)
            or requant is None
        ):
            continue
        req_start, req_end, result, output_init, output_type, multiplier, \
            activation_min, activation_max = requant
        stride_h, stride_w = pool.stride
        kernel_h, kernel_w = window_shape
        total_pad_h = max(
            0, (output_shape[1] - 1) * stride_h + kernel_h - input_shape[1]
        )
        total_pad_w = max(
            0, (output_shape[2] - 1) * stride_w + kernel_w - input_shape[2]
        )
        if total_pad_h or total_pad_w:
            continue
        prefix = f"cmsis_nn_{rewritten}"
        config_values = [
            input_shape[0], input_shape[1], input_shape[2], input_shape[3],
            output_shape[1], output_shape[2], kernel_h, kernel_w,
            stride_h, stride_w, 0, 0, activation_min, activation_max,
            multiplier,
        ]
        indent = re.match(r"\s*", lines[pool_index]).group(0)
        symbols = ", ".join(f"s{index}" for index in range(7))
        maps = [
            f"affine_map<()[{symbols}] -> (s0, s4, s5, s3)>",
            f"affine_map<()[{symbols}] -> (s6)>",
            f"affine_map<()[{symbols}] -> (s0, s1, s2, s3)>",
        ]
        generated = [
            f"{indent}%{prefix}_config = arith.constant dense<{config_values}> : tensor<15xi32>",
            f'{indent}{result} = iree_linalg_ext.custom_op {{indexing_maps = [{", ".join(maps)}], '
            f'iterator_types = []}} attributes {{'
            f'iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"oneliner_cmsis_nn_avg_pool_s8", bitcode>, '
            f'hal.executable.objects = [#hal.executable.object<{{path = "{BITCODE_PATH}"}}>]}} '
            f'ins({pool.input_value}, %{prefix}_config : {pool.input_type}, tensor<15xi32>) '
            f'outs({output_init} : {output_type}) {{',
            f'{indent}^bb0(%input: tensor<?x?x?x?xi8>, %config: tensor<?xi32>, '
            f'%out: tensor<?x?x?x?xi8>):',
            f"{indent}  iree_linalg_ext.yield %out : tensor<?x?x?x?xi8>",
            f"{indent}}} -> {output_type}",
        ]
        replacements.append((pool_index, req_end, generated))
        rewritten += 1

    for start, end, generated in sorted(
        replacements, key=lambda replacement: replacement[0], reverse=True
    ):
        lines[start:end] = generated
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), rewritten


def finalize_configured(text: str) -> tuple[str, int]:
    finalized = 0
    lines = []
    for line in text.splitlines():
        if "iree_codegen.ukernel.generic" in line:
            for name in UKERNEL_NAMES:
                descriptor = (
                    '{iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<'
                    f'"{name}", bitcode>}} '
                )
                if descriptor not in line:
                    continue
                line = line.replace(descriptor, "", 1)
                line = line.replace(
                    " strided_dims(",
                    " fn_def_attrs {hal.import.bitcode = true} strided_dims(",
                    1,
                )
                finalized += 1
                break
        lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), finalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-match", action="store_true")
    parser.add_argument("--finalize-configured", action="store_true")
    parser.add_argument(
        "--kernel-class",
        choices=("dsp", "mve"),
        default="dsp",
        help="CMSIS-NN kernel family of the target (affects scratch sizes)",
    )
    args = parser.parse_args()
    if args.finalize_configured:
        output, count = finalize_configured(sys.stdin.read())
    else:
        output, count = rewrite(sys.stdin.read(), args.kernel_class)
    if args.require_match and count == 0:
        print("no supported CMSIS-NN int8 operation found", file=sys.stderr)
        return 1
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
