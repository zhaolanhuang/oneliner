#!/usr/bin/env python3
"""
Extract IREE Stream command execution blocks and render Rust call flows.

The command graph is parsed with IREE's MLIR Python bindings. Text tokenization
is limited to opaque composite constant attributes whose payloads are not
exposed by the bindings.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

from iree.compiler import ir
from iree.compiler.dialects import cf, hal, stream, util


RUST_KEYWORDS = {
    "as", "async", "await", "break", "const", "continue", "crate", "dyn",
    "else", "enum", "extern", "false", "fn", "for", "gen", "if", "impl",
    "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
    "return", "self", "Self", "static", "struct", "super", "trait", "true",
    "try", "type", "unsafe", "use", "where", "while", "yield",
}


@dataclasses.dataclass
class ConstantBlob:
    name: str
    size: int
    data: bytes
    source: str


@dataclasses.dataclass
class ResourceBinding:
    arg: str
    source: str
    kind: str
    size_expr: str
    size: int | None
    role: str
    constant_name: str | None = None


@dataclasses.dataclass
class TensorRange:
    access: str
    arg: str
    kind: str
    tensor_name: str
    offset_expr: str
    offset: int | None
    length_expr: str
    length: int | None


@dataclasses.dataclass
class DispatchCall:
    kind: str
    callee: str
    executable: str
    function: str
    ordinal: int
    params: list[str]
    param_values: list[int | None]
    ranges: list[TensorRange]
    workload: tuple[int | None, ...]


@dataclasses.dataclass
class FillCommand:
    kind: str
    value_expr: str
    value: int | None
    value_type: str
    target: TensorRange


@dataclasses.dataclass
class CopyCommand:
    kind: str
    source: TensorRange
    target: TensorRange


@dataclasses.dataclass
class ConcurrentCommand:
    kind: str
    commands: list[Any]


@dataclasses.dataclass
class CmdExecute:
    name: str
    result: str | None
    line_no: int | None
    resources: list[ResourceBinding]
    commands: list[Any]


class StreamExtractionError(RuntimeError):
    pass


def rust_ident(raw: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", raw).strip("_").lower()
    ident = re.sub(r"_+", "_", ident)
    if not ident:
        ident = "value"
    if ident[0].isdigit():
        ident = f"v_{ident}"
    if ident in RUST_KEYWORDS:
        ident += "_"
    return ident


def const_ident(raw: str) -> str:
    return rust_ident(raw).upper()


def find_matching(text: str, start: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return index
    raise StreamExtractionError(f"unbalanced {open_ch}{close_ch}")


def split_balanced_items(text: str, separator: str = ",") -> list[str]:
    items: list[str] = []
    start = 0
    depth_angle = depth_square = depth_round = depth_brace = 0
    in_string = False
    escaped = False
    for index, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
        elif ch == "(":
            depth_round += 1
        elif ch == ")":
            depth_round -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        elif (
            ch == separator
            and depth_angle == 0
            and depth_square == 0
            and depth_round == 0
            and depth_brace == 0
        ):
            item = text[start:index].strip()
            if item:
                items.append(item)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items


def parse_tensor_type(type_text: str) -> tuple[int, str] | None:
    match = re.search(r"(?:tensor|vector)<(?P<body>[^>]+)>", type_text.strip())
    if not match:
        return None
    parts = match.group("body").split("x")
    if not parts:
        return None
    element_type = parts[-1]
    count = 1
    for dim in parts[:-1]:
        if dim == "?":
            return None
        count *= int(dim)
    return count, element_type


def element_width(element_type: str) -> int:
    if element_type in {"i1", "i8", "ui8"}:
        return 1
    if element_type in {"i16", "ui16", "f16", "bf16"}:
        return 2
    if element_type in {"i32", "ui32", "f32"}:
        return 4
    if element_type in {"i64", "ui64", "f64"}:
        return 8
    raise StreamExtractionError(f"unsupported dense element type: {element_type}")


def pack_scalar(value: str, element_type: str) -> bytes:
    if element_type.startswith("i") or element_type.startswith("ui"):
        bits = element_width(element_type) * 8
        return (int(value) & ((1 << bits) - 1)).to_bytes(bits // 8, "little", signed=False)
    if element_type == "f32":
        return struct.pack("<f", float(value))
    if element_type == "f64":
        return struct.pack("<d", float(value))
    raise StreamExtractionError(f"unsupported dense element type: {element_type}")


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)


def dense_payload_to_bytes(payload: str, type_text: str) -> bytes | None:
    type_info = parse_tensor_type(type_text)
    if type_info is None:
        return None
    count, element_type = type_info
    payload = payload.strip()

    if payload.startswith('"0x') and payload.endswith('"'):
        return bytes.fromhex(payload[3:-1])

    tokens = numeric_tokens(payload)
    if len(tokens) == 1:
        return pack_scalar(tokens[0], element_type) * count
    if len(tokens) == count:
        return b"".join(pack_scalar(token, element_type) for token in tokens)
    return None


def dense_attr_to_bytes(item: str, dense_resources: dict[str, bytes]) -> bytes | None:
    resource_match = re.search(r"dense_resource<(?P<alias>[\w.$-]+)>", item)
    if resource_match:
        colon_pos = item.find(":", resource_match.end())
        if colon_pos < 0:
            return None
        type_info = parse_tensor_type(item[colon_pos + 1 :])
        if type_info is None:
            return None
        count, element_type = type_info
        data = dense_resources.get(resource_match.group("alias"))
        if data is None or len(data) != count * element_width(element_type):
            return None
        return data

    dense_pos = item.find("dense<")
    if dense_pos < 0:
        return None
    open_pos = dense_pos + len("dense")
    close_pos = find_matching(item, open_pos, "<", ">")
    payload = item[open_pos + 1 : close_pos]
    colon_pos = item.find(":", close_pos)
    if colon_pos < 0:
        return None
    return dense_payload_to_bytes(payload, item[colon_pos + 1 :].strip())


def parse_dense_resources(text: str) -> dict[str, bytes]:
    resources: dict[str, bytes] = {}
    pattern = re.compile(
        r'^[ \t]*(?P<alias>[\w.$-]+)[ \t]*:[ \t]*'
        r'"0x(?P<data>[0-9A-Fa-f]*)"',
        re.MULTILINE,
        )
    for match in pattern.finditer(text):
        encoded = bytes.fromhex(match.group("data"))
        if len(encoded) >= 4:
            # MLIR prefixes resource blobs with a little-endian alignment field.
            resources[match.group("alias")] = encoded[4:]
    return resources


def composite_to_bytes(
    text: str, dense_resources: dict[str, bytes]
) -> tuple[int | None, bytes | None]:
    marker = text.find("#util.composite")
    if marker < 0:
        return None, None
    open_pos = text.find("<", marker)
    if open_pos < 0:
        return None, None
    close_pos = find_matching(text, open_pos, "<", ">")
    body = text[open_pos + 1 : close_pos]
    header, _, rest = body.partition("[")
    declared_match = re.search(r"(?P<size>\d+)xi8", header)
    declared_size = int(declared_match.group("size")) if declared_match else None
    if not rest:
        return declared_size, None
    list_open = body.find("[")
    list_close = find_matching(body, list_open, "[", "]")
    data = bytearray()
    for item in split_balanced_items(body[list_open + 1 : list_close]):
        payload = dense_attr_to_bytes(item, dense_resources)
        if payload is None:
            return declared_size, None
        data.extend(payload)
    if declared_size is not None and len(data) != declared_size:
        return declared_size, None
    return declared_size, bytes(data)


def parse_composite_constants(text: str) -> dict[str, ConstantBlob]:
    constants: dict[str, ConstantBlob] = {}
    dense_resources = parse_dense_resources(text)
    pattern = re.compile(r"(?P<alias>#[\w.$-]+)\s*=\s*#util\.composite")
    for match in pattern.finditer(text):
        marker = text.find("#util.composite", match.start())
        open_pos = text.find("<", marker)
        close_pos = find_matching(text, open_pos, "<", ">")
        composite_text = text[marker : close_pos + 1]
        declared_size, data = composite_to_bytes(composite_text, dense_resources)
        if data is None:
            continue
        name = f"constant_{rust_ident(match.group('alias'))}"
        constants[name] = ConstantBlob(
            name=name,
            size=declared_size if declared_size is not None else len(data),
            data=data,
            source=match.group("alias"),
        )
    return constants


def parse_composite_attribute_names(
    text: str, constants: dict[str, ConstantBlob]
) -> dict[str, str]:
    names: dict[str, str] = {}
    pattern = re.compile(r"(?P<alias>#[\w.$-]+)\s*=\s*#util\.composite")
    for match in pattern.finditer(text):
        constant_name = f"constant_{rust_ident(match.group('alias'))}"
        if constant_name not in constants:
            continue
        marker = text.find("#util.composite", match.start())
        open_pos = text.find("<", marker)
        close_pos = find_matching(text, open_pos, "<", ">")
        # Parsing in the module's context uniquifies already-registered
        # dense_resource names, so canonicalize each alias in a fresh context.
        attribute = ir.Attribute.parse(
            text[marker : close_pos + 1], context=ir.Context()
        )
        names[str(attribute)] = constant_name
    return names


def binding_name(binding: ResourceBinding) -> str:
    if binding.role == "input":
        return f"input_{rust_ident(binding.arg)}"
    if binding.role == "output":
        return f"output_{rust_ident(binding.arg)}"
    if binding.role == "inout":
        return f"inout_{rust_ident(binding.arg)}"
    if binding.role == "temporary":
        return f"temp_{rust_ident(binding.arg)}"
    if binding.role == "constant":
        return f"const_{rust_ident(binding.arg)}"
    return f"{rust_ident(binding.role)}_{rust_ident(binding.arg)}"


def command_ranges(command: Any) -> list[TensorRange]:
    if isinstance(command, DispatchCall):
        return command.ranges
    if isinstance(command, FillCommand):
        return [command.target]
    if isinstance(command, CopyCommand):
        return [command.source, command.target]
    if isinstance(command, ConcurrentCommand):
        ranges: list[TensorRange] = []
        for child in command.commands:
            ranges.extend(command_ranges(child))
        return ranges
    return []


def infer_external_roles(bindings: list[ResourceBinding], commands: list[Any]) -> None:
    access_by_arg: dict[str, set[str]] = {}
    for command in commands:
        for item in command_ranges(command):
            access_by_arg.setdefault(item.arg, set()).add(item.access)

    for binding in bindings:
        if binding.kind != "external":
            continue
        accesses = access_by_arg.get(binding.arg, set())
        has_read = bool(accesses & {"ro", "rw"})
        has_write = bool(accesses & {"wo", "rw"})
        if has_read and has_write:
            binding.role = "inout"
        elif binding.role != "external":
            continue
        elif has_write:
            binding.role = "output"
        elif has_read:
            binding.role = "input"


def apply_tensor_names(command: Any, bindings_by_arg: dict[str, ResourceBinding]) -> None:
    for item in command_ranges(command):
        binding = bindings_by_arg.get(item.arg)
        if binding is None:
            raise StreamExtractionError(f"resource binding for {item.arg} was not found")
        item.tensor_name = binding_name(binding)


def value_name(value: ir.Value) -> str:
    return value.get_name()


def resource_kind(value: ir.Value) -> str:
    type_name = str(value.type)
    prefix = "!stream.resource<"
    if not type_name.startswith(prefix) or not type_name.endswith(">"):
        raise StreamExtractionError(
            f"expected a stream resource, got {type_name} for {value_name(value)}"
        )
    return type_name[len(prefix) : -1]


def resolve_ir_int(value: ir.Value) -> int | None:
    if not isinstance(value, ir.OpResult):
        return None
    owner = value.owner
    if owner.name != "arith.constant" or "value" not in owner.attributes:
        return None
    try:
        return ir.IntegerAttr(owner.attributes["value"]).value
    except ValueError:
        return None


def source_line(operation: ir.Operation) -> int | None:
    location = operation.location
    return location.start_line if isinstance(location, ir.FileLineColLoc) else None


@dataclasses.dataclass(frozen=True)
class WorkloadDimension:
    """One export count result before dispatch workload substitution."""

    constant: int | None = None
    argument_index: int | None = None

    def resolve(self, arguments: list[int | None]) -> int | None:
        """Resolves a constant or a count-block argument at one call site."""
        if self.constant is not None:
            return self.constant
        if self.argument_index is None or self.argument_index >= len(arguments):
            return None
        return arguments[self.argument_index]


@dataclasses.dataclass(frozen=True)
class ExecutableExport:
    symbol_path: tuple[str, ...]
    ordinal: int
    local_ordinal: int
    workload: tuple[WorkloadDimension, ...]
    workload_argument_count: int


class StructuredStreamParser:
    def __init__(self, text: str):
        self.context = ir.Context()
        try:
            self.module = ir.Module.parse(text, context=self.context)
        except ir.MLIRError as exc:
            raise StreamExtractionError(f"invalid IREE MLIR: {exc}") from exc
        self.constant_blobs = parse_composite_constants(text)
        self.constant_by_attribute = parse_composite_attribute_names(
            text, self.constant_blobs
        )
        self.block_argument_sources = self._find_block_argument_sources()
        self.constant_by_value = self._find_constant_values()
        self.exports = self._find_exports()

    def _find_block_argument_sources(self) -> dict[ir.Value, list[ir.Value]]:
        sources: dict[ir.Value, list[ir.Value]] = {}

        def add_branch(operands, successor: ir.Block) -> None:
            for source, argument in zip(operands, successor.arguments, strict=True):
                sources.setdefault(argument, []).append(source)

        for operation in ir.get_ops_of_type(self.module, cf.BranchOp):
            add_branch(operation.destOperands, operation.successors[0])
        for operation in ir.get_ops_of_type(self.module, cf.CondBranchOp):
            add_branch(operation.trueDestOperands, operation.successors[0])
            add_branch(operation.falseDestOperands, operation.successors[1])
        return sources

    def _find_constant_values(self) -> dict[ir.Value, str]:
        by_value: dict[ir.Value, str] = {}

        def assign(value: ir.Value, constant_name: str) -> bool:
            existing = by_value.get(value)
            if existing is not None and existing != constant_name:
                raise StreamExtractionError(
                    f"ambiguous constant provenance for {value_name(value)}"
                )
            if existing is not None:
                return False
            by_value[value] = constant_name
            return True

        for operation in ir.get_ops_of_type(self.module, util.BufferConstantOp):
            constant_name = self.constant_by_attribute.get(str(operation.value))
            if constant_name is not None:
                assign(operation.result, constant_name)

        for operation in ir.get_ops_of_type(self.module, stream.ResourceTryMapOp):
            constant_name = by_value.get(operation.source)
            if constant_name is not None:
                assign(operation.result, constant_name)

        # File loads, awaits, and control flow preserve resource identity.
        # Iterate to account for joins and chains in generated initializers.
        changed = True
        while changed:
            changed = False
            for operation in ir.get_ops_of_type(self.module, stream.FileConstantOp):
                constant_name = by_value.get(operation.source)
                if constant_name is not None:
                    changed |= assign(operation.result, constant_name)
            for operation in ir.get_ops_of_type(self.module, stream.FileReadOp):
                constant_name = by_value.get(operation.source)
                if constant_name is not None:
                    changed |= assign(operation.target, constant_name)
            for argument, sources in self.block_argument_sources.items():
                constant_names = [by_value.get(source) for source in sources]
                if not constant_names or any(name is None for name in constant_names):
                    continue
                first_name = constant_names[0]
                if any(name != first_name for name in constant_names[1:]):
                    raise StreamExtractionError(
                        f"ambiguous constant provenance for {value_name(argument)}"
                    )
                changed |= assign(argument, first_name)
            for operation in ir.get_ops_of_type(self.module, stream.TimepointAwaitOp):
                for source, result in zip(
                    operation.resource_operands, operation.results_, strict=True
                ):
                    constant_name = by_value.get(source)
                    if constant_name is not None:
                        changed |= assign(result, constant_name)

        stores_by_global: dict[str, list[ir.Value]] = {}
        for operation in ir.get_ops_of_type(self.module, util.GlobalStoreOp):
            stores_by_global.setdefault(operation.global_.value, []).append(operation.value)
        constant_by_global: dict[str, str] = {}
        for global_name, values in stores_by_global.items():
            constant_names = [by_value.get(value) for value in values]
            if not constant_names or any(name is None for name in constant_names):
                continue
            first_name = constant_names[0]
            if any(name != first_name for name in constant_names[1:]):
                raise StreamExtractionError(
                    f"ambiguous constant stores to @{global_name}"
                )
            constant_by_global[global_name] = first_name
        for operation in ir.get_ops_of_type(self.module, util.GlobalLoadOp):
            constant_name = constant_by_global.get(operation.global_.value)
            if constant_name is not None:
                assign(operation.result, constant_name)
        return by_value

    def _find_exports(self) -> dict[tuple[str, ...], ExecutableExport]:
        pending: list[
            tuple[
                tuple[str, ...],
                int,
                tuple[WorkloadDimension, ...],
                int,
            ]
        ] = []
        seen_paths: set[tuple[str, ...]] = set()
        for operation in ir.get_ops_of_type(self.module, hal.ExecutableExportOp):
            path = self._export_path(operation)
            if path in seen_paths:
                raise StreamExtractionError(
                    f"duplicate executable export {'::'.join(path)}"
                )
            seen_paths.add(path)
            if operation.ordinal is None:
                raise StreamExtractionError(
                    f"executable export {'::'.join(path)} has no ordinal"
                )
            if len(operation.workgroup_count.blocks) != 1:
                raise StreamExtractionError(
                    f"executable export {'::'.join(path)} must have one count block"
                )
            block = operation.workgroup_count.blocks[0]
            block_operations = list(block.operations)
            if not block_operations:
                raise StreamExtractionError(
                    f"executable export {'::'.join(path)} has an empty count block"
                )
            terminator = block_operations[-1]
            if terminator.operation.name != "hal.return":
                raise StreamExtractionError(
                    f"executable export {'::'.join(path)} has no hal.return workload"
                )
            block_arguments = list(block.arguments)
            if not block_arguments or str(block_arguments[0].type) != "!hal.device":
                raise StreamExtractionError(
                    f"executable export {'::'.join(path)} count block has no device argument"
                )
            workload_arguments = block_arguments[1:]

            def parse_dimension(value: ir.Value) -> WorkloadDimension:
                constant = resolve_ir_int(value)
                if constant is not None:
                    return WorkloadDimension(constant=constant)
                for index, argument in enumerate(workload_arguments):
                    if value == argument:
                        return WorkloadDimension(argument_index=index)
                # Keep unsupported count expressions explicit so the call-site
                # diagnostic remains the existing "not static 3D" error.
                return WorkloadDimension()

            pending.append(
                (
                    path,
                    operation.ordinal.value,
                    tuple(parse_dimension(value) for value in terminator.operands),
                    len(workload_arguments),
                )
            )

        # IREE's static linked library concatenates each executable's local
        # export table in module order. The runtime consumes this flattened
        # ordinal, not the variant-local hal.executable.export ordinal.
        groups: dict[
            tuple[str, ...],
            list[
                tuple[
                    tuple[str, ...],
                    int,
                    tuple[WorkloadDimension, ...],
                    int,
                ]
            ],
        ] = {}
        for item in pending:
            path = item[0]
            if len(path) < 3:
                raise StreamExtractionError(
                    f"executable export {'::'.join(path)} has no variant scope"
                )
            groups.setdefault(path[:-2], []).append(item)

        exports: dict[tuple[str, ...], ExecutableExport] = {}
        base_ordinal = 0
        for executable_path, items in groups.items():
            local_ordinals = sorted(item[1] for item in items)
            if local_ordinals != list(range(len(items))):
                raise StreamExtractionError(
                    "executable "
                    f"{'::'.join(executable_path)} has non-contiguous or duplicate ordinals"
                )
            for path, local_ordinal, workload, workload_argument_count in items:
                exports[path] = ExecutableExport(
                    symbol_path=path,
                    ordinal=base_ordinal + local_ordinal,
                    local_ordinal=local_ordinal,
                    workload=workload,
                    workload_argument_count=workload_argument_count,
                )
            base_ordinal += len(items)
        return exports

    @staticmethod
    def _export_path(operation: hal.ExecutableExportOp) -> tuple[str, ...]:
        path = [operation.sym_name.value]
        parent = operation.operation.parent
        while parent is not None:
            attributes = parent.attributes
            if "sym_name" in attributes:
                path.append(ir.StringAttr(attributes["sym_name"]).value)
            parent = parent.parent
        path.reverse()
        return tuple(path)

    @staticmethod
    def _module_scope(operation: ir.Operation) -> tuple[str, ...]:
        scope: list[str] = []
        parent = operation.parent
        while parent is not None:
            if parent.name == "builtin.module" and "sym_name" in parent.attributes:
                scope.append(ir.StringAttr(parent.attributes["sym_name"]).value)
            parent = parent.parent
        scope.reverse()
        return tuple(scope)

    def _lookup_export(
        self, symbol_path: tuple[str, ...], operation: ir.Operation
    ) -> ExecutableExport:
        scoped_path = self._module_scope(operation) + symbol_path
        export = self.exports.get(scoped_path)
        if export is not None:
            return export
        rendered = "::".join(f"@{part}" for part in symbol_path)
        raise StreamExtractionError(f"executable export {rendered} was not found")

    def _source_role(
        self, source: ir.Value, kind: str, seen: set[ir.Value] | None = None
    ) -> tuple[str, str | None]:
        constant_name = self.constant_by_value.get(source)
        if constant_name is not None:
            return "constant", constant_name
        if kind == "constant":
            return "constant", None
        if kind == "transient":
            return "temporary", None
        if isinstance(source, ir.BlockArgument):
            incoming = self.block_argument_sources.get(source)
            if incoming:
                seen = set() if seen is None else seen
                if source in seen:
                    raise StreamExtractionError(
                        f"cyclic resource provenance for {value_name(source)}"
                    )
                seen.add(source)
                roles = [self._source_role(value, kind, seen) for value in incoming]
                seen.remove(source)
                if any(role != roles[0] for role in roles[1:]):
                    raise StreamExtractionError(
                        f"ambiguous resource provenance for {value_name(source)}"
                    )
                return roles[0]
        if not isinstance(source, ir.OpResult):
            return kind, None
        owner = source.owner
        if owner.name == "stream.tensor.import":
            return "input", None
        if owner.name == "stream.resource.alloca":
            return "output" if kind == "external" else "temporary", None
        if owner.name == "stream.timepoint.await":
            await_op = owner.opview
            source_index = source.result_number
            if source_index < len(await_op.resource_operands):
                return self._source_role(
                    await_op.resource_operands[source_index], kind, seen
                )
        return kind, None

    def _parse_execute(self, operation: stream.CmdExecuteOp, index: int) -> CmdExecute:
        if len(operation.body.blocks) != 1:
            raise StreamExtractionError("stream.cmd.execute must contain one block")
        block = operation.body.blocks[0]
        sources = list(operation.resource_operands)
        sizes = list(operation.resource_operand_sizes)
        arguments = list(block.arguments)
        if not (len(sources) == len(sizes) == len(arguments)):
            raise StreamExtractionError(
                "stream.cmd.execute resource operands, sizes, and block arguments differ"
            )

        bindings: list[ResourceBinding] = []
        bindings_by_arg: dict[str, ResourceBinding] = {}
        for source, size, argument in zip(sources, sizes, arguments, strict=True):
            kind = resource_kind(argument)
            role, constant_name = self._source_role(source, kind)
            resolved_size = resolve_ir_int(size)
            if resolved_size is None:
                raise StreamExtractionError(
                    f"dynamic resource size {value_name(size)} is unsupported"
                )
            binding = ResourceBinding(
                arg=value_name(argument),
                source=value_name(source),
                kind=kind,
                size_expr=value_name(size),
                size=resolved_size,
                role=role,
                constant_name=constant_name,
            )
            bindings.append(binding)
            bindings_by_arg[binding.arg] = binding

        commands = self._parse_command_block(block)
        infer_external_roles(bindings, commands)
        unsupported_roles = [
            binding
            for binding in bindings
            if binding.role not in {"constant", "temporary", "input", "output", "inout"}
        ]
        if unsupported_roles:
            rendered = ", ".join(
                f"{binding.arg} ({binding.role})" for binding in unsupported_roles
            )
            raise StreamExtractionError(f"unsupported resource roles: {rendered}")
        for command in commands:
            apply_tensor_names(command, bindings_by_arg)
        return CmdExecute(
            name=f"cmd_execute_{index}",
            result=value_name(operation.result_timepoint),
            line_no=source_line(operation.operation),
            resources=bindings,
            commands=commands,
        )

    def _parse_command_block(self, block: ir.Block) -> list[Any]:
        commands: list[Any] = []
        for operation in block.operations:
            name = operation.operation.name
            if name == "stream.yield":
                continue
            if name == "stream.cmd.dispatch":
                commands.append(self._parse_dispatch(operation))
            elif name == "stream.cmd.fill":
                commands.append(self._parse_fill(operation))
            elif name == "stream.cmd.copy":
                commands.append(self._parse_copy(operation))
            elif name == "stream.cmd.concurrent":
                if len(operation.body.blocks) != 1:
                    raise StreamExtractionError(
                        "stream.cmd.concurrent must contain one block"
                    )
                commands.append(
                    ConcurrentCommand(
                        kind="concurrent",
                        commands=self._parse_command_block(operation.body.blocks[0]),
                    )
                )
            elif name.startswith("stream.cmd."):
                raise StreamExtractionError(f"unsupported command operation {name}")
            else:
                raise StreamExtractionError(
                    f"unexpected operation {name} in stream command block"
                )
        return commands

    def _parse_dispatch(self, operation: stream.CmdDispatchOp) -> DispatchCall:
        if len(operation.entry_points) != 1:
            raise StreamExtractionError(
                "dispatches with multiple executable entry points are unsupported"
            )
        symbol_path = tuple(operation.entry_points[0].value)
        export = self._lookup_export(symbol_path, operation.operation)
        workload_operands = list(operation.workload)
        if len(workload_operands) != export.workload_argument_count:
            raise StreamExtractionError(
                f"dispatch workload arity for {'::'.join(symbol_path)} does not "
                "match its executable export"
            )
        workload_arguments = [resolve_ir_int(value) for value in workload_operands]
        resolved_workload = tuple(
            dimension.resolve(workload_arguments) for dimension in export.workload
        )
        if len(resolved_workload) != 3 or any(
            value is None for value in resolved_workload
        ):
            raise StreamExtractionError(
                f"dispatch workload for {'::'.join(symbol_path)} is not static 3D"
            )
        resources = list(operation.resources)
        offsets = list(operation.resource_offsets)
        lengths = list(operation.resource_lengths)
        accesses = list(operation.resource_accesses)
        if not (len(resources) == len(offsets) == len(lengths) == len(accesses)):
            raise StreamExtractionError("dispatch resource ranges have inconsistent lengths")

        ranges: list[TensorRange] = []
        access_names = {1: "ro", 2: "wo", 3: "rw"}
        for resource, offset, length, access in zip(
            resources, offsets, lengths, accesses, strict=True
        ):
            access_value = ir.IntegerAttr(access).value
            if access_value not in access_names:
                raise StreamExtractionError(
                    f"unsupported resource access flag {access_value}"
                )
            resolved_offset = resolve_ir_int(offset)
            resolved_length = resolve_ir_int(length)
            if resolved_offset is None or resolved_length is None:
                raise StreamExtractionError(
                    "dynamic resource range "
                    f"{value_name(offset)} for {value_name(length)} is unsupported"
                )
            ranges.append(
                TensorRange(
                    access=access_names[access_value],
                    arg=value_name(resource),
                    kind=resource_kind(resource),
                    tensor_name="",
                    offset_expr=value_name(offset),
                    offset=resolved_offset,
                    length_expr=value_name(length),
                    length=resolved_length,
                )
            )

        params = [value_name(value) for value in operation.uniform_operands]
        param_values = [resolve_ir_int(value) for value in operation.uniform_operands]
        if any(value is None for value in param_values):
            raise StreamExtractionError(
                f"dynamic dispatch uniforms for {'::'.join(symbol_path)} are unsupported"
            )
        return DispatchCall(
            kind="dispatch",
            callee="::".join(f"@{part}" for part in symbol_path),
            executable=symbol_path[0],
            function=symbol_path[-1],
            ordinal=export.ordinal,
            params=params,
            param_values=param_values,
            ranges=ranges,
            workload=resolved_workload,
        )

    def _parse_fill(self, operation: stream.CmdFillOp) -> FillCommand:
        value = operation.value
        target = operation.target
        offset = operation.target_offset
        length = operation.target_length
        resolved_value = resolve_ir_int(value)
        resolved_offset = resolve_ir_int(offset)
        resolved_length = resolve_ir_int(length)
        if resolved_value is None:
            raise StreamExtractionError(
                f"dynamic fill value {value_name(value)} is unsupported"
            )
        if resolved_offset is None or resolved_length is None:
            raise StreamExtractionError(
                "dynamic fill range "
                f"{value_name(offset)} for {value_name(length)} is unsupported"
            )
        return FillCommand(
            kind="fill",
            value_expr=value_name(value),
            value=resolved_value,
            value_type=str(value.type),
            target=TensorRange(
                access="wo",
                arg=value_name(target),
                kind=resource_kind(target),
                tensor_name="",
                offset_expr=value_name(offset),
                offset=resolved_offset,
                length_expr=value_name(length),
                length=resolved_length,
            ),
        )

    def _parse_copy(self, operation: stream.CmdCopyOp) -> CopyCommand:
        source_offset = resolve_ir_int(operation.source_offset)
        target_offset = resolve_ir_int(operation.target_offset)
        length = resolve_ir_int(operation.length)
        if source_offset is None or target_offset is None or length is None:
            raise StreamExtractionError("dynamic stream copy ranges are unsupported")
        return CopyCommand(
            kind="copy",
            source=TensorRange(
                access="ro",
                arg=value_name(operation.source),
                kind=resource_kind(operation.source),
                tensor_name="",
                offset_expr=value_name(operation.source_offset),
                offset=source_offset,
                length_expr=value_name(operation.length),
                length=length,
            ),
            target=TensorRange(
                access="wo",
                arg=value_name(operation.target),
                kind=resource_kind(operation.target),
                tensor_name="",
                offset_expr=value_name(operation.target_offset),
                offset=target_offset,
                length_expr=value_name(operation.length),
                length=length,
            ),
        )

    def parse(self) -> tuple[list[CmdExecute], dict[str, ConstantBlob]]:
        executes = [
            self._parse_execute(operation, index)
            for index, operation in enumerate(
                ir.get_ops_of_type(self.module, stream.CmdExecuteOp)
            )
        ]
        return executes, self.constant_blobs


def parse_cmd_executes(text: str) -> tuple[list[CmdExecute], dict[str, ConstantBlob]]:
    return StructuredStreamParser(text).parse()


def bytes_to_rust_array(data: bytes, indent: str = "    ", per_line: int = 16) -> list[str]:
    lines: list[str] = []
    for start in range(0, len(data), per_line):
        chunk = data[start : start + per_line]
        lines.append(f"{indent}{', '.join(f'0x{byte:02X}' for byte in chunk)},")
    return lines


def render_resource_static(binding: ResourceBinding, constant_blobs: dict[str, ConstantBlob]) -> list[str]:
    name = const_ident(binding_name(binding))
    if binding.role != "constant":
        raise StreamExtractionError(f"mutable resource {binding.arg} must be stored in Workspace")
    if binding.constant_name and binding.constant_name in constant_blobs:
        blob = constant_blobs[binding.constant_name]
        lines = [f"pub static {name}: Aligned<AlignedType,[u8; {blob.size}]> = Aligned(["]
        lines.extend(bytes_to_rust_array(blob.data))
        lines.append("]);")
        return lines
    raise StreamExtractionError(f"constant {binding.arg} could not be materialized")


def render_workspace_field(binding: ResourceBinding) -> str:
    name = const_ident(binding_name(binding))
    if binding.role != "temporary":
        raise StreamExtractionError(
            f"{binding.role} resource {binding.arg} cannot be a Workspace field"
        )

    if binding.size is None:
        raise StreamExtractionError(
            f"resource {binding.arg} size expression {binding.size_expr} could not be resolved"
        )
    return f"pub(super) {name}: Aligned<AlignedType, [u8; {binding.size}]>,"


def render_workspace_initializer(binding: ResourceBinding) -> str:
    name = const_ident(binding_name(binding))
    if binding.role != "temporary":
        raise StreamExtractionError(
            f"{binding.role} resource {binding.arg} cannot initialize Workspace"
        )
    if binding.size is None:
        raise StreamExtractionError(
            f"resource {binding.arg} size expression {binding.size_expr} could not be resolved"
        )
    return f"{name}: Aligned([0; {binding.size}]),"


def render_tensor_range(
    item: TensorRange,
    workspace_names: frozenset[str] = frozenset(),
    external_roles: dict[str, str] | None = None,
) -> str:
    access = {"ro": "Ro", "wo": "Wo", "rw": "Rw"}.get(item.access, "Unknown")
    if item.offset is None or item.length is None:
        raise StreamExtractionError(
            f"unresolved tensor range {item.arg}: {item.offset_expr} for {item.length_expr}"
        )
    offset = item.offset
    length = item.length
    storage_name = const_ident(item.tensor_name)
    role = (external_roles or {}).get(item.tensor_name)
    if item.tensor_name in workspace_names:
        storage = f"(*workspace.{storage_name}).to_buffer_mut()"
    elif role == "input":
        storage = "input"
    elif role == "output":
        storage = "output"
    elif role == "inout":
        storage = "inout"
    else:
        storage = f"(*{storage_name}).to_buffer_ref()"
    return (
        f"AnyBufferRange {{ buffer: {storage}.into(), access: Access::{access}, "
        f"offset: {offset}, length: {length} }}"
    )


def render_command(
    command: Any,
    indent: str,
    workspace_names: frozenset[str] = frozenset(),
    external_roles: dict[str, str] | None = None,
) -> list[str]:
    out: list[str] = []
    if isinstance(command, DispatchCall):
        if (
            any(value is None for value in command.param_values)
            or len(command.workload) != 3
            or any(value is None for value in command.workload)
        ):
            raise StreamExtractionError(f"unresolved dispatch values for {command.callee}")
        params = ", ".join(str(value) for value in command.param_values)
        workload = ", ".join(str(value) for value in command.workload)
        out.append(f"{indent}unsafe {{")
        out.append(
            f"{indent}    try_dispatch(dispatch_fn_from_library(QUERY_FN_PTR, {command.ordinal})?, &[{params}], &[{workload}], &["
        )
        for item in command.ranges:
            out.append(
                f"{indent}        {render_tensor_range(item, workspace_names, external_roles)},"
            )
        out.append(f"{indent}    ])?;")
        out.append(f"{indent}}}")
    elif isinstance(command, FillCommand):
        if command.value is None:
            raise StreamExtractionError(f"unresolved fill value {command.value_expr}")
        rendered = render_tensor_range(command.target, workspace_names, external_roles)
        out.append(f"{indent}unsafe {{ fill({rendered}, {command.value})?; }}")
    elif isinstance(command, CopyCommand):
        raise StreamExtractionError("stream.cmd.copy is unsupported")
    elif isinstance(command, ConcurrentCommand):
        out.append(f"{indent}concurrent(|| {{")
        for child in command.commands:
            out.extend(
                render_command(child, indent + "    ", workspace_names, external_roles)
            )
        out.append(f"{indent}    Ok(())")
        out.append(f"{indent}}})?;")
    return out

def render_rust(executes: list[CmdExecute], constant_blobs: dict[str, ConstantBlob]) -> str:
    out: list[str] = [
        "// Generated by iree_stream_flow_to_rust.py",
        "// Command flow was extracted with IREE's structured MLIR bindings.",
        "",
    ]

    emitted: set[str] = set()
    bindings: list[ResourceBinding] = []
    for execute in executes:
        for binding in execute.resources:
            name = binding_name(binding)
            if name in emitted:
                continue
            emitted.add(name)
            bindings.append(binding)

    constant_bindings = [binding for binding in bindings if binding.role == "constant"]
    workspace_bindings = [binding for binding in bindings if binding.role == "temporary"]
    external_bindings = [
        binding for binding in bindings if binding.role in {"input", "output", "inout"}
    ]
    unsupported_bindings = [
        binding
        for binding in bindings
        if binding.role not in {"constant", "temporary", "input", "output", "inout"}
    ]
    if unsupported_bindings:
        raise StreamExtractionError(
            "unsupported mutable resource roles: "
            + ", ".join(binding.role for binding in unsupported_bindings)
        )
    workspace_names = frozenset(binding_name(binding) for binding in workspace_bindings)
    external_roles = {
        binding_name(binding): binding.role for binding in external_bindings
    }

    for binding in constant_bindings:
        out.extend(render_resource_static(binding, constant_blobs))
        out.append("")

    out.append("pub struct Workspace {")
    for binding in workspace_bindings:
        out.append(f"    {render_workspace_field(binding)}")
    out.append("}")
    out.append("")
    out.append("impl Workspace {")
    out.append("    pub const fn new() -> Self {")
    out.append("        Self {")
    for binding in workspace_bindings:
        out.append(f"            {render_workspace_initializer(binding)}")
    out.append("        }")
    out.append("    }")
    out.append("}")
    out.append("")
    out.append("impl Default for Workspace {")
    out.append("    fn default() -> Self {")
    out.append("        Self::new()")
    out.append("    }")
    out.append("}")
    out.append("")

    for execute in executes:
        inout_count = sum(item.role == "inout" for item in execute.resources)
        if inout_count:
            if inout_count != 1 or any(
                item.role in {"input", "output"} for item in execute.resources
            ):
                raise StreamExtractionError("in-place execute must expose one external pool")
            signature = "workspace: &mut Workspace, inout: BufferMut"
        else:
            signature = "workspace: &mut Workspace, input: Buffer, output: BufferMut"
        out.append(
            f"pub fn {rust_ident(execute.name)}({signature}) -> Result<(), Error> {{"
        )
        if execute.line_no is not None:
            out.append(f"    // source MLIR line: {execute.line_no}")
        if execute.result:
            out.append(f"    // stream.cmd.execute result timepoint: {execute.result}")
        for command in execute.commands:
            out.extend(
                render_command(command, "    ", workspace_names, external_roles)
            )
        out.append("    Ok(())")
        out.append("}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def dataclass_to_json(value: Any) -> Any:
    if isinstance(value, ResourceBinding):
        rendered = {
            field.name: dataclass_to_json(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        rendered["static_ident"] = const_ident(binding_name(value))
        return rendered
    if dataclasses.is_dataclass(value):
        return {field.name: dataclass_to_json(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, list):
        return [dataclass_to_json(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_to_json(item) for key, item in value.items()}
    if isinstance(value, bytes):
        return value.hex()
    return value


def render_metadata_json(executes: list[CmdExecute]) -> str:
    document = {
        "schema_version": 1,
        "cmd_executes": [
            {
                "name": execute.name,
                "resources": [
                    {
                        "static_ident": const_ident(binding_name(binding)),
                        "kind": binding.kind,
                        "size": binding.size,
                        "role": binding.role,
                    }
                    for binding in execute.resources
                ],
            }
            for execute in executes
        ],
    }
    return json.dumps(document, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse IREE Stream MLIR and emit Rust dispatch flow.")
    parser.add_argument("input", type=Path, help="Input .mlir file")
    parser.add_argument("-o", "--output", type=Path, help="Output file, defaults to stdout")
    parser.add_argument("--format", choices=("rust", "json"), default="rust")
    parser.add_argument("--rust-output", type=Path, help="Write generated Rust to this file")
    parser.add_argument("--json-output", type=Path, help="Write generated metadata JSON to this file")
    args = parser.parse_args(argv)

    if args.output and (args.rust_output or args.json_output):
        parser.error("--output cannot be combined with --rust-output or --json-output")

    try:
        text = args.input.read_text(encoding="utf-8")
        executes, constant_blobs = parse_cmd_executes(text)
        rust_rendered = render_rust(executes, constant_blobs)

        if args.rust_output or args.json_output:
            if args.rust_output:
                args.rust_output.write_text(rust_rendered, encoding="utf-8")
            if args.json_output:
                args.json_output.write_text(render_metadata_json(executes), encoding="utf-8")
            return 0

        rendered = (
            json.dumps(
                {
                    "constants": dataclass_to_json(constant_blobs),
                    "cmd_executes": dataclass_to_json(executes),
                },
                indent=2,
            )
            + "\n"
            if args.format == "json"
            else rust_rendered
        )
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (OSError, StreamExtractionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
