"""Target-independent RAM usage analysis for lowered IREE MLIR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_ELEMENT_BYTES = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f32": 4, "f64": 8}


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def parse_stream_arena(stream_text: str) -> tuple[int, list[int]]:
    """Returns all simultaneously resident static Stream allocations."""
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"(%[\w.$-]+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*index",
            stream_text,
        )
    }
    allocations = []
    for token in re.findall(
        r"stream\.resource\.alloca[^\n]*resource<transient>\{(%[\w.$-]+)\}",
        stream_text,
    ):
        if token not in constants:
            raise ValueError(f"unresolved transient arena size: {token}")
        allocations.append(constants[token])
    return sum(allocations), allocations


def parse_llvm_static_allocas(executable_text: str) -> tuple[int, dict[str, int]]:
    """Conservatively aligns static LLVM allocas per lowered function."""
    totals: dict[str, int] = {}
    constants: dict[str, int] = {}
    current: str | None = None
    offset = 0
    maximum_alignment = 1

    def finish() -> None:
        if current is not None and offset:
            totals[current] = _align_up(offset, maximum_alignment)

    for line in executable_text.splitlines():
        function = re.search(r"llvm\.func\s+@([^\s(]+)", line)
        if function:
            finish()
            current = function.group(1)
            constants = {}
            offset = 0
            maximum_alignment = 1
            continue
        if current is None:
            continue
        constant = re.search(
            r"(%[\w.$-]+)\s*=\s*llvm\.mlir\.constant\((\d+)\s*:\s*index\)",
            line,
        )
        if constant:
            constants[constant.group(1)] = int(constant.group(2))
        allocation = re.search(
            r"llvm\.alloca\s+(%[\w.$-]+)\s+x\s+(i8|i16|i32|i64|f32|f64)"
            r"\s+\{alignment\s*=\s*(\d+)",
            line,
        )
        if not allocation:
            continue
        count_name, element_type, alignment_text = allocation.groups()
        if count_name not in constants:
            raise ValueError(f"unresolved static alloca element count: {count_name}")
        alignment = int(alignment_text)
        offset = _align_up(offset, alignment)
        offset += constants[count_name] * _ELEMENT_BYTES[element_type]
        maximum_alignment = max(maximum_alignment, alignment)
    finish()
    return max(totals.values(), default=0), totals


@dataclass(frozen=True)
class LoweringRamUsage:
    """Static RAM owned by generated IREE lowering artifacts."""

    transient_size: int
    transient_allocations: tuple[int, ...]
    stack_size: int
    function_stack_sizes: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "transient_size": self.transient_size,
            "transient_allocations": list(self.transient_allocations),
            "stack_size": self.stack_size,
            "function_stack_sizes": self.function_stack_sizes,
        }


def analyze_ram_usage(stream_text: str, executable_text: str) -> LoweringRamUsage:
    """Analyzes Stream allocations and per-dispatch static LLVM stack."""
    transient_size, allocations = parse_stream_arena(stream_text)
    stack_size, functions = parse_llvm_static_allocas(executable_text)
    return LoweringRamUsage(
        transient_size=transient_size,
        transient_allocations=tuple(allocations),
        stack_size=stack_size,
        function_stack_sizes=functions,
    )
