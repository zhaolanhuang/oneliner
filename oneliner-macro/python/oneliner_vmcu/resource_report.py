"""Post-lowering arena, stack, workspace, and total-SRAM analysis."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .memory import align_up


_ELEMENT_BYTES = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f32": 4, "f64": 8}


@dataclass(frozen=True)
class ObjectStackAnalysis:
    """Maximum stack frame recovered from final-object disassembly."""

    maximum_bytes: int
    function: str
    analyzer: str

    def to_dict(self) -> dict[str, Any]:
        """Returns stable JSON fields for the deployment plan."""
        return {
            "maximum_bytes": self.maximum_bytes,
            "function": self.function,
            "analyzer": self.analyzer,
        }


def parse_stream_arena(stream_text: str) -> tuple[int, list[int]]:
    """Returns the largest static transient allocation in Stream IR."""
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
    return max(allocations, default=0), allocations


def parse_llvm_static_allocas(executable_text: str) -> tuple[int, dict[str, int]]:
    """Conservatively aligns static LLVM allocas per lowered function."""
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"(%[\w.$-]+)\s*=\s*llvm\.mlir\.constant\((\d+)\s*:\s*index\)",
            executable_text,
        )
    }
    totals: dict[str, int] = {}
    current = "module"
    offset = 0
    maximum_alignment = 1

    def finish() -> None:
        """Stores the current function's final alignment-rounded estimate."""
        if offset:
            totals[current] = align_up(offset, maximum_alignment)

    for line in executable_text.splitlines():
        function = re.search(r"llvm\.func\s+@([^\s(]+)", line)
        if function:
            finish()
            current = function.group(1)
            offset = 0
            maximum_alignment = 1
            continue
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
        offset = align_up(offset, alignment)
        offset += constants[count_name] * _ELEMENT_BYTES[element_type]
        maximum_alignment = max(maximum_alignment, alignment)
    finish()
    return max(totals.values(), default=0), totals


def _mask_alignment(mask_text: str) -> int:
    """Decodes an x86 two's-complement stack alignment mask."""
    mask = int(mask_text, 16)
    bits = len(mask_text) * 4
    value = (1 << bits) - mask
    return value if value > 0 and value & (value - 1) == 0 else 1


def parse_objdump_stack(disassembly: str) -> ObjectStackAnalysis:
    """Extracts conservative x86 or ARM stack frames from final machine code."""
    frames: dict[str, int] = {}
    current: str | None = None
    pushes = 0
    stack_subtractions = 0
    alignment_slack = 0

    def finish() -> None:
        """Records the current symbol's conservative maximum frame."""
        if current is not None:
            frames[current] = pushes + stack_subtractions + alignment_slack

    for line in disassembly.splitlines():
        symbol = re.match(r"^[0-9a-fA-F]+\s+<([^>]+)>:$", line.strip())
        if symbol:
            finish()
            current = symbol.group(1)
            pushes = 0
            stack_subtractions = 0
            alignment_slack = 0
            continue
        if current is None:
            continue
        instruction = line.split("\t")[-1].strip() if "\t" in line else line.strip()
        if re.match(r"push[a-z]*\s+%", instruction):
            pushes += 8
        arm_push = re.search(r"\bpush(?:\.w)?\s+\{([^}]+)\}", instruction)
        if arm_push:
            pushes += 4 * len([item for item in arm_push.group(1).split(",") if item.strip()])
        x86_sub = re.search(r"\bsub[a-z]*\s+\$0x([0-9a-fA-F]+),%rsp", instruction)
        if x86_sub:
            stack_subtractions += int(x86_sub.group(1), 16)
        arm_sub = re.search(r"\bsub(?:\.w)?\s+sp,\s*(?:sp,\s*)?#(0x[0-9a-fA-F]+|\d+)", instruction)
        if arm_sub:
            stack_subtractions += int(arm_sub.group(1), 0)
        alignment = re.search(
            r"\band[a-z]*\s+\$0x([0-9a-fA-F]+),%rsp", instruction
        )
        if alignment:
            alignment_slack = max(
                alignment_slack, _mask_alignment(alignment.group(1)) - 1
            )
    finish()
    if not frames:
        raise ValueError("objdump output did not contain function symbols")
    function, maximum = max(frames.items(), key=lambda item: (item[1], item[0]))
    return ObjectStackAnalysis(maximum, function, "objdump-final-object")


def analyze_object_stack(object_path: Path, objdump: str = "objdump") -> ObjectStackAnalysis:
    """Disassembles one object and returns its maximum static stack frame."""
    completed = subprocess.run(
        [objdump, "-d", str(object_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or "objdump failed")
    return parse_objdump_stack(completed.stdout)


def finalize_resource_plan(
    plan: dict[str, Any],
    stream_text: str,
    executable_text: str,
    object_path: Path,
    *,
    objdump: str = "objdump",
    deployment: str = "rewritten",
) -> dict[str, Any]:
    """Adds authoritative lowering metrics and evaluates ``vmcu_sram``."""
    if deployment not in ("rewritten", "baseline-fallback"):
        raise ValueError(f"unsupported deployment kind: {deployment}")
    arena_bytes, transient_allocations = parse_stream_arena(stream_text)
    llvm_stack_bytes, llvm_functions = parse_llvm_static_allocas(executable_text)
    object_stack = analyze_object_stack(object_path, objdump)
    workspace_bytes = (
        int(plan.get("resources", {}).get("workspace_bytes", 0))
        if deployment == "rewritten"
        else 0
    )
    io_pool_bytes = (
        int(plan.get("resources", {}).get("io_pool_allocated_bytes") or 0)
        if deployment == "rewritten"
        else 0
    )
    # The current stock-IREE lowering materializes B/C/D as llvm.alloca, so
    # object stack already contains the fixed workspace. Reporting it remains
    # useful, but adding it again would double-count deployment SRAM.
    workspace_residency = (
        "stack-included" if deployment == "rewritten" and workspace_bytes else "none"
    )
    workspace_additional_bytes = 0 if workspace_residency == "stack-included" else workspace_bytes
    total = (
        io_pool_bytes
        + arena_bytes
        + object_stack.maximum_bytes
        + workspace_additional_bytes
    )
    budget = plan.get("resources", {}).get("vmcu_sram_budget")
    tolerance = max(256, llvm_stack_bytes // 4)
    difference = abs(object_stack.maximum_bytes - llvm_stack_bytes)
    status = "within-budget"
    if budget is not None and total > int(budget):
        status = "exceeds-budget"
    plan["applied"] = bool(plan.get("applied")) and deployment == "rewritten"
    plan["deployment"] = {
        "kind": deployment,
        "object": str(object_path),
    }
    plan["resources"] = {
        "logical_i32_accumulator_bytes_eliminated": plan.get("totals", {}).get(
            "eliminated_i32_accumulator_bytes", 0
        ),
        "arena_bytes": arena_bytes,
        "io_pool_allocated_bytes": io_pool_bytes,
        "transient_allocations": transient_allocations,
        "stack_bytes": object_stack.maximum_bytes,
        "stack_analysis": object_stack.to_dict(),
        "llvm_static_alloca_estimate_bytes": llvm_stack_bytes,
        "llvm_function_estimates": llvm_functions,
        "stack_estimate_difference_bytes": difference,
        "stack_estimate_tolerance_bytes": tolerance,
        "stack_estimate_within_tolerance": difference <= tolerance,
        "workspace_bytes": workspace_bytes,
        "workspace_residency": workspace_residency,
        "workspace_additional_sram_bytes": workspace_additional_bytes,
        "total_sram_bytes": total,
        "vmcu_sram_budget": budget,
        "status": status,
    }
    return plan
