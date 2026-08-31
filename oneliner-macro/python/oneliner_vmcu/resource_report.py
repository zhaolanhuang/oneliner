"""Post-lowering arena, stack, workspace, and total-SRAM analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .memory import align_up


_ELEMENT_BYTES = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f32": 4, "f64": 8}


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
        """Stores the current function's final alignment-rounded estimate."""
        if current is not None and offset:
            totals[current] = align_up(offset, maximum_alignment)

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
        offset = align_up(offset, alignment)
        offset += constants[count_name] * _ELEMENT_BYTES[element_type]
        maximum_alignment = max(maximum_alignment, alignment)
    finish()
    return max(totals.values(), default=0), totals


def finalize_resource_plan(
    plan: dict[str, Any],
    stream_text: str,
    executable_text: str,
    object_path: Path,
    *,
    deployment: str = "rewritten",
) -> dict[str, Any]:
    """Adds post-lowering resource metrics and evaluates ``vmcu_sram``."""
    if deployment not in ("rewritten", "baseline-fallback"):
        raise ValueError(f"unsupported deployment kind: {deployment}")
    arena_bytes, transient_allocations = parse_stream_arena(stream_text)
    llvm_stack_bytes, llvm_functions = parse_llvm_static_allocas(executable_text)
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
    # The current stock-IREE lowering materializes B/C/D as llvm.alloca, so the
    # static LLVM stack estimate already contains the fixed workspace.
    workspace_residency = (
        "stack-included" if deployment == "rewritten" and workspace_bytes else "none"
    )
    workspace_additional_bytes = 0 if workspace_residency == "stack-included" else workspace_bytes
    total = (
        io_pool_bytes
        + arena_bytes
        + llvm_stack_bytes
        + workspace_additional_bytes
    )
    budget = plan.get("resources", {}).get("vmcu_sram_budget")
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
        "stack_bytes": llvm_stack_bytes,
        "llvm_static_alloca_estimate_bytes": llvm_stack_bytes,
        "llvm_function_estimates": llvm_functions,
        "workspace_bytes": workspace_bytes,
        "workspace_residency": workspace_residency,
        "workspace_additional_sram_bytes": workspace_additional_bytes,
        "total_sram_bytes": total,
        "vmcu_sram_budget": budget,
        "status": status,
    }
    return plan
