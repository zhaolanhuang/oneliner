"""Finalizes vMCU post-lowering resource measurements."""

from __future__ import annotations

from typing import Any

from oneliner_iree import analyze_ram_usage


def finalize_resource_plan(
    plan: dict[str, Any],
    stream_text: str,
    executable_text: str,
) -> dict[str, Any]:
    """Adds post-lowering pool, transient arena, workspace, and stack metrics."""
    applied = bool(plan.get("applied"))
    lowering = analyze_ram_usage(stream_text, executable_text)
    workspace_bytes = (
        int(plan.get("resources", {}).get("workspace_bytes", 0))
        if applied
        else 0
    )
    io_pool_bytes = (
        int(plan.get("resources", {}).get("io_pool_allocated_bytes") or 0)
        if applied
        else 0
    )
    # The current stock-IREE lowering materializes B/C/D as llvm.alloca, so the
    # static LLVM stack estimate already contains the fixed workspace.
    workspace_residency = (
        "stack-included" if applied and workspace_bytes else "none"
    )
    workspace_additional_bytes = 0 if workspace_residency == "stack-included" else workspace_bytes
    total = (
        io_pool_bytes
        + lowering.transient_size
        + lowering.stack_size
        + workspace_additional_bytes
    )
    plan["resources"] = {
        "logical_i32_accumulator_bytes_eliminated": plan.get("totals", {}).get(
            "eliminated_i32_accumulator_bytes", 0
        ),
        "arena_bytes": lowering.transient_size,
        "io_pool_allocated_bytes": io_pool_bytes,
        "transient_allocations": list(lowering.transient_allocations),
        "stack_bytes": lowering.stack_size,
        "llvm_static_alloca_estimate_bytes": lowering.stack_size,
        "llvm_function_estimates": lowering.function_stack_sizes,
        "workspace_bytes": workspace_bytes,
        "workspace_residency": workspace_residency,
        "workspace_additional_sram_bytes": workspace_additional_bytes,
        "total_sram_bytes": total,
    }
    return plan
