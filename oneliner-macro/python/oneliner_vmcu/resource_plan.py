"""Finalizes vMCU deployment resources and SRAM budget status."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oneliner_iree import analyze_ram_usage


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
    lowering = analyze_ram_usage(stream_text, executable_text)
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
        + lowering.transient_size
        + lowering.stack_size
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
        "vmcu_sram_budget": budget,
        "status": status,
    }
    return plan
