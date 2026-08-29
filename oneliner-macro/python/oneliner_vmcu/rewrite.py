"""Transactional analysis and rewrite driver."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import prod
from typing import Any, Literal

from iree.compiler import ir
from .ir_utils import verify
from .model import Analysis, PatternMatch, RejectedCandidate
from .normalize import NormalizedModule, normalize_module
from .registry import PatternRegistry, create_default_registry
from .versioning import CompilerVersionDiagnostics, diagnose_compiler_versions
from .compact_analysis import CompactAnalysis, build_compact_analysis
from .pool_emitter import emit_compact_graph


# Only enabled modes reach Python; Rust handles the disabled `off` path without
# starting a preprocessing split or Python process.
Mode = Literal["auto", "strict"]


class RewriteError(RuntimeError):
    """A safe rewrite could not be completed."""


@dataclass(frozen=True)
class RewriteResult:
    """Verified output text and its machine-readable analysis/rewrite plan."""

    text: str
    plan: dict[str, Any]


def _parse(text: str) -> tuple[ir.Context, ir.Module]:
    """Parses and verifies textual MLIR in a context with known dialects only.

    Rejecting unregistered dialects prevents the rewriter from accepting an
    operation whose verifier or semantics are unavailable in this IREE build.
    The returned context must stay alive as long as any returned MLIR object.
    """
    context = ir.Context()
    context.allow_unregistered_dialects = False
    try:
        with context:
            module = ir.Module.parse(text)
            verify(module.operation, "input module")
    except (ir.MLIRError, ValueError) as error:
        raise RewriteError(f"failed to parse preprocessing MLIR: {error}") from error
    return context, module


def _analyze(
    module: ir.Module, registry: PatternRegistry
) -> tuple[Analysis, NormalizedModule]:
    """Builds the SSA graph and runs every enabled semantic pattern matcher."""
    try:
        normalized = normalize_module(module)
        return registry.analyze(normalized), normalized
    except ValueError as error:
        raise RewriteError(f"failed to build the preprocessing SSA graph: {error}") from error


def _apply_sram_budget(analysis: Analysis, sram_budget: int | None) -> Analysis:
    """Rejects fixed schedules whose known workspace alone exceeds the cap.

    Arena and object stack are unavailable at preprocessing time and are added
    by the post-lowering resource reporter. This early gate guarantees that an
    impossible 11-segment workspace never reaches mutation.
    """
    if sram_budget is None:
        return analysis
    accepted = []
    rejected = list(analysis.rejected)
    for candidate in analysis.matches:
        if candidate.workspace_bytes <= sram_budget:
            accepted.append(candidate)
            continue
        rejected.append(
            RejectedCandidate(
                candidate.root,
                candidate.kind,
                "fixed schedule workspace exceeds vmcu_sram: "
                f"required={candidate.workspace_bytes} budget={sram_budget}",
                "preprocessing resource planner",
            )
        )
    return Analysis(accepted, rejected)


def _immutable_signature(value: Any) -> Any:
    """Copies plan data into a comparable shape without unstable match IDs."""
    if isinstance(value, dict):
        return tuple(
            (key, _immutable_signature(item))
            for key, item in sorted(value.items())
            if key != "id"
        )
    if isinstance(value, list):
        return tuple(_immutable_signature(item) for item in value)
    return value


def _candidate_signature(candidate: PatternMatch) -> Any:
    """Returns immutable semantic facts that survive operation-index changes."""
    return _immutable_signature(candidate.to_dict())


def _require_next_candidate(
    analysis: Analysis,
    expected_count: int,
    expected_kind: str,
    expected_signature: Any,
    emitted_count: int,
) -> PatternMatch:
    """Guards iterative re-analysis against lost, new, or changed matches."""
    actual_count = len(analysis.matches)
    if actual_count != expected_count:
        raise RewriteError(
            "iterative re-analysis changed the remaining candidate count after "
            f"{emitted_count} rewrites: expected={expected_count} "
            f"actual={actual_count}"
        )
    candidate = analysis.matches[0]
    actual_signature = _candidate_signature(candidate)
    if candidate.kind != expected_kind or actual_signature != expected_signature:
        raise RewriteError(
            "iterative re-analysis changed the next candidate after "
            f"{emitted_count} rewrites: expected_kind={expected_kind} "
            f"actual_kind={candidate.kind} actual_id={candidate.root.identifier}"
        )
    return candidate


def _plan(
    source_text: str,
    mode: Mode,
    analysis: Analysis,
    applied: bool,
    versions: CompilerVersionDiagnostics,
    normalized: NormalizedModule,
    registry: PatternRegistry,
    sram_budget: int | None,
    schedule_search: str,
    search_state_limit: int,
    compact: CompactAnalysis | None,
) -> dict[str, Any]:
    """Builds the stable JSON plan without serializing live MLIR objects.

    The byte metric is a logical graph quantity, not a promise about the final
    target arena; IREE's later scheduling and bufferization remain authoritative.
    """
    accepted = [candidate.to_dict() for candidate in analysis.matches]
    rejected = [candidate.to_dict() for candidate in analysis.rejected]
    workspace_bytes = max(
        (candidate.workspace_bytes for candidate in analysis.matches), default=0
    )
    compact_graph = compact.plan.to_dict() if compact is not None else {
        "status": "not-planned",
        "search": {
            "mode": schedule_search,
            "state_limit": search_state_limit
            if schedule_search == "bounded"
            else None,
            "explored_states": 0,
            "optimal": False,
        },
        "reason": "candidate emitters have not supplied a unified compact graph",
    }
    if compact is not None:
        compact_graph["boundaries"] = [
            item.to_dict() for item in compact.boundaries
        ]
        compact_graph["materialized_boundaries"] = [
            item.to_dict() for item in compact.boundaries if item.direct_kind is None
        ]
    return {
        "schema_version": 4,
        "mode": mode,
        "applied": applied,
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "iree_versions": versions.to_dict(),
        "normalization": normalized.to_dict(),
        "pattern_registry": list(registry.names),
        "compact_graph": compact_graph,
        "accepted": accepted,
        "rejected": rejected,
        "totals": {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "eliminated_i32_accumulator_bytes": sum(
                candidate.eliminated_accumulator_bytes
                for candidate in analysis.matches
            ),
        },
        "resources": {
            "logical_i32_accumulator_bytes_eliminated": sum(
                candidate.eliminated_accumulator_bytes
                for candidate in analysis.matches
            ),
            "arena_bytes": None,
            "stack_bytes": None,
            "workspace_bytes": workspace_bytes,
            "io_pool_logical_bytes": (
                compact.plan.logical_pool_bytes if compact is not None else None
            ),
            "io_pool_allocated_bytes": (
                compact.plan.allocated_pool_bytes if compact is not None else None
            ),
            "unsupported_transient_bytes": (
                max(
                    (
                        prod(item.output_shape)
                        for item in compact.boundaries
                        if item.direct_kind is None
                    ),
                    default=0,
                )
                if compact is not None
                else None
            ),
            "total_sram_bytes": None,
            "vmcu_sram_budget": sram_budget,
            "status": "awaiting-post-lowering-analysis" if applied else "not-applied",
        },
    }


def rewrite_text(
    source_text: str,
    mode: Mode = "auto",
    iree_compile: str | None = None,
    registry: PatternRegistry | None = None,
    sram_budget: int | None = None,
    schedule_search: str = "bounded",
    search_state_limit: int = 1_000_000,
) -> RewriteResult:
    """Analyzes and transactionally rewrites preprocessing-phase MLIR.

    The first parse is never mutated. If candidates exist, a second parse must
    produce the exact same candidate identities before any edit is made. The
    final serialized module is verified and reparsed before being returned.
    """
    if mode not in ("auto", "strict"):
        raise RewriteError(f"unsupported rewrite mode: {mode}")
    if sram_budget is not None and sram_budget <= 0:
        raise RewriteError("sram_budget must be a positive byte count")
    if schedule_search not in ("bounded", "optimal", "greedy"):
        raise RewriteError(f"unsupported schedule search mode: {schedule_search}")
    if search_state_limit <= 0:
        raise RewriteError("search_state_limit must be positive")
    versions = diagnose_compiler_versions(iree_compile)
    if versions.compatible is False:
        raise RewriteError(versions.diagnostic or "incompatible IREE versions")
    active_registry = registry or create_default_registry()
    # Direct tensor-ABI fixtures keep the legacy transactional emitter. The
    # destructive in-place ABI applies at IREE's real hal.buffer_view boundary.
    use_compact_emitter = (
        registry is None
        and source_text.count("hal.tensor.import") == 1
        and source_text.count("hal.tensor.export") == 1
    )
    # Pass one is analysis-only and preserves the caller's exact source text.
    source_context, source_module = _parse(source_text)
    with source_context:
        source_analysis, source_normalized = _analyze(source_module, active_registry)
        source_analysis = _apply_sram_budget(source_analysis, sram_budget)
        source_signatures = tuple(
            _candidate_signature(candidate) for candidate in source_analysis.matches
        )
        source_compact = None
        compact_error = None
        if source_analysis.matches and use_compact_emitter:
            try:
                source_compact = build_compact_analysis(
                    source_analysis,
                    search_mode=schedule_search,
                    search_state_limit=search_state_limit,
                )
            except ValueError as error:
                compact_error = str(error)
            if (
                source_compact is not None
                and mode == "strict"
                and any(item.direct_kind is None for item in source_compact.boundaries)
            ):
                materialized_count = sum(
                    item.direct_kind is None for item in source_compact.boundaries
                )
                compact_error = (
                    "strict mode requires full compact coverage, but found "
                    f"{materialized_count} materialized boundaries"
                )
                source_compact = None
        if compact_error is not None and mode == "strict":
            raise RewriteError(compact_error)
    if not source_analysis.matches:
        if mode == "strict":
            reasons = "; ".join(item.reason for item in source_analysis.rejected[:3])
            suffix = f": {reasons}" if reasons else ""
            raise RewriteError(f"strict mode found no safe vMCU patterns{suffix}")
        # Auto mode promises a byte-for-byte fallback, including formatting and
        # dialect aliases, when no candidate is proven safe.
        return RewriteResult(
            source_text,
            _plan(
                source_text,
                mode,
                source_analysis,
                False,
                versions,
                source_normalized,
                active_registry,
                sram_budget,
                schedule_search,
                search_state_limit,
                source_compact,
            ),
        )

    if use_compact_emitter and source_compact is None:
        # Auto mode preserves the original module when no complete safe pool
        # plan exists. A partial full-tensor rewrite would defeat the ABI and
        # make the schema-v4 SRAM report misleading.
        return RewriteResult(
            source_text,
            _plan(
                source_text,
                mode,
                source_analysis,
                False,
                versions,
                source_normalized,
                active_registry,
                sram_budget,
                schedule_search,
                search_state_limit,
                None,
            ),
        )

    # Pass two reparses the immutable input. Requiring identical candidate IDs
    # ensures no mutation starts from analysis state that cannot be reproduced.
    rewrite_context, rewrite_module = _parse(source_text)
    with rewrite_context, ir.Location.unknown():
        rewrite_analysis, _ = _analyze(rewrite_module, active_registry)
        rewrite_analysis = _apply_sram_budget(rewrite_analysis, sram_budget)
        if rewrite_analysis.match_ids != source_analysis.match_ids:
            raise RewriteError("transactional re-analysis produced different candidates")
        try:
            if use_compact_emitter:
                rewrite_compact = build_compact_analysis(
                    rewrite_analysis,
                    search_mode=schedule_search,
                    search_state_limit=search_state_limit,
                )
                if rewrite_compact.plan.to_dict() != source_compact.plan.to_dict():
                    raise RewriteError("transactional compact plan changed after reparse")
                emit_compact_graph(
                    rewrite_module, tuple(rewrite_analysis.matches), rewrite_compact
                )
                del rewrite_compact
                del rewrite_analysis
            else:
                current_analysis = rewrite_analysis
                del rewrite_analysis
                total_candidates = len(source_analysis.matches)
                for emitted_count, (expected_candidate, expected_signature) in enumerate(
                    zip(source_analysis.matches, source_signatures, strict=True)
                ):
                    candidate = _require_next_candidate(
                        current_analysis,
                        total_candidates - emitted_count,
                        expected_candidate.kind,
                        expected_signature,
                        emitted_count,
                    )
                    active_registry.emit(candidate)
                    del candidate
                    del current_analysis
                    current_analysis, _ = _analyze(rewrite_module, active_registry)
                    current_analysis = _apply_sram_budget(current_analysis, sram_budget)
                if current_analysis.matches:
                    raise RewriteError(
                        "iterative re-analysis found new candidates after all planned "
                        f"rewrites: ids={current_analysis.match_ids}"
                    )
            verify(rewrite_module.operation, "rewritten module")
            rewritten_text = str(rewrite_module)
        except (ir.MLIRError, ValueError, IndexError) as error:
            raise RewriteError(f"failed to rewrite preprocessing MLIR: {error}") from error

    # Serialization can reveal region or ownership errors hidden by live Python
    # handles, so the emitted text gets one final independent parse and verify.
    final_context, final_module = _parse(rewritten_text)
    with final_context:
        verify(final_module.operation, "serialized rewritten module")
    return RewriteResult(
        rewritten_text,
        _plan(
            source_text,
            mode,
            source_analysis,
            True,
            versions,
            source_normalized,
            active_registry,
            sram_budget,
            schedule_search,
            search_state_limit,
            source_compact,
        ),
    )
