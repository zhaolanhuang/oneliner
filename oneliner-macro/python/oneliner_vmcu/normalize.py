"""Normalization boundary between dialect inspection and semantic matching.

Phase 1 intentionally performs a read-only normalization.  It records stable
structural facts and leaves textual MLIR untouched, preserving auto-mode's
byte-for-byte fallback and Phase 0's exact FC output.  Later dialect adapters
can canonicalize equivalent source forms behind this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from iree.compiler import ir

from .graph import FunctionGraph, build_graphs


@dataclass(frozen=True)
class NormalizedModule:
    """SSA graphs plus name-independent normalization diagnostics."""

    graphs: list[FunctionGraph]
    operation_histogram: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serializes only structural information, never model identities."""
        return {
            "strategy": "read_only_preprocessing_v1",
            "function_count": len(self.graphs),
            "operation_histogram": dict(self.operation_histogram),
        }


def normalize_module(module: ir.Module) -> NormalizedModule:
    """Builds canonical SSA views without depending on symbol or SSA names."""
    graphs = build_graphs(module)
    histogram: dict[str, int] = {}
    for graph in graphs:
        for node in graph.nodes:
            name = node.operation.name
            histogram[name] = histogram.get(name, 0) + 1
    return NormalizedModule(graphs, tuple(sorted(histogram.items())))
