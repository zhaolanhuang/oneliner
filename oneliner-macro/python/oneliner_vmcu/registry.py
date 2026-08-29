"""Ordered registration and dispatch for independent vMCU pattern families."""

from __future__ import annotations

from dataclasses import dataclass

from iree.compiler import ir

from .model import Analysis, PatternMatch, RejectedCandidate
from .normalize import NormalizedModule
from .patterns.base import PatternAnalyzer, PatternEmitter


@dataclass(frozen=True)
class PatternRegistration:
    """One semantic analyzer and the emitter for matches it owns."""

    name: str
    analyze: PatternAnalyzer
    emit: PatternEmitter


class PatternRegistry:
    """Deterministic, append-only registry used by analysis and rewriting."""

    def __init__(self) -> None:
        """Creates an empty registry with no model-specific assumptions."""
        self._registrations: list[PatternRegistration] = []

    @property
    def names(self) -> tuple[str, ...]:
        """Returns registration order for diagnostics and reproducibility."""
        return tuple(item.name for item in self._registrations)

    def register(
        self, name: str, analyzer: PatternAnalyzer, emitter: PatternEmitter
    ) -> None:
        """Adds one pattern without requiring a driver code change."""
        if not name or any(item.name == name for item in self._registrations):
            raise ValueError(f"duplicate or empty vMCU pattern name: {name!r}")
        self._registrations.append(PatternRegistration(name, analyzer, emitter))

    def analyze(self, normalized: NormalizedModule) -> Analysis:
        """Runs patterns in registration order with shared overlap ownership."""
        matches: list[PatternMatch] = []
        rejected: list[RejectedCandidate] = []
        occupied: set[ir.Operation] = set()
        for registration in self._registrations:
            result = registration.analyze(normalized.graphs, occupied)
            matches.extend(result.matches)
            rejected.extend(result.rejected)
        return Analysis(matches, rejected)

    def emit(self, match: PatternMatch) -> None:
        """Dispatches by semantic kind and rejects missing ownership."""
        for registration in self._registrations:
            if registration.name == match.kind:
                registration.emit(match)
                return
        raise ValueError(f"no emitter registered for vMCU pattern {match.kind!r}")


def create_default_registry() -> PatternRegistry:
    """Creates the built-in registry without introducing module-global mutation."""
    from .emitters.conv2d import emit_conv2d
    from .emitters.depthwise import emit_depthwise
    from .emitters.fully_connected import emit_fully_connected
    from .emitters.inverted_bottleneck import emit_inverted_bottleneck
    from .patterns.conv2d import analyze_conv2d
    from .patterns.depthwise import analyze_depthwise
    from .patterns.fully_connected import analyze_fully_connected
    from .patterns.inverted_bottleneck import analyze_inverted_bottleneck

    registry = PatternRegistry()
    # Composite patterns must claim their whole subgraph before standalone
    # operators inspect the same roots.
    registry.register(
        "inverted_bottleneck_k2_plus_2_segment",
        analyze_inverted_bottleneck,
        emit_inverted_bottleneck,
    )
    registry.register("quantized_conv2d", analyze_conv2d, emit_conv2d)
    registry.register("quantized_depthwise_conv2d", analyze_depthwise, emit_depthwise)
    registry.register(
        FullyConnectedMatchKind.NAME, analyze_fully_connected, emit_fully_connected
    )
    return registry


class FullyConnectedMatchKind:
    """Avoids importing the concrete match class during registry declaration."""

    NAME = "quantized_fully_connected"
