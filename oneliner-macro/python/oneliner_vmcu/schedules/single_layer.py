"""Deterministic segment schedule shared by standalone operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..memory import FixedSegmentMemoryPlan, SegmentSpec


@dataclass(frozen=True)
class SingleLayerSegmentSchedule:
    """One fixed input/output segment schedule with no alternative search."""

    input_elements: int
    output_elements: int
    segment_lanes: int
    pool_segments: int
    storage_type: str = "i8"
    alignment: int = 4

    def __post_init__(self) -> None:
        """Requires a fully static, non-empty single-layer problem."""
        if min(
            self.input_elements,
            self.output_elements,
            self.segment_lanes,
            self.pool_segments,
        ) <= 0:
            raise ValueError("single-layer segment schedule requires positive sizes")

    @property
    def input_segment_count(self) -> int:
        """Returns ceil(input_elements / segment_lanes)."""
        return (self.input_elements + self.segment_lanes - 1) // self.segment_lanes

    @property
    def output_segment_count(self) -> int:
        """Returns ceil(output_elements / segment_lanes)."""
        return (self.output_elements + self.segment_lanes - 1) // self.segment_lanes

    def memory_plan(self) -> FixedSegmentMemoryPlan:
        """Builds the only workspace layout accepted by this schedule."""
        state = SegmentSpec(
            "loop_state",
            1,
            self.segment_lanes,
            self.storage_type,
            self.alignment,
            0,
            self.input_segment_count + self.output_segment_count,
        )
        return FixedSegmentMemoryPlan(
            self.segment_lanes, self.pool_segments, (state,)
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns deterministic schedule and partial-lane information."""
        plan = self.memory_plan()
        return {
            "kind": "single_layer_fixed_segment",
            "input_segments": self.input_segment_count,
            "output_segments": self.output_segment_count,
            "last_input_valid_lanes": plan.valid_lanes(
                self.input_segment_count - 1, self.input_elements
            ),
            "last_output_valid_lanes": plan.valid_lanes(
                self.output_segment_count - 1, self.output_elements
            ),
            "memory": plan.to_dict(),
        }
