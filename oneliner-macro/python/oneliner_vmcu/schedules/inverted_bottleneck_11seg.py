"""The single vMCU inverted-bottleneck schedule: exactly eleven segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..memory import (
    CircularMemoryPlan,
    FixedSegmentMemoryPlan,
    SegmentLifetime,
    SegmentSpec,
    align_up,
    plan_minimum_circular_pool,
)


@dataclass(frozen=True)
class InvertedBottleneck11SegmentSchedule:
    """Fixed 9×B + 1×C + 1×D workspace with no alternatives."""

    module_input_channels: int
    expanded_channels: int
    module_output_channels: int
    alignment: int = 4

    def __post_init__(self) -> None:
        """Requires a statically sized module and non-empty expansion."""
        if min(
            self.module_input_channels,
            self.expanded_channels,
            self.module_output_channels,
        ) <= 0:
            raise ValueError("IBN channel dimensions must be positive")

    @property
    def segment_lanes(self) -> int:
        """Implements vMCU Section 5.3's ``min(Cin, Cout)`` rule."""
        return min(self.module_input_channels, self.module_output_channels)

    @property
    def expansion_chunks(self) -> int:
        """Returns the number of fixed-width expanded-channel chunks."""
        return (
            self.expanded_channels + self.segment_lanes - 1
        ) // self.segment_lanes

    def memory_plan(self) -> FixedSegmentMemoryPlan:
        """Returns exactly 9 B, 1 C, and 1 D logical segments.

        B and C contain requantized i8 activations. D carries projection i32
        accumulators across expanded-channel chunks; its wider dtype is included
        in bytes without changing the paper's logical eleven-segment count.
        """
        specs = (
            SegmentSpec("B", 9, self.segment_lanes, "i8", self.alignment, 0, 3),
            SegmentSpec("C", 1, self.segment_lanes, "i8", self.alignment, 1, 3),
            # B/C are expanded-channel lanes. D is indexed by projection
            # output channel, so it must not inherit min(Cin, Cout) when a
            # block changes its module channel count.
            SegmentSpec("D", 1, self.module_output_channels, "i32", self.alignment, 0, 4),
        )
        return FixedSegmentMemoryPlan(self.segment_lanes, 11, specs)

    def activation_memory_plan(
        self,
        input_shape: tuple[int, int, int, int],
        output_shape: tuple[int, int, int, int],
        depthwise_stride: tuple[int, int],
        padding_low: tuple[int, ...],
        *,
        residual: bool,
    ) -> CircularMemoryPlan:
        """Plans deterministic in-place activation addresses from true last-use.

        Each spatial output step has seven ordered phases matching the fixed
        emitter. Expansion consumes every input-channel segment in its 3x3 A
        patch; a residual extends the center A segment lifetime through add.
        """
        _, input_h, input_w, input_channels = input_shape
        _, output_h, output_w, output_channels = output_shape
        input_chunks = (input_channels + self.segment_lanes - 1) // self.segment_lanes
        output_chunks = (
            output_channels + self.segment_lanes - 1
        ) // self.segment_lanes
        last_uses = [-1] * (input_h * input_w * input_chunks)
        first_writes: list[SegmentLifetime] = []
        for oh in range(output_h):
            for ow in range(output_w):
                output_pixel = oh * output_w + ow
                step = output_pixel * 7
                for kh in range(3):
                    for kw in range(3):
                        ih = oh * depthwise_stride[0] + kh - padding_low[1]
                        iw = ow * depthwise_stride[1] + kw - padding_low[2]
                        if not (0 <= ih < input_h and 0 <= iw < input_w):
                            continue
                        for chunk in range(input_chunks):
                            logical = (ih * input_w + iw) * input_chunks + chunk
                            last_uses[logical] = max(last_uses[logical], step + 2)
                if residual:
                    for chunk in range(input_chunks):
                        logical = (oh * input_w + ow) * input_chunks + chunk
                        last_uses[logical] = max(last_uses[logical], step + 5)
                for chunk in range(output_chunks):
                    first_writes.append(
                        SegmentLifetime(output_pixel * output_chunks + chunk, step + 6)
                    )
        if any(step < 0 for step in last_uses):
            raise ValueError("IBN activation planner found an unread input segment")
        input_lifetimes = tuple(
            SegmentLifetime(index, step) for index, step in enumerate(last_uses)
        )
        segment_bytes = align_up(self.segment_lanes, self.alignment)
        return plan_minimum_circular_pool(
            input_lifetimes, tuple(first_writes), segment_bytes
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the fixed schedule with hard eleven-segment assertions."""
        plan = self.memory_plan()
        workspace_segments = sum(spec.count for spec in plan.specs)
        if workspace_segments != 11:
            raise AssertionError("IBN fixed schedule must contain exactly 11 segments")
        return {
            "kind": "inverted_bottleneck_11_segment",
            "segment_lanes": self.segment_lanes,
            "expansion_chunks": self.expansion_chunks,
            "workspace_segments": workspace_segments,
            "workspace_bytes": plan.workspace_bytes,
            "buffers": [spec.to_dict() for spec in plan.specs],
            "schedule_search": False,
            "recomputation_fallback": False,
        }
