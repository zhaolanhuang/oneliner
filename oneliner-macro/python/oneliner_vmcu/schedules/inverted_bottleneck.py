"""Paper-derived K²+2 segment schedule for inverted bottlenecks.

Paper correspondence: vMCU §5.2, PDF pp.6-7, Figure 6. The paper's 3×3
schedule keeps nine B segments, one C segment, and one D segment, totaling
11=3×3+1+1 workspace segments. Section 5.3 (PDF p.7) selects
``min(Cin,Cout)`` as the Conv/IBN segment width.

Engineering extension: the paper evaluates several kernel sizes but states the
workspace derivation explicitly for 3×3. This implementation applies the same
construction to arbitrary static ``Kh×Kw``, yielding ``Kh×Kw+2`` segments, and
sizes D by ``Cout`` i32 lanes so unequal input/output channels are supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..memory import FixedSegmentMemoryPlan, SegmentSpec


@dataclass(frozen=True)
class InvertedBottleneckSegmentSchedule:
    """K² expansion-patch segments, one depthwise segment, and one D segment.

    Paper correspondence: §5.2, PDF p.7, paragraph beginning "For each step of
    computation" through the explicit 11-segment workspace statement.
    """

    module_input_channels: int
    expanded_channels: int
    module_output_channels: int
    kernel_height: int
    kernel_width: int
    alignment: int = 4

    def __post_init__(self) -> None:
        if min(
            self.module_input_channels,
            self.expanded_channels,
            self.module_output_channels,
            self.kernel_height,
            self.kernel_width,
        ) <= 0:
            raise ValueError("IBN schedule dimensions must be positive")

    @property
    def segment_lanes(self) -> int:
        return min(self.module_input_channels, self.module_output_channels)

    @property
    def patch_segments(self) -> int:
        return self.kernel_height * self.kernel_width

    @property
    def workspace_segments(self) -> int:
        return self.patch_segments + 2

    @property
    def expansion_chunks(self) -> int:
        return (
            self.expanded_channels + self.segment_lanes - 1
        ) // self.segment_lanes

    def memory_plan(self) -> FixedSegmentMemoryPlan:
        specs = (
            SegmentSpec(
                "B",
                self.patch_segments,
                self.segment_lanes,
                "i8",
                self.alignment,
                0,
                3,
            ),
            SegmentSpec(
                "C", 1, self.segment_lanes, "i8", self.alignment, 1, 3
            ),
            SegmentSpec(
                "D",
                1,
                self.module_output_channels,
                "i32",
                self.alignment,
                0,
                4,
            ),
        )
        return FixedSegmentMemoryPlan(
            self.segment_lanes, self.workspace_segments, specs
        )

    def to_dict(self) -> dict[str, Any]:
        memory = self.memory_plan()
        return {
            "kind": "inverted_bottleneck_k2_plus_2_segment",
            "kernel_shape": [self.kernel_height, self.kernel_width],
            "segment_lanes": self.segment_lanes,
            "expansion_chunks": self.expansion_chunks,
            "workspace_segments": self.workspace_segments,
            "workspace_bytes": memory.workspace_bytes,
            "buffers": [item.to_dict() for item in memory.specs],
        }
