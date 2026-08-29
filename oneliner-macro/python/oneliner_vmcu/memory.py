"""Deterministic segment workspace descriptions and safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_STORAGE_BYTES = {"i8": 1, "ui8": 1, "i16": 2, "i32": 4}


def align_up(value: int, alignment: int) -> int:
    """Rounds ``value`` to a positive power-of-two alignment."""
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("segment alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


@dataclass(frozen=True)
class SegmentSpec:
    """One statically sized family of equal-width segment buffers."""

    name: str
    count: int
    lanes: int
    storage_type: str
    alignment: int = 1
    lifetime_start: int = 0
    lifetime_end: int = 1

    def __post_init__(self) -> None:
        """Rejects invalid sizes and empty/reversed lifetimes."""
        if not self.name:
            raise ValueError("segment name cannot be empty")
        if self.count <= 0 or self.lanes <= 0:
            raise ValueError("segment count and lanes must be positive")
        if self.storage_type not in _STORAGE_BYTES:
            raise ValueError(f"unsupported segment storage type: {self.storage_type}")
        align_up(0, self.alignment)
        if self.lifetime_start < 0 or self.lifetime_end <= self.lifetime_start:
            raise ValueError("segment lifetime must be a non-empty half-open interval")

    @property
    def element_bytes(self) -> int:
        """Returns the storage width of one lane."""
        return _STORAGE_BYTES[self.storage_type]

    @property
    def segment_bytes(self) -> int:
        """Returns one segment's aligned byte stride."""
        return align_up(self.lanes * self.element_bytes, self.alignment)

    @property
    def total_bytes(self) -> int:
        """Returns the statically reserved bytes for every segment instance."""
        return self.count * self.segment_bytes

    def to_dict(self) -> dict[str, Any]:
        """Returns the exact plan representation used by resource reports."""
        return {
            "name": self.name,
            "count": self.count,
            "lanes": self.lanes,
            "storage_type": self.storage_type,
            "alignment": self.alignment,
            "segment_bytes": self.segment_bytes,
            "total_bytes": self.total_bytes,
            "lifetime": [self.lifetime_start, self.lifetime_end],
        }


@dataclass(frozen=True)
class SegmentAddress:
    """One logical segment placement in a circular pool."""

    logical_index: int
    physical_index: int
    byte_offset: int


@dataclass(frozen=True)
class SegmentLifetime:
    """Read/write times for one logical input or output segment."""

    logical_index: int
    access_step: int

    def __post_init__(self) -> None:
        """Rejects negative indices and schedule steps."""
        if self.logical_index < 0 or self.access_step < 0:
            raise ValueError("segment lifetime indices and steps cannot be negative")


@dataclass(frozen=True)
class CircularMemoryPlan:
    """Deterministic input/output bases proven safe for one circular pool."""

    pool_segments: int
    segment_bytes: int
    b_in: int
    b_out: int
    input_last_uses: tuple[SegmentLifetime, ...]
    output_first_writes: tuple[SegmentLifetime, ...]

    def __post_init__(self) -> None:
        """Checks dimensions and replays every possible overwrite hazard."""
        if self.pool_segments <= 0 or self.segment_bytes <= 0:
            raise ValueError("circular pool dimensions must be positive")
        if not 0 <= self.b_in < self.pool_segments:
            raise ValueError("b_in must address the circular segment pool")
        if not 0 <= self.b_out < self.pool_segments:
            raise ValueError("b_out must address the circular segment pool")
        last_use_by_slot: dict[int, int] = {}
        for item in self.input_last_uses:
            slot = (self.b_in + item.logical_index) % self.pool_segments
            last_use_by_slot[slot] = max(
                item.access_step, last_use_by_slot.get(slot, -1)
            )
        for output in self.output_first_writes:
            slot = (self.b_out + output.logical_index) % self.pool_segments
            last_use = last_use_by_slot.get(slot)
            if last_use is not None and output.access_step <= last_use:
                raise ValueError(
                    "output overwrites an input segment before its last read: "
                    f"slot={slot} write={output.access_step} last_read={last_use}"
                )

    def input_address(self, logical_index: int) -> SegmentAddress:
        """Returns one deterministic modulo input address."""
        physical = (self.b_in + logical_index) % self.pool_segments
        return SegmentAddress(logical_index, physical, physical * self.segment_bytes)

    def output_address(self, logical_index: int) -> SegmentAddress:
        """Returns one deterministic modulo output address."""
        physical = (self.b_out + logical_index) % self.pool_segments
        return SegmentAddress(logical_index, physical, physical * self.segment_bytes)

    def to_dict(self) -> dict[str, Any]:
        """Serializes bases, lifetimes, and every planned physical address."""
        return {
            "pool_segments": self.pool_segments,
            "segment_bytes": self.segment_bytes,
            "b_in": self.b_in,
            "b_out": self.b_out,
            "input_last_uses": [
                {
                    "logical_index": item.logical_index,
                    "access_step": item.access_step,
                    "physical_index": self.input_address(item.logical_index).physical_index,
                }
                for item in self.input_last_uses
            ],
            "output_first_writes": [
                {
                    "logical_index": item.logical_index,
                    "access_step": item.access_step,
                    "physical_index": self.output_address(item.logical_index).physical_index,
                }
                for item in self.output_first_writes
            ],
        }


def plan_circular_memory(
    input_last_uses: tuple[SegmentLifetime, ...],
    output_first_writes: tuple[SegmentLifetime, ...],
    pool_segments: int,
    segment_bytes: int,
    *,
    b_in: int = 0,
) -> CircularMemoryPlan:
    """Selects the smallest safe ``b_out`` by deterministic linear scan.

    A physical slot may be reused only when its input's last read is strictly
    earlier than the output's first write. Residual schedules express their
    longer input lifetime by assigning the add step as ``access_step``.
    """
    if pool_segments <= 0 or segment_bytes <= 0:
        raise ValueError("circular pool dimensions must be positive")
    if len({item.logical_index for item in input_last_uses}) != len(input_last_uses):
        raise ValueError("input lifetime indices must be unique")
    if len({item.logical_index for item in output_first_writes}) != len(
        output_first_writes
    ):
        raise ValueError("output lifetime indices must be unique")
    for b_out in range(pool_segments):
        try:
            return CircularMemoryPlan(
                pool_segments,
                segment_bytes,
                b_in % pool_segments,
                b_out,
                input_last_uses,
                output_first_writes,
            )
        except ValueError:
            continue
    raise ValueError("no safe circular output offset satisfies segment last-use")


def plan_minimum_circular_pool(
    input_last_uses: tuple[SegmentLifetime, ...],
    output_first_writes: tuple[SegmentLifetime, ...],
    segment_bytes: int,
) -> CircularMemoryPlan:
    """Finds the smallest safe pool, then its smallest safe output offset."""
    minimum = max(len(input_last_uses), len(output_first_writes), 1)
    maximum = max(1, len(input_last_uses) + len(output_first_writes))
    for pool_segments in range(minimum, maximum + 1):
        try:
            return plan_circular_memory(
                input_last_uses,
                output_first_writes,
                pool_segments,
                segment_bytes,
            )
        except ValueError:
            continue
    raise ValueError("no finite circular pool satisfies segment lifetimes")


@dataclass(frozen=True)
class FixedSegmentMemoryPlan:
    """Static workspace plus deterministic circular addressing parameters."""

    segment_lanes: int
    pool_segments: int
    specs: tuple[SegmentSpec, ...]

    def __post_init__(self) -> None:
        """Validates that every component is static and names are unique."""
        if self.segment_lanes <= 0 or self.pool_segments <= 0:
            raise ValueError("memory plan dimensions must be positive")
        names = [spec.name for spec in self.specs]
        if len(names) != len(set(names)):
            raise ValueError("segment spec names must be unique")

    @property
    def workspace_bytes(self) -> int:
        """Returns explicit scratch without arena or stack estimates."""
        return sum(spec.total_bytes for spec in self.specs)

    def circular_address(
        self, logical_index: int, segment_bytes: int, base_segment: int = 0
    ) -> SegmentAddress:
        """Maps an unbounded logical index into the finite segment pool."""
        if logical_index < 0 or segment_bytes <= 0:
            raise ValueError("logical index and segment byte width must be valid")
        physical = (base_segment + logical_index) % self.pool_segments
        return SegmentAddress(logical_index, physical, physical * segment_bytes)

    def valid_lanes(self, logical_segment: int, logical_elements: int) -> int:
        """Returns the lane mask prefix length for a possibly partial segment."""
        if logical_segment < 0 or logical_elements < 0:
            raise ValueError("logical segment and element count cannot be negative")
        remaining = logical_elements - logical_segment * self.segment_lanes
        return max(0, min(self.segment_lanes, remaining))

    def to_dict(self) -> dict[str, Any]:
        """Returns a machine-readable static memory plan."""
        return {
            "segment_lanes": self.segment_lanes,
            "pool_segments": self.pool_segments,
            "workspace_bytes": self.workspace_bytes,
            "buffers": [spec.to_dict() for spec in self.specs],
        }


def assert_non_overlapping_live_buffers(specs: tuple[SegmentSpec, ...]) -> None:
    """Checks that same-name buffers never describe conflicting live storage.

    Unique names are normally enforced by ``FixedSegmentMemoryPlan``.  This
    helper additionally serves schedule builders that concatenate subplans.
    """
    by_name: dict[str, SegmentSpec] = {}
    for spec in specs:
        prior = by_name.get(spec.name)
        if prior is not None:
            overlap = max(prior.lifetime_start, spec.lifetime_start) < min(
                prior.lifetime_end, spec.lifetime_end
            )
            if overlap:
                raise ValueError(f"live segment buffers overlap: {spec.name}")
        by_name[spec.name] = spec
