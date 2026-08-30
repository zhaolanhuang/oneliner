"""Static byte-accurate circular activation-pool planning for vMCU.

Paper correspondence (vMCU, MLSys 2024):
  * §2.4, PDF pp.3-4, Figure 1(c): partial input/output overlap and the
    seven-segment GEMM example.
  * §4, PDF pp.4-5, Equation (1): circular segment pool, row-major address
    mapping, input/output base offsets, and the no-overwrite constraint.
  * §5.2, PDF p.6, Equation (2): extending the overwrite constraint across a
    producer-consumer graph.

Engineering extension: the paper formulates offset selection as ILP and gives
an informal graph generalization. This module instead performs deterministic
greedy/bounded/exhaustive DAG search and independently replays physical range
overlaps at activation-segment granularity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from .memory import align_up


class ScheduleSearchMode(str, Enum):
    """Search policies exposed by the Python and Rust build interfaces.

    These modes are engineering alternatives to the ILP formulation suggested
    in vMCU §4 (PDF p.5); the paper does not define bounded/optimal/greedy modes.
    """

    BOUNDED = "bounded"
    OPTIMAL = "optimal"
    GREEDY = "greedy"


@dataclass(frozen=True)
class VirtualTensor:
    """One static activation represented as circular logical bytes.

    Paper correspondence: §4, PDF p.4, ``Pool[MemCap / Seg]`` and the
    row-major tensor-to-pool address formulation. ``producer``/``consumers``
    are the graph form of §5.2, PDF p.6, ``G=(V,E)``.
    """

    name: str
    size_bytes: int
    producer: str | None
    consumers: tuple[str, ...]
    shape: tuple[int, ...] = ()
    segment_bytes: int = 1
    is_graph_input: bool = False
    is_graph_output: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("virtual tensor name cannot be empty")
        if self.size_bytes <= 0 or self.segment_bytes <= 0:
            raise ValueError("virtual tensor byte sizes must be positive")
        if len(set(self.consumers)) != len(self.consumers):
            raise ValueError(f"virtual tensor {self.name!r} has duplicate consumers")
        if self.is_graph_input and self.producer is not None:
            raise ValueError("graph input cannot have a producer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "size_bytes": self.size_bytes,
            "segment_bytes": self.segment_bytes,
            "producer": self.producer,
            "consumers": list(self.consumers),
            "is_graph_input": self.is_graph_input,
            "is_graph_output": self.is_graph_output,
        }


@dataclass(frozen=True)
class SegmentReadSchedule:
    """Last-read events stored once per activation segment."""

    size_bytes: int
    segment_bytes: int
    segment_events: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.size_bytes <= 0 or self.segment_bytes <= 0:
            raise ValueError("read schedule byte sizes must be positive")
        segment_count = (self.size_bytes + self.segment_bytes - 1) // self.segment_bytes
        if len(self.segment_events) not in (1, segment_count):
            raise ValueError("read schedule does not cover its input segments")
        if min(self.segment_events) < 0:
            raise ValueError("read schedule events must be non-negative")

    @classmethod
    def constant(
        cls, size_bytes: int, segment_bytes: int, event: int
    ) -> "SegmentReadSchedule":
        return cls(size_bytes, segment_bytes, (event,))

    def event_at(self, logical_byte: int) -> int:
        if not 0 <= logical_byte < self.size_bytes:
            raise IndexError(logical_byte)
        if len(self.segment_events) == 1:
            return self.segment_events[0]
        return self.segment_events[logical_byte // self.segment_bytes]

    def next_event_change(self, logical_byte: int) -> int:
        if len(self.segment_events) == 1:
            return self.size_bytes
        return min(
            self.size_bytes,
            (logical_byte // self.segment_bytes + 1) * self.segment_bytes,
        )

    def __iter__(self):
        return (self.event_at(index) for index in range(self.size_bytes))


@dataclass(frozen=True)
class OutputWriteSchedule:
    """Compact grouped-affine first-write events with an explicit fallback."""

    size_bytes: int
    first_event: int
    byte_stride: int
    group_bytes: int
    group_gap: int = 0
    explicit_events: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.size_bytes <= 0 or self.group_bytes <= 0:
            raise ValueError("write schedule byte sizes must be positive")
        if self.explicit_events:
            if (
                len(self.explicit_events) != self.size_bytes
                or min(self.explicit_events) < 0
            ):
                raise ValueError("explicit write schedule is invalid")
        elif self.first_event < 0 or self.byte_stride < 0 or self.group_gap < 0:
            raise ValueError("affine write schedule is invalid")

    @classmethod
    def affine(
        cls,
        size_bytes: int,
        first_event: int,
        byte_stride: int,
        *,
        group_bytes: int | None = None,
        group_gap: int = 0,
    ) -> "OutputWriteSchedule":
        return cls(
            size_bytes,
            first_event,
            byte_stride,
            group_bytes or size_bytes,
            group_gap,
        )

    @classmethod
    def explicit(cls, events: Iterable[int]) -> "OutputWriteSchedule":
        event_items = tuple(events)
        return cls(len(event_items), 0, 0, max(1, len(event_items)), 0, event_items)

    def event_at(self, logical_byte: int) -> int:
        if not 0 <= logical_byte < self.size_bytes:
            raise IndexError(logical_byte)
        if self.explicit_events:
            return self.explicit_events[logical_byte]
        return (
            self.first_event
            + logical_byte * self.byte_stride
            + (logical_byte // self.group_bytes) * self.group_gap
        )

    def minimum_event(self, start: int, end: int) -> int:
        if not 0 <= start < end <= self.size_bytes:
            raise IndexError((start, end))
        if self.explicit_events:
            return min(self.explicit_events[start:end])
        return self.event_at(start)

    def __iter__(self):
        return (self.event_at(index) for index in range(self.size_bytes))

    def to_dict(self) -> dict[str, object]:
        if self.explicit_events:
            return {"kind": "explicit", "events": list(self.explicit_events)}
        return {
            "kind": "grouped_affine",
            "size_bytes": self.size_bytes,
            "first_event": self.first_event,
            "byte_stride": self.byte_stride,
            "group_bytes": self.group_bytes,
            "group_gap": self.group_gap,
        }


@dataclass(frozen=True)
class KernelAccessSchedule:
    """Last-read and first-write times for one fixed kernel schedule.

    Read lifetimes are compressed to one event per segment. Output writes use
    a grouped-affine formula for production kernels.

    Paper correspondence: §4, PDF pp.4-5, iteration instances ``S[i]``, affine
    access functions, row-major linear addresses, and Equation (1)'s ordering
    constraint. Compressed segment events and grouped-affine writes are this
    implementation's executable representation of those relations, not data
    structures prescribed by the paper.
    """

    name: str
    inputs: tuple[str, ...]
    output: str
    input_last_reads: tuple[tuple[str, SegmentReadSchedule], ...]
    output_first_writes: OutputWriteSchedule
    segment_bytes: int
    workspace_bytes: int = 0
    kind: str = "unknown"

    def __post_init__(self) -> None:
        if not self.name or not self.output:
            raise ValueError("kernel and output names cannot be empty")
        if not self.inputs or len(set(self.inputs)) != len(self.inputs):
            raise ValueError("kernel inputs must be non-empty and unique")
        if self.segment_bytes <= 0 or self.workspace_bytes < 0:
            raise ValueError("kernel segment/workspace sizes are invalid")
        if tuple(name for name, _ in self.input_last_reads) != self.inputs:
            raise ValueError("kernel read-event order must match its inputs")

    def reads_for(self, tensor: str) -> SegmentReadSchedule:
        for name, events in self.input_last_reads:
            if name == tensor:
                return events
        raise KeyError(tensor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "inputs": list(self.inputs),
            "output": self.output,
            "segment_bytes": self.segment_bytes,
            "workspace_bytes": self.workspace_bytes,
            "output_write_schedule": self.output_first_writes.to_dict(),
        }


@dataclass(frozen=True)
class TensorPlacement:
    """One virtual tensor's base in a circular byte pool.

    Paper correspondence: §4, PDF p.5, input/output offsets ``bIn`` and
    ``bOut``. ``base`` is the physical realization of one such offset.
    """

    tensor: str
    base: int
    size_bytes: int
    pool_bytes: int

    def __post_init__(self) -> None:
        if not self.tensor or self.size_bytes <= 0 or self.pool_bytes <= 0:
            raise ValueError("tensor placement dimensions are invalid")
        if self.size_bytes > self.pool_bytes:
            raise ValueError("one virtual tensor cannot exceed the circular pool")
        if not 0 <= self.base < self.pool_bytes:
            raise ValueError("tensor base must address the circular pool")

    @property
    def wraps(self) -> bool:
        return self.base + self.size_bytes > self.pool_bytes

    def physical_byte(self, logical_byte: int) -> int:
        if not 0 <= logical_byte < self.size_bytes:
            raise ValueError("logical byte is outside the virtual tensor")
        return (self.base + logical_byte) % self.pool_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor,
            "base": self.base,
            "size_bytes": self.size_bytes,
            "wraps": self.wraps,
        }


@dataclass(frozen=True)
class KernelPlacement:
    kernel: str
    order: int
    output_base: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "kernel": self.kernel,
            "order": self.order,
            "output_base": self.output_base,
        }


@dataclass(frozen=True)
class CompactGraphPlan:
    """Verified full-DAG activation-pool plan."""

    logical_pool_bytes: int
    allocated_pool_bytes: int
    alignment: int
    search_mode: ScheduleSearchMode
    optimal: bool
    explored_states: int
    state_limit: int | None
    tensors: tuple[VirtualTensor, ...]
    kernels: tuple[KernelAccessSchedule, ...]
    placements: tuple[TensorPlacement, ...]
    execution: tuple[KernelPlacement, ...]

    @property
    def maximum_workspace_bytes(self) -> int:
        return max((item.workspace_bytes for item in self.kernels), default=0)

    @property
    def output_requires_normalization(self) -> bool:
        return False

    def placement_for(self, tensor: str) -> TensorPlacement:
        for placement in self.placements:
            if placement.tensor == tensor:
                return placement
        raise KeyError(tensor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_pool_bytes": self.logical_pool_bytes,
            "allocated_pool_bytes": self.allocated_pool_bytes,
            "alignment": self.alignment,
            "search": {
                "mode": self.search_mode.value,
                "optimal": self.optimal,
                "explored_states": self.explored_states,
                "state_limit": self.state_limit,
            },
            "maximum_workspace_bytes": self.maximum_workspace_bytes,
            "output_view_base": self.placement_for(
                next(item.name for item in self.tensors if item.is_graph_output)
            ).base,
            "output_requires_normalization": self.output_requires_normalization,
            "tensors": [item.to_dict() for item in self.tensors],
            "kernels": [item.to_dict() for item in self.kernels],
            "placements": [item.to_dict() for item in self.placements],
            "execution": [item.to_dict() for item in self.execution],
        }


def segment_last_reads(
    element_last_reads: Iterable[int], segment_bytes: int
) -> SegmentReadSchedule:
    """Compresses element lifetimes to one maximum event per segment.

    Paper correspondence: Introduction, PDF p.2, paragraph beginning "the data
    elements in input tensors may have different lifetime": segment lifetime
    is defined as the maximum lifetime of its elements. Section 5.3, PDF p.7,
    discusses the footprint/latency trade-off controlled by segment size.
    """
    events = tuple(element_last_reads)
    if not events or segment_bytes <= 0 or min(events) < 0:
        raise ValueError("segment lifetime inputs must be non-empty and non-negative")
    segment_events = []
    for start in range(0, len(events), segment_bytes):
        end = min(start + segment_bytes, len(events))
        segment_events.append(max(events[start:end]))
    compressed = tuple(segment_events)
    if len(set(compressed)) == 1:
        compressed = compressed[:1]
    return SegmentReadSchedule(len(events), segment_bytes, compressed)


def _validate_graph(
    tensors: tuple[VirtualTensor, ...], kernels: tuple[KernelAccessSchedule, ...]
) -> tuple[dict[str, VirtualTensor], dict[str, KernelAccessSchedule]]:
    tensor_by_name = {item.name: item for item in tensors}
    kernel_by_name = {item.name: item for item in kernels}
    if len(tensor_by_name) != len(tensors) or len(kernel_by_name) != len(kernels):
        raise ValueError("compact graph names must be unique")
    if sum(item.is_graph_input for item in tensors) != 1:
        raise ValueError("compact graph requires exactly one input")
    if sum(item.is_graph_output for item in tensors) != 1:
        raise ValueError("compact graph requires exactly one output")
    for kernel in kernels:
        output = tensor_by_name.get(kernel.output)
        if output is None or output.producer != kernel.name:
            raise ValueError(f"kernel {kernel.name!r} output edge is invalid")
        if kernel.output_first_writes.size_bytes != output.size_bytes:
            raise ValueError(f"kernel {kernel.name!r} output event size is invalid")
        for input_name in kernel.inputs:
            tensor = tensor_by_name.get(input_name)
            if tensor is None or kernel.name not in tensor.consumers:
                raise ValueError(f"kernel {kernel.name!r} input edge is invalid")
            if kernel.reads_for(input_name).size_bytes != tensor.size_bytes:
                raise ValueError(f"kernel {kernel.name!r} input event size is invalid")
    for tensor in tensors:
        if tensor.producer is not None and tensor.producer not in kernel_by_name:
            raise ValueError(f"tensor {tensor.name!r} producer is missing")
        if any(name not in kernel_by_name for name in tensor.consumers):
            raise ValueError(f"tensor {tensor.name!r} consumer is missing")
    return tensor_by_name, kernel_by_name


@dataclass(frozen=True)
class _PhysicalSpan:
    start: int
    end: int
    logical_start: int


def _physical_spans(
    base: int, size_bytes: int, pool_bytes: int
) -> tuple[_PhysicalSpan, ...]:
    tail = min(size_bytes, pool_bytes - base)
    first = _PhysicalSpan(base, base + tail, 0)
    if tail == size_bytes:
        return (first,)
    return (first, _PhysicalSpan(0, size_bytes - tail, tail))


def _live_layout(
    live: Mapping[str, TensorPlacement],
) -> tuple[tuple[str, tuple[_PhysicalSpan, ...]], ...]:
    return tuple(
        (
            tensor_name,
            _physical_spans(placement.base, placement.size_bytes, placement.pool_bytes),
        )
        for tensor_name, placement in live.items()
    )


def _safe_output_base(
    *,
    pool_bytes: int,
    output_base: int,
    output: VirtualTensor,
    kernel: KernelAccessSchedule,
    live_layout: tuple[tuple[str, tuple[_PhysicalSpan, ...]], ...],
    remaining: Mapping[str, frozenset[str]],
) -> bool:
    """Checks one base through circular interval and segment intersections.

    Paper correspondence: §4, PDF p.5, Equation (1), and §2.4, PDF pp.3-4,
    Figure 1(c)'s warning that too few empty output segments silently overwrite
    live input. The ``remaining`` test is the §5.2/Equation (2) DAG extension:
    a branch tensor cannot be overwritten before its final consumer.
    """
    if output.is_graph_output and output_base + output.size_bytes > pool_bytes:
        return False
    output_spans = _physical_spans(output_base, output.size_bytes, pool_bytes)
    for tensor_name, input_spans in live_layout:
        reads = None
        for output_span in output_spans:
            for input_span in input_spans:
                overlap_start = max(output_span.start, input_span.start)
                overlap_end = min(output_span.end, input_span.end)
                if overlap_start >= overlap_end:
                    continue
                if (
                    remaining[tensor_name] != frozenset((kernel.name,))
                    or tensor_name not in kernel.inputs
                ):
                    return False
                if reads is None:
                    reads = kernel.reads_for(tensor_name)
                input_byte = input_span.logical_start + overlap_start - input_span.start
                output_byte = (
                    output_span.logical_start + overlap_start - output_span.start
                )
                bytes_left = overlap_end - overlap_start
                while bytes_left:
                    run_bytes = min(
                        bytes_left,
                        reads.next_event_change(input_byte) - input_byte,
                    )
                    if reads.event_at(input_byte) >= kernel.output_first_writes.minimum_event(
                        output_byte, output_byte + run_bytes
                    ):
                        return False
                    input_byte += run_bytes
                    output_byte += run_bytes
                    bytes_left -= run_bytes
    return True


@dataclass
class _SearchCounter:
    explored: int = 0
    exhausted: bool = False


def _search_capacity(
    tensors: tuple[VirtualTensor, ...],
    kernels: tuple[KernelAccessSchedule, ...],
    pool_bytes: int,
    *,
    counter: _SearchCounter,
    state_limit: int | None,
    greedy: bool,
) -> tuple[tuple[TensorPlacement, ...], tuple[KernelPlacement, ...]] | None:
    tensor_by_name, _ = _validate_graph(tensors, kernels)
    graph_input = next(item for item in tensors if item.is_graph_input)
    initial = TensorPlacement(graph_input.name, 0, graph_input.size_bytes, pool_bytes)
    solution: tuple[dict[str, TensorPlacement], list[KernelPlacement]] | None = None

    def visit(
        produced: frozenset[str],
        executed: frozenset[str],
        remaining: dict[str, frozenset[str]],
        live: dict[str, TensorPlacement],
        placements: dict[str, TensorPlacement],
        execution: list[KernelPlacement],
    ) -> bool:
        nonlocal solution
        if state_limit is not None and counter.explored >= state_limit:
            counter.exhausted = True
            return False
        counter.explored += 1
        if len(executed) == len(kernels):
            solution = placements.copy(), execution.copy()
            return True
        ready = sorted(
            (
                item
                for item in kernels
                if item.name not in executed
                and all(name in produced for name in item.inputs)
            ),
            key=lambda item: item.name,
        )
        if greedy:
            ready = ready[:1]
        current_layout = _live_layout(live)
        for kernel in ready:
            output = tensor_by_name[kernel.output]
            if greedy or state_limit is not None:
                base_candidates = sorted(
                    {
                        0,
                        *(
                            value % pool_bytes
                            for placement in live.values()
                            for value in (
                                placement.base,
                                placement.base + placement.size_bytes,
                                placement.base - output.size_bytes,
                            )
                        ),
                    }
                )
            else:
                preferred = {
                    0,
                    *(
                        value % pool_bytes
                        for placement in live.values()
                        for value in (
                            placement.base,
                            placement.base + placement.size_bytes,
                            placement.base - output.size_bytes,
                        )
                    ),
                }
                base_candidates = sorted(preferred) + [
                    base for base in range(pool_bytes) if base not in preferred
                ]
            for base in base_candidates:
                if state_limit is not None and counter.explored >= state_limit:
                    counter.exhausted = True
                    return False
                counter.explored += 1
                if not _safe_output_base(
                    pool_bytes=pool_bytes,
                    output_base=base,
                    output=output,
                    kernel=kernel,
                    live_layout=current_layout,
                    remaining=remaining,
                ):
                    continue
                placement = TensorPlacement(
                    output.name, base, output.size_bytes, pool_bytes
                )
                next_remaining = remaining.copy()
                next_live = live.copy()
                for input_name in kernel.inputs:
                    next_remaining[input_name] -= {kernel.name}
                    if not next_remaining[input_name]:
                        next_live.pop(input_name, None)
                next_live[output.name] = placement
                next_placements = placements.copy()
                next_placements[output.name] = placement
                if visit(
                    produced | {output.name},
                    executed | {kernel.name},
                    next_remaining,
                    next_live,
                    next_placements,
                    execution + [KernelPlacement(kernel.name, len(execution), base)],
                ):
                    return True
            if greedy:
                break
        return False

    found = visit(
        frozenset((graph_input.name,)),
        frozenset(),
        {item.name: frozenset(item.consumers) for item in tensors},
        {graph_input.name: initial},
        {graph_input.name: initial},
        [],
    )
    if not found or solution is None:
        return None
    placements, execution = solution
    return (
        tuple(placements[item.name] for item in tensors),
        tuple(execution),
    )


def replay_compact_graph_plan(plan: CompactGraphPlan) -> None:
    """Independently replays topology and every physical overlap.

    Paper correspondence: §4, PDF p.5, Equation (1), and §5.2, PDF p.6,
    Equation (2). Engineering extension: the paper relies on solved affine/ILP
    constraints; vMCU-on-IREE replays the concrete schedule by circular range
    and activation segment as a compiler safety proof before emission.
    """
    tensor_by_name, kernel_by_name = _validate_graph(plan.tensors, plan.kernels)
    placement_by_name = {item.tensor: item for item in plan.placements}
    graph_input = next(item for item in plan.tensors if item.is_graph_input)
    if placement_by_name[graph_input.name].base != 0:
        raise ValueError("graph input must start at byte zero")
    produced = {graph_input.name}
    remaining = {
        item.name: frozenset(item.consumers) for item in plan.tensors
    }
    live = {graph_input.name: placement_by_name[graph_input.name]}
    for expected_order, item in enumerate(plan.execution):
        if item.order != expected_order:
            raise ValueError("kernel execution order is not contiguous")
        kernel = kernel_by_name[item.kernel]
        if any(name not in produced for name in kernel.inputs):
            raise ValueError(f"kernel {kernel.name!r} executes before its inputs")
        output = tensor_by_name[kernel.output]
        placement = placement_by_name[output.name]
        if placement.base != item.output_base or not _safe_output_base(
            pool_bytes=plan.logical_pool_bytes,
            output_base=placement.base,
            output=output,
            kernel=kernel,
            live_layout=_live_layout(live),
            remaining=remaining,
        ):
            raise ValueError(f"kernel {kernel.name!r} contains an overwrite hazard")
        for input_name in kernel.inputs:
            remaining[input_name] -= {kernel.name}
            if not remaining[input_name]:
                live.pop(input_name, None)
        live[output.name] = placement
        produced.add(output.name)
    graph_output = next(item for item in plan.tensors if item.is_graph_output)
    if graph_output.name not in produced:
        raise ValueError("graph output was never produced")


def plan_compact_graph(
    tensors: Iterable[VirtualTensor],
    kernels: Iterable[KernelAccessSchedule],
    *,
    search_mode: ScheduleSearchMode | str = ScheduleSearchMode.BOUNDED,
    search_state_limit: int = 1_000_000,
    alignment: int = 64,
) -> CompactGraphPlan:
    """Finds the smallest verified pool reached by the selected search mode.

    Paper correspondence: §4, PDF p.5, minimizes ``bIn-bOut`` under Equation
    (1), and derives ``max(MN,MK)+min(N,K)-1`` for GEMM/Figure 3.

    Engineering extension: ``greedy``, ``bounded``, and ``optimal`` are not
    algorithms named by the paper. They replace its ILP suggestion with an
    explicit DAG topology/base search suitable for this compiler pipeline.
    """
    tensor_items = tuple(tensors)
    kernel_items = tuple(kernels)
    _validate_graph(tensor_items, kernel_items)
    try:
        mode = ScheduleSearchMode(search_mode)
    except ValueError as error:
        raise ValueError(f"unsupported schedule search mode: {search_mode}") from error
    if search_state_limit <= 0:
        raise ValueError("schedule search state limit must be positive")
    align_up(0, alignment)
    lower = max(item.size_bytes for item in tensor_items)
    upper = sum(item.size_bytes for item in tensor_items)
    counter = _SearchCounter()
    result = None
    capacity = lower
    state_limit = search_state_limit if mode == ScheduleSearchMode.BOUNDED else None
    if mode in (ScheduleSearchMode.GREEDY, ScheduleSearchMode.BOUNDED):
        # Greedy feasibility is monotone for the non-wrapping boundary bases
        # tried above. Binary search avoids a byte-by-byte capacity sweep on
        # real activation tensors while retaining deterministic placement.
        low_capacity = lower
        high_capacity = upper
        best = None
        # Bounded mode starts from the deterministic greedy best-known plan.
        # It then permits full topology/base branching at each binary probe
        # until the explicit state budget is consumed.
        if mode == ScheduleSearchMode.BOUNDED:
            greedy_counter = _SearchCounter()
            greedy_low = lower
            greedy_high = upper
            while greedy_low <= greedy_high:
                candidate_capacity = (greedy_low + greedy_high) // 2
                candidate_result = _search_capacity(
                    tensor_items,
                    kernel_items,
                    candidate_capacity,
                    counter=greedy_counter,
                    state_limit=None,
                    greedy=True,
                )
                if candidate_result is None:
                    greedy_low = candidate_capacity + 1
                else:
                    best = (candidate_capacity, candidate_result)
                    greedy_high = candidate_capacity - 1
            if best is None:
                raise ValueError("no safe greedy seed for bounded compact scheduling")
            high_capacity = best[0] - 1
        while low_capacity <= high_capacity:
            candidate_capacity = (low_capacity + high_capacity) // 2
            candidate_result = _search_capacity(
                tensor_items,
                kernel_items,
                candidate_capacity,
                counter=counter,
                state_limit=state_limit,
                greedy=mode == ScheduleSearchMode.GREEDY,
            )
            if counter.exhausted:
                break
            if candidate_result is None:
                low_capacity = candidate_capacity + 1
            else:
                best = (candidate_capacity, candidate_result)
                high_capacity = candidate_capacity - 1
        if best is not None:
            capacity, result = best
    else:
        for capacity in range(lower, upper + 1):
            result = _search_capacity(
                tensor_items,
                kernel_items,
                capacity,
                counter=counter,
                state_limit=state_limit,
                greedy=False,
            )
            if result is not None:
                break
            if counter.exhausted:
                break
    if result is None and mode == ScheduleSearchMode.BOUNDED:
        # A deterministic greedy seed guarantees a safe best-known fallback.
        counter.exhausted = True
        greedy_counter = _SearchCounter()
        low_capacity = lower
        high_capacity = upper
        best = None
        while low_capacity <= high_capacity:
            candidate_capacity = (low_capacity + high_capacity) // 2
            candidate_result = _search_capacity(
                tensor_items,
                kernel_items,
                candidate_capacity,
                counter=greedy_counter,
                state_limit=None,
                greedy=True,
            )
            if candidate_result is None:
                low_capacity = candidate_capacity + 1
            else:
                best = (candidate_capacity, candidate_result)
                high_capacity = candidate_capacity - 1
        if best is not None:
            capacity, result = best
    if result is None:
        raise ValueError("no safe compact activation-pool schedule was found")
    placements, execution = result
    plan = CompactGraphPlan(
        logical_pool_bytes=capacity,
        allocated_pool_bytes=align_up(capacity, alignment),
        alignment=alignment,
        search_mode=mode,
        optimal=mode == ScheduleSearchMode.OPTIMAL,
        explored_states=counter.explored,
        state_limit=search_state_limit if mode == ScheduleSearchMode.BOUNDED else None,
        tensors=tensor_items,
        kernels=kernel_items,
        placements=placements,
        execution=execution,
    )
    replay_compact_graph_plan(plan)
    return plan
