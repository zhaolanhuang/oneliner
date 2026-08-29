"""Static byte-accurate circular activation-pool planning for vMCU."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from .memory import align_up


class ScheduleSearchMode(str, Enum):
    """Search policies exposed by the Python and Rust build interfaces."""

    BOUNDED = "bounded"
    OPTIMAL = "optimal"
    GREEDY = "greedy"


@dataclass(frozen=True)
class VirtualTensor:
    """One static activation represented as circular logical bytes."""

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
class KernelAccessSchedule:
    """Last-read and first-write times for one fixed kernel schedule.

    Event arrays are byte addressed. Callers raise element lifetimes to the
    maximum lifetime of their segment before constructing this object.
    """

    name: str
    inputs: tuple[str, ...]
    output: str
    input_last_reads: tuple[tuple[str, tuple[int, ...]], ...]
    output_first_writes: tuple[int, ...]
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
        if not self.output_first_writes or any(
            step < 0 for step in self.output_first_writes
        ):
            raise ValueError("kernel output write events are invalid")
        if tuple(name for name, _ in self.input_last_reads) != self.inputs:
            raise ValueError("kernel read-event order must match its inputs")
        if any(not events or min(events) < 0 for _, events in self.input_last_reads):
            raise ValueError("kernel input read events are invalid")

    def reads_for(self, tensor: str) -> tuple[int, ...]:
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
            "input_last_reads": {
                name: list(events) for name, events in self.input_last_reads
            },
            "output_first_writes": list(self.output_first_writes),
        }


@dataclass(frozen=True)
class TensorPlacement:
    """One virtual tensor's base in a circular byte pool."""

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
) -> tuple[int, ...]:
    """Uses the maximum element lifetime for every byte in one segment."""
    events = tuple(element_last_reads)
    if not events or segment_bytes <= 0 or min(events) < 0:
        raise ValueError("segment lifetime inputs must be non-empty and non-negative")
    result = list(events)
    for start in range(0, len(events), segment_bytes):
        end = min(start + segment_bytes, len(events))
        result[start:end] = [max(events[start:end])] * (end - start)
    return tuple(result)


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
        if len(kernel.output_first_writes) != output.size_bytes:
            raise ValueError(f"kernel {kernel.name!r} output event size is invalid")
        for input_name in kernel.inputs:
            tensor = tensor_by_name.get(input_name)
            if tensor is None or kernel.name not in tensor.consumers:
                raise ValueError(f"kernel {kernel.name!r} input edge is invalid")
            if len(kernel.reads_for(input_name)) != tensor.size_bytes:
                raise ValueError(f"kernel {kernel.name!r} input event size is invalid")
    for tensor in tensors:
        if tensor.producer is not None and tensor.producer not in kernel_by_name:
            raise ValueError(f"tensor {tensor.name!r} producer is missing")
        if any(name not in kernel_by_name for name in tensor.consumers):
            raise ValueError(f"tensor {tensor.name!r} consumer is missing")
    return tensor_by_name, kernel_by_name


def _safe_output_base(
    *,
    pool_bytes: int,
    output_base: int,
    output: VirtualTensor,
    kernel: KernelAccessSchedule,
    live: Mapping[str, TensorPlacement],
    remaining: Mapping[str, frozenset[str]],
) -> bool:
    owners: dict[int, list[tuple[str, int]]] = {}
    for tensor_name, placement in live.items():
        for logical_byte in range(placement.size_bytes):
            owners.setdefault(placement.physical_byte(logical_byte), []).append(
                (tensor_name, logical_byte)
            )
    for output_byte, write_step in enumerate(kernel.output_first_writes):
        physical = (output_base + output_byte) % pool_bytes
        for tensor_name, logical_byte in owners.get(physical, ()):
            if remaining[tensor_name] != frozenset((kernel.name,)):
                return False
            if tensor_name not in kernel.inputs:
                return False
            if kernel.reads_for(tensor_name)[logical_byte] >= write_step:
                return False
    return not output.is_graph_output or output_base + output.size_bytes <= pool_bytes


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
                    live=live,
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
    """Independently replays topology and every physical overwrite."""
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
            live=live,
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
    """Finds the smallest verified pool reached by the selected search mode."""
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
