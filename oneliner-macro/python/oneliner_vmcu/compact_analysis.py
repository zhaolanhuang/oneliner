"""Builds a byte-addressed compact activation DAG from semantic matches."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable

from iree.compiler import ir

from .compact_memory import (
    CompactGraphPlan,
    KernelAccessSchedule,
    ScheduleSearchMode,
    VirtualTensor,
    plan_compact_graph,
    segment_last_reads,
)
from .ir_utils import body_operations, generic_io, owner_operation, tensor_shape
from .model import (
    Analysis,
    Conv2DMatch,
    DepthwiseConv2DMatch,
    FullyConnectedMatch,
    InvertedBottleneckMatch,
    PatternMatch,
)


_VIEW_OPERATIONS = frozenset(
    (
        "tensor.cast",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.reshape",
        "flow.tensor.reshape",
    )
)


@dataclass(frozen=True)
class CompactAnalysis:
    """Planner result plus stable candidate/tensor correspondence."""

    plan: CompactGraphPlan
    candidate_order: tuple[str, ...]
    input_values: tuple[tuple[str, ...], ...]
    output_tensors: tuple[str, ...]
    boundaries: tuple["MaterializedBoundary", ...]


@dataclass(frozen=True)
class MaterializedBoundary:
    name: str
    source_tensors: tuple[str, ...]
    source_values: tuple[ir.Value, ...]
    target_value: ir.Value
    output_shape: tuple[int, ...]
    direct_kind: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serializes boundary semantics without retaining an MLIR handle."""
        operation = owner_operation(self.target_value)
        return {
            "name": self.name,
            "source_tensors": list(self.source_tensors),
            "target_operation": operation.name if operation is not None else None,
            "output_shape": list(self.output_shape),
            "output_bytes": prod(self.output_shape),
            "lowering": self.direct_kind or "materialized",
        }


_DIRECT_SCALAR_OPERATIONS = frozenset(
    (
        "arith.addi",
        "arith.andi",
        "arith.cmpi",
        "arith.extsi",
        "arith.extui",
        "arith.maxsi",
        "arith.minsi",
        "arith.muli",
        "arith.select",
        "arith.shrsi",
        "arith.shrui",
        "arith.subi",
        "arith.trunci",
    )
)


def _direct_boundary_kind(
    target: ir.Value, sources: frozenset[ir.Value]
) -> str | None:
    """Recognizes scalarizable identity residual trees and terminal pooling."""
    visiting: set[ir.Value] = set()

    def visit(value: ir.Value) -> tuple[bool, bool]:
        if value in sources:
            return True, False
        if value in visiting:
            return False, False
        visiting.add(value)
        owner = owner_operation(value)
        if owner is None:
            return False, False
        if owner.name in _VIEW_OPERATIONS:
            return visit(owner.operands[0])
        if owner.name == "linalg.generic":
            inputs, outputs = generic_io(owner)
            if len(outputs) != 1 or len(owner.results) != 1:
                return False, False
            output_shape = tensor_shape(owner.results[0], str(ir.RankedTensorType(owner.results[0].type).element_type))
            identity = f"affine_map<({', '.join(f'd{i}' for i in range(len(output_shape)))}) -> ({', '.join(f'd{i}' for i in range(len(output_shape)))})>"
            maps = tuple(str(item) for item in owner.attributes["indexing_maps"])
            iterators = tuple(str(item) for item in owner.attributes["iterator_types"])
            block, operations = body_operations(owner, "compact boundary")
            output_argument = block.arguments[len(inputs)]
            if (
                maps != (identity,) * (len(inputs) + 1)
                or iterators != ("#linalg.iterator_type<parallel>",) * len(output_shape)
                or list(output_argument.uses)
                or not operations
                or operations[-1].name != "linalg.yield"
                or any(item.name not in _DIRECT_SCALAR_OPERATIONS for item in operations[:-1])
            ):
                return False, False
            results = [visit(item) for item in inputs]
            return all(item[0] for item in results), any(item[1] for item in results)
        if owner.name == "linalg.pooling_nhwc_sum":
            inputs = list(owner.opview.inputs)
            if len(inputs) != 2 or ir.RankedTensorType(inputs[0].type).rank != 4:
                return False, False
            kernel_shape = tuple(int(item) for item in ir.RankedTensorType(inputs[1].type).shape)
            if len(kernel_shape) != 2 or any(item <= 0 for item in kernel_shape):
                return False, False
            supported, nested_pool = visit(inputs[0])
            return supported and not nested_pool, True
        return False, False

    supported, has_pooling = visit(target)
    if not supported:
        return None
    return "pooling_sum_expression" if has_pooling else "elementwise_expression"


def _output_value(candidate: PatternMatch) -> ir.Value:
    if isinstance(candidate, InvertedBottleneckMatch):
        if candidate.residual is not None:
            return candidate.residual.final_operation.results[0]
        return candidate.projection.rescale.results[0]
    if isinstance(candidate, (Conv2DMatch, DepthwiseConv2DMatch, FullyConnectedMatch)):
        return candidate.rescale.results[0]
    raise ValueError(f"unsupported compact candidate {candidate.kind!r}")


def _input_values(candidate: PatternMatch) -> tuple[ir.Value, ...]:
    if isinstance(candidate, InvertedBottleneckMatch):
        values = [candidate.expansion.input]
        residual = candidate.residual_input
        if residual is not None and residual != values[0]:
            values.append(residual)
        return tuple(values)
    if isinstance(candidate, (Conv2DMatch, DepthwiseConv2DMatch, FullyConnectedMatch)):
        return (candidate.input,)
    raise ValueError(f"unsupported compact candidate {candidate.kind!r}")


def _strip_views_and_padding(value: ir.Value) -> ir.Value:
    """Folds no-copy views and explicit static padding into kernel indexing."""
    current = value
    while True:
        owner = owner_operation(current)
        if owner is None:
            return current
        if owner.name == "tensor.pad":
            current = owner.operands[0]
            continue
        if owner.name in _VIEW_OPERATIONS and owner.operands:
            current = owner.operands[0]
            continue
        return current


def _shape(value: ir.Value) -> tuple[int, ...]:
    return tensor_shape(value, "i8")


def _activation_sources(
    value: ir.Value,
    candidate_outputs: dict[ir.Value, PatternMatch],
    seen: set[ir.Value] | None = None,
) -> set[ir.Value]:
    """Finds nearest candidate/import activations through an unsupported op."""
    if value in candidate_outputs:
        return {value}
    visited = set() if seen is None else seen
    if value in visited:
        return set()
    visited.add(value)
    owner = owner_operation(value)
    if owner is None or owner.name == "hal.tensor.import":
        return {value}
    if owner.name == "arith.constant" or owner.name == "tensor.empty":
        return set()
    if owner.name == "tensor.pad" or owner.name in _VIEW_OPERATIONS:
        return _activation_sources(owner.operands[0], candidate_outputs, visited)
    try:
        operands = list(owner.opview.inputs)
    except (AttributeError, ValueError):
        operands = list(owner.operands)
    result: set[ir.Value] = set()
    for operand in operands:
        try:
            ir.RankedTensorType(operand.type)
        except ValueError:
            continue
        result.update(_activation_sources(operand, candidate_outputs, visited))
    return result


def _flatten_nhwc(shape: tuple[int, int, int, int], indices: tuple[int, int, int, int]) -> int:
    n, h, w, c = indices
    return ((n * shape[1] + h) * shape[2] + w) * shape[3] + c


def _conv_events(
    input_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    kernel_shape: tuple[int, int, int, int],
    strides: tuple[int, int],
    dilations: tuple[int, int],
    padding_low: tuple[int, ...],
    *,
    depthwise: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    reads = [-1] * prod(input_shape)
    writes = [0] * prod(output_shape)
    step = 0
    kh_extent, kw_extent = kernel_shape[:2]
    for n in range(output_shape[0]):
        for oh in range(output_shape[1]):
            for ow in range(output_shape[2]):
                for oc in range(output_shape[3]):
                    channels: Iterable[int] = (oc,) if depthwise else range(input_shape[3])
                    for kh in range(kh_extent):
                        ih = oh * strides[0] + kh * dilations[0] - padding_low[1]
                        if not 0 <= ih < input_shape[1]:
                            continue
                        for kw in range(kw_extent):
                            iw = ow * strides[1] + kw * dilations[1] - padding_low[2]
                            if not 0 <= iw < input_shape[2]:
                                continue
                            for ic in channels:
                                step += 1
                                reads[_flatten_nhwc(input_shape, (n, ih, iw, ic))] = step
                    step += 1
                    writes[_flatten_nhwc(output_shape, (n, oh, ow, oc))] = step
    # Every logical input byte belongs to the activation even if geometry does
    # not read it. Such bytes expire at entry and may be overwritten at step 0.
    return tuple(max(0, value) for value in reads), tuple(writes)


def _ibn_events(
    candidate: InvertedBottleneckMatch,
    input_shape: tuple[int, int, int, int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    output_shape = candidate.output_shape
    depthwise = candidate.depthwise
    kernel_h, kernel_w = depthwise.weight_shape[:2]
    reads = [-1] * prod(input_shape)
    writes = [0] * prod(output_shape)
    step = 0
    segment_lanes = min(input_shape[3], output_shape[3])
    expanded_channels = candidate.expansion.output_shape[3]
    for n in range(output_shape[0]):
        for oh in range(output_shape[1]):
            for ow in range(output_shape[2]):
                # Match the emitter's chunk -> patch -> lane -> input-channel
                # order exactly. Last-read timestamps are only a valid overwrite
                # proof when they use the same sequence as the physical kernel.
                for chunk in range(0, expanded_channels, segment_lanes):
                    for kh in range(kernel_h):
                        ih = (
                            oh * depthwise.strides[0]
                            + kh * depthwise.dilations[0]
                            - depthwise.padding_low[1]
                        )
                        if not 0 <= ih < input_shape[1]:
                            continue
                        for kw in range(kernel_w):
                            iw = (
                                ow * depthwise.strides[1]
                                + kw * depthwise.dilations[1]
                                - depthwise.padding_low[2]
                            )
                            if not 0 <= iw < input_shape[2]:
                                continue
                            valid_lanes = min(segment_lanes, expanded_channels - chunk)
                            for _lane in range(valid_lanes):
                                for ic in range(input_shape[3]):
                                    step += 1
                                    reads[
                                        _flatten_nhwc(input_shape, (n, ih, iw, ic))
                                    ] = step
                for oc in range(output_shape[3]):
                    if candidate.residual is not None:
                        step += 1
                        reads[_flatten_nhwc(input_shape, (n, oh, ow, oc))] = step
                    step += 1
                    writes[_flatten_nhwc(output_shape, (n, oh, ow, oc))] = step
    return tuple(max(0, value) for value in reads), tuple(writes)


def _fc_events(candidate: FullyConnectedMatch) -> tuple[tuple[int, ...], tuple[int, ...]]:
    reads = [0] * (candidate.rows * candidate.input_channels)
    writes = [0] * prod(candidate.output_shape)
    step = 0
    for row in range(candidate.rows):
        for output_channel in range(candidate.output_channels):
            for input_channel in range(candidate.input_channels):
                step += 1
                reads[row * candidate.input_channels + input_channel] = step
            step += 1
            writes[row * candidate.output_channels + output_channel] = step
    return tuple(reads), tuple(writes)


def _candidate_events(
    candidate: PatternMatch, input_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    if isinstance(candidate, InvertedBottleneckMatch):
        reads, writes = _ibn_events(candidate, input_shape)  # type: ignore[arg-type]
        return reads, writes, candidate.output_shape, min(
            input_shape[-1], candidate.output_shape[-1]
        )
    if isinstance(candidate, Conv2DMatch):
        reads, writes = _conv_events(
            input_shape,  # type: ignore[arg-type]
            candidate.output_shape,
            candidate.weight_shape,
            candidate.strides,
            candidate.dilations,
            candidate.padding_low,
            depthwise=False,
        )
        return reads, writes, candidate.output_shape, min(
            input_shape[-1], candidate.output_shape[-1]
        )
    if isinstance(candidate, DepthwiseConv2DMatch):
        reads, writes = _conv_events(
            input_shape,  # type: ignore[arg-type]
            candidate.output_shape,
            candidate.weight_shape,
            candidate.strides,
            candidate.dilations,
            candidate.padding_low,
            depthwise=True,
        )
        return reads, writes, candidate.output_shape, candidate.output_shape[-1]
    if isinstance(candidate, FullyConnectedMatch):
        reads, writes = _fc_events(candidate)
        return reads, writes, candidate.output_shape, min(
            candidate.input_channels, candidate.output_channels
        )
    raise ValueError(f"unsupported compact candidate {candidate.kind!r}")


def _candidate_output_shape(candidate: PatternMatch) -> tuple[int, ...]:
    if isinstance(candidate, InvertedBottleneckMatch):
        return candidate.output_shape
    if isinstance(candidate, (Conv2DMatch, DepthwiseConv2DMatch)):
        return candidate.output_shape
    if isinstance(candidate, FullyConnectedMatch):
        return candidate.output_shape
    raise ValueError(f"unsupported compact candidate {candidate.kind!r}")


def build_compact_analysis(
    analysis: Analysis,
    *,
    search_mode: ScheduleSearchMode | str,
    search_state_limit: int,
) -> CompactAnalysis:
    """Builds and verifies a single fully-supported compact model region."""
    candidates = tuple(
        sorted(
            analysis.matches,
            key=lambda item: (item.root.function, item.root.index, item.root.name),
        )
    )
    if not candidates:
        raise ValueError("compact graph requires at least one candidate")
    output_value_to_candidate = {
        _output_value(item): item for item in candidates
    }
    names = {item.root.identifier: f"kernel_{index}" for index, item in enumerate(candidates)}
    tensor_names = {
        item.root.identifier: f"activation_{index + 1}"
        for index, item in enumerate(candidates)
    }
    dependencies: dict[str, str] = {}
    stripped_inputs: dict[str, tuple[ir.Value, ...]] = {}
    graph_inputs: set[ir.Value] = set()
    boundaries: dict[ir.Value, MaterializedBoundary] = {}
    for candidate in candidates:
        values = tuple(_strip_views_and_padding(item) for item in _input_values(candidate))
        stripped_inputs[candidate.root.identifier] = values
        predecessor_names: list[str] = []
        for value in values:
            producer = output_value_to_candidate.get(value)
            if producer is not None:
                predecessor_names.append(tensor_names[producer.root.identifier])
                continue
            sources = _activation_sources(value, output_value_to_candidate)
            if not sources:
                raise ValueError(
                    f"candidate {candidate.root.identifier} crosses an unsupported "
                    "boundary with no activation source"
                )
            source_candidates = [output_value_to_candidate.get(source) for source in sources]
            if all(item is None for item in source_candidates) and len(sources) == 1:
                source = next(iter(sources))
                graph_inputs.add(source)
                predecessor_names.append("activation_0")
                continue
            boundary = boundaries.get(value)
            if boundary is None:
                boundary_index = len(boundaries)
                source_pairs = []
                for source, source_candidate in zip(sources, source_candidates, strict=True):
                    if source_candidate is None:
                        graph_inputs.add(source)
                        source_pairs.append(("activation_0", source))
                    else:
                        source_pairs.append(
                            (tensor_names[source_candidate.root.identifier], source)
                        )
                source_pairs.sort(key=lambda item: item[0])
                boundary = MaterializedBoundary(
                    f"materialization_{boundary_index}",
                    tuple(item[0] for item in source_pairs),
                    tuple(item[1] for item in source_pairs),
                    value,
                    _shape(value),
                    _direct_boundary_kind(
                        value, frozenset(item[1] for item in source_pairs)
                    ),
                )
                boundaries[value] = boundary
            predecessor_names.append(boundary.name)
        unique_predecessors = tuple(dict.fromkeys(predecessor_names))
        if len(unique_predecessors) != 1:
            raise ValueError(
                f"kernel {candidate.root.identifier} has unsupported distinct activation inputs"
            )
        dependencies[candidate.root.identifier] = unique_predecessors[0]
    if len(graph_inputs) != 1:
        raise ValueError(
            "compact region must have exactly one external activation input; "
            f"found {len(graph_inputs)}"
        )
    graph_input = next(iter(graph_inputs))
    graph_input_shape = _shape(graph_input)
    consumers: dict[str, list[str]] = {"activation_0": []}
    for tensor_name in tensor_names.values():
        consumers[tensor_name] = []
    for boundary in boundaries.values():
        consumers[boundary.name] = []
    kernels: list[KernelAccessSchedule] = []
    output_shapes: dict[str, tuple[int, ...]] = {
        tensor_names[item.root.identifier]: _candidate_output_shape(item)
        for item in candidates
    }
    for candidate in candidates:
        candidate_id = candidate.root.identifier
        kernel_name = names[candidate_id]
        input_name = dependencies[candidate_id]
        if input_name == "activation_0":
            input_shape = graph_input_shape
        elif input_name in output_shapes:
            input_shape = output_shapes[input_name]
        else:
            input_shape = next(
                item.output_shape for item in boundaries.values() if item.name == input_name
            )
        reads, writes, output_shape, segment_bytes = _candidate_events(candidate, input_shape)
        reads = segment_last_reads(reads, segment_bytes)
        output_name = tensor_names[candidate_id]
        consumers[input_name].append(kernel_name)
        if output_shapes[output_name] != output_shape:
            raise ValueError(f"kernel {candidate_id} output shape drifted during planning")
        kernels.append(
            KernelAccessSchedule(
                name=kernel_name,
                inputs=(input_name,),
                output=output_name,
                input_last_reads=((input_name, reads),),
                output_first_writes=writes,
                segment_bytes=segment_bytes,
                workspace_bytes=candidate.workspace_bytes,
                kind=candidate.kind,
            )
        )
    for boundary in boundaries.values():
        boundary_name = boundary.name
        source_names = boundary.source_tensors
        boundary_shape = boundary.output_shape
        source_sizes = tuple(
            prod(graph_input_shape if name == "activation_0" else output_shapes[name])
            for name in source_names
        )
        source_size = sum(source_sizes)
        output_size = prod(boundary_shape)
        boundary_kernel = f"kernel_{boundary_name}"
        for source_name in source_names:
            consumers[source_name].append(boundary_kernel)
        kernels.append(
            KernelAccessSchedule(
                name=boundary_kernel,
                inputs=source_names,
                output=boundary_name,
                input_last_reads=tuple(
                    (name, (source_size,) * size)
                    for name, size in zip(source_names, source_sizes, strict=True)
                ),
                output_first_writes=tuple(
                    source_size + index + 1 for index in range(output_size)
                ),
                segment_bytes=1,
                kind="materialized_boundary",
            )
        )
    terminal = [name for name in consumers if name != "activation_0" and not consumers[name]]
    if len(terminal) != 1:
        raise ValueError(
            "compact region must have exactly one external activation output; "
            f"found {len(terminal)}"
        )
    tensors = [
        VirtualTensor(
            "activation_0",
            prod(graph_input_shape),
            None,
            tuple(consumers["activation_0"]),
            graph_input_shape,
            graph_input_shape[-1],
            is_graph_input=True,
        )
    ]
    for candidate in candidates:
        candidate_id = candidate.root.identifier
        name = tensor_names[candidate_id]
        shape = output_shapes[name]
        tensors.append(
            VirtualTensor(
                name,
                prod(shape),
                names[candidate_id],
                tuple(consumers[name]),
                shape,
                kernels[len(tensors) - 1].segment_bytes,
                is_graph_output=name == terminal[0],
            )
        )
    for boundary in boundaries.values():
        boundary_name = boundary.name
        boundary_shape = boundary.output_shape
        tensors.append(
            VirtualTensor(
                boundary_name,
                prod(boundary_shape),
                f"kernel_{boundary_name}",
                tuple(consumers[boundary_name]),
                boundary_shape,
                1,
                is_graph_output=boundary_name == terminal[0],
            )
        )
    plan = plan_compact_graph(
        tensors,
        kernels,
        search_mode=search_mode,
        search_state_limit=search_state_limit,
    )
    return CompactAnalysis(
        plan,
        tuple(item.root.identifier for item in candidates),
        tuple(tuple(str(value.type) for value in stripped_inputs[item.root.identifier]) for item in candidates),
        tuple(tensor_names[item.root.identifier] for item in candidates),
        tuple(boundaries.values()),
    )
