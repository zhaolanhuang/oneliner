"""Builds a byte-addressed compact activation DAG from semantic matches.

Paper correspondence (vMCU, MLSys 2024): §3 (PDF p.4) separates memory
management, segment-aware kernels, and compiler support; §4 (PDF pp.4-5)
defines iteration/access functions and offset constraints; §5.2 (PDF pp.6-7,
Figure 6 and Equation (2)) generalizes scheduling to multi-layer graphs.

Engineering extension: the paper demonstrates selected linear/fused modules.
This analysis recovers a static DAG from MLIR SSA, folds no-copy views and
padding, and represents unsupported/directly-scalarizable boundaries explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from iree.compiler import ir

from .compact_memory import (
    CompactGraphPlan,
    KernelAccessSchedule,
    OutputWriteSchedule,
    ScheduleSearchMode,
    SegmentReadSchedule,
    VirtualTensor,
    plan_compact_graph,
    replay_compact_graph_plan,
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
    """Context-free compact plan safe to carry across MLIR parses."""

    plan: CompactGraphPlan
    candidate_order: tuple[str, ...]
    candidate_signatures: tuple[Any, ...]
    graph_signature: Any
    boundaries: tuple["BoundaryDescriptor", ...]


@dataclass(frozen=True)
class CompactBindings:
    """Live MLIR values rebound in the module that will be mutated."""

    boundaries: tuple["MaterializedBoundary", ...]


@dataclass(frozen=True)
class BoundaryDescriptor:
    """Context-free boundary identity and semantics."""

    name: str
    source_tensors: tuple[str, ...]
    source_types: tuple[str, ...]
    target_type: str
    target_operation: str | None
    output_shape: tuple[int, ...]
    direct_kind: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_tensors": list(self.source_tensors),
            "target_operation": self.target_operation,
            "output_shape": list(self.output_shape),
            "output_bytes": prod(self.output_shape),
            "lowering": self.direct_kind or "materialized",
        }


@dataclass(frozen=True)
class MaterializedBoundary:
    descriptor: BoundaryDescriptor
    source_values: tuple[ir.Value, ...]
    target_value: ir.Value

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def source_tensors(self) -> tuple[str, ...]:
        return self.descriptor.source_tensors

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.descriptor.output_shape

    @property
    def direct_kind(self) -> str | None:
        return self.descriptor.direct_kind


@dataclass(frozen=True)
class _CompactGraphDiscovery:
    candidates: tuple[PatternMatch, ...]
    candidate_order: tuple[str, ...]
    candidate_signatures: tuple[Any, ...]
    names: dict[str, str]
    tensor_names: dict[str, str]
    dependencies: dict[str, str]
    graph_input_shape: tuple[int, ...]
    boundaries: tuple[MaterializedBoundary, ...]
    output_shapes: dict[str, tuple[int, ...]]
    consumers: dict[str, tuple[str, ...]]
    terminal: str
    graph_signature: Any


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


def _immutable_signature(value: Any) -> Any:
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
    return _immutable_signature(candidate.to_dict())


def _boundary_descriptor(
    name: str,
    source_tensors: tuple[str, ...],
    source_values: tuple[ir.Value, ...],
    target_value: ir.Value,
    output_shape: tuple[int, ...],
    direct_kind: str | None,
) -> BoundaryDescriptor:
    operation = owner_operation(target_value)
    return BoundaryDescriptor(
        name,
        source_tensors,
        tuple(str(value.type) for value in source_values),
        str(target_value.type),
        operation.name if operation is not None else None,
        output_shape,
        direct_kind,
    )


def _direct_boundary_kind(
    target: ir.Value, sources: frozenset[ir.Value]
) -> str | None:
    """Recognizes scalarizable identity residual trees and terminal pooling.

    Paper correspondence: §5.2, PDF pp.6-7, Figure 6 includes the residual add
    in the fused inverted bottleneck and Equation (2) covers graph edges.
    Engineering extension: generic identity-map residual trees and terminal
    pooling are broader than the concrete fused kernel described by the paper.
    """
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


def _last_axis_consumer(
    input_index: int,
    output_extent: int,
    kernel_extent: int,
    stride: int,
    dilation: int,
    padding_low: int,
) -> tuple[int, int] | None:
    """Returns the lexicographically last output/kernel coordinate using an input."""
    result = None
    for kernel_index in range(kernel_extent):
        numerator = input_index + padding_low - kernel_index * dilation
        if numerator % stride:
            continue
        output_index = numerator // stride
        if 0 <= output_index < output_extent:
            candidate = (output_index, kernel_index)
            if result is None or candidate > result:
                result = candidate
    return result


def _last_spatial_consumers(
    input_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    kernel_shape: tuple[int, int, int, int],
    strides: tuple[int, int],
    dilations: tuple[int, int],
    padding_low: tuple[int, ...],
) -> tuple[tuple[tuple[int, int, int, int] | None, ...], ...]:
    height = tuple(
        _last_axis_consumer(
            index,
            output_shape[1],
            kernel_shape[0],
            strides[0],
            dilations[0],
            padding_low[1],
        )
        for index in range(input_shape[1])
    )
    width = tuple(
        _last_axis_consumer(
            index,
            output_shape[2],
            kernel_shape[1],
            strides[1],
            dilations[1],
            padding_low[2],
        )
        for index in range(input_shape[2])
    )
    return tuple(
        tuple(
            None
            if height_item is None or width_item is None
            else (height_item[0], width_item[0], height_item[1], width_item[1])
            for width_item in width
        )
        for height_item in height
    )


def _conv_events(
    input_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    kernel_shape: tuple[int, int, int, int],
    strides: tuple[int, int],
    dilations: tuple[int, int],
    padding_low: tuple[int, ...],
    *,
    depthwise: bool,
) -> tuple[tuple[int, ...], OutputWriteSchedule]:
    """Encodes Conv2D reads/writes directly from emitter loop coordinates.

    Paper correspondence: §4, PDF pp.4-5, iteration domain/access functions and
    Equation (1); §5.1, PDF pp.5-6, Figure 5's two-level Conv2D kernel. The
    fixed-radix timestamps preserve that loop's total order without enumerating
    its MAC operations.
    """
    kh_extent, kw_extent = kernel_shape[:2]
    input_channels = input_shape[3]
    reduction_span = kh_extent * kw_extent * (1 if depthwise else input_channels)
    output_span = reduction_span + 1
    writes = OutputWriteSchedule.affine(
        prod(output_shape),
        1 + reduction_span,
        output_span,
    )
    spatial = _last_spatial_consumers(
        input_shape,
        output_shape,
        kernel_shape,
        strides,
        dilations,
        padding_low,
    )
    reads = [0] * prod(input_shape)
    for n in range(input_shape[0]):
        for ih in range(input_shape[1]):
            for iw in range(input_shape[2]):
                consumer = spatial[ih][iw]
                if consumer is None:
                    continue
                oh, ow, kh, kw = consumer
                for channel in range(input_channels):
                    output_channel = channel if depthwise else output_shape[3] - 1
                    output_ordinal = _flatten_nhwc(
                        output_shape, (n, oh, ow, output_channel)
                    )
                    reduction_ordinal = (
                        kh * kw_extent + kw
                        if depthwise
                        else (kh * kw_extent + kw) * input_channels + channel
                    )
                    reads[_flatten_nhwc(input_shape, (n, ih, iw, channel))] = (
                        1 + output_ordinal * output_span + reduction_ordinal
                    )
    return tuple(reads), writes


def _ibn_events(
    candidate: InvertedBottleneckMatch,
    input_shape: tuple[int, int, int, int],
) -> tuple[tuple[int, ...], OutputWriteSchedule]:
    """Encodes the fused inverted-bottleneck access sequence analytically.

    Paper correspondence: §5.2, PDF pp.6-7, Figure 6: load A patches, produce
    ``K×K`` B segments, reduce to one C segment, accumulate D, add residual A,
    and store E. The paper's 3×3 case uses 11=9+1+1 workspace segments; this
    implementation generalizes the same schedule to ``K²+2``.
    """
    output_shape = candidate.output_shape
    depthwise = candidate.depthwise
    kernel_h, kernel_w = depthwise.weight_shape[:2]
    segment_lanes = min(input_shape[3], output_shape[3])
    expanded_channels = candidate.expansion.output_shape[3]
    input_channels = input_shape[3]
    patch_segments = kernel_h * kernel_w
    chunk_count = (expanded_channels + segment_lanes - 1) // segment_lanes
    expansion_span = chunk_count * patch_segments * segment_lanes * input_channels
    pixel_span = expansion_span + 2 * output_shape[3]
    output_pixels = prod(output_shape[:3])
    writes = OutputWriteSchedule.affine(
        output_pixels * output_shape[3],
        expansion_span + 2,
        2,
        group_bytes=output_shape[3],
        group_gap=expansion_span,
    )
    spatial = _last_spatial_consumers(
        input_shape,
        output_shape,
        depthwise.weight_shape,
        depthwise.strides,
        depthwise.dilations,
        depthwise.padding_low,
    )
    reads = [0] * prod(input_shape)
    last_expanded_channel = expanded_channels - 1
    chunk = last_expanded_channel // segment_lanes
    lane = last_expanded_channel % segment_lanes
    for n in range(input_shape[0]):
        for ih in range(input_shape[1]):
            for iw in range(input_shape[2]):
                consumer = spatial[ih][iw]
                if consumer is not None:
                    oh, ow, kh, kw = consumer
                    pixel = (n * output_shape[1] + oh) * output_shape[2] + ow
                    patch = kh * kernel_w + kw
                    expansion_base = (
                        (chunk * patch_segments + patch) * segment_lanes + lane
                    ) * input_channels
                    for channel in range(input_channels):
                        reads[_flatten_nhwc(input_shape, (n, ih, iw, channel))] = (
                            1 + pixel * pixel_span + expansion_base + channel
                        )
                if (
                    candidate.residual is not None
                    and ih < output_shape[1]
                    and iw < output_shape[2]
                ):
                    pixel = (n * output_shape[1] + ih) * output_shape[2] + iw
                    for channel in range(min(input_channels, output_shape[3])):
                        index = _flatten_nhwc(input_shape, (n, ih, iw, channel))
                        residual_event = (
                            1 + pixel * pixel_span + expansion_span + 2 * channel
                        )
                        reads[index] = max(reads[index], residual_event)
    return tuple(reads), writes


def _fc_events(
    candidate: FullyConnectedMatch,
) -> tuple[tuple[int, ...], OutputWriteSchedule]:
    """Encodes fully-connected reads and writes without enumerating MACs.

    Paper correspondence: §2.4, PDF p.3, Figure 1(c); §4, PDF p.5, Figure 3 and
    the GEMM minimum-footprint derivation; §5.1, PDF pp.5-6, Figure 4.
    """
    output_span = candidate.input_channels + 1
    reads = [0] * (candidate.rows * candidate.input_channels)
    for row in range(candidate.rows):
        output_ordinal = row * candidate.output_channels + candidate.output_channels - 1
        for input_channel in range(candidate.input_channels):
            reads[row * candidate.input_channels + input_channel] = (
                1 + output_ordinal * output_span + input_channel
            )
    writes = OutputWriteSchedule.affine(
        prod(candidate.output_shape),
        1 + candidate.input_channels,
        output_span,
    )
    return tuple(reads), writes


def _candidate_events(
    candidate: PatternMatch, input_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], OutputWriteSchedule, tuple[int, ...], int]:
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


def _discover_compact_graph(analysis: Analysis) -> _CompactGraphDiscovery:
    """Recovers compact DAG structure and live boundary values without planning."""
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
    graph_inputs: set[ir.Value] = set()
    boundaries: dict[ir.Value, MaterializedBoundary] = {}
    for candidate in candidates:
        values = tuple(_strip_views_and_padding(item) for item in _input_values(candidate))
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
                source_tensors = tuple(item[0] for item in source_pairs)
                source_values = tuple(item[1] for item in source_pairs)
                output_shape = _shape(value)
                direct_kind = _direct_boundary_kind(value, frozenset(source_values))
                boundary = MaterializedBoundary(
                    _boundary_descriptor(
                        f"materialization_{boundary_index}",
                        source_tensors,
                        source_values,
                        value,
                        output_shape,
                        direct_kind,
                    ),
                    source_values,
                    value,
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
    output_shapes: dict[str, tuple[int, ...]] = {
        tensor_names[item.root.identifier]: _candidate_output_shape(item)
        for item in candidates
    }
    for candidate in candidates:
        candidate_id = candidate.root.identifier
        kernel_name = names[candidate_id]
        input_name = dependencies[candidate_id]
        consumers[input_name].append(kernel_name)
    for boundary in boundaries.values():
        boundary_kernel = f"kernel_{boundary.name}"
        for source_name in boundary.source_tensors:
            consumers[source_name].append(boundary_kernel)
    terminal = [name for name in consumers if name != "activation_0" and not consumers[name]]
    if len(terminal) != 1:
        raise ValueError(
            "compact region must have exactly one external activation output; "
            f"found {len(terminal)}"
        )
    immutable_consumers = {
        name: tuple(items) for name, items in consumers.items()
    }
    boundary_items = tuple(boundaries.values())
    candidate_order = tuple(item.root.identifier for item in candidates)
    candidate_signatures = tuple(_candidate_signature(item) for item in candidates)
    graph_signature = (
        graph_input_shape,
        tuple(
            (
                candidate_id,
                names[candidate_id],
                tensor_names[candidate_id],
                dependencies[candidate_id],
                output_shapes[tensor_names[candidate_id]],
            )
            for candidate_id in candidate_order
        ),
        tuple(item.descriptor for item in boundary_items),
        tuple((name, immutable_consumers[name]) for name in immutable_consumers),
        terminal[0],
    )
    return _CompactGraphDiscovery(
        candidates,
        candidate_order,
        candidate_signatures,
        names,
        tensor_names,
        dependencies,
        graph_input_shape,
        boundary_items,
        output_shapes,
        immutable_consumers,
        terminal[0],
        graph_signature,
    )


def _build_access_schedules(
    graph: _CompactGraphDiscovery,
) -> tuple[tuple[VirtualTensor, ...], tuple[KernelAccessSchedule, ...]]:
    """Builds analytical access schedules once for the immutable source graph."""
    kernels: list[KernelAccessSchedule] = []
    segment_bytes_by_output: dict[str, int] = {}
    boundary_shapes = {item.name: item.output_shape for item in graph.boundaries}
    for candidate in graph.candidates:
        candidate_id = candidate.root.identifier
        kernel_name = graph.names[candidate_id]
        input_name = graph.dependencies[candidate_id]
        if input_name == "activation_0":
            input_shape = graph.graph_input_shape
        elif input_name in graph.output_shapes:
            input_shape = graph.output_shapes[input_name]
        else:
            input_shape = boundary_shapes[input_name]
        # vMCU §5.3 (PDF p.7): Conv/IBN segment width is min(input channels,
        # output channels), while depthwise uses its channel width. Event
        # generation above lifts that kernel-specific choice into §4 lifetimes.
        reads, writes, output_shape, segment_bytes = _candidate_events(candidate, input_shape)
        reads = segment_last_reads(reads, segment_bytes)
        output_name = graph.tensor_names[candidate_id]
        if graph.output_shapes[output_name] != output_shape:
            raise ValueError(f"kernel {candidate_id} output shape drifted during planning")
        segment_bytes_by_output[output_name] = segment_bytes
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
    for boundary in graph.boundaries:
        # Engineering extension to vMCU §5.2/Equation (2): unknown graph regions
        # get a conservative read-all-then-write schedule. Direct residual and
        # pooling emitters may execute more finely, but never violate this proof.
        boundary_name = boundary.name
        source_names = boundary.source_tensors
        boundary_shape = boundary.output_shape
        source_sizes = tuple(
            prod(
                graph.graph_input_shape
                if name == "activation_0"
                else graph.output_shapes[name]
            )
            for name in source_names
        )
        source_size = sum(source_sizes)
        output_size = prod(boundary_shape)
        boundary_kernel = f"kernel_{boundary_name}"
        kernels.append(
            KernelAccessSchedule(
                name=boundary_kernel,
                inputs=source_names,
                output=boundary_name,
                input_last_reads=tuple(
                    (name, SegmentReadSchedule.constant(size, 1, source_size))
                    for name, size in zip(source_names, source_sizes, strict=True)
                ),
                output_first_writes=OutputWriteSchedule.affine(
                    output_size,
                    source_size + 1,
                    1,
                ),
                segment_bytes=1,
                kind="materialized_boundary",
            )
        )
    tensors = [
        VirtualTensor(
            "activation_0",
            prod(graph.graph_input_shape),
            None,
            graph.consumers["activation_0"],
            graph.graph_input_shape,
            graph.graph_input_shape[-1],
            is_graph_input=True,
        )
    ]
    for candidate in graph.candidates:
        candidate_id = candidate.root.identifier
        name = graph.tensor_names[candidate_id]
        shape = graph.output_shapes[name]
        tensors.append(
            VirtualTensor(
                name,
                prod(shape),
                graph.names[candidate_id],
                graph.consumers[name],
                shape,
                segment_bytes_by_output[name],
                is_graph_output=name == graph.terminal,
            )
        )
    for boundary in graph.boundaries:
        boundary_name = boundary.name
        boundary_shape = boundary.output_shape
        tensors.append(
            VirtualTensor(
                boundary_name,
                prod(boundary_shape),
                f"kernel_{boundary_name}",
                graph.consumers[boundary_name],
                boundary_shape,
                1,
                is_graph_output=boundary_name == graph.terminal,
            )
        )
    return tuple(tensors), tuple(kernels)


def build_compact_analysis(
    analysis: Analysis,
    *,
    search_mode: ScheduleSearchMode | str,
    search_state_limit: int,
) -> CompactAnalysis:
    """Builds and solves one context-free compact model plan.

    Paper correspondence:
      * §3, PDF p.4, Figure 2: coordinates compiler, memory manager, and kernel
        schedule instead of treating them as independent components.
      * §4, PDF pp.4-5, Equation (1): creates virtual tensors plus read/write
        events needed to solve input/output offsets.
      * §5.2, PDF p.6, Equation (2): builds producer-consumer graph constraints.
      * §5.3, PDF p.7: assigns kernel-dependent segment widths.

    Engineering extension: MLIR SSA recovery, arbitrary static-DAG boundaries,
    search modes, and replay verification are additions in this implementation.
    The returned analysis contains no MLIR handles and is safe to reuse after a
    transactional reparse.
    """
    graph = _discover_compact_graph(analysis)
    tensors, kernels = _build_access_schedules(graph)
    # Solves §4's bIn/bOut placement problem, generalized to the §5.2 DAG.
    plan = plan_compact_graph(
        tensors,
        kernels,
        search_mode=search_mode,
        search_state_limit=search_state_limit,
    )
    return CompactAnalysis(
        plan,
        graph.candidate_order,
        graph.candidate_signatures,
        graph.graph_signature,
        tuple(item.descriptor for item in graph.boundaries),
    )


def rebind_compact_analysis(
    analysis: Analysis, expected: CompactAnalysis
) -> CompactBindings:
    """Rebinds a source plan to live values without regenerating or searching it."""
    graph = _discover_compact_graph(analysis)
    if graph.candidate_order != expected.candidate_order:
        raise ValueError("compact candidate identities drifted after reparse")
    if graph.candidate_signatures != expected.candidate_signatures:
        raise ValueError("compact candidate semantics drifted after reparse")
    if graph.graph_signature != expected.graph_signature:
        raise ValueError("compact DAG or boundary semantics drifted after reparse")
    replay_compact_graph_plan(expected.plan)
    return CompactBindings(graph.boundaries)
