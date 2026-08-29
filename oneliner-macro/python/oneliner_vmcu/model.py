"""Immutable descriptions produced by the analysis phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from iree.compiler import ir

from .quantization import AffineQuantization


@dataclass(frozen=True)
class OpKey:
    """Stable identity of a direct operation within one function.

    The index is based on the original preprocessing IR. It deliberately does
    not use a Python object address, so the analysis and transactional reparse
    can compare candidate identities deterministically.
    """

    function: str
    index: int
    name: str

    @property
    def identifier(self) -> str:
        """Returns the human- and JSON-readable candidate identifier."""
        return f"{self.function}:{self.index}:{self.name}"


@dataclass(frozen=True)
class RejectedCandidate:
    """Records why a pattern root was inspected but not safe to rewrite."""

    root: OpKey
    pattern: str
    reason: str
    location: str

    def to_dict(self) -> dict[str, Any]:
        """Converts the rejection into the stable plan-file schema."""
        return {
            "root": self.root.identifier,
            "pattern": self.pattern,
            "reason": self.reason,
            "location": self.location,
        }


@dataclass
class PatternMatch:
    """Base class shared by every registered semantic pattern match."""

    kind: ClassVar[str] = "unknown"
    root: OpKey

    @property
    def claimed_operations(self) -> set[ir.Operation]:
        """Returns graph operations made unavailable to later patterns."""
        raise NotImplementedError

    @property
    def eliminated_accumulator_bytes(self) -> int:
        """Returns a conservative logical i32 tensor elimination metric."""
        return 0

    @property
    def workspace_bytes(self) -> int:
        """Returns explicit fixed-schedule scratch known before lowering."""
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Returns common JSON fields; concrete patterns add semantic data."""
        return {"id": self.root.identifier, "kind": self.kind}


@dataclass
class FullyConnectedMatch(PatternMatch):
    """All proven operations, values, and constants needed for one FC rewrite.

    MLIR objects are kept only for the rewrite pass. Static dimensions and
    zero-points are copied into ordinary Python values so diagnostics never
    need to infer them again after operations have been erased.
    """

    kind: ClassVar[str] = "quantized_fully_connected"
    matmul: ir.Operation
    expand: ir.Operation | None
    rescale: ir.Operation
    input: ir.Value
    output_major_weight: ir.Value
    bias: ir.Value
    multiplier: ir.Value
    shift: ir.Value
    input_quantization: AffineQuantization
    weight_quantization: AffineQuantization
    output_quantization: AffineQuantization
    rows: int
    input_channels: int
    output_channels: int
    output_shape: tuple[int, ...]

    @property
    def claimed_operations(self) -> set[ir.Operation]:
        """Returns the complete subgraph erased or replaced by this emitter."""
        operations = {self.matmul, self.rescale}
        if self.expand is not None:
            operations.add(self.expand)
        return operations

    @property
    def input_zero_point(self) -> int:
        """Returns the scalar input zero-point used by this FC operation."""
        return self.input_quantization.zero_point_at()

    @property
    def weight_zero_point(self) -> int:
        """Returns the scalar weight zero-point encoded by quantized_matmul."""
        return self.weight_quantization.zero_point_at()

    @property
    def output_zero_point(self) -> int:
        """Returns the scalar output zero-point reconstructed after scaling."""
        return self.output_quantization.zero_point_at()

    @property
    def eliminated_accumulator_bytes(self) -> int:
        """Returns the logical size of the removed [rows, Cout] i32 tensor."""
        return self.rows * self.output_channels * 4

    def to_dict(self) -> dict[str, Any]:
        """Returns the public, MLIR-object-free description used in the plan."""
        return {
            "id": self.root.identifier,
            "kind": self.kind,
            "rows": self.rows,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "output_shape": list(self.output_shape),
            "quantization": {
                "input": self.input_quantization.to_dict(),
                "weight": self.weight_quantization.to_dict(),
                "output": self.output_quantization.to_dict(),
            },
            "eliminated_i32_accumulator_bytes": self.eliminated_accumulator_bytes,
        }


@dataclass
class Conv2DMatch(PatternMatch):
    """Proven quantized NHWC/HWCF convolution and terminal requantization."""

    kind: ClassVar[str] = "quantized_conv2d"
    conv: ir.Operation
    bias_initializer: ir.Operation
    rescale: ir.Operation
    input: ir.Value
    weight: ir.Value
    bias: ir.Value
    multiplier: ir.Value
    shift: ir.Value
    input_quantization: AffineQuantization
    weight_quantization: AffineQuantization
    output_quantization: AffineQuantization
    input_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    strides: tuple[int, int]
    dilations: tuple[int, int]
    padding_low: tuple[int, ...]
    padding_high: tuple[int, ...]

    @property
    def claimed_operations(self) -> set[ir.Operation]:
        """Claims convolution and requantization; dead initializers remain cleanup."""
        return {self.conv, self.bias_initializer, self.rescale}

    @property
    def eliminated_accumulator_bytes(self) -> int:
        """Returns the removed full NHWC i32 accumulator size."""
        elements = 1
        for dimension in self.output_shape:
            elements *= dimension
        return elements * 4

    def to_dict(self) -> dict[str, Any]:
        """Returns geometry, quantization, and fixed segment schedule data."""
        segment_lanes = min(self.input_shape[3], self.output_shape[3])
        return {
            "id": self.root.identifier,
            "kind": self.kind,
            "input_shape": list(self.input_shape),
            "weight_shape": list(self.weight_shape),
            "output_shape": list(self.output_shape),
            "strides": list(self.strides),
            "dilations": list(self.dilations),
            "padding_low": list(self.padding_low),
            "padding_high": list(self.padding_high),
            "segment_lanes": segment_lanes,
            "quantization": {
                "input": self.input_quantization.to_dict(),
                "weight": self.weight_quantization.to_dict(),
                "output": self.output_quantization.to_dict(),
            },
            "eliminated_i32_accumulator_bytes": self.eliminated_accumulator_bytes,
        }


@dataclass
class DepthwiseConv2DMatch(PatternMatch):
    """Proven multiplier-one depthwise convolution through requantization."""

    kind: ClassVar[str] = "quantized_depthwise_conv2d"
    conv: ir.Operation
    accumulator_initializer: ir.Operation
    collapse: ir.Operation
    bias_add: ir.Operation
    rescale: ir.Operation
    input: ir.Value
    weight: ir.Value
    bias: ir.Value
    multiplier: ir.Value
    shift: ir.Value
    input_quantization: AffineQuantization
    weight_quantization: AffineQuantization
    output_quantization: AffineQuantization
    input_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    strides: tuple[int, int]
    dilations: tuple[int, int]
    padding_low: tuple[int, ...]
    padding_high: tuple[int, ...]

    @property
    def claimed_operations(self) -> set[ir.Operation]:
        """Claims the rewrite chain, leaving shareable zero-fill storage free.

        IREE's preprocessing CSE can feed several depthwise operations from
        one zero-filled destination.  The fill is a pure initializer and is
        erased independently only after its last user disappears, so it must
        not make otherwise disjoint candidates overlap.
        """
        return {
            self.conv,
            self.collapse,
            self.bias_add,
            self.rescale,
        }

    @property
    def eliminated_accumulator_bytes(self) -> int:
        """Returns the removed full depthwise i32 tensor size."""
        elements = 1
        for dimension in self.output_shape:
            elements *= dimension
        return elements * 4

    def to_dict(self) -> dict[str, Any]:
        """Returns geometry and quantization without live MLIR handles."""
        return {
            "id": self.root.identifier,
            "kind": self.kind,
            "input_shape": list(self.input_shape),
            "weight_shape": list(self.weight_shape),
            "output_shape": list(self.output_shape),
            "strides": list(self.strides),
            "dilations": list(self.dilations),
            "padding_low": list(self.padding_low),
            "padding_high": list(self.padding_high),
            "segment_lanes": self.output_shape[3],
            "channel_multiplier": 1,
            "quantization": {
                "input": self.input_quantization.to_dict(),
                "weight": self.weight_quantization.to_dict(),
                "output": self.output_quantization.to_dict(),
            },
            "eliminated_i32_accumulator_bytes": self.eliminated_accumulator_bytes,
        }


@dataclass(frozen=True)
class ResidualScale:
    """Immutable scalar facts extracted from one split residual generic."""

    input_type: str
    output_type: str
    input_zero_point: int | None
    output_zero_point: int | None
    multiplier: int
    shift: int


@dataclass
class ResidualMatch:
    """Validated residual path attached to an inverted bottleneck.

    ``fused`` is the compact same-quantization form emitted directly by the
    vMCU schedule.  ``split`` is the sequence of elementwise generics emitted
    by IREE around an i32 residual add.  The latter is retained as a structured
    path during emission so its already-proven scalar arithmetic can be
    preserved without retaining an old analysis object across rewrites.
    """

    mode: str
    operations: tuple[ir.Operation, ...]
    final_operation: ir.Operation
    input: ir.Value | None
    scales: tuple[ResidualScale, ...] = ()


@dataclass
class InvertedBottleneckMatch(PatternMatch):
    """Complete expansion→depthwise→projection→optional residual proof."""

    kind: ClassVar[str] = "inverted_bottleneck_k2_plus_2_segment"
    expansion: Conv2DMatch
    depthwise: DepthwiseConv2DMatch
    projection: Conv2DMatch
    depthwise_padding: ir.Operation
    residual: ResidualMatch | None

    @property
    def residual_input(self) -> ir.Value | None:
        """Returns the module input used by a validated residual path."""
        return self.residual.input if self.residual is not None else None

    @property
    def claimed_operations(self) -> set[ir.Operation]:
        """Claims every intermediate operator so no single-layer rewrite overlaps."""
        operations = (
            self.expansion.claimed_operations
            | self.depthwise.claimed_operations
            | self.projection.claimed_operations
            | {self.depthwise_padding}
        )
        if self.residual is not None:
            operations.update(self.residual.operations)
        return operations

    @property
    def eliminated_accumulator_bytes(self) -> int:
        """Returns all three removed full i32 accumulator tensors."""
        return sum(
            item.eliminated_accumulator_bytes
            for item in (self.expansion, self.depthwise, self.projection)
        )

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """Returns the module's final NHWC shape."""
        return self.projection.output_shape

    @property
    def workspace_bytes(self) -> int:
        """Returns the aligned K² B + 1 C + 1 D workspace payload."""
        from .schedules import InvertedBottleneckSegmentSchedule

        return InvertedBottleneckSegmentSchedule(
            self.expansion.input_shape[3],
            self.expansion.output_shape[3],
            self.projection.output_shape[3],
            self.depthwise.weight_shape[0],
            self.depthwise.weight_shape[1],
        ).memory_plan().workspace_bytes

    def to_dict(self) -> dict[str, Any]:
        """Returns the forced schedule and every layer's quantization boundary."""
        from .schedules import InvertedBottleneckSegmentSchedule

        schedule_builder = InvertedBottleneckSegmentSchedule(
            self.expansion.input_shape[3],
            self.expansion.output_shape[3],
            self.projection.output_shape[3],
            self.depthwise.weight_shape[0],
            self.depthwise.weight_shape[1],
        )
        schedule = schedule_builder.to_dict()
        return {
            "id": self.root.identifier,
            "kind": self.kind,
            "input_shape": list(self.expansion.input_shape),
            "expanded_shape": list(self.expansion.output_shape),
            "output_shape": list(self.output_shape),
            "residual": self.residual is not None,
            "layers": {
                "expansion": self.expansion.to_dict(),
                "depthwise": self.depthwise.to_dict(),
                "projection": self.projection.to_dict(),
            },
            "schedule": schedule,
            "eliminated_i32_accumulator_bytes": self.eliminated_accumulator_bytes,
        }


@dataclass
class Analysis:
    """Complete result of a side-effect-free pattern analysis."""

    matches: list[PatternMatch]
    rejected: list[RejectedCandidate]

    @property
    def match_ids(self) -> tuple[str, ...]:
        """Returns ordered IDs used to guard the transactional second parse."""
        return tuple(candidate.root.identifier for candidate in self.matches)
