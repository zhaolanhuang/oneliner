"""Standard-MLIR construction primitives for fixed segment buffers."""

from __future__ import annotations

from iree.compiler import ir
from iree.compiler.dialects import arith, scf, tensor

from .common import constant


def emit_static_segment_buffer(
    segments: int,
    lanes: int,
    element_type: ir.Type,
    fill_value: ir.Value,
) -> ir.Value:
    """Creates a statically shaped tensor buffer and initializes every lane."""
    if segments <= 0 or lanes <= 0:
        raise ValueError("static segment dimensions must be positive")
    buffer_type = ir.RankedTensorType.get([segments, lanes], element_type)
    return tensor.SplatOp(buffer_type, fill_value, []).result


def emit_modulo_index(index: ir.Value, capacity: int) -> ir.Value:
    """Maps an index to a finite positive capacity using unsigned remainder."""
    if capacity <= 0:
        raise ValueError("modulo capacity must be positive")
    divisor = constant(ir.IndexType.get(), capacity)
    return arith.RemUIOp(index, divisor).result


def emit_masked_segment_load(
    source: ir.Value,
    source_index: ir.Value,
    logical_length: int,
    padding_value: ir.Value,
) -> ir.Value:
    """Loads one source lane or yields the affine input zero-point.

    The ``tensor.extract`` is placed inside ``scf.if`` so an invalid final lane
    cannot speculatively form an out-of-bounds access before selection.
    """
    if logical_length < 0:
        raise ValueError("logical source length cannot be negative")
    bound = constant(ir.IndexType.get(), logical_length)
    condition = arith.CmpIOp(arith.CmpIPredicate.ult, source_index, bound).result
    branch = scf.IfOp(condition, [padding_value.type], has_else=True)
    with ir.InsertionPoint(branch.then_block):
        value = tensor.ExtractOp(source, [source_index]).result
        scf.YieldOp([value])
    with ir.InsertionPoint(branch.else_block):
        scf.YieldOp([padding_value])
    return branch.results[0]


def emit_segment_store(
    value: ir.Value,
    buffer: ir.Value,
    segment_index: ir.Value,
    lane_index: ir.Value,
) -> ir.Value:
    """Functionally updates one lane of a loop-carried tensor buffer."""
    return tensor.InsertOp(value, buffer, [segment_index, lane_index]).result
