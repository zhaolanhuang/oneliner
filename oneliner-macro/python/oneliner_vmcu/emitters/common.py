"""Small construction helpers shared by standard-MLIR emitters."""

from __future__ import annotations

from iree.compiler import ir
from iree.compiler.dialects import arith


def constant(value_type: ir.Type, value: int) -> ir.Value:
    """Materializes a scalar integer constant at the active insertion point."""
    return arith.ConstantOp(value_type, ir.IntegerAttr.get(value_type, value)).result


def flatten_prefix(indices: list[ir.Value], shape: tuple[int, ...]) -> ir.Value:
    """Flattens all output dimensions except the final channel dimension."""
    result = indices[0]
    index_type = ir.IndexType.get()
    for dimension, index in zip(shape[1:-1], indices[1:-1], strict=True):
        extent = constant(index_type, dimension)
        result = arith.AddIOp(arith.MulIOp(result, extent).result, index).result
    return result
