"""Small, side-effect-free helpers for inspecting MLIR Python objects."""

from __future__ import annotations

from iree.compiler import ir


def operation(value) -> ir.Operation:
    """Normalizes an OpView or Operation to the underlying Operation object."""
    return value.operation if hasattr(value, "operation") else value


def owner_operation(value: ir.Value) -> ir.Operation | None:
    """Returns an SSA result's defining operation, or None for block arguments."""
    if not isinstance(value, ir.OpResult):
        return None
    return operation(value.owner)


def scalar_integer(value: ir.Value) -> int | None:
    """Extracts an integer arith.constant without raising on non-constants."""
    owner = owner_operation(value)
    if owner is None or owner.name != "arith.constant" or "value" not in owner.attributes:
        return None
    try:
        return int(ir.IntegerAttr(owner.attributes["value"]).value)
    except ValueError:
        return None


def require_scalar(value: ir.Value, label: str) -> int:
    """Requires a scalar integer constant and returns its Python value."""
    result = scalar_integer(value)
    if result is None:
        raise ValueError(f"{label} must be a scalar arith.constant")
    return result


def require_zero_point(value: ir.Value, label: str) -> int:
    """Requires a scalar zero-point representable by the rewritten int8 path."""
    result = require_scalar(value, label)
    if result < -128 or result > 127:
        raise ValueError(f"{label} must be in [-128, 127]")
    return result


def tensor_shape(value: ir.Value, element_type: str, rank: int | None = None) -> tuple[int, ...]:
    """Returns a static ranked tensor shape after type, rank, and dtype checks."""
    try:
        value_type = ir.RankedTensorType(value.type)
    except ValueError as error:
        expected = f"rank-{rank} " if rank is not None else ""
        raise ValueError(f"expected a {expected}{element_type} tensor, got {value.type}") from error
    shape = tuple(int(dimension) for dimension in value_type.shape)
    if (
        (rank is not None and value_type.rank != rank)
        or str(value_type.element_type) != element_type
        or any(dimension < 0 for dimension in shape)
    ):
        expected = f"static rank-{rank}" if rank is not None else "static ranked"
        raise ValueError(f"expected a {expected} {element_type} tensor, got {value.type}")
    return shape


def dense_ints(attribute: ir.Attribute) -> tuple[int, ...]:
    """Copies an MLIR dense integer attribute into immutable Python values."""
    return tuple(int(item) for item in attribute)


def require_constant_tensor(
    value: ir.Value,
    element_type: str,
    shape: tuple[int, ...],
    label: str,
) -> tuple[int, ...]:
    """Requires a dense arith.constant tensor with an exact type and shape."""
    owner = owner_operation(value)
    if owner is None or owner.name != "arith.constant":
        raise ValueError(f"{label} must be an arith.constant tensor")
    if tensor_shape(value, element_type, len(shape)) != shape:
        raise ValueError(f"{label} has an unexpected tensor type")
    if "value" not in owner.attributes:
        raise ValueError(f"{label} has no dense value")
    return dense_ints(owner.attributes["value"])


def generic_io(candidate: ir.Operation) -> tuple[list[ir.Value], list[ir.Value]]:
    """Returns the DPS input/output partitions of a linalg.generic op."""
    if candidate.name != "linalg.generic":
        raise ValueError(f"expected linalg.generic, got {candidate.name}")
    return list(candidate.opview.inputs), list(candidate.opview.outputs)


def body_operations(candidate: ir.Operation, label: str) -> tuple[ir.Block, list[ir.Operation]]:
    """Returns the sole scalar block and its operations after region checks."""
    if len(candidate.regions) != 1 or len(candidate.regions[0].blocks) != 1:
        raise ValueError(f"{label} must have exactly one scalar block")
    block = candidate.regions[0].blocks[0]
    return block, [operation(item) for item in block.operations]


def unique_user(value: ir.Value, expected_name: str, label: str) -> ir.Operation:
    """Requires one exact semantic SSA consumer.

    Destination-style MLIR operations may use a tensor only as an output
    buffer.  Those uses are not data-flow consumers when the corresponding
    region argument is never read.  Counting them here makes a legal CSE
    reuse look like an ambiguous graph and, more importantly, makes IBN
    matching depend on buffer-allocation details rather than semantics.
    """
    users = semantic_users(value)
    if len(users) != 1 or users[0].name != expected_name:
        raise ValueError(f"{label} must have exactly one {expected_name} user")
    return users[0]


def semantic_users(value: ir.Value) -> list[ir.Operation]:
    """Returns direct users that consume the tensor value semantically.

    For ``linalg.generic`` the output operands are only semantic when the
    matching output block argument is read by the scalar region.  Other
    operations retain the conservative rule that every operand is a semantic
    use; their destination-style accumulation semantics are pattern-specific
    and must not be discarded by this generic helper.
    """
    users: list[ir.Operation] = []
    for use in value.uses:
        owner = operation(use.owner)
        if owner.name in ("util.return", "func.return"):
            continue
        if owner.name == "linalg.generic":
            try:
                input_count = len(owner.opview.inputs)
                if use.operand_number >= input_count:
                    output_index = use.operand_number - input_count
                    block = owner.regions[0].blocks[0]
                    block_argument = block.arguments[input_count + output_index]
                    if not list(block_argument.uses):
                        continue
            except (IndexError, ValueError, AttributeError):
                # A malformed operation must remain visible to the matcher so
                # that it is rejected rather than silently treated as dead.
                pass
        if owner not in users:
            users.append(owner)
    return users


def expected_map(context: ir.Context, dimensions: int, results: tuple[int, ...]) -> ir.AffineMapAttr:
    """Builds a projected-permutation affine map from selected dimensions."""
    return ir.AffineMapAttr.get(
        ir.AffineMap.get(
            dimensions,
            0,
            [ir.AffineDimExpr.get(index, context=context) for index in results],
            context=context,
        )
    )


def validate_maps(candidate: ir.Operation, results: tuple[tuple[int, ...], ...], label: str) -> None:
    """Requires exact indexing maps and parallel iterators for an elementwise op."""
    dimensions = len(candidate.opview.iterator_types)
    expected = [expected_map(candidate.context, dimensions, result) for result in results]
    if list(candidate.opview.indexing_maps) != expected:
        raise ValueError(f"{label} has non-canonical indexing maps")
    parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>", context=candidate.context)
    if any(iterator != parallel for iterator in candidate.opview.iterator_types):
        raise ValueError(f"{label} must contain only parallel iterators")


def replace_all_uses(old: ir.Value, new: ir.Value) -> None:
    """Redirects a snapshot of SSA uses so mutation cannot invalidate iteration."""
    for use in list(old.uses):
        use.owner.operands[use.operand_number] = new


def verify(candidate: ir.Operation, label: str) -> None:
    """Runs MLIR verification and converts MLIR exceptions into diagnostics."""
    try:
        candidate.verify()
    except ir.MLIRError as error:
        raise ValueError(f"{label} failed verification: {error}") from error


def erase_dead_operation(candidate: ir.Operation) -> bool:
    """Erases an operation iff all results are dead and reports whether it did."""
    if any(list(result.uses) for result in candidate.results):
        return False
    candidate.erase()
    return True
