"""Strict semantic matcher for quantized fully-connected subgraphs."""

from __future__ import annotations

from iree.compiler import ir

from ..graph import FunctionGraph
from ..ir_utils import (
    body_operations,
    dense_ints,
    generic_io,
    owner_operation,
    require_constant_tensor,
    require_scalar,
    require_zero_point,
    tensor_shape,
    unique_user,
    validate_maps,
)
from ..model import Analysis, FullyConnectedMatch, OpKey, RejectedCandidate
from ..quantization import AffineQuantization


def _validate_bias(candidate: ir.Operation, channels: int) -> ir.Value:
    """Proves an exact one-dimensional bias broadcast into a rank-2 tensor.

    The scalar region must yield the bias argument unchanged. This excludes
    fused activation, arithmetic, or output-dependent initialization that the
    replacement's simple bias seed would not preserve.
    """
    inputs, outputs = generic_io(candidate)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("bias initializer must have one input and one output")
    require_constant_tensor(inputs[0], "i32", (channels,), "bias")
    validate_maps(candidate, ((1,), (0, 1)), "bias initializer")
    block, operations = body_operations(candidate, "bias initializer")
    if (
        len(block.arguments) != 2
        or tuple(item.name for item in operations) != ("linalg.yield",)
        or operations[0].operands[0] != block.arguments[0]
    ):
        raise ValueError("bias initializer has modified scalar semantics")
    return inputs[0]


def _validate_clamp(
    operations: list[ir.Operation],
    cursor: ir.Value,
    offset: int,
    label: str,
) -> None:
    """Proves the terminal max(-128), min(127), trunc-i8 dataflow."""
    if len(operations) != offset + 4:
        raise ValueError(f"{label} has unexpected operations after scaling")
    lower, upper, truncation, terminator = operations[offset:]
    if (
        lower.name != "arith.maxsi"
        or lower.operands[0] != cursor
        or require_scalar(lower.operands[1], f"{label} clamp minimum") != -128
        or upper.name != "arith.minsi"
        or upper.operands[0] != lower.results[0]
        or require_scalar(upper.operands[1], f"{label} clamp maximum") != 127
        or truncation.name != "arith.trunci"
        or truncation.operands[0] != upper.results[0]
        or terminator.name != "linalg.yield"
        or terminator.operands[0] != truncation.results[0]
    ):
        raise ValueError(f"{label} is not the canonical int8 clamp")


def _validate_expanded_scale(
    operations: list[ir.Operation], block: ir.Block, with_zero_point: bool
) -> tuple[ir.Value, int]:
    """Validates IREE's exact arith expansion of TOSA DOUBLE_ROUND.

    Matching operation names alone is insufficient: every operand/result edge,
    comparison predicate, and rounding constant is checked below. The returned
    zero-point is copied for reconstruction by the replacement operation.
    """
    # IREE expands apply_scale into this 15-operation prefix before the optional
    # output offset and the common int8 clamp tail.
    prefix = operations[:15]
    expected = (
        "arith.extui",
        "arith.extsi",
        "arith.extsi",
        "arith.muli",
        "arith.extui",
        "arith.shli",
        "arith.shrui",
        "arith.addi",
        "arith.cmpi",
        "arith.select",
        "arith.addi",
        "arith.cmpi",
        "arith.select",
        "arith.shrsi",
        "arith.trunci",
    )
    if tuple(item.name for item in prefix) != expected or len(block.arguments) != 4:
        raise ValueError("requantize has modified expanded DOUBLE_ROUND semantics")
    (
        extui_shift,
        ext_value,
        ext_multiplier,
        multiply,
        ext_shift,
        shift_left,
        shift_half,
        base_add,
        compare_positive,
        select_direction,
        direction_add,
        compare_shift,
        select_round,
        shift_right,
        trunc_scale,
    ) = prefix
    value_arg, multiplier_arg, shift_arg = block.arguments[:3]
    predicate_positive = int(ir.IntegerAttr(compare_positive.attributes["predicate"]).value)
    predicate_shift = int(ir.IntegerAttr(compare_shift.attributes["predicate"]).value)
    # Validate the scalar SSA dataflow, including the sign-dependent second
    # rounding term required by DOUBLE_ROUND.
    if (
        extui_shift.operands[0] != shift_arg
        or ext_value.operands[0] != value_arg
        or ext_multiplier.operands[0] != multiplier_arg
        or list(multiply.operands) != [ext_value.results[0], ext_multiplier.results[0]]
        or ext_shift.operands[0] != shift_arg
        or require_scalar(shift_left.operands[0], "requantize scale base") != 1
        or shift_left.operands[1] != ext_shift.results[0]
        or shift_half.operands[0] != shift_left.results[0]
        or require_scalar(shift_half.operands[1], "requantize scale halving") != 1
        or list(base_add.operands) != [multiply.results[0], shift_half.results[0]]
        or compare_positive.operands[0] != value_arg
        or predicate_positive != 5
        or require_scalar(compare_positive.operands[1], "requantize zero") != 0
        or select_direction.operands[0] != compare_positive.results[0]
        or require_scalar(select_direction.operands[1], "requantize round up") != 1073741824
        or require_scalar(select_direction.operands[2], "requantize round down") != -1073741824
        or list(direction_add.operands) != [select_direction.results[0], base_add.results[0]]
        or predicate_shift != 4
        or compare_shift.operands[0] != extui_shift.results[0]
        or require_scalar(compare_shift.operands[1], "requantize shift bound") != 31
        or select_round.operands[0] != compare_shift.results[0]
        or select_round.operands[1] != direction_add.results[0]
        or select_round.operands[2] != base_add.results[0]
        or list(shift_right.operands) != [select_round.results[0], ext_shift.results[0]]
        or trunc_scale.operands[0] != shift_right.results[0]
    ):
        raise ValueError("requantize has modified expanded DOUBLE_ROUND dataflow")
    cursor = trunc_scale.results[0]
    output_zero_point = 0
    offset = 15
    if with_zero_point:
        addition = operations[offset]
        if addition.name != "arith.addi" or addition.operands[0] != cursor:
            raise ValueError("requantize output zero point has modified dataflow")
        output_zero_point = require_zero_point(addition.operands[1], "output zero point")
        cursor = addition.results[0]
        offset += 1
    _validate_clamp(operations, cursor, offset, "requantize")
    return trunc_scale.results[0], output_zero_point


def _validate_rescale(
    candidate: ir.Operation,
    channels: int,
    source: ir.Value,
) -> tuple[ir.Value, ir.Value, int, tuple[int, ...]]:
    """Validates per-channel requantization and returns rewrite operands.

    Two semantically equivalent representations are accepted: the original
    ``tosa.apply_scale`` or the exact arithmetic expansion emitted by IREE.
    Both must end in the canonical signed-int8 saturation sequence.
    """
    inputs, outputs = generic_io(candidate)
    if len(inputs) != 3 or len(outputs) != 1 or inputs[0] != source:
        raise ValueError("requantize must have accumulator, multiplier and shift inputs")
    output_shape = tensor_shape(candidate.results[0], "i8")
    if tensor_shape(source, "i32") != output_shape:
        raise ValueError("requantize input and output shapes must agree")
    rank = len(output_shape)
    if rank not in (2, 4) or output_shape[-1] != channels:
        raise ValueError("requantize output must be a rank-2 or rank-4 int8 tensor")
    multiplier, shift = inputs[1], inputs[2]
    multiplier_values = require_constant_tensor(multiplier, "i32", (channels,), "multiplier")
    shift_values = require_constant_tensor(shift, "i8", (channels,), "shift")
    if any(value < 0 or value >= (1 << 31) for value in multiplier_values):
        raise ValueError("multiplier values must be in [0, 2^31)")
    if any(value < 1 or value > 62 for value in shift_values):
        raise ValueError("shift values must be in [1, 62]")
    identity = tuple(range(rank))
    channel = (rank - 1,)
    validate_maps(candidate, (identity, channel, channel, identity), "requantize")
    block, operations = body_operations(candidate, "requantize")
    names = tuple(item.name for item in operations)
    # Enumerating complete bodies makes extra arithmetic or reordered clamping
    # a rejection instead of a potentially incorrect best-effort rewrite.
    tosa_without_zp = (
        "tosa.apply_scale",
        "arith.maxsi",
        "arith.minsi",
        "arith.trunci",
        "linalg.yield",
    )
    tosa_with_zp = (
        "tosa.apply_scale",
        "arith.addi",
        "arith.maxsi",
        "arith.minsi",
        "arith.trunci",
        "linalg.yield",
    )
    arith_without_zp = (
        "arith.extui",
        "arith.extsi",
        "arith.extsi",
        "arith.muli",
        "arith.extui",
        "arith.shli",
        "arith.shrui",
        "arith.addi",
        "arith.cmpi",
        "arith.select",
        "arith.addi",
        "arith.cmpi",
        "arith.select",
        "arith.shrsi",
        "arith.trunci",
        "arith.maxsi",
        "arith.minsi",
        "arith.trunci",
        "linalg.yield",
    )
    arith_with_zp = arith_without_zp[:15] + ("arith.addi",) + arith_without_zp[15:]
    if names in (tosa_without_zp, tosa_with_zp):
        if len(block.arguments) != 4:
            raise ValueError("requantize scalar block must have four arguments")
        scale = operations[0]
        rounding = ir.Attribute.parse(
            "#tosa.rounding_mode<DOUBLE_ROUND>", context=candidate.context
        )
        if list(scale.operands) != list(block.arguments[:3]) or scale.attributes.get(
            "rounding_mode"
        ) != rounding:
            raise ValueError("requantize must use per-channel DOUBLE_ROUND scaling")
        cursor = scale.results[0]
        output_zero_point = 0
        offset = 1
        if names == tosa_with_zp:
            addition = operations[1]
            if addition.operands[0] != cursor:
                raise ValueError("requantize output zero point has modified dataflow")
            output_zero_point = require_zero_point(addition.operands[1], "output zero point")
            cursor = addition.results[0]
            offset = 2
        _validate_clamp(operations, cursor, offset, "requantize")
    elif names in (arith_without_zp, arith_with_zp):
        _, output_zero_point = _validate_expanded_scale(
            operations, block, names == arith_with_zp
        )
    else:
        raise ValueError("requantize has unsupported scalar semantics")
    return multiplier, shift, output_zero_point, output_shape


def _match(root: ir.Operation, key: OpKey) -> FullyConnectedMatch:
    """Proves one complete quantized FC subgraph rooted at ``root``.

    The function is intentionally side-effect free. It either returns every
    value required for reconstruction or raises ``ValueError`` with the first
    failed semantic invariant.
    """
    # linalg.quantized_matmul has lhs, rhs, two zero-points, and one DPS init.
    if len(root.operands) != 5 or len(root.results) != 1:
        raise ValueError("quantized matmul must have five operands and one result")
    input_shape = tensor_shape(root.operands[0], "i8", 2)
    transposed_shape = tensor_shape(root.operands[1], "i8", 2)
    accumulator_shape = tensor_shape(root.results[0], "i32", 2)
    rows, input_channels = input_shape
    if transposed_shape[0] != input_channels:
        raise ValueError("input and weight reduction dimensions disagree")
    output_channels = transposed_shape[1]
    if accumulator_shape != (rows, output_channels):
        raise ValueError("matmul accumulator dimensions are inconsistent")

    # The replacement reads the original output-major constant directly. This
    # is safe only for the canonical [1, 0] frontend transpose.
    transpose = owner_operation(root.operands[1])
    if (
        transpose is None
        or transpose.name != "linalg.transpose"
        or "permutation" not in transpose.attributes
        or dense_ints(transpose.attributes["permutation"]) != (1, 0)
    ):
        raise ValueError("weight must use a [1, 0] linalg.transpose")
    output_major_weight = transpose.operands[0]
    require_constant_tensor(
        output_major_weight,
        "i8",
        (output_channels, input_channels),
        "output-major weight",
    )
    input_zero_point = require_zero_point(root.operands[2], "input zero point")
    weight_zero_point = require_zero_point(root.operands[3], "weight zero point")
    bias_initializer = owner_operation(root.operands[4])
    if bias_initializer is None or bias_initializer.name != "linalg.generic":
        raise ValueError("accumulator must be initialized by a bias broadcast")
    bias = _validate_bias(bias_initializer, output_channels)

    # A unique consumer is required because erasing the accumulator must not
    # change an independent user that still expects the full i32 tensor.
    users = [
        use.owner.operation if hasattr(use.owner, "operation") else use.owner
        for use in root.results[0].uses
    ]
    if len(users) != 1:
        raise ValueError("matmul accumulator must have exactly one user")
    expand: ir.Operation | None
    if users[0].name == "tensor.expand_shape":
        expand = users[0]
        source = expand.results[0]
        expanded_shape = tensor_shape(source, "i32")
        # Only row-major [rows, Cout] -> [...prefix, Cout] expansion preserves
        # the flattened row formula emitted by the replacement.
        expected_reassociation = (
            tuple(range(len(expanded_shape) - 1)),
            (len(expanded_shape) - 1,),
        )
        reassociation = tuple(
            tuple(int(index) for index in group)
            for group in expand.attributes["reassociation"]
        )
        if reassociation != expected_reassociation:
            raise ValueError("accumulator expand_shape has non-canonical reassociation")
        rescale = unique_user(source, "linalg.generic", "expanded accumulator")
    elif users[0].name == "linalg.generic":
        expand = None
        source = root.results[0]
        rescale = users[0]
    else:
        raise ValueError("matmul accumulator must feed expand_shape or requantize")
    multiplier, shift, output_zero_point, output_shape = _validate_rescale(
        rescale, output_channels, source
    )
    # The output prefix must cover exactly the collapsed matmul row extent.
    prefix_elements = 1
    for dimension in output_shape[:-1]:
        prefix_elements *= dimension
    if prefix_elements != rows:
        raise ValueError("output reshape does not preserve matmul rows")
    return FullyConnectedMatch(
        root=key,
        matmul=root,
        expand=expand,
        rescale=rescale,
        input=root.operands[0],
        output_major_weight=output_major_weight,
        bias=bias,
        multiplier=multiplier,
        shift=shift,
        input_quantization=AffineQuantization("i8", None, input_zero_point, None),
        weight_quantization=AffineQuantization("i8", None, weight_zero_point, None),
        output_quantization=AffineQuantization("i8", None, output_zero_point, None),
        rows=rows,
        input_channels=input_channels,
        output_channels=output_channels,
        output_shape=output_shape,
    )


def analyze_fully_connected(
    graphs: list[FunctionGraph], occupied: set[ir.Operation]
) -> Analysis:
    """Finds deterministic, non-overlapping FC matches and rejection reasons.

    Every quantized matmul is treated as a possible root. Rejections are data,
    not fatal errors, so auto mode can safely preserve unsupported subgraphs.
    """
    matches: list[FullyConnectedMatch] = []
    rejected: list[RejectedCandidate] = []
    for graph in graphs:
        for node in graph.nodes:
            if node.operation.name != "linalg.quantized_matmul":
                continue
            try:
                candidate = _match(node.operation, node.key)
                # Prevent future pattern families from selecting an operation
                # already owned by an earlier deterministic candidate.
                operations = candidate.claimed_operations
                if occupied.intersection(operations):
                    # Composite patterns have higher priority. Their internal
                    # operators are skipped silently instead of being reported
                    # as false semantic failures.
                    continue
                occupied.update(operations)
                matches.append(candidate)
            except (ValueError, IndexError, KeyError) as error:
                rejected.append(
                    RejectedCandidate(
                        node.key,
                        FullyConnectedMatch.kind,
                        str(error),
                        str(node.operation.location),
                    )
                )
    return Analysis(matches, rejected)
