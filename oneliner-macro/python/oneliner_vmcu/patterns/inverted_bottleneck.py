"""Semantic matcher for the paper-faithful K²+2 inverted bottleneck."""

from __future__ import annotations

from iree.compiler import ir

from ..graph import FunctionGraph
from ..ir_utils import (
    body_operations,
    generic_io,
    owner_operation,
    require_scalar,
    semantic_users,
    tensor_shape,
    validate_maps,
)
from ..model import (
    Analysis,
    InvertedBottleneckMatch,
    RejectedCandidate,
    ResidualMatch,
    ResidualScale,
)
from .conv2d import _match as match_conv2d
from .depthwise import _match as match_depthwise


_CONV2D_NAMES = frozenset(
    ("linalg.conv_2d_nhwc_hwcf_q", "linalg.conv_2d_nhwc_hwcf")
)


def _single_non_return_user(value: ir.Value) -> ir.Operation | None:
    """Returns one semantic graph consumer, ignoring storage-only uses."""
    users = semantic_users(value)
    if not users:
        return None
    if len(users) != 1:
        raise ValueError("IBN intermediate value has multiple fusion-external users")
    return users[0]


def _single_named_user(value: ir.Value, name: str) -> ir.Operation | None:
    """Returns one named semantic consumer, or ``None`` for non-matches."""
    users = semantic_users(value)
    if len(users) != 1 or users[0].name != name:
        return None
    return users[0]


def _find_structural_projection(
    depthwise_root: ir.Operation,
) -> ir.Operation | None:
    """Finds a real IBN projection by walking a depthwise spine forward.

    A pointwise expansion and a terminal pointwise convolution are both
    1x1 Conv2D operations.  Starting at every such operation was the source
    of the plan's twelve spurious IBN rejections.  The depthwise spine is the
    discriminating structure, so only a complete
    ``depthwise -> collapse -> bias -> requantize -> 1x1 projection`` chain is
    eligible for IBN matching.
    """
    if depthwise_root.name != "linalg.depthwise_conv_2d_nhwc_hwcm_q":
        return None
    try:
        padding = owner_operation(depthwise_root.operands[0])
        if padding is None or padding.name != "tensor.pad" or not padding.operands:
            return None
        expansion_rescale = owner_operation(padding.operands[0])
        if (
            expansion_rescale is None
            or expansion_rescale.name != "linalg.generic"
            or not expansion_rescale.operands
        ):
            return None
        expansion_root = owner_operation(expansion_rescale.operands[0])
        if (
            expansion_root is None
            or expansion_root.name not in _CONV2D_NAMES
        ):
            return None
        expansion_shape = tensor_shape(expansion_root.operands[1], "i8", 4)
        if expansion_shape[:2] != (1, 1):
            return None

        collapse = _single_named_user(depthwise_root.results[0], "tensor.collapse_shape")
        if collapse is None:
            return None
        bias_add = _single_named_user(collapse.results[0], "linalg.generic")
        if bias_add is None:
            return None
        depthwise_rescale = _single_named_user(bias_add.results[0], "linalg.generic")
        if depthwise_rescale is None:
            return None
        projections = [
            user
            for user in semantic_users(depthwise_rescale.results[0])
            if user.name in _CONV2D_NAMES
        ]
        if len(projections) != 1:
            return None
        projection_shape = tensor_shape(projections[0].operands[1], "i8", 4)
        if projection_shape[:2] != (1, 1):
            return None
        return projections[0]
    except (IndexError, ValueError):
        # Standalone analyzers own malformed or incomplete depthwise chains;
        # this structural probe must not turn them into duplicate IBN noise.
        return None


def _validate_residual(
    candidate: ir.Operation,
    projection: ir.Value,
    shape: tuple[int, int, int, int],
    output_zero_point: int,
) -> ir.Value:
    """Proves same-quantization residual centering, add, clamp, and store."""
    inputs, outputs = generic_io(candidate)
    if len(inputs) != 2 or len(outputs) != 1 or inputs[0] != projection:
        raise ValueError("residual add must consume projection and skip tensors")
    if tensor_shape(inputs[1], "i8", 4) != shape or tensor_shape(candidate.results[0], "i8", 4) != shape:
        raise ValueError("residual input/output shapes must equal projection")
    # The scalar arithmetic below is only equivalent to an elementwise skip
    # connection when every operand uses the four-dimensional identity map.
    identity = (0, 1, 2, 3)
    validate_maps(candidate, (identity, identity, identity), "IBN residual add")
    block, operations = body_operations(candidate, "IBN residual add")
    names = tuple(item.name for item in operations)
    expected = (
        "arith.extsi",
        "arith.extsi",
        "arith.subi",
        "arith.subi",
        "arith.addi",
        "arith.addi",
        "arith.maxsi",
        "arith.minsi",
        "arith.trunci",
        "linalg.yield",
    )
    if len(block.arguments) != 3 or names != expected:
        raise ValueError("residual add has unsupported affine scalar semantics")
    p_ext, r_ext, p_center, r_center, addition, offset, lower, upper, trunc, terminator = operations
    if (
        p_ext.operands[0] != block.arguments[0]
        or r_ext.operands[0] != block.arguments[1]
        or p_center.operands[0] != p_ext.results[0]
        or r_center.operands[0] != r_ext.results[0]
        or require_scalar(p_center.operands[1], "projection residual zero point") != output_zero_point
        or require_scalar(r_center.operands[1], "skip residual zero point") != output_zero_point
        or list(addition.operands) != [p_center.results[0], r_center.results[0]]
        or offset.operands[0] != addition.results[0]
        or require_scalar(offset.operands[1], "residual output zero point") != output_zero_point
        or lower.operands[0] != offset.results[0]
        or require_scalar(lower.operands[1], "residual clamp minimum") != -128
        or upper.operands[0] != lower.results[0]
        or require_scalar(upper.operands[1], "residual clamp maximum") != 127
        or trunc.operands[0] != upper.results[0]
        or terminator.operands[0] != trunc.results[0]
    ):
        raise ValueError("residual add dataflow or affine parameters changed")
    return inputs[1]


_SPLIT_RESIDUAL_SCALARS = {
    "arith.addi",
    "arith.cmpi",
    "arith.extsi",
    "arith.extui",
    "arith.maxsi",
    "arith.minsi",
    "arith.muli",
    "arith.select",
    "arith.shli",
    "arith.shrsi",
    "arith.shrui",
    "arith.subi",
    "arith.trunci",
}


def _tensor_shape_i8_or_i32(value: ir.Value) -> tuple[int, ...]:
    """Returns a static tensor shape for an elementwise residual value."""
    for element_type in ("i8", "i32"):
        try:
            return tensor_shape(value, element_type)
        except ValueError:
            pass
    raise ValueError("IBN split residual tensors must be static i8 or i32")


def _tensor_element_type(value: ir.Value) -> str:
    """Returns the scalar element type of a validated ranked tensor."""
    return str(ir.RankedTensorType(value.type).element_type)


def _validate_split_generic(
    candidate: ir.Operation, shape: tuple[int, int, int, int]
) -> tuple[list[ir.Value], ir.Value]:
    """Validates one pure elementwise generic in an IREE residual pipeline."""
    inputs, outputs = generic_io(candidate)
    if not outputs or len(outputs) != 1:
        raise ValueError("IBN split residual generic must have one output")
    if any(_tensor_shape_i8_or_i32(value) != shape for value in (*inputs, *outputs)):
        raise ValueError("IBN split residual shapes must remain unchanged")
    identity = tuple(range(len(shape)))
    validate_maps(candidate, (identity,) * (len(inputs) + 1), "IBN split residual")
    block, operations = body_operations(candidate, "IBN split residual")
    if len(block.arguments) != len(inputs) + 1 or not operations:
        raise ValueError("IBN split residual block shape is invalid")
    if operations[-1].name != "linalg.yield" or len(operations[-1].operands) != 1:
        raise ValueError("IBN split residual must end in one linalg.yield")
    scalar_operations = operations[:-1]
    if any(item.name not in _SPLIT_RESIDUAL_SCALARS for item in scalar_operations):
        raise ValueError("IBN split residual has unsupported scalar semantics")
    known = set(block.arguments)
    for item in scalar_operations:
        if len(item.results) != 1:
            raise ValueError("IBN split residual has invalid scalar dataflow")
        for operand in item.operands:
            if operand in known:
                continue
            owner = owner_operation(operand)
            if owner is None or owner.name != "arith.constant":
                raise ValueError("IBN split residual has invalid scalar dataflow")
        known.add(item.results[0])
    if operations[-1].operands[0] not in known or operations[-1].operands[0] in block.arguments:
        raise ValueError("IBN split residual yield must use computed scalar data")
    return inputs, operations[-1].operands[0]


def _direct_scalar_constant(value: ir.Value) -> int | None:
    """Returns a direct integer constant used by split scalar arithmetic."""
    owner = owner_operation(value)
    if owner is None or owner.name != "arith.constant":
        return None
    try:
        return require_scalar(value, "IBN split residual scalar")
    except ValueError:
        return None


def _validate_split_scale(
    candidate: ir.Operation,
    shape: tuple[int, int, int, int],
    *,
    expected_input_zero_point: int | None,
    expected_output_zero_point: int | None,
) -> ResidualScale:
    """Validates one centered/scaled unary generic and records its integers.

    IREE may fold the TOSA scale into several ``arith`` operations before the
    residual add.  The exact operation sequence is intentionally allowed to
    vary, but the dataflow must still contain one constant multiplier, one
    bounded right shift, a final i32 truncation, and (for the terminal i8
    value) the canonical output offset and clamp.
    """
    inputs, yielded = _validate_split_generic(candidate, shape)
    if len(inputs) != 1:
        raise ValueError("IBN split residual scaling must have one tensor input")
    input_type = str(candidate.regions[0].blocks[0].arguments[0].type)
    output_type = str(yielded.type)
    if input_type not in ("i8", "i32") or output_type not in ("i8", "i32"):
        raise ValueError("IBN split residual scaling must use i8/i32 tensors")
    block, operations = body_operations(candidate, "IBN split residual scaling")
    scalar_operations = operations[:-1]
    if not scalar_operations:
        raise ValueError("IBN split residual scaling has no scalar computation")

    # Every non-constant scalar operand must be a prior result or a block
    # argument.  This prevents an unrelated tensor or a later operation from
    # being smuggled into an otherwise shape-correct residual generic.
    known = set(block.arguments)
    for item in scalar_operations:
        if len(item.results) != 1:
            raise ValueError("IBN split residual scaling has invalid scalar dataflow")
        for operand in item.operands:
            if operand in known or _direct_scalar_constant(operand) is not None:
                continue
            raise ValueError("IBN split residual scaling has invalid scalar dataflow")
        known.add(item.results[0])

    def depends_on(value: ir.Value, target: ir.Value, seen: set[ir.Value] | None = None) -> bool:
        """Checks scalar expression dependence without following constants."""
        if value == target:
            return True
        if _direct_scalar_constant(value) is not None:
            return False
        owner = owner_operation(value)
        if owner is None or owner not in scalar_operations:
            return False
        visited = set() if seen is None else seen
        if value in visited:
            return False
        visited.add(value)
        return any(depends_on(operand, target, visited) for operand in owner.operands)

    input_zero_point = None
    if input_type == "i8":
        if expected_input_zero_point is None:
            raise ValueError("IBN split residual i8 input lacks an expected zero point")
        extensions = [
            item
            for item in scalar_operations
            if item.name == "arith.extsi"
            and len(item.operands) == 1
            and item.operands[0] == block.arguments[0]
            and str(item.results[0].type) == "i32"
        ]
        centers = [
            item
            for item in scalar_operations
            if item.name == "arith.subi"
            and len(item.operands) == 2
            and _direct_scalar_constant(item.operands[1]) is not None
        ]
        if len(extensions) != 1 or len(centers) != 1:
            raise ValueError("IBN split residual i8 path must center exactly once")
        center = centers[0]
        if (
            center.operands[0] != extensions[0].results[0]
            or _direct_scalar_constant(center.operands[1]) != expected_input_zero_point
        ):
            raise ValueError("IBN split residual i8 path has the wrong zero point")
        input_zero_point = expected_input_zero_point
    elif expected_input_zero_point is not None:
        raise ValueError("IBN split residual i32 path cannot consume an i8 zero point")

    multiplications = [item for item in scalar_operations if item.name == "arith.muli"]
    scale_values = []
    for item in multiplications:
        constants = [_direct_scalar_constant(operand) for operand in item.operands]
        if sum(value is not None for value in constants) != 1:
            continue
        multiplier = next(value for value in constants if value is not None)
        if not 0 <= multiplier < (1 << 31):
            raise ValueError("IBN split residual multiplier is out of range")
        scale_values.append(multiplier)
    if len(scale_values) != 1:
        raise ValueError("IBN split residual scaling must contain one constant multiplier")

    shifts = [
        _direct_scalar_constant(item.operands[1])
        for item in scalar_operations
        if item.name in ("arith.shrsi", "arith.shrui") and len(item.operands) == 2
    ]
    shifts = [value for value in shifts if value is not None]
    if len(shifts) != 1 or not 1 <= shifts[0] <= 62:
        raise ValueError("IBN split residual scaling must contain one shift in [1, 62]")
    multiplier = scale_values[0]
    shift = shifts[0]

    if output_type == "i32":
        if expected_output_zero_point is not None:
            raise ValueError("IBN split residual intermediate scale cannot clamp to i8")
        if any(item.name in ("arith.maxsi", "arith.minsi") for item in scalar_operations):
            raise ValueError("IBN split residual i32 scale must not clamp")
        truncations = [item for item in scalar_operations if item.name == "arith.trunci"]
        if (
            not truncations
            or truncations[-1].results[0] != yielded
            or str(truncations[-1].results[0].type) != "i32"
        ):
            raise ValueError("IBN split residual i32 scale must end in i32 truncation")
        output_zero_point = None
    else:
        if len(scalar_operations) < 4:
            raise ValueError("IBN split residual i8 scale is missing its clamp")
        # IREE omits the affine add when the output zero point is zero. Accept
        # both canonical suffixes without indexing operands before checking the
        # operation shape (the old order was the source of an IndexError).
        truncation, lower, upper, final_truncation = scalar_operations[-4:]
        offset = None
        output_zero_point = 0
        if len(scalar_operations) >= 5 and scalar_operations[-4].name == "arith.addi":
            truncation, offset, lower, upper, final_truncation = scalar_operations[-5:]
            if len(offset.operands) == 2:
                output_zero_point = _direct_scalar_constant(offset.operands[1])
            else:
                output_zero_point = None
        lower_input = truncation.results[0] if offset is None else offset.results[0]
        if (
            truncation.name != "arith.trunci"
            or str(truncation.results[0].type) != "i32"
            or (offset is not None and offset.operands[0] != truncation.results[0])
            or output_zero_point is None
            or not -128 <= output_zero_point <= 127
            or (
                expected_output_zero_point is not None
                and output_zero_point != expected_output_zero_point
            )
            or lower.name != "arith.maxsi"
            or lower.operands[0] != lower_input
            or _direct_scalar_constant(lower.operands[1]) != -128
            or upper.name != "arith.minsi"
            or upper.operands[0] != lower.results[0]
            or _direct_scalar_constant(upper.operands[1]) != 127
            or final_truncation.name != "arith.trunci"
            or final_truncation.operands[0] != upper.results[0]
            or str(final_truncation.results[0].type) != "i8"
            or yielded != final_truncation.results[0]
        ):
            raise ValueError("IBN split residual terminal scale has invalid offset or clamp")

    return ResidualScale(
        input_type,
        output_type,
        input_zero_point,
        output_zero_point,
        multiplier,
        shift,
    )


def _validate_split_add(
    candidate: ir.Operation, shape: tuple[int, int, int, int]
) -> tuple[ir.Value, ir.Value]:
    """Validates the exact two-input i32 residual add and returns its inputs."""
    inputs, yielded = _validate_split_generic(candidate, shape)
    block, operations = body_operations(candidate, "IBN split residual add")
    if len(inputs) != 2 or any(str(value.type) != "i32" for value in block.arguments[:2]):
        raise ValueError("IBN split residual add must consume two i32 tensors")
    if str(yielded.type) != "i32":
        raise ValueError("IBN split residual add must produce i32")
    if (
        len(block.arguments) != 3
        or tuple(item.name for item in operations) != ("arith.addi", "linalg.yield")
        or list(operations[0].operands) != list(block.arguments[:2])
        or yielded != operations[0].results[0]
    ):
        raise ValueError("IBN split residual add has modified i32 dataflow")
    return inputs[0], inputs[1]


def _depends_on(value: ir.Value, target: ir.Value, seen: set[ir.Operation] | None = None) -> bool:
    """Checks tensor-level dependency through elementwise generic inputs."""
    if value == target:
        return True
    owner = owner_operation(value)
    if owner is None:
        return False
    visited = set() if seen is None else seen
    if owner in visited or owner.name != "linalg.generic":
        return False
    visited.add(owner)
    inputs, _ = generic_io(owner)
    return any(_depends_on(item, target, visited) for item in inputs)


def _match_split_residual(
    projection: ir.Value,
    shape: tuple[int, int, int, int],
    module_input: ir.Value,
    module_input_zero_point: int,
    projection_output_zero_point: int,
) -> ResidualMatch | None:
    """Recognizes IREE's split residual conversion/add/requantize path.

    The path is intentionally retained during emission.  All its scalar
    operations are validated here, while preserving the original chain keeps
    IREE's exact rounding sequence instead of approximating it with a
    same-zero-point add.
    """
    first = _single_non_return_user(projection)
    if first is None or first.name != "linalg.generic":
        return None
    current_value = projection
    projection_chain: list[ir.Operation] = []
    projection_scales: list[ResidualScale] = []
    add_operation: ir.Operation | None = None
    residual_input: ir.Value | None = None
    for _ in range(8):
        current = _single_non_return_user(current_value)
        if current is None or current.name != "linalg.generic":
            break
        inputs, _ = _validate_split_generic(current, shape)
        if current_value not in inputs:
            raise ValueError("IBN split residual chain lost projection dataflow")
        if len(inputs) == 2:
            if add_operation is not None:
                raise ValueError("IBN split residual chain has more than one add")
            left, right = _validate_split_add(current, shape)
            projection_index = inputs.index(current_value)
            residual_input = inputs[1 - projection_index]
            if left != inputs[0] or right != inputs[1]:
                raise ValueError("IBN split residual add input order changed")
            add_operation = current
            break
        if len(inputs) != 1:
            raise ValueError("IBN split residual chain has an unsupported arity")
        scale = _validate_split_scale(
            current,
            shape,
            expected_input_zero_point=(
                projection_output_zero_point
                if _tensor_element_type(current_value) == "i8"
                else None
            ),
            expected_output_zero_point=None,
        )
        # The first projection generic is the only i8-input operation on this
        # branch.  Its expected zero point is checked below using the parsed
        # immutable quantization fact; subsequent stages are i32-only scales.
        projection_chain.append(current)
        projection_scales.append(scale)
        current_value = current.results[0]

    if add_operation is None or residual_input is None:
        return None
    if not projection_chain or projection_scales[0].input_type != "i8":
        raise ValueError("IBN split residual projection path lacks i8 centering")
    projection_zero_point = projection_scales[0].input_zero_point
    if projection_zero_point is None:
        raise ValueError("IBN split residual projection path lacks a zero point")

    # Walk the skip side backwards.  Each producer must be a validated unary
    # scale and the root must be the exact module input; a merely related
    # tensor is not enough to claim the residual edge.
    skip_chain_reverse: list[ir.Operation] = []
    skip_scales_reverse: list[ResidualScale] = []
    skip_value = residual_input
    seen: set[ir.Operation] = set()
    for _ in range(8):
        if skip_value == module_input:
            break
        owner = owner_operation(skip_value)
        if owner is None or owner.name != "linalg.generic":
            break
        if owner in seen:
            raise ValueError("IBN split residual skip path contains a cycle")
        seen.add(owner)
        users = semantic_users(owner.results[0])
        if not users or (add_operation not in users and len(users) != 1):
            raise ValueError("IBN split residual skip path has an external user")
        inputs, _ = _validate_split_generic(owner, shape)
        if len(inputs) != 1:
            raise ValueError("IBN split residual skip path has an unsupported arity")
        scale = _validate_split_scale(
            owner,
            shape,
            expected_input_zero_point=(
                module_input_zero_point
                if _tensor_element_type(inputs[0]) == "i8"
                else None
            ),
            expected_output_zero_point=None,
        )
        skip_chain_reverse.append(owner)
        skip_scales_reverse.append(scale)
        skip_value = inputs[0]
    if skip_value != module_input or not skip_chain_reverse:
        raise ValueError("IBN split residual skip path must scale the exact module input")
    if skip_scales_reverse[-1].input_type != "i8":
        raise ValueError("IBN split residual skip path lacks i8 centering")

    # Follow the add's result to the terminal i8 requantization.  The terminal
    # result can fan out to the next block and to a later skip path, so stop at
    # the verified i8 generic rather than treating that legal fan-out as an
    # ambiguous internal edge.
    final_chain: list[ir.Operation] = []
    final_scales: list[ResidualScale] = []
    current_value = add_operation.results[0]
    for _ in range(8):
        current = _single_non_return_user(current_value)
        if current is None or current.name != "linalg.generic":
            raise ValueError("IBN split residual chain has no final requantization")
        inputs, _ = _validate_split_generic(current, shape)
        if len(inputs) != 1 or inputs[0] != current_value:
            raise ValueError("IBN split residual final path lost add dataflow")
        scale = _validate_split_scale(
            current,
            shape,
            expected_input_zero_point=None,
            # The residual output may intentionally use the next block's
            # affine zero point (it need not equal the projection's i8 point).
            # The validator still extracts and range-checks the terminal
            # offset, while the immutable match records it in ``scales``.
            expected_output_zero_point=None,
        )
        final_chain.append(current)
        final_scales.append(scale)
        current_value = current.results[0]
        if _tensor_element_type(current_value) == "i8":
            break
    if not final_chain or _tensor_element_type(current_value) != "i8":
        raise ValueError("IBN split residual chain has no terminal i8 value")
    if _depends_on(residual_input, projection):
        raise ValueError("IBN split residual skip path depends on projection")
    if not _depends_on(residual_input, module_input):
        raise ValueError("IBN split residual skip path must consume module input")
    operations = tuple(
        projection_chain
        + skip_chain_reverse
        + [add_operation]
        + final_chain
    )
    return ResidualMatch(
        "split",
        operations,
        final_chain[-1],
        skip_value,
        tuple(projection_scales + list(reversed(skip_scales_reverse)) + final_scales),
    )


def _match_chain(
    projection_root: ir.Operation,
    projection_key,
    keys: dict[ir.Operation, object],
) -> InvertedBottleneckMatch:
    """Walks projection producers backward and proves the entire IBN chain."""
    projection = match_conv2d(projection_root, projection_key)
    if projection.weight_shape[:2] != (1, 1) or projection.strides != (1, 1):
        raise ValueError("IBN projection must be stride-1 1x1 Conv2D")
    depthwise_output = projection.input
    depthwise_rescale = owner_operation(depthwise_output)
    if depthwise_rescale is None or depthwise_rescale.name != "linalg.generic":
        raise ValueError("IBN projection input must come from depthwise requantization")
    depthwise_bias = owner_operation(depthwise_rescale.operands[0])
    if depthwise_bias is None or depthwise_bias.name != "linalg.generic":
        raise ValueError("IBN depthwise requantization must consume bias add")
    depthwise_collapse = owner_operation(depthwise_bias.operands[1])
    if depthwise_collapse is None or depthwise_collapse.name != "tensor.collapse_shape":
        raise ValueError("IBN depthwise bias must consume collapsed accumulator")
    depthwise_root = owner_operation(depthwise_collapse.operands[0])
    if depthwise_root is None or depthwise_root.name != "linalg.depthwise_conv_2d_nhwc_hwcm_q":
        raise ValueError("IBN middle operator must be quantized depthwise Conv2D")
    depthwise = match_depthwise(depthwise_root, keys[depthwise_root])
    if depthwise.dilations != (1, 1):
        raise ValueError("IBN K²+2 schedule requires unit depthwise dilation")
    padding = owner_operation(depthwise.input)
    if padding is None or padding.name != "tensor.pad":
        raise ValueError("IBN depthwise input must use explicit quantized padding")
    expansion_output = padding.operands[0]
    expansion_rescale = owner_operation(expansion_output)
    if expansion_rescale is None or expansion_rescale.name != "linalg.generic":
        raise ValueError("IBN padding must consume expansion requantization")
    expansion_root = owner_operation(expansion_rescale.operands[0])
    if expansion_root is None or expansion_root.name not in _CONV2D_NAMES:
        raise ValueError("IBN expansion must be quantized Conv2D")
    expansion = match_conv2d(expansion_root, keys[expansion_root])
    if expansion.weight_shape[:2] != (1, 1) or expansion.strides != (1, 1):
        raise ValueError("IBN expansion must be stride-1 1x1 Conv2D")
    if depthwise.output_shape[3] != expansion.output_shape[3]:
        raise ValueError("IBN expansion/depthwise channel dimensions disagree")
    if (
        expansion.output_quantization.zero_point_at()
        != depthwise.input_quantization.zero_point_at()
    ):
        raise ValueError("IBN expansion/depthwise affine zero-points disagree")
    if projection.input_shape != depthwise.output_shape:
        raise ValueError("IBN depthwise/projection shapes disagree")
    if (
        depthwise.output_quantization.zero_point_at()
        != projection.input_quantization.zero_point_at()
    ):
        raise ValueError("IBN depthwise/projection affine zero-points disagree")
    if expansion.input_shape[0] != 1 or projection.output_shape[0] != 1:
        raise ValueError("IBN fixed schedule currently requires batch 1")
    if _single_non_return_user(expansion.rescale.results[0]) != padding:
        raise ValueError("IBN expansion has a fusion-external consumer")
    if _single_non_return_user(depthwise.rescale.results[0]) != projection.conv:
        raise ValueError("IBN depthwise has a fusion-external consumer")

    residual = None
    projection_users = semantic_users(projection.rescale.results[0])
    # A block output can feed both the next expansion and a later skip path.
    # In that fan-out case the current IBN still has a safe projection
    # boundary, but no single residual chain can be claimed by this match.
    projection_user = projection_users[0] if len(projection_users) == 1 else None
    if projection_user is not None:
        if projection_user.name != "linalg.generic":
            # Any non-elementwise user is a legal compact-region boundary.
            # It remains outside this candidate and consumes the replacement
            # projection value after emission.
            projection_user = None
        if projection_user is not None:
            try:
                residual_input = _validate_residual(
                    projection_user,
                    projection.rescale.results[0],
                    projection.output_shape,
                    projection.output_quantization.zero_point_at(),
                )
                residual = ResidualMatch(
                    "fused", (projection_user,), projection_user, residual_input
                )
            except ValueError:
                try:
                    # Split residual arithmetic is validated here, but remains
                    # an explicit materialization boundary until the pool
                    # emitter can reproduce every i64 rounding variant inside
                    # a tied dispatch. It must never be approximated as the
                    # same-zero-point fused form.
                    _match_split_residual(
                        projection.rescale.results[0],
                        projection.output_shape,
                        expansion.input,
                        expansion.input_quantization.zero_point_at(),
                        projection.output_quantization.zero_point_at(),
                    )
                    residual = None
                except (ValueError, IndexError):
                    # A generic that is not a validated residual is an
                    # unsupported/materialized boundary, not an IBN failure.
                    residual = None
    if residual is not None:
        residual_input = residual.input
        if depthwise.strides != (1, 1):
            raise ValueError("strided IBN cannot contain a shape-preserving residual")
        if residual_input != expansion.input:
            raise ValueError("IBN residual must consume the exact module input SSA value")
        if (
            expansion.input_quantization.zero_point_at()
            != projection.output_quantization.zero_point_at()
        ) and residual.mode == "fused":
            raise ValueError("IBN residual input/output affine zero-points disagree")
    return InvertedBottleneckMatch(
        root=projection_key,
        expansion=expansion,
        depthwise=depthwise,
        projection=projection,
        depthwise_padding=padding,
        residual=residual,
    )


def analyze_inverted_bottleneck(
    graphs: list[FunctionGraph], occupied: set[ir.Operation]
) -> Analysis:
    """Matches IBNs before standalone patterns claim component operations.

    Discovery is rooted at depthwise operators so ordinary 1x1 expansion and
    terminal pointwise convolutions never become fake projection candidates.
    """
    matches = []
    rejected = []
    for graph in graphs:
        keys = {node.operation: node.key for node in graph.nodes}
        for node in graph.nodes:
            if node.operation.name != "linalg.depthwise_conv_2d_nhwc_hwcm_q":
                continue
            projection_root = _find_structural_projection(node.operation)
            if projection_root is None:
                continue
            projection_key = keys.get(projection_root)
            if projection_key is None:
                continue
            try:
                candidate = _match_chain(projection_root, projection_key, keys)
                if occupied.intersection(candidate.claimed_operations):
                    # A higher-priority composite already owns this exact SSA
                    # region. Overlap is selection bookkeeping, not a semantic
                    # rejection that should appear in the user-facing plan.
                    continue
                occupied.update(candidate.claimed_operations)
                matches.append(candidate)
            except (ValueError, IndexError, KeyError) as error:
                rejected.append(
                    RejectedCandidate(
                        projection_key,
                        InvertedBottleneckMatch.kind,
                        str(error),
                        str(projection_root.location),
                    )
                )
    return Analysis(matches, rejected)
