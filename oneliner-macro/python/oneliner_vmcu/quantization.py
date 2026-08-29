"""Model-independent descriptions of affine integer quantization.

The matcher copies quantization facts out of MLIR into these immutable Python
objects.  Keeping the facts separate from live ``ir.Value`` handles makes them
safe to serialize in plans and reuse after transactional reparsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias


ScalarOrVector: TypeAlias = int | float | tuple[int, ...] | tuple[float, ...]


@dataclass(frozen=True)
class AffineQuantization:
    """Complete affine quantization metadata for one tensor boundary.

    ``scale`` may be unavailable at preprocessing time when IREE has already
    converted it into integer multiplier/shift operands.  In that case it is
    ``None`` and those exact SSA operands remain authoritative.  A tuple denotes
    per-axis values and therefore requires ``quantized_axis``.
    """

    storage_type: str
    scale: ScalarOrVector | None
    zero_point: int | tuple[int, ...]
    quantized_axis: int | None
    rounding_mode: str = "DOUBLE_ROUND"
    clamp_min: int = -128
    clamp_max: int = 127

    def __post_init__(self) -> None:
        """Rejects ambiguous or out-of-range descriptions immediately."""
        if self.storage_type not in ("i8", "ui8"):
            raise ValueError(f"unsupported quantized storage type: {self.storage_type}")
        lower, upper = (-128, 127) if self.storage_type == "i8" else (0, 255)
        zero_points = (
            self.zero_point if isinstance(self.zero_point, tuple) else (self.zero_point,)
        )
        if not zero_points:
            raise ValueError("zero-point vector cannot be empty")
        if any(value < lower or value > upper for value in zero_points):
            raise ValueError(
                f"zero point must be representable by {self.storage_type}: {zero_points}"
            )
        vector_scale = isinstance(self.scale, tuple)
        vector_zero_point = isinstance(self.zero_point, tuple)
        if (vector_scale or vector_zero_point) and self.quantized_axis is None:
            raise ValueError("per-axis quantization requires quantized_axis")
        if vector_scale and vector_zero_point and len(self.scale) != len(self.zero_point):
            raise ValueError("per-axis scale and zero-point lengths disagree")
        if self.clamp_min > self.clamp_max:
            raise ValueError("quantization clamp minimum exceeds maximum")

    @property
    def is_per_axis(self) -> bool:
        """Returns whether any quantization parameter varies by axis."""
        return isinstance(self.scale, tuple) or isinstance(self.zero_point, tuple)

    def zero_point_at(self, axis_index: int = 0) -> int:
        """Returns a scalar or selected per-axis zero-point."""
        if isinstance(self.zero_point, tuple):
            return self.zero_point[axis_index]
        return self.zero_point

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-safe plan representation."""
        return {
            "storage_type": self.storage_type,
            "scale": list(self.scale) if isinstance(self.scale, tuple) else self.scale,
            "zero_point": (
                list(self.zero_point)
                if isinstance(self.zero_point, tuple)
                else self.zero_point
            ),
            "quantized_axis": self.quantized_axis,
            "rounding_mode": self.rounding_mode,
            "clamp": [self.clamp_min, self.clamp_max],
        }
