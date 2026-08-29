"""Common protocol implemented by model-independent pattern analyzers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from iree.compiler import ir

from ..graph import FunctionGraph
from ..model import Analysis, PatternMatch


PatternAnalyzer: TypeAlias = Callable[
    [list[FunctionGraph], set[ir.Operation]], Analysis
]
PatternEmitter: TypeAlias = Callable[[PatternMatch], None]
