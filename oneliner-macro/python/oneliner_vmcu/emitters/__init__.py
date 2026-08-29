"""Standard-MLIR emitters selected through the pattern registry."""

from .fully_connected import emit_fully_connected

__all__ = ["emit_fully_connected"]
