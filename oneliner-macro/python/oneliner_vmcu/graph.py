"""SSA graph construction independent of textual operation adjacency."""

from __future__ import annotations

from dataclasses import dataclass, field

from iree.compiler import ir

from .ir_utils import operation, owner_operation
from .model import OpKey


@dataclass
class GraphNode:
    """One direct operation plus its function-local SSA graph edges."""

    key: OpKey
    operation: ir.Operation
    producers: set[OpKey] = field(default_factory=set)
    consumers: set[OpKey] = field(default_factory=set)


@dataclass
class FunctionGraph:
    """Direct-operation graph for a supported single-block MLIR function."""

    name: str
    nodes: list[GraphNode]
    by_operation: dict[ir.Operation, GraphNode]


def _function_name(function: ir.Operation, fallback: int) -> str:
    """Returns a stable function label even for operations without sym_name."""
    for attribute_name in ("sym_name", "function_ref"):
        if attribute_name in function.attributes:
            return str(function.attributes[attribute_name]).strip('"')
    return f"function_{fallback}"


def _direct_operations(function: ir.Operation) -> list[ir.Operation]:
    """Returns operations eligible for first-generation straight-line matching."""
    if len(function.regions) != 1 or len(function.regions[0].blocks) != 1:
        # Control-flow functions are outside the first pattern set. Treat them
        # as unmatched instead of making auto mode reject an otherwise valid
        # module.
        return []
    return [operation(item) for item in function.regions[0].blocks[0].operations]


def build_graphs(module: ir.Module) -> list[FunctionGraph]:
    """Builds producer/consumer edges from SSA operands for every function.

    Only direct operations become nodes. Nested scalar operations inside a
    linalg region remain part of their parent operation's semantic validation,
    rather than being mistaken for graph-level operators.
    """
    functions: list[ir.Operation] = []

    def collect(candidate: ir.Operation) -> ir.WalkResult:
        """Collects function roots while preventing a redundant nested walk."""
        if candidate.name in ("util.func", "func.func"):
            functions.append(candidate)
            return ir.WalkResult.SKIP
        return ir.WalkResult.ADVANCE

    module.operation.walk(collect)
    graphs: list[FunctionGraph] = []
    for function_index, function in enumerate(functions):
        name = _function_name(function, function_index)
        direct = _direct_operations(function)
        nodes = [
            GraphNode(OpKey(name, index, candidate.name), candidate)
            for index, candidate in enumerate(direct)
        ]
        by_operation = {node.operation: node for node in nodes}
        # Resolve every operand's defining operation. This is what makes the
        # matcher independent of textual adjacency and unrelated inserted ops.
        for node in nodes:
            for operand in node.operation.operands:
                producer = owner_operation(operand)
                if producer in by_operation:
                    producer_node = by_operation[producer]
                    node.producers.add(producer_node.key)
                    producer_node.consumers.add(node.key)
        graphs.append(FunctionGraph(name, nodes, by_operation))
    return graphs
