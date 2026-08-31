#!/usr/bin/env python3
"""Imports a torch.export ExportedProgram through IREE Turbine."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a PyTorch .pt2 ExportedProgram into IREE MLIR"
    )
    parser.add_argument("input", type=Path, help="input .pt2 file")
    parser.add_argument("--output", required=True, type=Path, help="output MLIR file")
    parser.add_argument(
        "--module-name", required=True, help="sanitized IREE module name"
    )
    return parser.parse_args()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def load_toolchain() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch support requires the 'torch' host package; install a compatible "
            "torch and iree-turbine pair"
        ) from error

    try:
        import iree.turbine.aot as aot
    except ImportError as error:
        raise RuntimeError(
            "PyTorch support requires the 'iree-turbine' host package; install it "
            "in the active Python environment"
        ) from error

    return torch, aot


def validate_exported_program(program: Any, torch: Any) -> None:
    exported_program_type = getattr(torch.export, "ExportedProgram", None)
    if exported_program_type is None or not isinstance(program, exported_program_type):
        raise TypeError(
            "the .pt2 file did not contain a torch.export.ExportedProgram"
        )

    signature = program.graph_signature
    user_inputs = tuple(
        spec
        for spec in signature.input_specs
        if getattr(spec.kind, "name", str(spec.kind).rsplit(".", 1)[-1]) == "USER_INPUT"
    )
    user_outputs = tuple(
        spec
        for spec in signature.output_specs
        if getattr(spec.kind, "name", str(spec.kind).rsplit(".", 1)[-1]) == "USER_OUTPUT"
    )
    if len(user_inputs) != 1:
        raise ValueError(
            "OneLiner ModelInference requires exactly one PyTorch user input, "
            f"but the exported program declares {len(user_inputs)}"
        )
    if len(user_outputs) != 1:
        raise ValueError(
            "OneLiner ModelInference requires exactly one PyTorch user output, "
            f"but the exported program declares {len(user_outputs)}"
        )

    for label, spec in (("input", user_inputs[0]), ("output", user_outputs[0])):
        argument = spec.arg
        if type(argument).__name__ != "TensorArgument":
            raise TypeError(
                f"OneLiner requires a tensor {label}, but the exported program uses "
                f"{type(argument).__name__}"
            )

    range_constraints = getattr(program, "range_constraints", {})
    if range_constraints:
        raise ValueError(
            "OneLiner currently requires fixed tensor shapes; the exported program "
            "contains dynamic shape constraints"
        )


def import_model(input_path: Path, output_path: Path, module_name: str) -> None:
    if input_path.suffix.lower() != ".pt2":
        raise ValueError(f"expected a .pt2 PyTorch model, got: {input_path}")

    torch, aot = load_toolchain()
    try:
        program = torch.export.load(input_path)
        validate_exported_program(program, torch)
        exported = aot.export(
            program,
            module_name=module_name,
            function_name="main",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exported.save_mlir(output_path)
    except Exception as error:
        versions = (
            f"torch={getattr(torch, '__version__', 'unknown')}, "
            f"iree-turbine={package_version('iree-turbine')}, "
            f"iree-base-compiler={package_version('iree-base-compiler')}"
        )
        raise RuntimeError(f"{error} ({versions})") from error

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"PyTorch importer did not produce MLIR at {output_path}")


def main() -> None:
    args = parse_args()
    import_model(args.input.resolve(), args.output.resolve(), args.module_name)


if __name__ == "__main__":
    main()
