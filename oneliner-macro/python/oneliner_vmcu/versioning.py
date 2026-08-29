"""IREE Python-package and command-line version consistency diagnostics."""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_VERSION = re.compile(r"compiler version\s+([^\s]+)")


@dataclass(frozen=True)
class CompilerVersionDiagnostics:
    """Versions used to parse MLIR and resume the compiler pipeline."""

    python_package: str
    executable: str | None
    executable_version: str | None
    compatible: bool | None
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns a stable JSON representation for ``vmcu.plan.json``."""
        return {
            "python_package": self.python_package,
            "executable": self.executable,
            "executable_version": self.executable_version,
            "compatible": self.compatible,
            "diagnostic": self.diagnostic,
        }


def diagnose_compiler_versions(
    executable: str | Path | None,
) -> CompilerVersionDiagnostics:
    """Compares the installed binding wheel with one ``iree-compile`` binary.

    Failure to execute the optional command is reported as data.  The caller
    decides whether a missing diagnostic is fatal; an observed mismatch is
    always explicit and can therefore never be silently mistaken for a valid
    split-pipeline configuration.
    """
    package_version = importlib.metadata.version("iree-base-compiler")
    if executable is None:
        return CompilerVersionDiagnostics(package_version, None, None, None)
    executable_text = str(executable)
    try:
        completed = subprocess.run(
            [executable_text, "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return CompilerVersionDiagnostics(
            package_version,
            executable_text,
            None,
            None,
            f"failed to query iree-compile: {error}",
        )
    combined = f"{completed.stdout}\n{completed.stderr}"
    match = _VERSION.search(combined)
    if completed.returncode != 0 or match is None:
        return CompilerVersionDiagnostics(
            package_version,
            executable_text,
            None,
            None,
            "iree-compile did not report a parseable compiler version",
        )
    executable_version = match.group(1)
    compatible = executable_version == package_version
    diagnostic = None if compatible else (
        "IREE Python bindings and iree-compile must use the same version"
    )
    return CompilerVersionDiagnostics(
        package_version,
        executable_text,
        executable_version,
        compatible,
        diagnostic,
    )
