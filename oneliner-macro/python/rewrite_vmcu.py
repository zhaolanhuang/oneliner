#!/usr/bin/env python3
"""Entry point for the post-preprocessing vMCU graph rewriter."""

from __future__ import annotations

import sys
from pathlib import Path


# The package is shipped beside this script rather than installed separately.
# Prepending the directory also guarantees that Cargo uses this checked-in
# implementation instead of an unrelated globally installed module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from oneliner_vmcu.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
