#!/usr/bin/env python3
"""Reports target-independent RAM usage from lowered IREE MLIR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oneliner_iree import analyze_ram_usage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    args = parser.parse_args()
    try:
        resources = analyze_ram_usage(
            args.stream.read_text(encoding="utf-8"),
            args.executable.read_text(encoding="utf-8"),
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"IREE resource analysis failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(resources.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
