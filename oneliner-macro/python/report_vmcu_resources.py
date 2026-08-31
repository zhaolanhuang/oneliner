#!/usr/bin/env python3
"""Updates one vMCU plan with post-lowering resource evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from oneliner_vmcu.resource_plan import finalize_resource_plan  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Defines the build-time interface consumed by the Rust macro."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument(
        "--deployment", choices=("rewritten", "baseline-fallback"), default="rewritten"
    )
    return parser


def run(arguments: list[str] | None = None) -> int:
    """Updates the plan atomically and returns 3 when the budget is exceeded."""
    args = _parser().parse_args(arguments)
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        finalize_resource_plan(
            plan,
            args.stream.read_text(encoding="utf-8"),
            args.executable.read_text(encoding="utf-8"),
            args.object,
            deployment=args.deployment,
        )
        temporary = args.plan.with_suffix(args.plan.suffix + ".tmp")
        temporary.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(args.plan)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        print(f"vMCU resource analysis failed: {error}", file=sys.stderr)
        return 2
    return 3 if plan["resources"]["status"] == "exceeds-budget" else 0


def main() -> None:
    """Maps the resource status to the process exit code."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
