"""Command-line interface used by the Rust macro build pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Cargo executes this checked-in file directly, so make its package parent
# importable without requiring oneliner_vmcu to be installed site-wide.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oneliner_vmcu.rewrite import RewriteError, rewrite_text  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Defines the stable command line invoked by the Rust build pipeline."""
    parser = argparse.ArgumentParser(
        description="Match and rewrite vMCU patterns after IREE preprocessing"
    )
    parser.add_argument("input", type=Path, help="preprocessing-phase textual MLIR")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "strict"), default="auto")
    parser.add_argument(
        "--schedule-search",
        choices=("bounded", "optimal", "greedy"),
        default="bounded",
        help="compact activation-pool topology/base search policy",
    )
    parser.add_argument(
        "--search-state-limit",
        type=int,
        default=1_000_000,
        help="maximum explored states in bounded scheduling mode",
    )
    parser.add_argument(
        "--sram-budget",
        type=int,
        help="maximum deployable SRAM bytes; fixed workspace is checked before rewrite",
    )
    parser.add_argument(
        "--iree-compile",
        help="iree-compile executable whose version must match the Python package",
    )
    return parser


def run(arguments: list[str] | None = None) -> int:
    """Executes one rewrite and returns a process-style status code.

    Output files are written only after ``rewrite_text`` has completed all
    verification steps. Exit code 2 keeps diagnostics compatible with argparse
    and causes the procedural macro's command wrapper to fail the Cargo build.
    """
    args = _parser().parse_args(arguments)
    try:
        source = args.input.read_text(encoding="utf-8")
        result = rewrite_text(
            source,
            args.mode,
            args.iree_compile,
            sram_budget=args.sram_budget,
            schedule_search=args.schedule_search,
            search_state_limit=args.search_state_limit,
        )
        args.output.write_text(result.text, encoding="utf-8")
        args.plan_output.write_text(
            json.dumps(result.plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, UnicodeError, RewriteError) as error:
        print(f"vMCU rewrite failed: {error}", file=sys.stderr)
        return 2
    totals = result.plan["totals"]
    print(
        "vMCU rewrite: "
        f"accepted={totals['accepted']} rejected={totals['rejected']} "
        f"eliminated_i32_accumulator_bytes="
        f"{totals['eliminated_i32_accumulator_bytes']}"
    )
    return 0


def main() -> None:
    """CLI entry point that maps ``run``'s status to the process exit code."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
