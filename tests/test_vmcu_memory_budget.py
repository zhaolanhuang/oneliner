import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PYTHON_DIR = ROOT / "oneliner-macro" / "python"
FIXTURE = ROOT / "tests" / "fixtures" / "vmcu_ibn_11seg.mlir"
REWRITER = PYTHON_DIR / "oneliner_vmcu" / "cli.py"
REPORTER = PYTHON_DIR / "oneliner_vmcu" / "resource_cli.py"
sys.path.insert(0, str(PYTHON_DIR))

from oneliner_vmcu import rewrite_text  # noqa: E402
from oneliner_vmcu.memory import (  # noqa: E402
    CircularMemoryPlan,
    SegmentLifetime,
    plan_circular_memory,
)
from oneliner_iree.ram_usage_analysis import (  # noqa: E402
    parse_llvm_static_allocas,
    parse_stream_arena,
)


class CircularMemoryAndBudgetTests(unittest.TestCase):
    """Last-use safety, deterministic offsets, budgets, and lowering evidence."""

    def setUp(self):
        """Loads the fixed-schedule residual IBN source."""
        self.source = FIXTURE.read_text(encoding="utf-8")

    def test_smallest_safe_output_offset_is_deterministic(self):
        """Linear scanning selects b_out=1 and never overwrites a live input."""
        inputs = (SegmentLifetime(0, 5), SegmentLifetime(1, 1))
        outputs = (SegmentLifetime(0, 2), SegmentLifetime(1, 6))
        plans = [plan_circular_memory(inputs, outputs, 2, 16) for _ in range(20)]
        self.assertTrue(all(plan.b_in == 0 and plan.b_out == 1 for plan in plans))
        self.assertTrue(all(plan.to_dict() == plans[0].to_dict() for plan in plans))

    def test_unsafe_manual_plan_and_impossible_pool_are_rejected(self):
        """A write at or before last-use is a hard read-after-write violation."""
        inputs = (SegmentLifetime(0, 5), SegmentLifetime(1, 5))
        outputs = (SegmentLifetime(0, 2), SegmentLifetime(1, 2))
        with self.assertRaisesRegex(ValueError, "before its last read"):
            CircularMemoryPlan(2, 4, 0, 0, inputs, outputs)
        with self.assertRaisesRegex(ValueError, "no safe circular output offset"):
            plan_circular_memory(inputs, outputs, 2, 4)

    def test_workspace_budget_falls_back_below_fixed_schedule_size(self):
        """A 47-byte cap cannot silently select the fixed 48-byte workspace."""
        result = rewrite_text(self.source, sram_budget=47)
        self.assertEqual(result.text, self.source)
        self.assertFalse(result.plan["applied"])
        self.assertIn("required=48 budget=47", result.plan["rejected"][0]["reason"])
        accepted = rewrite_text(self.source, sram_budget=48)
        self.assertEqual(accepted.plan["resources"]["workspace_bytes"], 48)

    def test_resource_parsers_measure_arena_and_alloca_stack(self):
        """Synthetic lowering snippets cover all non-IREE parser branches."""
        stream = """
          %c96 = arith.constant 96 : index
          %c32 = arith.constant 32 : index
          %a = stream.resource.alloca : !stream.resource<transient>{%c96}
          %b = stream.resource.alloca : !stream.resource<transient>{%c32}
        """
        self.assertEqual(parse_stream_arena(stream), (128, [96, 32]))
        executable = """
          llvm.func @small() {
            %c18 = llvm.mlir.constant(18 : index) : i64
            %c2 = llvm.mlir.constant(2 : index) : i64
            %0 = llvm.alloca %c18 x i8 {alignment = 4 : i64} : (i64) -> !llvm.ptr
            %1 = llvm.alloca %c2 x i32 {alignment = 8 : i64} : (i64) -> !llvm.ptr
          }
          llvm.func @large() {
            %c18 = llvm.mlir.constant(96 : index) : i32
            %c2 = llvm.mlir.constant(4704 : index) : i32
            %c160 = llvm.mlir.constant(160 : index) : i32
            %0 = llvm.alloca %c18 x i8 {alignment = 64 : i64} : (i32) -> !llvm.ptr
            %1 = llvm.alloca %c2 x i8 {alignment = 64 : i64} : (i32) -> !llvm.ptr
            %2 = llvm.alloca %c160 x i32 {alignment = 64 : i64} : (i32) -> !llvm.ptr
          }
        """
        maximum, functions = parse_llvm_static_allocas(executable)
        self.assertEqual(functions, {"small": 32, "large": 5504})
        self.assertEqual(maximum, 5504)
    @unittest.skipUnless(
        shutil.which("iree-compile"),
        "iree-compile is required",
    )
    def test_post_lowering_report_updates_total_and_enforces_budget(self):
        """Executable stack, Stream arena, and workspace form one SRAM total."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pre = directory / "pre.mlir"
            rewritten = directory / "rewritten.mlir"
            plan = directory / "plan.json"
            vmfb = directory / "model.vmfb"
            object_file = directory / "model.o"
            dumps = directory / "dumps"
            dumps.mkdir()
            common = [
                "--iree-hal-target-device=local",
                "--iree-hal-local-target-device-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=generic",
                "--iree-llvmcpu-link-embedded=false",
                "--iree-llvmcpu-link-static",
                f"--iree-llvmcpu-static-library-output-path={object_file}",
            ]
            subprocess.run(
                [
                    "iree-compile",
                    str(FIXTURE),
                    "--compile-to=preprocessing",
                    "--emit-mlir-bytecode=false",
                    *common,
                    "-o",
                    str(pre),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(REWRITER),
                    str(pre),
                    "-o",
                    str(rewritten),
                    "--plan-output",
                    str(plan),
                    "--sram-budget",
                    "4096",
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "iree-compile",
                    str(rewritten),
                    "--compile-from=preprocessing",
                    *common,
                    f"--dump-compilation-phases-to={dumps}",
                    "-o",
                    str(vmfb),
                ],
                check=True,
                capture_output=True,
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPORTER),
                    "--plan",
                    str(plan),
                    "--stream",
                    str(next(dumps.glob("*.7.stream.mlir"))),
                    "--executable",
                    str(next(dumps.glob("*.10.executable-targets.mlir"))),
                    "--object",
                    str(object_file),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(plan.read_text(encoding="utf-8"))
            resources = report["resources"]
            self.assertEqual(resources["workspace_bytes"], 48)
            self.assertGreater(resources["stack_bytes"], 0)
            self.assertEqual(
                resources["total_sram_bytes"],
                resources["io_pool_allocated_bytes"]
                + resources["arena_bytes"]
                + resources["stack_bytes"]
                + resources["workspace_additional_sram_bytes"],
            )
            self.assertEqual(resources["workspace_residency"], "stack-included")
            self.assertEqual(resources["status"], "within-budget")
            # Re-run the same authoritative evidence with a deliberately small
            # cap to verify the nonzero process status consumed by Rust.
            report["resources"]["vmcu_sram_budget"] = 500
            plan.write_text(json.dumps(report), encoding="utf-8")
            exceeded = subprocess.run(
                [
                    sys.executable,
                    str(REPORTER),
                    "--plan",
                    str(plan),
                    "--stream",
                    str(next(dumps.glob("*.7.stream.mlir"))),
                    "--executable",
                    str(next(dumps.glob("*.10.executable-targets.mlir"))),
                    "--object",
                    str(object_file),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(exceeded.returncode, 3, exceeded.stderr)
            self.assertEqual(
                json.loads(plan.read_text(encoding="utf-8"))["resources"]["status"],
                "exceeds-budget",
            )

    @unittest.skipUnless(shutil.which("iree-compile"), "iree-compile is required")
    def test_fixed_ibn_compiles_to_a_cortex_m4_object(self):
        """The native dispatch and static buffers survive thumbv7em lowering."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pre = directory / "pre.mlir"
            rewritten = directory / "rewrite.mlir"
            plan = directory / "plan.json"
            object_file = directory / "model.o"
            vmfb = directory / "model.vmfb"
            dumps = directory / "dumps"
            dumps.mkdir()
            common = [
                "--iree-hal-target-device=local",
                "--iree-hal-local-target-device-backends=llvm-cpu",
                "--iree-llvmcpu-target-triple=thumbv7em-none-eabi",
                "--iree-llvmcpu-target-cpu=cortex-m4",
                "--iree-llvmcpu-stack-allocation-limit=65536",
                "--iree-llvmcpu-link-embedded=false",
                "--iree-llvmcpu-link-static",
                f"--iree-llvmcpu-static-library-output-path={object_file}",
            ]
            subprocess.run(
                [
                    "iree-compile",
                    str(FIXTURE),
                    "--compile-to=preprocessing",
                    "--emit-mlir-bytecode=false",
                    *common,
                    "-o",
                    str(pre),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(REWRITER),
                    str(pre),
                    "-o",
                    str(rewritten),
                    "--plan-output",
                    str(plan),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "iree-compile",
                    str(rewritten),
                    "--compile-from=preprocessing",
                    *common,
                    f"--dump-compilation-phases-to={dumps}",
                    "-o",
                    str(vmfb),
                ],
                check=True,
                capture_output=True,
            )
            self.assertGreater(object_file.stat().st_size, 0)
            lowered = next(dumps.glob("*.10.executable-targets.mlir")).read_text(
                encoding="utf-8"
            )
            self.assertIn("llvm.func @semantic_ibn_dispatch_0", lowered)
            self.assertNotIn("memref.alloc", lowered)
            self.assertNotIn("tensor<?", lowered)


if __name__ == "__main__":
    unittest.main()
