import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PYTHON_DIR = ROOT / "oneliner-macro" / "python"
FIXTURE = ROOT / "tests" / "fixtures" / "vmcu_segment_primitives.mlir"

import sys

sys.path.insert(0, str(PYTHON_DIR))

from oneliner_vmcu.memory import (  # noqa: E402
    FixedSegmentMemoryPlan,
    SegmentSpec,
    assert_non_overlapping_live_buffers,
)
from oneliner_vmcu.schedules import SingleLayerSegmentSchedule  # noqa: E402


class SegmentMemoryTests(unittest.TestCase):
    """Static sizing, mask, modulo, and stock-IREE validation."""

    def test_segment_bytes_include_dtype_and_alignment(self):
        """Workspace accounting never assumes one byte or ignores alignment."""
        spec = SegmentSpec("acc", 3, 5, "i32", alignment=16)
        self.assertEqual(spec.segment_bytes, 32)
        self.assertEqual(spec.total_bytes, 96)

    def test_partial_lane_mask_is_a_prefix(self):
        """The final segment contains only the remaining logical elements."""
        plan = FixedSegmentMemoryPlan(4, 3, ())
        self.assertEqual([plan.valid_lanes(i, 10) for i in range(4)], [4, 4, 2, 0])

    def test_circular_address_is_deterministic(self):
        """Logical indices wrap to a stable physical byte offset."""
        plan = FixedSegmentMemoryPlan(4, 3, ())
        addresses = [plan.circular_address(i, 16, 1) for i in range(7)]
        self.assertEqual([item.physical_index for item in addresses], [1, 2, 0, 1, 2, 0, 1])
        self.assertEqual([item.byte_offset for item in addresses], [16, 32, 0, 16, 32, 0, 16])

    def test_single_layer_schedule_has_one_fixed_state(self):
        """The schedule reports one loop-carried segment and no alternatives."""
        schedule = SingleLayerSegmentSchedule(10, 7, 4, 3)
        report = schedule.to_dict()
        self.assertEqual(report["input_segments"], 3)
        self.assertEqual(report["output_segments"], 2)
        self.assertEqual(report["last_input_valid_lanes"], 2)
        self.assertEqual(report["last_output_valid_lanes"], 3)
        self.assertEqual(len(report["memory"]["buffers"]), 1)

    def test_live_duplicate_buffer_is_rejected(self):
        """Two same-name live intervals cannot claim one implicit buffer."""
        specs = (
            SegmentSpec("B", 1, 8, "i8", lifetime_start=0, lifetime_end=3),
            SegmentSpec("B", 1, 8, "i8", lifetime_start=2, lifetime_end=4),
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_non_overlapping_live_buffers(specs)

    @unittest.skipUnless(shutil.which("iree-compile"), "iree-compile is unavailable")
    def test_segment_fixture_compiles_for_host_and_cortex_m(self):
        """Only stock dialects are required for host and thumbv7em objects."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            targets = (
                ("host", "x86_64-unknown-linux-gnu", "generic"),
                ("cortex_m4", "thumbv7em-none-eabi", "cortex-m4"),
            )
            for name, triple, cpu in targets:
                vmfb = directory / f"{name}.vmfb"
                obj = directory / f"{name}.o"
                dumps = directory / f"{name}-dumps"
                dumps.mkdir()
                completed = subprocess.run(
                    [
                        "iree-compile",
                        str(FIXTURE),
                        "--iree-hal-target-device=local",
                        "--iree-hal-local-target-device-backends=llvm-cpu",
                        f"--iree-llvmcpu-target-triple={triple}",
                        f"--iree-llvmcpu-target-cpu={cpu}",
                        "--iree-llvmcpu-link-embedded=false",
                        "--iree-llvmcpu-link-static",
                        f"--iree-llvmcpu-static-library-output-path={obj}",
                        "--iree-llvmcpu-stack-allocation-limit=16384",
                        "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation",
                        f"--dump-compilation-phases-to={dumps}",
                        "-o",
                        str(vmfb),
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertGreater(vmfb.stat().st_size, 0)
                self.assertGreater(obj.stat().st_size, 0)
                phase_ten = next(dumps.glob("*.10.executable-targets.mlir"))
                lowered = phase_ten.read_text(encoding="utf-8")
                self.assertNotIn("memref.alloc(", lowered)
                self.assertNotIn("tensor<?", lowered)


if __name__ == "__main__":
    unittest.main()
