import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PYTHON_DIR = ROOT / "oneliner-macro" / "python"
FIXTURE = ROOT / "tests" / "fixtures" / "vmcu_ibn_11seg.mlir"
REAL_PREPROCESSING = (
    ROOT
    / "examples"
    / "ariel-os-minimal"
    / "target"
    / "oneliner"
    / "model_iree_mcunet_10fps_vww"
    / "vmcu.preprocessing.mlir"
)
SCRIPT = PYTHON_DIR / "oneliner_vmcu" / "cli.py"
sys.path.insert(0, str(PYTHON_DIR))

from oneliner_vmcu import rewrite_text  # noqa: E402
from oneliner_vmcu.reference import (  # noqa: E402
    ReferenceQuantization,
    emitted_style_inverted_bottleneck_11seg,
    tensor_style_inverted_bottleneck,
)
from oneliner_vmcu.schedules import InvertedBottleneck11SegmentSchedule  # noqa: E402


def _remove_residual(source: str) -> str:
    """Builds a no-residual fixture without maintaining a duplicate MLIR file."""
    marker = source.index("    // Same-quantization residual")
    return source[:marker] + (
        "    util.return %proj_output : tensor<1x4x4x2xi8>\n"
        "  }\n"
        "}\n"
    )


def _make_stride_two(source: str) -> str:
    """Changes only the depthwise boundary and its downstream result shapes."""
    source = _remove_residual(source)
    source = source.replace(
        "-> tensor<1x4x4x2xi8> {", "-> tensor<1x2x2x2xi8> {", 1
    )
    split = source.index("    // Depthwise boundary")
    prefix, tail = source[:split], source[split:]
    tail = tail.replace(
        "strides = dense<1> : tensor<2xi64>",
        "strides = dense<2> : tensor<2xi64>",
        1,
    )
    for old, new in (
        ("tensor<1x4x4x4x1xi32>", "tensor<1x2x2x4x1xi32>"),
        ("tensor<1x4x4x4xi32>", "tensor<1x2x2x4xi32>"),
        ("tensor<1x4x4x4xi8>", "tensor<1x2x2x4xi8>"),
        ("tensor<1x4x4x2xi32>", "tensor<1x2x2x2xi32>"),
        ("tensor<1x4x4x2xi8>", "tensor<1x2x2x2xi8>"),
    ):
        tail = tail.replace(old, new)
    return prefix + tail


def _make_unequal_channels(source: str) -> str:
    """Builds a no-residual IBN with two input and three output channels."""
    source = _remove_residual(source)
    source = source.replace(
        ") -> tensor<1x4x4x2xi8> {", ") -> tensor<1x4x4x3xi8> {", 1
    )
    source = source.replace(
        "util.return %proj_output : tensor<1x4x4x2xi8>",
        "util.return %proj_output : tensor<1x4x4x3xi8>",
        1,
    )
    marker = source.index("    // Projection boundary")
    prefix, tail = source[:marker], source[marker:]
    for old, new in (
        ("tensor<1x1x4x2xi8>", "tensor<1x1x4x3xi8>"),
        ("tensor<2xi32>", "tensor<3xi32>"),
        ("tensor<2xi8>", "tensor<3xi8>"),
        ("tensor<1x4x4x2xi32>", "tensor<1x4x4x3xi32>"),
        ("tensor<1x4x4x2xi8>", "tensor<1x4x4x3xi8>"),
        ("dense<[29, -31]>", "dense<[29, -31, 17]>"),
        (
            "dense<[1073741824, 1073741824]>" ,
            "dense<[1073741824, 1073741824, 1073741824]>",
        ),
        ("dense<[31, 31]>", "dense<[31, 31, 31]>"),
    ):
        tail = tail.replace(old, new)
    return prefix + tail


class FixedInvertedBottleneckTests(unittest.TestCase):
    """Fixed-schedule matching, affine correctness, and stock-IREE coverage."""

    def setUp(self):
        """Loads the model-independent residual IBN fixture."""
        self.source = FIXTURE.read_text(encoding="utf-8")

    def test_schedule_is_exactly_nine_b_one_c_one_d(self):
        """Logical segment identity and byte accounting are both explicit."""
        schedule = InvertedBottleneck11SegmentSchedule(2, 4, 2)
        report = schedule.to_dict()
        self.assertEqual(report["workspace_segments"], 11)
        self.assertEqual(
            [
                (item["name"], item["count"], item["storage_type"])
                for item in report["buffers"]
            ],
            [("B", 9, "i8"), ("C", 1, "i8"), ("D", 1, "i32")],
        )
        self.assertEqual(report["workspace_bytes"], 48)
        self.assertFalse(report["schedule_search"])
        self.assertFalse(report["recomputation_fallback"])

    def test_residual_ibn_rewrites_as_one_composite_transaction(self):
        """The three operators and add become the sole eleven-segment pattern."""
        result = rewrite_text(self.source)
        self.assertEqual(result.plan["totals"]["accepted"], 1)
        self.assertEqual(result.plan["totals"]["rejected"], 0)
        accepted = result.plan["accepted"][0]
        self.assertEqual(accepted["kind"], "inverted_bottleneck_k2_plus_2_segment")
        self.assertEqual(accepted["schedule"]["workspace_segments"], 11)
        self.assertNotIn("linalg.conv_2d_nhwc_hwcf_q", result.text)
        self.assertNotIn("linalg.depthwise_conv_2d_nhwc_hwcm_q", result.text)
        self.assertIn("tensor<9x2xi8>", result.text)
        self.assertIn("tensor<2xi8>", result.text)
        self.assertIn("tensor<2xi32>", result.text)
        # Full spatial B/C/D layer outputs must not survive the transaction.
        self.assertNotIn("tensor<1x4x4x4xi8>", result.text)
        self.assertNotIn("tensor<1x4x4x4xi32>", result.text)
        self.assertNotIn("tensor<1x4x4x2xi32>", result.text)

    def test_no_residual_stride_two_boundary_padding_is_supported(self):
        """The same unique schedule handles downsampling without a skip edge."""
        source = _make_stride_two(self.source)
        result = rewrite_text(source)
        self.assertEqual(result.plan["totals"]["accepted"], 1)
        accepted = result.plan["accepted"][0]
        self.assertFalse(accepted["residual"])
        self.assertEqual(accepted["layers"]["depthwise"]["strides"], [2, 2])
        self.assertEqual(accepted["schedule"]["workspace_segments"], 11)
        self.assertIn("tensor<1x2x2x2xi8>", result.text)

    def test_unequal_module_channels_use_cout_for_the_d_segment(self):
        """Cin != Cout keeps B/C lanes at min(Cin, Cout) and sizes D by Cout."""
        source = _make_unequal_channels(self.source)
        result = rewrite_text(source)
        self.assertEqual(result.plan["totals"]["accepted"], 1)
        self.assertEqual(result.plan["totals"]["rejected"], 0)
        accepted = result.plan["accepted"][0]
        self.assertEqual(accepted["schedule"]["segment_lanes"], 2)
        buffers = accepted["schedule"]["buffers"]
        self.assertEqual([(item["name"], item["lanes"]) for item in buffers], [
            ("B", 2),
            ("C", 2),
            ("D", 3),
        ])
        self.assertEqual(accepted["schedule"]["workspace_bytes"], 52)
        self.assertIn("tensor<1x4x4x3xi8>", result.text)

    def test_model_function_and_ssa_names_do_not_affect_matching(self):
        """Only SSA dataflow and proven scalar semantics identify an IBN."""
        renamed = self.source.replace("@semantic_ibn", "@unrelated_network_block")
        renamed = renamed.replace("%input", "%arbitrary_activation")
        renamed = renamed.replace("%exp_", "%stage_alpha_")
        baseline = rewrite_text(self.source)
        result = rewrite_text(renamed)
        self.assertEqual(result.plan["totals"], baseline.plan["totals"])
        self.assertEqual(
            result.plan["accepted"][0]["schedule"],
            baseline.plan["accepted"][0]["schedule"],
        )

    def test_nonzero_affine_boundaries_are_bit_exact(self):
        """Residual stride-1 and no-residual stride-2 agree on 100 random cases."""
        generator = random.Random(0x11B0FF)
        for residual, stride in ((True, (1, 1)), (False, (2, 2))):
            for _ in range(50):
                final_zp = generator.randint(-19, 19)
                input_zp = final_zp if residual else generator.randint(-19, 19)
                expansion_output_zp = generator.randint(-19, 19)
                depthwise_output_zp = generator.randint(-19, 19)
                image = [
                    [
                        [generator.randint(-128, 127) for _ in range(2)]
                        for _ in range(4)
                    ]
                    for _ in range(4)
                ]
                expansion_weights = [
                    [
                        [
                            [generator.randint(-8, 8) for _ in range(4)]
                            for _ in range(2)
                        ]
                    ]
                ]
                depthwise_weights = [
                    [
                        [generator.randint(-8, 8) for _ in range(4)]
                        for _ in range(3)
                    ]
                    for _ in range(3)
                ]
                projection_weights = [
                    [
                        [
                            [generator.randint(-8, 8) for _ in range(2)]
                            for _ in range(4)
                        ]
                    ]
                ]
                expansion_q = ReferenceQuantization(
                    input_zp,
                    tuple(generator.randint(-5, 5) for _ in range(4)),
                    expansion_output_zp,
                    [1 << 30] * 4,
                    [31] * 4,
                )
                depthwise_q = ReferenceQuantization(
                    expansion_output_zp,
                    tuple(generator.randint(-5, 5) for _ in range(4)),
                    depthwise_output_zp,
                    [1 << 30] * 4,
                    [31] * 4,
                )
                projection_q = ReferenceQuantization(
                    depthwise_output_zp,
                    tuple(generator.randint(-5, 5) for _ in range(2)),
                    final_zp,
                    [1 << 30] * 2,
                    [31] * 2,
                )
                arguments = (
                    image,
                    expansion_weights,
                    [generator.randint(-100, 100) for _ in range(4)],
                    expansion_q,
                    depthwise_weights,
                    [generator.randint(-100, 100) for _ in range(4)],
                    depthwise_q,
                    projection_weights,
                    [generator.randint(-100, 100) for _ in range(2)],
                    projection_q,
                )
                expected = tensor_style_inverted_bottleneck(
                    *arguments, depthwise_stride=stride, residual=residual
                )
                actual, counters = emitted_style_inverted_bottleneck_11seg(
                    *arguments, depthwise_stride=stride, residual=residual
                )
                self.assertEqual(actual, expected)
                positions = len(expected) * len(expected[0])
                self.assertEqual(counters["b_slots"], positions * 9 * 4)
                self.assertEqual(counters["c_values"], positions * 4)
                self.assertEqual(counters["projection_products"], positions * 4 * 2)

    @unittest.skipUnless(shutil.which("iree-compile"), "iree-compile is unavailable")
    def test_split_pipeline_compiles_fixed_schedule_with_stock_iree(self):
        """Preprocessing pause, Python fusion, and resumed IREE all succeed."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pre = directory / "pre.mlir"
            rewritten = directory / "rewrite.mlir"
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
                    str(SCRIPT),
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
            report = json.loads(plan.read_text(encoding="utf-8"))
            self.assertEqual(report["accepted"][0]["schedule"]["workspace_segments"], 11)
            self.assertGreater(vmfb.stat().st_size, 0)
            self.assertGreater(object_file.stat().st_size, 0)
            dispatch_dump = next(dumps.glob("*.5.dispatch-creation.mlir")).read_text(
                encoding="utf-8"
            )
            self.assertEqual(dispatch_dump.count("flow.dispatch.workgroups"), 1)


if __name__ == "__main__":
    unittest.main()
