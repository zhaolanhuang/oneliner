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
FIXTURE = ROOT / "tests" / "fixtures" / "vmcu_conv_depthwise.mlir"
CHAINED_FIXTURE = ROOT / "tests" / "fixtures" / "vmcu_chained_conv.mlir"
SHARED_DEPTHWISE_FIXTURE = ROOT / "tests" / "fixtures" / "vmcu_shared_depthwise.mlir"
SCRIPT = PYTHON_DIR / "rewrite_vmcu.py"
sys.path.insert(0, str(PYTHON_DIR))

from oneliner_vmcu import rewrite_text  # noqa: E402
from oneliner_vmcu.model import Analysis  # noqa: E402
from oneliner_vmcu.reference import (  # noqa: E402
    emitted_style_conv2d,
    emitted_style_depthwise,
    pad_nhwc,
    tensor_style_conv2d,
    tensor_style_depthwise,
)
from oneliner_vmcu.registry import create_default_registry  # noqa: E402


class QuantizedConvolutionTests(unittest.TestCase):
    """Semantic matching, geometry, affine correctness, and IREE lowering."""

    def setUp(self):
        """Loads the two unrelated synthetic network functions."""
        self.source = FIXTURE.read_text(encoding="utf-8")

    def test_rewrites_conv_and_depthwise_without_full_accumulators(self):
        """Both operators become direct i8-producing scalar reductions."""
        result = rewrite_text(self.source, "strict")
        kinds = [item["kind"] for item in result.plan["accepted"]]
        self.assertEqual(kinds, ["quantized_conv2d", "quantized_depthwise_conv2d"])
        self.assertNotIn("linalg.conv_2d_nhwc_hwcf_q", result.text)
        self.assertNotIn("linalg.depthwise_conv_2d_nhwc_hwcm_q", result.text)
        self.assertNotIn("tensor<1x3x3x3xi32>", result.text)
        self.assertNotIn("tensor<1x4x4x3x1xi32>", result.text)
        self.assertNotIn("tensor<1x4x4x3xi32>", result.text)

    def test_model_and_function_names_do_not_affect_convolution_matching(self):
        """Synthetic network identity is irrelevant to semantic acceptance."""
        renamed = self.source.replace("@synthetic_conv", "@not_mcunet")
        renamed = renamed.replace("@unrelated_depthwise_network", "@any_model")
        result = rewrite_text(renamed, "strict")
        self.assertEqual(result.plan["totals"]["accepted"], 2)

    def test_complete_registry_is_reanalyzed_after_each_emitter(self):
        """Two rewrites trigger source, transactional, and per-emitter analyses."""
        registry = create_default_registry()
        analysis_calls = 0

        def count_analysis(graphs, occupied):
            nonlocal analysis_calls
            analysis_calls += 1
            return Analysis([], [])

        registry.register("analysis_counter", count_analysis, lambda match: None)
        result = rewrite_text(self.source, "strict", registry=registry)
        accepted = result.plan["totals"]["accepted"]
        self.assertEqual(accepted, 2)
        self.assertEqual(analysis_calls, accepted + 2)

    def test_chained_conv_cli_discards_stale_matches_between_emitters(self):
        """A consumer of a rewritten Conv2D result must not retain its old Value."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rewritten = directory / "rewritten.mlir"
            plan = directory / "plan.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(CHAINED_FIXTURE),
                    "-o",
                    str(rewritten),
                    "--plan-output",
                    str(plan),
                    "--mode",
                    "strict",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(plan.read_text(encoding="utf-8"))
            self.assertEqual(report["totals"]["accepted"], 2)
            self.assertEqual(
                [item["kind"] for item in report["accepted"]],
                ["quantized_conv2d", "quantized_conv2d"],
            )
            rewritten_text = rewritten.read_text(encoding="utf-8")
            self.assertNotIn("linalg.conv_2d_nhwc_hwcf_q", rewritten_text)

    def test_wrong_quantized_padding_preserves_the_rejected_conv(self):
        """Padding with another value rejects and preserves only that candidate."""
        modified = self.source.replace(
            "tensor.yield %input_zp_i8 : i8", "tensor.yield %weight_zp_i8 : i8", 1
        ).replace(
            "%weight_zp = arith.constant 2 : i32",
            "%weight_zp_i8 = arith.constant 2 : i8\n    %weight_zp = arith.constant 2 : i32",
            1,
        )
        result = rewrite_text(modified, "auto")
        self.assertEqual(result.plan["totals"]["accepted"], 1)
        self.assertIn("linalg.conv_2d_nhwc_hwcf_q", result.text)
        self.assertTrue(any("padding value" in item["reason"] for item in result.plan["rejected"]))

    def test_static_attribute_padding_accepts_asymmetric_boundaries(self):
        """Modern tensor.pad static_low/static_high forms are fully semantic."""
        modified = self.source.replace(
            "low[%c0, %c1, %c1, %c0] high[%c0, %c1, %c1, %c0]",
            "low[0, 1, 0, 0] high[0, 1, 2, 0]",
        )
        result = rewrite_text(modified, "strict")
        self.assertEqual(result.plan["totals"]["accepted"], 2)
        self.assertEqual(result.plan["totals"]["rejected"], 0)

    def test_dynamic_padding_extent_is_rejected(self):
        """A runtime padding operand cannot enter a fixed scalar schedule."""
        modified = self.source.replace(
            "%c1 = arith.constant 1 : index",
            "%c1 = arith.constant 1 : index\n    %dynamic = arith.addi %c1, %c0 : index",
            1,
        ).replace(
            "low[%c0, %c1, %c1, %c0]",
            "low[%c0, %dynamic, %c1, %c0]",
            1,
        )
        result = rewrite_text(modified, "auto")
        self.assertEqual(result.plan["totals"]["accepted"], 1)
        self.assertTrue(any("padding low index" in item["reason"] for item in result.plan["rejected"]))

    def test_shared_zero_fill_is_reused_until_the_last_depthwise_emit(self):
        """Two depthwise roots may safely share one pure zero initializer."""
        source = SHARED_DEPTHWISE_FIXTURE.read_text(encoding="utf-8")
        result = rewrite_text(source, "strict")
        self.assertEqual(result.plan["totals"]["accepted"], 2)
        self.assertEqual(result.plan["totals"]["rejected"], 0)
        self.assertNotIn("linalg.depthwise_conv_2d_nhwc_hwcm_q", result.text)
        self.assertNotIn("linalg.fill", result.text)

    def test_random_depthwise_kernels_are_bit_exact(self):
        """Standalone scalar reductions remain exact for 5x5 and 7x7 kernels."""
        generator = random.Random(0xD3E75)
        for kernel, stride, padding in (
            (5, (1, 1), (1, 2, 0, 4)),
            (7, (2, 2), (3, 4, 2, 5)),
        ):
            input_zp = generator.randint(-20, 20)
            image = [
                [[generator.randint(-128, 127) for _ in range(3)] for _ in range(10)]
                for _ in range(9)
            ]
            padded = pad_nhwc(image, padding, input_zp)
            weights = [
                [
                    [generator.randint(-8, 8) for _ in range(3)]
                    for _ in range(kernel)
                ]
                for _ in range(kernel)
            ]
            args = (padded, weights, [1, 2, 3], [1 << 30] * 3, [31] * 3)
            expected = tensor_style_depthwise(
                *args,
                input_zp,
                (-3, 0, 4),
                9,
                stride=stride,
            )
            actual = emitted_style_depthwise(
                *args,
                input_zp,
                (-3, 0, 4),
                9,
                stride=stride,
            )
            self.assertEqual(actual, expected)

    def test_standalone_depthwise_match_accepts_5x5_and_7x7(self):
        """The matcher delegates arbitrary static positive kernels to fallback."""
        tail_start = self.source.index("  util.func public @unrelated_depthwise_network")
        for kernel, pad in ((5, 2), (7, 3)):
            prefix, tail = self.source[:tail_start], self.source[tail_start:]
            tail = tail.replace(
                "%c1 = arith.constant 1 : index",
                "%c1 = arith.constant 1 : index\n    %cpad = arith.constant "
                f"{pad} : index",
                1,
            )
            tail = tail.replace(
                "low[%c0, %c1, %c1, %c0] high[%c0, %c1, %c1, %c0]",
                "low[%c0, %cpad, %cpad, %c0] high[%c0, %cpad, %cpad, %c0]",
                1,
            )
            tail = tail.replace("tensor<1x6x6x3xi8>", f"tensor<1x{4 + 2 * pad}x{4 + 2 * pad}x3xi8>")
            tail = tail.replace("tensor<3x3x3x1xi8>", f"tensor<{kernel}x{kernel}x3x1xi8>")
            result = rewrite_text(prefix + tail, "strict")
            self.assertEqual(result.plan["totals"]["accepted"], 2)
            self.assertEqual(result.plan["totals"]["rejected"], 0)
            self.assertNotIn("linalg.depthwise_conv_2d_nhwc_hwcm_q", result.text)

    def test_random_conv_geometries_are_bit_exact(self):
        """1x1/3x3, stride/dilation, and valid/same/explicit padding agree."""
        generator = random.Random(0xC0A2D)
        cases = (
            (1, (1, 1), (1, 1), (0, 0, 0, 0)),
            (1, (2, 2), (1, 1), (1, 0, 2, 0)),
            (3, (1, 1), (1, 1), (1, 1, 1, 1)),
            (3, (2, 2), (1, 1), (1, 1, 1, 1)),
            (3, (1, 1), (2, 2), (2, 2, 2, 2)),
        )
        for kernel, stride, dilation, padding in cases:
            for _ in range(20):
                input_zp = generator.randint(-20, 20)
                output_zp = generator.randint(-20, 20)
                image = [
                    [
                        [generator.randint(-128, 127) for _ in range(2)]
                        for _ in range(6)
                    ]
                    for _ in range(6)
                ]
                padded = pad_nhwc(image, padding, input_zp)
                weights = [
                    [
                        [
                            [generator.randint(-8, 8) for _ in range(3)]
                            for _ in range(2)
                        ]
                        for _ in range(kernel)
                    ]
                    for _ in range(kernel)
                ]
                args = (padded, weights, [3, -5, 7], [1 << 30] * 3, [31] * 3)
                weight_zp = (generator.randint(-4, 4),) * 3
                expected = tensor_style_conv2d(
                    *args, input_zp, weight_zp, output_zp, stride, dilation
                )
                actual = emitted_style_conv2d(
                    *args, input_zp, weight_zp, output_zp, stride, dilation
                )
                self.assertEqual(actual, expected)

    def test_random_depthwise_geometries_are_bit_exact(self):
        """Multiplier-one 3x3 depthwise keeps affine per-channel offsets."""
        generator = random.Random(0xD3E7)
        for stride, dilation, padding in (
            ((1, 1), (1, 1), (1, 1, 1, 1)),
            ((2, 2), (1, 1), (1, 1, 1, 1)),
            ((1, 1), (2, 2), (2, 2, 2, 2)),
        ):
            for _ in range(20):
                input_zp = generator.randint(-20, 20)
                image = [
                    [
                        [generator.randint(-128, 127) for _ in range(3)]
                        for _ in range(6)
                    ]
                    for _ in range(6)
                ]
                padded = pad_nhwc(image, padding, input_zp)
                weights = [
                    [
                        [generator.randint(-8, 8) for _ in range(3)]
                        for _ in range(3)
                    ]
                    for _ in range(3)
                ]
                args = (padded, weights, [1, 2, 3], [1 << 30] * 3, [31] * 3)
                weight_zp = (-3, 0, 4)
                expected = tensor_style_depthwise(
                    *args,
                    input_zp,
                    weight_zp,
                    9,
                    stride=stride,
                    dilation=dilation,
                )
                actual = emitted_style_depthwise(
                    *args,
                    input_zp,
                    weight_zp,
                    9,
                    stride=stride,
                    dilation=dilation,
                )
                self.assertEqual(actual, expected)

    @unittest.skipUnless(shutil.which("iree-compile"), "iree-compile is unavailable")
    def test_split_pipeline_compiles_rewritten_convolutions(self):
        """Preprocessing, Python rewrite, and resumed stock IREE all succeed."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pre = directory / "pre.mlir"
            rewritten = directory / "rewritten.mlir"
            plan = directory / "plan.json"
            vmfb = directory / "model.vmfb"
            subprocess.run(
                [
                    "iree-compile",
                    str(FIXTURE),
                    "--compile-to=preprocessing",
                    "--emit-mlir-bytecode=false",
                    "--iree-hal-target-device=local",
                    "--iree-hal-local-target-device-backends=llvm-cpu",
                    "--iree-llvmcpu-target-cpu=generic",
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
                    "--mode",
                    "strict",
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "iree-compile",
                    str(rewritten),
                    "--compile-from=preprocessing",
                    "--iree-hal-target-device=local",
                    "--iree-hal-local-target-device-backends=llvm-cpu",
                    "--iree-llvmcpu-target-cpu=generic",
                    "-o",
                    str(vmfb),
                ],
                check=True,
                capture_output=True,
            )
            report = json.loads(plan.read_text(encoding="utf-8"))
            self.assertEqual(report["totals"]["accepted"], 2)
            self.assertGreater(vmfb.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
