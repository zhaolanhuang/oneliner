import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import random
from collections import Counter
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).parents[1]
# Resolve the checked-in package exactly as the Cargo entry-point script does.
PYTHON_DIR = ROOT / "oneliner-macro" / "python"
SCRIPT = PYTHON_DIR / "oneliner_vmcu" / "cli.py"
FIXTURE = ROOT / "tests" / "fixtures" / "vmcu_fc.mlir"
REAL_PREPROCESSING = (
    ROOT
    / "examples"
    / "ariel-os-minimal"
    / "target"
    / "oneliner"
    / "model_iree_mcunet_10fps_vww"
    / "vmcu.preprocessing.mlir"
)
REAL_TARGET_DIRECTORY = REAL_PREPROCESSING.parent
sys.path.insert(0, str(PYTHON_DIR))

from oneliner_vmcu import RewriteError, rewrite_text  # noqa: E402
from oneliner_vmcu import compact_analysis as compact_analysis_module  # noqa: E402
from oneliner_vmcu import compact_memory as compact_memory_module  # noqa: E402
from oneliner_vmcu import rewrite as rewrite_module  # noqa: E402
from oneliner_vmcu.reference import (  # noqa: E402
    emitted_style_fully_connected,
    tensor_style_fully_connected,
)
from oneliner_vmcu.versioning import diagnose_compiler_versions  # noqa: E402
from oneliner_vmcu.registry import PatternRegistry, create_default_registry  # noqa: E402


class VmcuRewriterTests(unittest.TestCase):
    """Safety, matching, CLI, and split-IREE integration coverage."""

    def setUp(self):
        """Loads a fresh canonical source string for mutation-isolation tests."""
        self.source = FIXTURE.read_text(encoding="utf-8")

    def test_rewrites_canonical_fc_without_an_i32_tensor_result(self):
        """A proven FC becomes a scalar reduction that directly yields int8."""
        result = rewrite_text(self.source, "strict")

        self.assertTrue(result.plan["applied"])
        self.assertEqual(result.plan["totals"]["accepted"], 1)
        self.assertEqual(
            result.plan["totals"]["eliminated_i32_accumulator_bytes"], 32
        )
        self.assertNotIn("linalg.quantized_matmul", result.text)
        self.assertIn("scf.for", result.text)
        self.assertIn("tosa.apply_scale", result.text)
        self.assertNotIn("tensor.generate", result.text)
        self.assertIn("arith.subi", result.text)
        self.assertEqual(
            result.plan["accepted"][0]["quantization"]["weight"]["zero_point"],
            2,
        )

    def test_nonzero_weight_zero_point_is_bit_exact_for_1000_inputs(self):
        """Tensor and emitted scalar schedules agree on randomized affine FC."""
        generator = random.Random(0x564D4355)
        weights = [
            [generator.randint(-128, 127) for _ in range(7)] for _ in range(5)
        ]
        bias = [generator.randint(-100_000, 100_000) for _ in range(5)]
        multiplier = [generator.randint(1 << 27, (1 << 31) - 1) for _ in range(5)]
        shift = [generator.randint(28, 42) for _ in range(5)]
        for _ in range(1000):
            inputs = [[generator.randint(-128, 127) for _ in range(7)]]
            input_zero_point = generator.randint(-127, 126)
            weight_zero_point = generator.randint(-127, 126)
            output_zero_point = generator.randint(-127, 126)
            expected = tensor_style_fully_connected(
                inputs,
                weights,
                bias,
                multiplier,
                shift,
                input_zero_point,
                weight_zero_point,
                output_zero_point,
            )
            actual = emitted_style_fully_connected(
                inputs,
                weights,
                bias,
                multiplier,
                shift,
                input_zero_point,
                weight_zero_point,
                output_zero_point,
            )
            self.assertEqual(actual, expected)

    def test_per_axis_reference_weight_zero_points_are_bit_exact(self):
        """The common affine model also represents per-output-channel offsets."""
        inputs = [[-128, -1, 0, 127], [4, 3, 2, 1]]
        weights = [[127, -128, 1, 0], [3, 4, 5, 6], [-7, -8, -9, -10]]
        arguments = (inputs, weights, [0, 17, -23], [1 << 30] * 3, [31] * 3)
        expected = tensor_style_fully_connected(
            *arguments, -3, (-5, 0, 11), 7
        )
        actual = emitted_style_fully_connected(*arguments, -3, (-5, 0, 11), 7)
        self.assertEqual(actual, expected)

    def test_iree_package_and_executable_versions_match(self):
        """The MLIR binding and split compiler must be the exact same build."""
        compiler = shutil.which("iree-compile")
        if compiler is None:
            self.skipTest("iree-compile is unavailable")
        diagnostics = diagnose_compiler_versions(compiler)
        self.assertTrue(diagnostics.compatible, diagnostics.diagnostic)
        result = rewrite_text(self.source, "strict", compiler)
        self.assertTrue(result.plan["iree_versions"]["compatible"])

    def test_matching_uses_ssa_edges_not_textual_adjacency(self):
        """An unrelated operation between producers and users cannot break a match."""
        separated = self.source.replace(
            "    %output_init = tensor.empty()",
            "    %unrelated = arith.constant 99 : i32\n"
            "    %output_init = tensor.empty()",
        )

        result = rewrite_text(separated, "strict")

        self.assertEqual(result.plan["totals"]["accepted"], 1)
        self.assertIn("%c99_i32", result.text)

    def test_unrelated_function_declarations_do_not_block_auto_analysis(self):
        """Unsupported declarations are ignored without hiding supported functions."""
        source = self.source.replace(
            "module {", "module {\n  func.func private @helper(i32) -> i32", 1
        )

        result = rewrite_text(source, "strict")

        self.assertEqual(result.plan["totals"]["accepted"], 1)

    def test_auto_mode_leaves_unmatched_input_byte_for_byte(self):
        """Auto mode preserves formatting and content when no root is accepted."""
        source = (ROOT / "examples" / "models" / "abs2.mlir").read_text(
            encoding="utf-8"
        )

        result = rewrite_text(source, "auto")

        self.assertEqual(result.text, source)
        self.assertFalse(result.plan["applied"])
        self.assertEqual(result.plan["totals"]["accepted"], 0)

    def test_strict_mode_rejects_an_unmatched_module(self):
        """Strict mode turns an empty accepted set into an actionable error."""
        source = (ROOT / "examples" / "models" / "abs2.mlir").read_text(
            encoding="utf-8"
        )

        with self.assertRaisesRegex(RewriteError, "no safe vMCU patterns"):
            rewrite_text(source, "strict")

    def test_modified_clamp_is_rejected_without_partial_mutation(self):
        """Changing scalar semantics records a rejection and preserves the source."""
        modified = self.source.replace("arith.maxsi", "arith.minsi", 1)

        result = rewrite_text(modified, "auto")

        self.assertEqual(result.text, modified)
        self.assertEqual(result.plan["totals"]["accepted"], 0)
        self.assertEqual(result.plan["totals"]["rejected"], 1)
        self.assertIn("unsupported scalar semantics", result.plan["rejected"][0]["reason"])

    def test_cli_writes_machine_readable_plan(self):
        """The standalone entry point emits verified MLIR and schema-versioned JSON."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rewritten.mlir"
            plan = Path(directory) / "plan.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(FIXTURE),
                    "-o",
                    str(output),
                    "--plan-output",
                    str(plan),
                    "--mode",
                    "strict",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            report = json.loads(plan.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], 4)
            self.assertEqual(report["accepted"][0]["kind"], "quantized_fully_connected")

    @unittest.skipUnless(
        REAL_PREPROCESSING.is_file() and shutil.which("iree-compile"),
        "the checked-in MCUNet preprocessing IR and iree-compile are required",
    )
    def test_real_mcunet_rewrite_isolated_from_repository_outputs(self):
        """The reported model rewrites in a subprocess and compiles from temp."""
        original_target_state = {
            path.name: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in REAL_TARGET_DIRECTORY.iterdir()
            if path.is_file()
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rewritten = directory / "vmcu.rewritten.mlir"
            plan = directory / "vmcu.plan.json"
            vmfb = directory / "model.vmfb"
            object_file = directory / "model.o"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(REAL_PREPROCESSING),
                    "-o",
                    str(rewritten),
                    "--plan-output",
                    str(plan),
                    "--mode",
                    "auto",
                    "--iree-compile",
                    "iree-compile",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(plan.read_text(encoding="utf-8"))
            accepted = report["accepted"]
            self.assertEqual(report["schema_version"], 4)
            self.assertEqual(report["totals"]["accepted"], 17)
            self.assertEqual(report["totals"]["rejected"], 0)
            self.assertEqual(
                Counter(item["kind"] for item in accepted),
                Counter(
                    {
                        "inverted_bottleneck_k2_plus_2_segment": 13,
                        "quantized_conv2d": 3,
                        "quantized_depthwise_conv2d": 1,
                    }
                ),
            )
            self.assertEqual(len({item["id"] for item in accepted}), 17)
            conv_roots = [
                item["id"]
                for item in accepted
                if item["kind"] == "quantized_conv2d"
            ]
            depthwise_roots = [
                item["id"]
                for item in accepted
                if item["kind"] == "quantized_depthwise_conv2d"
            ]
            for item in accepted:
                if item["kind"] != "inverted_bottleneck_k2_plus_2_segment":
                    continue
                conv_roots.extend(
                    item["layers"][name]["id"]
                    for name in ("expansion", "projection")
                )
                depthwise_roots.append(item["layers"]["depthwise"]["id"])
            self.assertEqual(len(conv_roots), 29)
            self.assertEqual(len(depthwise_roots), 14)
            self.assertEqual(len(set(conv_roots)), 29)
            self.assertEqual(len(set(depthwise_roots)), 14)
            compact = report["compact_graph"]
            self.assertEqual(compact["allocated_pool_bytes"] % 64, 0)
            self.assertFalse(compact["output_requires_normalization"])
            self.assertEqual(len(compact["boundaries"]), 8)
            self.assertEqual(compact["materialized_boundaries"], [])

            # The preprocessing module carries the target's original static
            # library path.  Compile a temporary copy with that path redirected
            # so this acceptance test cannot mutate the checked-in target tree.
            compile_input = directory / "compile.mlir"
            compile_text, replacements = re.subn(
                r'(static_library_output\s*=\s*)"[^"]*"',
                lambda match: f'{match.group(1)}"{object_file.as_posix()}"',
                rewritten.read_text(encoding="utf-8"),
                count=1,
            )
            self.assertEqual(replacements, 1)
            compile_input.write_text(compile_text, encoding="utf-8")

            subprocess.run(
                [
                    "iree-compile",
                    str(compile_input),
                    "--compile-from=preprocessing",
                    "--iree-hal-target-device=local",
                    "--iree-hal-local-target-device-backends=llvm-cpu",
                    "--iree-llvmcpu-target-cpu=generic",
                    "--iree-llvmcpu-link-embedded=false",
                    "--iree-llvmcpu-link-static",
                    f"--iree-llvmcpu-static-library-output-path={object_file}",
                    "-o",
                    str(vmfb),
                ],
                check=True,
                capture_output=True,
            )
            self.assertGreater(vmfb.stat().st_size, 0)
            self.assertGreater(object_file.stat().st_size, 0)
        self.assertEqual(
            original_target_state,
            {
                path.name: (path.stat().st_mtime_ns, path.stat().st_size)
                for path in REAL_TARGET_DIRECTORY.iterdir()
                if path.is_file()
            },
        )

    @unittest.skipUnless(
        REAL_PREPROCESSING.is_file(),
        "the checked-in MCUNet preprocessing IR is required",
    )
    def test_compact_transaction_plans_once_and_rebinds_second_parse(self):
        """Planning is source-only; mutation uses rebound values and one old plan."""
        counts = Counter()
        contexts = []
        original_analyze = PatternRegistry.analyze
        original_build_schedules = compact_analysis_module._build_access_schedules
        original_plan = compact_analysis_module.plan_compact_graph
        original_first_replay = compact_memory_module.replay_compact_graph_plan
        original_second_replay = compact_analysis_module.replay_compact_graph_plan
        original_parse = rewrite_module._parse
        original_emit = rewrite_module.emit_compact_graph

        def analyze(registry, normalized):
            counts["registry"] += 1
            return original_analyze(registry, normalized)

        def build_schedules(graph):
            counts["schedules"] += 1
            return original_build_schedules(graph)

        def plan(*args, **kwargs):
            counts["plan"] += 1
            return original_plan(*args, **kwargs)

        def first_replay(plan):
            counts["replay"] += 1
            return original_first_replay(plan)

        def second_replay(plan):
            counts["replay"] += 1
            return original_second_replay(plan)

        def parse(text):
            context, module = original_parse(text)
            contexts.append(context)
            return context, module

        def emit(module, candidates, compact, bindings):
            counts["emit"] += 1
            self.assertEqual(len(contexts), 2)
            self.assertTrue(
                all(isinstance(item.target_type, str) for item in compact.boundaries)
            )
            self.assertTrue(bindings.boundaries)
            self.assertTrue(
                all(item.target_value.context == contexts[1] for item in bindings.boundaries)
            )
            return original_emit(module, candidates, compact, bindings)

        with (
            mock.patch.object(PatternRegistry, "analyze", analyze),
            mock.patch.object(
                compact_analysis_module, "_build_access_schedules", build_schedules
            ),
            mock.patch.object(compact_analysis_module, "plan_compact_graph", plan),
            mock.patch.object(
                compact_memory_module, "replay_compact_graph_plan", first_replay
            ),
            mock.patch.object(
                compact_analysis_module,
                "replay_compact_graph_plan",
                second_replay,
            ),
            mock.patch.object(rewrite_module, "_parse", parse),
            mock.patch.object(rewrite_module, "emit_compact_graph", emit),
        ):
            result = rewrite_module.rewrite_text(
                REAL_PREPROCESSING.read_text(encoding="utf-8"), "auto"
            )

        self.assertTrue(result.plan["applied"])
        self.assertEqual(
            counts,
            Counter(
                {"registry": 2, "replay": 2, "schedules": 1, "plan": 1, "emit": 1}
            ),
        )

    def test_matching_is_independent_of_function_and_ssa_names(self):
        """Renaming every user-visible identity preserves semantic acceptance."""
        renamed = self.source.replace("@main", "@completely_unrelated_model")
        renamed = renamed.replace("%input", "%model_argument")
        renamed = renamed.replace("%weight", "%constant_filter")
        renamed = renamed.replace("%output", "%final_value")
        baseline = rewrite_text(self.source, "strict")
        result = rewrite_text(renamed, "strict")
        self.assertEqual(result.plan["totals"], baseline.plan["totals"])
        self.assertEqual(
            result.plan["accepted"][0]["quantization"],
            baseline.plan["accepted"][0]["quantization"],
        )

    def test_registry_can_be_extended_without_driver_changes(self):
        """A new analyzer registration is visible without editing rewrite.py."""
        registry = create_default_registry()

        def no_matches(graphs, occupied):
            from oneliner_vmcu.model import Analysis

            return Analysis([], [])

        def unreachable_emitter(match):
            raise AssertionError("no-match emitter must not run")

        registry.register("test_extension", no_matches, unreachable_emitter)
        result = rewrite_text(self.source, "strict", registry=registry)
        self.assertEqual(result.plan["pattern_registry"][-1], "test_extension")
        self.assertIn("quantized_fully_connected", result.plan["pattern_registry"])

    def test_default_registry_rejects_duplicate_pattern_names(self):
        """Duplicate semantic ownership cannot silently replace an emitter."""
        registry = PatternRegistry()

        def no_matches(graphs, occupied):
            from oneliner_vmcu.model import Analysis

            return Analysis([], [])

        registry.register("kind", no_matches, lambda match: None)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            registry.register("kind", no_matches, lambda match: None)

    @unittest.skipUnless(shutil.which("iree-compile"), "iree-compile is unavailable")
    def test_post_preprocessing_output_compiles_from_preprocessing(self):
        """The real split pipeline emits VMFB, static object, and expected dump."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            preprocessing = directory / "preprocessing.mlir"
            rewritten = directory / "rewritten.mlir"
            plan = directory / "plan.json"
            vmfb = directory / "model.vmfb"
            object_file = directory / "model.o"
            dumps = directory / "dumps"
            dumps.mkdir()
            environment = dict(os.environ)
            # Phase 1 creates target-aware textual preprocessing IR. Static-link
            # settings are intentionally passed here because IREE persists them
            # in the executable target before the pipeline is paused.
            subprocess.run(
                [
                    "iree-compile",
                    str(FIXTURE),
                    "--compile-to=preprocessing",
                    "--emit-mlir-bytecode=false",
                    "--iree-hal-target-device=local",
                    "--iree-hal-local-target-device-backends=llvm-cpu",
                    "--iree-llvmcpu-target-cpu=generic",
                    "--iree-llvmcpu-link-embedded=false",
                    "--iree-llvmcpu-link-static",
                    f"--iree-llvmcpu-static-library-output-path={object_file}",
                    "-o",
                    str(preprocessing),
                ],
                env=environment,
                check=True,
                capture_output=True,
            )
            # The same CLI that Cargo invokes performs analysis and rewriting.
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(preprocessing),
                    "-o",
                    str(rewritten),
                    "--plan-output",
                    str(plan),
                    "--mode",
                    "strict",
                ],
                env=environment,
                check=True,
                capture_output=True,
            )
            # Resume the stock compiler without a plugin or custom dialect.
            subprocess.run(
                [
                    "iree-compile",
                    str(rewritten),
                    "--compile-from=preprocessing",
                    "--iree-hal-target-device=local",
                    "--iree-hal-local-target-device-backends=llvm-cpu",
                    "--iree-llvmcpu-target-cpu=generic",
                    "--iree-llvmcpu-link-embedded=false",
                    "--iree-llvmcpu-link-static",
                    f"--iree-llvmcpu-static-library-output-path={object_file}",
                    f"--dump-compilation-phases-to={dumps}",
                    "-o",
                    str(vmfb),
                ],
                env=environment,
                check=True,
                capture_output=True,
            )

            self.assertGreater(vmfb.stat().st_size, 0)
            self.assertGreater(object_file.stat().st_size, 0)
            self.assertTrue(
                any(path.name.endswith(".10.executable-targets.mlir") for path in dumps.iterdir())
            )
            self.assertEqual(
                json.loads(plan.read_text(encoding="utf-8"))["totals"]["accepted"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
