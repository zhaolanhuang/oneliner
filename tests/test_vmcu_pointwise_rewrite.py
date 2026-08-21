import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "oneliner-macro"
    / "python"
    / "rewrite_vmcu_pointwise.py"
)
FIXTURE = Path(__file__).parent / "fixtures" / "vmcu_pointwise_pair.preprocessing.mlir"
SPEC = importlib.util.spec_from_file_location("oneliner_vmcu_rewriter", SCRIPT)
REWRITER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REWRITER
SPEC.loader.exec_module(REWRITER)


class VmcuPointwiseRewriteTests(unittest.TestCase):
    def test_plans_one_intermediate_segment(self):
        plan = REWRITER.plan_pointwise_pair(FIXTURE.read_text(encoding="utf-8"))

        self.assertEqual(plan.rows, 4)
        self.assertEqual(plan.intermediate_channels, 5)
        self.assertEqual(plan.full_intermediate_bytes, 20)
        self.assertEqual(plan.segment_bytes, 5)
        self.assertEqual(plan.saved_intermediate_bytes, 15)

    def test_rewrites_pair_as_one_bitcode_ukernel(self):
        rewritten, plan = REWRITER.rewrite(FIXTURE.read_text(encoding="utf-8"))
        context = REWRITER.ir.Context()
        module = REWRITER.ir.Module.parse(rewritten, context=context)
        custom_ops = REWRITER.operations_named(module, "iree_linalg_ext.custom_op")

        self.assertEqual(len(custom_ops), 1)
        self.assertTrue(module.operation.verify())
        self.assertEqual(len(custom_ops[0].operands), 6)
        self.assertEqual(len(custom_ops[0].results), 2)
        self.assertEqual(str(custom_ops[0].results[1].type), "tensor<5xi8>")
        indexing_maps = REWRITER.ir.ArrayAttr(custom_ops[0].attributes["indexing_maps"])
        self.assertTrue(
            all(REWRITER.ir.AffineMapAttr(item).value.n_symbols == 5 for item in indexing_maps)
        )
        self.assertEqual(rewritten.count(REWRITER.UKERNEL_NAME), 1)
        self.assertNotIn("linalg.quantized_matmul", rewritten)
        self.assertEqual(plan.segment_bytes, 5)

    def test_rejects_nonzero_zero_point(self):
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "%c0_i32 = arith.constant 0 : i32",
            "%c0_i32 = arith.constant 1 : i32",
            1,
        )

        with self.assertRaisesRegex(ValueError, "zero zero-points"):
            REWRITER.plan_pointwise_pair(text)

    def test_rejects_non_saturating_clamp(self):
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "%c-128_i32 = arith.constant -128 : i32",
            "%c-128_i32 = arith.constant -127 : i32",
            1,
        )

        with self.assertRaisesRegex(ValueError, "canonical int8 clamp"):
            REWRITER.plan_pointwise_pair(text)

    def test_rejects_nonzero_accumulator_initializer(self):
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "%2 = linalg.fill ins(%c0_i32",
            "%one = arith.constant 1 : i32\n    %2 = linalg.fill ins(%one",
            1,
        )

        with self.assertRaisesRegex(ValueError, "initialized by zero fills"):
            REWRITER.plan_pointwise_pair(text)

    def test_rejects_clamp_with_different_yield(self):
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "linalg.yield %14 : i8",
            "linalg.yield %out : i8",
            1,
        )

        with self.assertRaisesRegex(ValueError, "canonical int8 clamp"):
            REWRITER.plan_pointwise_pair(text)

    def test_rejects_non_identity_clamp_map(self):
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "affine_map<(d0, d1) -> (d0, d1)>",
            "affine_map<(d0, d1) -> (d0, 0)>",
            1,
        )

        with self.assertRaisesRegex(ValueError, "identity saturation maps"):
            REWRITER.plan_pointwise_pair(text)

    def test_rejects_reused_intermediate(self):
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "%6 = tensor.empty()",
            "%reuse = tensor.cast %5 : tensor<4x5xi8> to tensor<4x5xi8>\n    %6 = tensor.empty()",
            1,
        )

        with self.assertRaisesRegex(ValueError, "contiguous canonical chain"):
            REWRITER.plan_pointwise_pair(text)

    def test_rejects_external_use_of_erased_initializer(self):
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "    %11 = hal.tensor.export",
            "    %reuse = tensor.cast %4 : tensor<4x5xi8> to tensor<4x5xi8>\n"
            "    %11 = hal.tensor.export",
            1,
        )

        with self.assertRaisesRegex(ValueError, "fresh empty tensor"):
            REWRITER.plan_pointwise_pair(text)

    def test_ignores_nested_constant_with_reused_ssa_name(self):
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "%c0_i32 = arith.constant 0 : i32",
            "%c0_i32 = arith.constant 1 : i32",
            1,
        ).replace(
            "    %11 = hal.tensor.export",
            "    scf.execute_region {\n"
            "      %c0_i32 = arith.constant 0 : i32\n"
            "      scf.yield\n"
            "    }\n"
            "    %11 = hal.tensor.export",
            1,
        )

        with self.assertRaisesRegex(ValueError, "invalid IREE preprocessing MLIR"):
            REWRITER.plan_pointwise_pair(text)

    def test_structured_rewrite_handles_existing_ssa_names(self):
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "    %1 = tensor.empty()",
            "    %vmcu_0_config = arith.constant dense<[0, 0, 0, 0]> : tensor<4xi32>\n"
            "    %1 = tensor.empty()",
            1,
        )

        rewritten, _ = REWRITER.rewrite(text)

        context = REWRITER.ir.Context()
        module = REWRITER.ir.Module.parse(rewritten, context=context)
        self.assertTrue(module.operation.verify())
        self.assertEqual(
            len(REWRITER.operations_named(module, "iree_linalg_ext.custom_op")), 1
        )

    def test_finalizes_exactly_one_configured_ukernel(self):
        configured = f'''module {{
  func.func @test(%input: tensor<4x4xi8>, %w0: tensor<4x5xi8>,
      %w1: tensor<5x3xi8>, %segment: tensor<5xi8>, %config: tensor<4xi32>,
      %output: tensor<4x3xi8>) -> tensor<4x3xi8> {{
    %0 = iree_codegen.ukernel.generic
      {{iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"{REWRITER.UKERNEL_NAME}", bitcode>}}
      "{REWRITER.UKERNEL_NAME}"
      ins(%input, %w0, %w1, %segment, %config : tensor<4x4xi8>, tensor<4x5xi8>,
          tensor<5x3xi8>, tensor<5xi8>, tensor<4xi32>)
      outs(%output : tensor<4x3xi8>) fn_def_attrs {{test.keep = "yes"}}
      strided_dims([[], [], [], [], [], []])
      -> tensor<4x3xi8>
    return %0 : tensor<4x3xi8>
  }}
}}
'''

        finalized, count = REWRITER.finalize_configured(configured)

        self.assertEqual(count, 1)
        self.assertIn("hal.import.bitcode = true", finalized)
        self.assertIn('test.keep = "yes"', finalized)
        self.assertNotIn("ukernel_descriptor", finalized)
        context = REWRITER.ir.Context()
        module = REWRITER.ir.Module.parse(finalized, context=context)
        self.assertTrue(module.operation.verify())

    def test_plan_json_is_serializable(self):
        rewritten, plan = REWRITER.rewrite(FIXTURE.read_text(encoding="utf-8"))
        self.assertIn(REWRITER.UKERNEL_NAME, rewritten)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(
                json.dumps({"schema_version": 1, **REWRITER.asdict(plan)}),
                encoding="utf-8",
            )
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["segment_bytes"], 5)


if __name__ == "__main__":
    unittest.main()
