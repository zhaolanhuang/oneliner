import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "oneliner-macro"
    / "python"
    / "rewrite_cmsis_nn.py"
)
SPEC = importlib.util.spec_from_file_location("oneliner_cmsis_nn_rewrite", SCRIPT)
REWRITER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REWRITER
SPEC.loader.exec_module(REWRITER)


class CmsisNNRewriteTests(unittest.TestCase):
    def test_rewrites_static_int8_conv_and_converts_shift(self):
        fixture = """
module {
  func.func @main() {
    %c0_i32 = arith.constant 0 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %weights = arith.constant dense<0> : tensor<6x5x5x1xi8>
    %bias = arith.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
    %mult = arith.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
    %shift = arith.constant dense<[41, 41, 40, 40, 41, 42]> : tensor<6xi8>
    %input = tensor.empty() : tensor<1x28x28x1xi8>
    %filter_init = tensor.empty() : tensor<5x5x1x6xi8>
    %filter = linalg.transpose ins(%weights : tensor<6x5x5x1xi8>) outs(%filter_init : tensor<5x5x1x6xi8>) permutation = [1, 2, 3, 0]
    %acc = tensor.empty() : tensor<1x24x24x6xi32>
    %bias_init = linalg.generic {indexing_maps = [], iterator_types = []} ins(%bias : tensor<6xi32>) outs(%acc : tensor<1x24x24x6xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x24x24x6xi32>
    %conv = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %filter, %c-128_i32, %c0_i32 : tensor<1x28x28x1xi8>, tensor<5x5x1x6xi8>, i32, i32) outs(%bias_init : tensor<1x24x24x6xi32>) -> tensor<1x24x24x6xi32>
    %output = tensor.empty() : tensor<1x24x24x6xi8>
    %result = linalg.generic {indexing_maps = [], iterator_types = []} ins(%conv, %mult, %shift : tensor<1x24x24x6xi32>, tensor<6xi32>, tensor<6xi8>) outs(%output : tensor<1x24x24x6xi8>) {
    ^bb0(%in: i32, %m: i32, %s: i8, %out: i8):
      %scaled = tosa.apply_scale %in, %m, %s {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %offset = arith.addi %scaled, %c-128_i32 : i32
      %clamped_min = arith.maxsi %offset, %c-128_i32 : i32
      %clamped = arith.minsi %clamped_min, %c127_i32 : i32
      %value = arith.trunci %clamped : i32 to i8
      linalg.yield %value : i8
    } -> tensor<1x24x24x6xi8>
    return
  }
}
"""

        output, count = REWRITER.rewrite(fixture)

        self.assertEqual(count, 1)
        self.assertIn(
            'iree_codegen.ukernel_descriptor<"oneliner_cmsis_nn_conv_s8", bitcode>',
            output,
        )
        self.assertIn('path = "oneliner_cmsis_nn_conv_s8.bc"', output)
        self.assertIn("iree_linalg_ext.custom_op", output)
        self.assertIn("dense<[-10, -10, -9, -9, -10, -11]>", output)
        self.assertIn("tensor<20xi32>", output)
        self.assertIn("112]>", output)
        self.assertNotIn("tosa.apply_scale", output)

    def test_leaves_nonzero_weight_zero_point_unchanged(self):
        fixture = "%c1 = arith.constant 1 : i32\n%conv = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%a, %b, %c1, %c1 : tensor<1x1x1x1xi8>, tensor<1x1x1x1xi8>, i32, i32) outs(%c : tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>\n"

        output, count = REWRITER.rewrite(fixture)

        self.assertEqual(count, 0)
        self.assertEqual(output, fixture)

    def test_finalizes_configured_generic_ukernel(self):
        fixture = (
            '%0 = iree_codegen.ukernel.generic '
            '{iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<'
            '"oneliner_cmsis_nn_conv_s8", bitcode>} '
            '"oneliner_cmsis_nn_conv_s8" ins(%arg0 : tensor<1xi8>) '
            'outs(%arg1 : tensor<1xi8>) strided_dims([[], []]) -> tensor<1xi8>\n'
        )

        output, count = REWRITER.finalize_configured(fixture)

        self.assertEqual(count, 1)
        self.assertIn("fn_def_attrs {hal.import.bitcode = true}", output)
        self.assertNotIn("ukernel_descriptor", output)


if __name__ == "__main__":
    unittest.main()
