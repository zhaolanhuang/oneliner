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


def depthwise_fixture(channel_multiplier=1, input_channels=2, filter_size=3):
    output_channels = input_channels * channel_multiplier
    input_size = filter_size + 2
    output_size = 3
    filter_count = filter_size * filter_size * output_channels
    filter_values = ", ".join("0" for _ in range(filter_count))
    bias_values = ", ".join(str(value) for value in range(output_channels))
    multiplier_values = ", ".join("1073741824" for _ in range(output_channels))
    shift_hex = "29" * output_channels
    return f"""
module {{
  func.func @main() {{
    %c0_i32 = arith.constant 0 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %filter = arith.constant dense<[{filter_values}]> : tensor<{filter_size}x{filter_size}x{input_channels}x{channel_multiplier}xi8>
    %bias = arith.constant dense<[{bias_values}]> : tensor<{output_channels}xi32>
    %mult = arith.constant dense<[{multiplier_values}]> : tensor<{output_channels}xi32>
    %shift = arith.constant dense<"0x{shift_hex}"> : tensor<{output_channels}xi8>
    %input = tensor.empty() : tensor<1x{input_size}x{input_size}x{input_channels}xi8>
    %acc_init = tensor.empty() : tensor<1x{output_size}x{output_size}x{input_channels}x{channel_multiplier}xi32>
    %acc = linalg.fill ins(%c0_i32 : i32) outs(%acc_init : tensor<1x{output_size}x{output_size}x{input_channels}x{channel_multiplier}xi32>) -> tensor<1x{output_size}x{output_size}x{input_channels}x{channel_multiplier}xi32>
    %conv = linalg.depthwise_conv_2d_nhwc_hwcm_q {{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}} ins(%input, %filter, %c-128_i32, %c0_i32 : tensor<1x{input_size}x{input_size}x{input_channels}xi8>, tensor<{filter_size}x{filter_size}x{input_channels}x{channel_multiplier}xi8>, i32, i32) outs(%acc : tensor<1x{output_size}x{output_size}x{input_channels}x{channel_multiplier}xi32>) -> tensor<1x{output_size}x{output_size}x{input_channels}x{channel_multiplier}xi32>
    %collapsed = tensor.collapse_shape %conv [[0], [1], [2], [3, 4]] : tensor<1x{output_size}x{output_size}x{input_channels}x{channel_multiplier}xi32> into tensor<1x{output_size}x{output_size}x{output_channels}xi32>
    %bias_init = tensor.empty() : tensor<1x{output_size}x{output_size}x{output_channels}xi32>
    %biased = linalg.generic {{indexing_maps = [], iterator_types = []}} ins(%bias, %collapsed : tensor<{output_channels}xi32>, tensor<1x{output_size}x{output_size}x{output_channels}xi32>) outs(%bias_init : tensor<1x{output_size}x{output_size}x{output_channels}xi32>) {{
    ^bb0(%bias_in: i32, %in: i32, %out: i32):
      %sum = arith.addi %in, %bias_in : i32
      linalg.yield %sum : i32
    }} -> tensor<1x{output_size}x{output_size}x{output_channels}xi32>
    %output = tensor.empty() : tensor<1x{output_size}x{output_size}x{output_channels}xi8>
    %result = linalg.generic {{indexing_maps = [], iterator_types = []}} ins(%biased, %mult, %shift : tensor<1x{output_size}x{output_size}x{output_channels}xi32>, tensor<{output_channels}xi32>, tensor<{output_channels}xi8>) outs(%output : tensor<1x{output_size}x{output_size}x{output_channels}xi8>) {{
    ^bb0(%in: i32, %m: i32, %s: i8, %out: i8):
      %scaled = tosa.apply_scale %in, %m, %s {{rounding_mode = DOUBLE_ROUND}} : (i32, i32, i8) -> i32
      %offset = arith.addi %scaled, %c-128_i32 : i32
      %clamped_min = arith.maxsi %offset, %c-128_i32 : i32
      %clamped = arith.minsi %clamped_min, %c127_i32 : i32
      %value = arith.trunci %clamped : i32 to i8
      linalg.yield %value : i8
    }} -> tensor<1x{output_size}x{output_size}x{output_channels}xi8>
    return
  }}
}}
"""


def conv_fixture(dilation=1, stride=1, output_size=4):
    return f"""
module {{
  func.func @main() {{
    %c0_i32 = arith.constant 0 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %weights = arith.constant dense<1> : tensor<1x1x1x1xi8>
    %bias = arith.constant dense<[3]> : tensor<1xi32>
    %mult = arith.constant dense<[1073741824]> : tensor<1xi32>
    %shift = arith.constant dense<[40]> : tensor<1xi8>
    %input = tensor.empty() : tensor<1x{output_size}x{output_size}x1xi8>
    %filter_init = tensor.empty() : tensor<1x1x1x1xi8>
    %filter = linalg.transpose ins(%weights : tensor<1x1x1x1xi8>) outs(%filter_init : tensor<1x1x1x1xi8>) permutation = [1, 2, 3, 0]
    %acc = tensor.empty() : tensor<1x{output_size}x{output_size}x1xi32>
    %bias_init = linalg.generic {{indexing_maps = [], iterator_types = []}} ins(%bias : tensor<1xi32>) outs(%acc : tensor<1x{output_size}x{output_size}x1xi32>) {{
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }} -> tensor<1x{output_size}x{output_size}x1xi32>
    %conv = linalg.conv_2d_nhwc_hwcf_q {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins(%input, %filter, %c-128_i32, %c0_i32 : tensor<1x{output_size}x{output_size}x1xi8>, tensor<1x1x1x1xi8>, i32, i32) outs(%bias_init : tensor<1x{output_size}x{output_size}x1xi32>) -> tensor<1x{output_size}x{output_size}x1xi32>
    %output = tensor.empty() : tensor<1x{output_size}x{output_size}x1xi8>
    %result = linalg.generic {{indexing_maps = [], iterator_types = []}} ins(%conv, %mult, %shift : tensor<1x{output_size}x{output_size}x1xi32>, tensor<1xi32>, tensor<1xi8>) outs(%output : tensor<1x{output_size}x{output_size}x1xi8>) {{
    ^bb0(%in: i32, %m: i32, %s: i8, %out: i8):
      %scaled = tosa.apply_scale %in, %m, %s {{rounding_mode = DOUBLE_ROUND}} : (i32, i32, i8) -> i32
      %offset = arith.addi %scaled, %c-128_i32 : i32
      %clamped_min = arith.maxsi %offset, %c-128_i32 : i32
      %clamped = arith.minsi %clamped_min, %c127_i32 : i32
      %value = arith.trunci %clamped : i32 to i8
      linalg.yield %value : i8
    }} -> tensor<1x{output_size}x{output_size}x1xi8>
    return
  }}
}}
"""


def avgpool_fixture(input_size=2, output_size=1, window_size=2, stride=2, channels=4):
    return f"""
module {{
  func.func @main() {{
    %c0_i32 = arith.constant 0 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %cmult_i64 = arith.constant 1073741825 : i64
    %cround_i64 = arith.constant 2147483648 : i64
    %c32_i64 = arith.constant 32 : i64
    %input = tensor.empty() : tensor<1x{input_size}x{input_size}x{channels}xi8>
    %window = tensor.empty() : tensor<{window_size}x{window_size}xi32>
    %acc_empty = tensor.empty() : tensor<1x{output_size}x{output_size}x{channels}xi32>
    %acc = linalg.fill ins(%c0_i32 : i32) outs(%acc_empty : tensor<1x{output_size}x{output_size}x{channels}xi32>) -> tensor<1x{output_size}x{output_size}x{channels}xi32>
    %pool = linalg.pooling_nhwc_sum {{dilations = dense<1> : vector<2xi64>, strides = dense<{stride}> : vector<2xi64>}} ins(%input, %window : tensor<1x{input_size}x{input_size}x{channels}xi8>, tensor<{window_size}x{window_size}xi32>) outs(%acc : tensor<1x{output_size}x{output_size}x{channels}xi32>) -> tensor<1x{output_size}x{output_size}x{channels}xi32>
    %out_empty = tensor.empty() : tensor<1x{output_size}x{output_size}x{channels}xi8>
    %result = linalg.generic {{indexing_maps = [], iterator_types = []}} ins(%pool : tensor<1x{output_size}x{output_size}x{channels}xi32>) outs(%out_empty : tensor<1x{output_size}x{output_size}x{channels}xi8>) {{
    ^bb0(%in: i32, %out: i8):
      %e = arith.extsi %in : i32 to i64
      %m = arith.muli %e, %cmult_i64 : i64
      %r = arith.addi %m, %cround_i64 : i64
      %s = arith.shrui %r, %c32_i64 : i64
      %t = arith.trunci %s : i64 to i32
      %lo = arith.maxsi %t, %c-128_i32 : i32
      %hi = arith.minsi %lo, %c127_i32 : i32
      %v = arith.trunci %hi : i32 to i8
      linalg.yield %v : i8
    }} -> tensor<1x{output_size}x{output_size}x{channels}xi8>
    return
  }}
}}
"""


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
        self.assertIn('path = "oneliner_cmsis_nn.bc"', output)
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

    def test_1x1_conv_with_dilation_allocates_generic_scratch(self):
        output, count = REWRITER.rewrite(conv_fixture(dilation=2))

        self.assertEqual(count, 1)
        self.assertIn(
            'iree_codegen.ukernel_descriptor<"oneliner_cmsis_nn_conv_s8", bitcode>',
            output,
        )
        self.assertIn("tensor<16xi8>", output)
        self.assertIn(
            "dense<[1, 4, 4, 1, 4, 4, 1, 1, 1, 1, 1, 2, 2, 0, 0, "
            "128, -128, -128, 127, 16]>",
            output,
        )

    def test_1x1_conv_without_dilation_uses_no_scratch(self):
        output, count = REWRITER.rewrite(conv_fixture(dilation=1))

        self.assertEqual(count, 1)
        self.assertIn("tensor<1xi8>", output)
        self.assertIn(
            "dense<[1, 4, 4, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1, 0, 0, "
            "128, -128, -128, 127, 0]>",
            output,
        )

    def test_rewrites_static_int8_max_pool(self):
        fixture = """
module {
  func.func @main() {
    %c-128_i8 = arith.constant -128 : i8
    %input = tensor.empty() : tensor<1x8x8x4xi8>
    %window = tensor.empty() : tensor<2x2xi8>
    %empty = tensor.empty() : tensor<1x4x4x4xi8>
    %init = linalg.fill ins(%c-128_i8 : i8) outs(%empty : tensor<1x4x4x4xi8>) -> tensor<1x4x4x4xi8>
    %result = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%input, %window : tensor<1x8x8x4xi8>, tensor<2x2xi8>) outs(%init : tensor<1x4x4x4xi8>) -> tensor<1x4x4x4xi8>
    return
  }
}
"""

        output, count = REWRITER.rewrite(fixture)

        self.assertEqual(count, 1)
        self.assertIn(
            'iree_codegen.ukernel_descriptor<"oneliner_cmsis_nn_max_pool_s8", bitcode>',
            output,
        )
        self.assertIn("tensor<14xi32>", output)
        self.assertNotIn("linalg.pooling_nhwc_max", output)

    def test_rewrites_fully_connected_form(self):
        fixture = """
module {
  func.func @main() {
    %c0_i32 = arith.constant 0 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %cmult_i64 = arith.constant 1073741824 : i64
    %cshift_i64 = arith.constant 39 : i64
    %weights = arith.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<2x1x1x3xi8>
    %bias = arith.constant dense<[7, 8]> : tensor<2xi32>
    %input = tensor.empty() : tensor<1x1x1x3xi8>
    %filter_empty = tensor.empty() : tensor<1x1x3x2xi8>
    %filter = linalg.transpose ins(%weights : tensor<2x1x1x3xi8>) outs(%filter_empty : tensor<1x1x3x2xi8>) permutation = [1, 2, 3, 0]
    %acc_empty = tensor.empty() : tensor<1x1x1x2xi32>
    %acc_init = linalg.generic {indexing_maps = [], iterator_types = []} ins(%bias : tensor<2xi32>) outs(%acc_empty : tensor<1x1x1x2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x1x1x2xi32>
    %conv = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %filter, %c-128_i32, %c0_i32 : tensor<1x1x1x3xi8>, tensor<1x1x3x2xi8>, i32, i32) outs(%acc_init : tensor<1x1x1x2xi32>) -> tensor<1x1x1x2xi32>
    %output = tensor.empty() : tensor<1x1x1x2xi8>
    %result = linalg.generic {indexing_maps = [], iterator_types = []} ins(%conv : tensor<1x1x1x2xi32>) outs(%output : tensor<1x1x1x2xi8>) {
    ^bb0(%in: i32, %out: i8):
      %extended = arith.extsi %in : i32 to i64
      %scaled = arith.muli %extended, %cmult_i64 : i64
      %rounded = arith.shrsi %scaled, %cshift_i64 : i64
      %value32 = arith.trunci %rounded : i64 to i32
      %offset = arith.addi %value32, %c-128_i32 : i32
      %clamped_min = arith.maxsi %offset, %c-128_i32 : i32
      %clamped = arith.minsi %clamped_min, %c127_i32 : i32
      %value = arith.trunci %clamped : i32 to i8
      linalg.yield %value : i8
    } -> tensor<1x1x1x2xi8>
    return
  }
}
"""

        output, count = REWRITER.rewrite(fixture)

        self.assertEqual(count, 1)
        self.assertIn(
            'iree_codegen.ukernel_descriptor<'
            '"oneliner_cmsis_nn_fully_connected_s8", bitcode>',
            output,
        )
        self.assertIn("tensor<10xi32>", output)
        self.assertIn("ins(%input, %weights, %bias,", output)
        self.assertNotIn("arith.muli", output)

    def test_rewrites_depthwise_with_hex_encoded_shifts(self):
        output, count = REWRITER.rewrite(depthwise_fixture())

        self.assertEqual(count, 1)
        self.assertIn(
            'iree_codegen.ukernel_descriptor<'
            '"oneliner_cmsis_nn_depthwise_conv_s8", bitcode>',
            output,
        )
        self.assertIn("tensor<21xi32>", output)
        self.assertIn(
            "ins(%input, %filter, %bias, %mult, %cmsis_nn_0_shift,",
            output,
        )
        self.assertIn(
            "dense<[1, 5, 5, 2, 3, 3, 2, 3, 3, 1, 1, 1, 1, 0, 0, "
            "128, -128, -128, 127, 1, 0]>",
            output,
        )
        self.assertNotIn("tosa.apply_scale", output)

    def test_leaves_depthwise_channel_multiplier_unchanged(self):
        fixture = depthwise_fixture(channel_multiplier=2)

        output, count = REWRITER.rewrite(fixture)

        self.assertEqual(count, 0)
        self.assertEqual(output, fixture)

    def test_rewrites_avg_pool_with_requant(self):
        output, count = REWRITER.rewrite(avgpool_fixture())

        self.assertEqual(count, 1)
        self.assertIn(
            'iree_codegen.ukernel_descriptor<'
            '"oneliner_cmsis_nn_avg_pool_s8", bitcode>',
            output,
        )
        self.assertIn("tensor<15xi32>", output)
        self.assertIn(
            "dense<[1, 2, 2, 4, 1, 1, 2, 2, 2, 2, 0, 0, "
            "-128, 127, 1073741825]>",
            output,
        )
        self.assertNotIn("linalg.pooling_nhwc_sum", output)

    def test_leaves_avg_pool_with_padding_unchanged(self):
        fixture = avgpool_fixture(input_size=3, output_size=2)

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
