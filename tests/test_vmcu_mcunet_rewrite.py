import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "oneliner-macro" / "python" / "rewrite_vmcu_mcunet.py"
MODEL = ROOT / "examples" / "models" / "mcunet_10fps_vww.mlir"
SPEC = importlib.util.spec_from_file_location("oneliner_vmcu_mcunet_rewriter", SCRIPT)
REWRITER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REWRITER
SPEC.loader.exec_module(REWRITER)


def constant_values(value):
    owner = REWRITER.owner_operation(value)
    return [int(item) for item in owner.attributes["value"]]


def replace_occurrence(text, old, new, occurrence):
    start = -1
    for _ in range(occurrence):
        start = text.index(old, start + 1)
    return text[:start] + new + text[start + len(old) :]


def configured_fixture(count=13):
    operations = []
    previous = "%output"
    for index in range(count):
        operations.append(
            f'''    %{index} = iree_codegen.ukernel.generic
      {{iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"{REWRITER.UKERNEL_NAME}", bitcode>}}
      "{REWRITER.UKERNEL_NAME}"
      ins(%input : tensor<1xi8>) outs({previous} : tensor<1xi8>)
      fn_def_attrs {{test.keep = "yes"}} strided_dims([[], []]) -> tensor<1xi8>'''
        )
        previous = f"%{index}"
    return f'''module {{
  func.func @test(%input: tensor<1xi8>, %output: tensor<1xi8>) -> tensor<1xi8> {{
{chr(10).join(operations)}
    return {previous} : tensor<1xi8>
  }}
}}
'''


def module_config_kind(custom):
    if len(custom.operands) == 16:
        return "ibn"
    if len(custom.operands) == 8:
        return "conv2d"
    return "pair"


class VmcuMcunetRewriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module_config_kind = staticmethod(module_config_kind)
        cls.source = MODEL.read_text(encoding="utf-8")
        cls.rewritten, cls.plan = REWRITER.rewrite_generic(cls.source)
        cls.context = REWRITER.ir.Context()
        cls.module = REWRITER.ir.Module.parse(cls.rewritten, context=cls.context)
        cls.custom_ops = REWRITER.operations_named(cls.module, "iree_linalg_ext.custom_op")
        cls.ibn_ops = [op for op in cls.custom_ops if module_config_kind(op) == "ibn"]

    def test_matches_all_blocks_and_plan_metrics(self):
        matched = REWRITER.match_generic(self.source)

        kinds = [module.kind for module in matched.modules]
        self.assertEqual(kinds, ["ibn"] * 13 + ["conv2d"])
        self.assertEqual(sum(module.residual for module in matched.modules), 7)
        self.assertEqual(self.plan.mode, "auto")
        self.assertEqual(self.plan.block_count, 14)
        self.assertEqual(self.plan.residual_block_count, 7)
        self.assertEqual(self.plan.in_place_block_count, 7)
        self.assertEqual(self.plan.standard_peak_intermediate_bytes, 49152)
        self.assertEqual(self.plan.max_segment_bytes, 8448)
        self.assertEqual(self.plan.total_segment_bytes, 60800)
        self.assertEqual(self.plan.full_intermediate_bytes, 49152)
        self.assertEqual(self.plan.segment_bytes, 8448)
        self.assertEqual(self.plan.saved_intermediate_bytes, 40704)

    def test_rewrites_all_blocks_and_removes_old_block_contractions(self):
        self.assertTrue(self.module.operation.verify())
        self.assertEqual(len(self.custom_ops), 14)
        self.assertEqual(len(REWRITER.operations_named(self.module, REWRITER.DEPTHWISE)), 1)
        self.assertEqual(len(REWRITER.operations_named(self.module, REWRITER.MATMUL)), 2)
        self.assertEqual(self.rewritten.count(REWRITER.UKERNEL_NAME), 13)
        self.assertEqual(self.rewritten.count(REWRITER.CONV2D_KERNEL_NAME), 1)
        self.assertEqual(self.rewritten.count(REWRITER.GENERIC_BITCODE_PATH), 14)

        reparsed_context = REWRITER.ir.Context()
        reparsed = REWRITER.ir.Module.parse(self.module.operation.get_asm(), context=reparsed_context)
        self.assertTrue(reparsed.operation.verify())

    def test_custom_op_abi_uses_original_constants_and_in_place_outputs(self):
        in_place = 0
        for custom in self.ibn_ops:
            self.assertEqual(len(custom.operands), 16)
            self.assertEqual(len(custom.results), 2)
            self.assertEqual(
                [int(item) for item in custom.attributes["operandSegmentSizes"]],
                [14, 2],
            )
            for operand_number in range(1, 13):
                self.assertEqual(
                    REWRITER.owner_operation(custom.operands[operand_number]).name,
                    "arith.constant",
                )
            config = constant_values(custom.operands[13])
            if custom.operands[14] == custom.operands[0]:
                in_place += 1
                self.assertEqual(config[1], 3)
            else:
                self.assertEqual(config[1], 0)
                self.assertEqual(REWRITER.owner_operation(custom.operands[14]).name, "tensor.empty")
        self.assertEqual(in_place, 7)

    def test_maps_cover_all_symbols_and_keep_output_major_weights(self):
        for custom in self.ibn_ops:
            maps = [REWRITER.ir.AffineMapAttr(item).value for item in custom.attributes["indexing_maps"]]
            self.assertEqual(len(maps), 16)
            self.assertTrue(all(item.n_symbols == 13 and item.n_dims == 0 for item in maps))
            symbols = {
                REWRITER.ir.AffineSymbolExpr(expression).position
                for affine_map in maps
                for expression in affine_map.results
            }
            self.assertEqual(symbols, set(range(13)))

            config = constant_values(custom.operands[13])
            cin, cexp, cout = config[5], config[8], config[9]
            self.assertEqual(
                tuple(REWRITER.ir.RankedTensorType(custom.operands[1].type).shape),
                (cexp, cin),
            )
            self.assertEqual(
                tuple(REWRITER.ir.RankedTensorType(custom.operands[3].type).shape),
                (cout, cexp),
            )

    def test_configs_have_exact_schema_geometry_and_residual_scales(self):
        expected_residual = {
            2: [1073741824, 10, 1785103811, 32, 1073741824, 11, 0, 0, 1938331585, 50],
            4: [1073741824, 10, 2086460944, 32, 1073741824, 11, 0, 0, 1899219683, 50],
            5: [1073741824, 10, 1765608801, 33, 1073741824, 11, 0, 0, 1980469084, 50],
            7: [1073741824, 10, 1825212155, 32, 1073741824, 11, 0, 0, 1942687389, 50],
            8: [1073741824, 10, 1391170981, 33, 1073741824, 11, 0, 0, 2136511470, 50],
            10: [1073741824, 11, 0, 0, 1073741824, 10, 1608900243, 32, 2140498155, 50],
            12: [1073741824, 10, 1935178768, 32, 1073741824, 11, 0, 0, 1104644408, 49],
        }
        expected_scratch = [8448, 4608, 4480, 3456, 3456, 3360, 2880, 4480, 1920, 3456, 7200, 6336, 6144]
        expected_final_zp = [-22, -7, 5, 1, 7, -18, -9, -11, -2, -1, -15, 0, 12]

        for number, custom in enumerate(self.ibn_ops, 1):
            config = constant_values(custom.operands[13])
            self.assertEqual(len(config), 37)
            self.assertEqual(config[0], 1)
            self.assertEqual(config[2], 1)
            self.assertEqual(config[7], config[6])
            self.assertEqual(config[36], REWRITER.CONFIG_MAGIC)
            scratch = config[10] * config[4] * config[8] + config[7] * config[8]
            if config[1] == 3:
                scratch += (config[18] + 1) * config[7] * config[9]
                self.assertEqual(config[25:35], expected_residual[number])
            else:
                self.assertEqual(config[25:35], [0] * 10)
                self.assertEqual(config[24], config[23])
            self.assertEqual(config[24], expected_final_zp[number - 1])
            self.assertEqual(config[35], scratch)
            self.assertEqual(config[35], expected_scratch[number - 1])
            self.assertEqual(str(custom.results[1].type), f"tensor<{scratch}xi8>")

    def test_plan_json_schema_is_compatible_with_loader_fields(self):
        encoded = json.dumps(REWRITER.plan_json(self.plan))
        decoded = json.loads(encoded)

        self.assertEqual(decoded["schema_version"], 2)
        self.assertEqual(decoded["mode"], "auto")
        self.assertEqual(decoded["full_intermediate_bytes"], 49152)
        self.assertEqual(decoded["segment_bytes"], 8448)
        self.assertEqual(decoded["saved_intermediate_bytes"], 40704)

    def test_rejects_modified_stage_semantics(self):
        modified = replace_occurrence(
            self.source,
            "%350 = arith.maxsi %349, %c-128_i32 : i32",
            "%350 = arith.minsi %349, %c-128_i32 : i32",
            3,
        )

        self.assertLess(len(REWRITER.match_generic(modified).modules), 14)

    def test_rejects_modified_residual_semantics(self):
        modified = replace_occurrence(
            self.source,
            "%348 = arith.addi %in, %in_282 : i32",
            "%348 = arith.subi %in, %in_282 : i32",
            4,
        )

        self.assertLess(len(REWRITER.match_generic(modified).modules), 14)

    @unittest.skipUnless(shutil.which("iree-compile"), "iree-compile is required")
    def test_rewrites_fresh_iree_preprocessing_form(self):
        with tempfile.TemporaryDirectory() as directory:
            preprocessing = Path(directory) / "mcunet.preprocessing.mlir"
            subprocess.run(
                [
                    "iree-compile",
                    str(MODEL),
                    "--iree-hal-target-device=local",
                    "--iree-hal-local-target-device-backends=llvm-cpu",
                    "--iree-llvmcpu-target-triple=thumbv7em-none-eabi",
                    "--compile-to=preprocessing",
                    "--emit-mlir-bytecode=false",
                    "-o",
                    str(preprocessing),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            rewritten, plan = REWRITER.rewrite_generic(preprocessing.read_text(encoding="utf-8"))

        context = REWRITER.ir.Context()
        module = REWRITER.ir.Module.parse(rewritten, context=context)
        self.assertTrue(module.operation.verify())
        self.assertEqual(len(REWRITER.operations_named(module, "iree_linalg_ext.custom_op")), 14)
        self.assertEqual(plan.in_place_block_count, 7)

    def test_finalizes_all_thirteen_configured_ukernels(self):
        finalized, count = REWRITER.finalize_generic(configured_fixture())

        self.assertEqual(count, 13)
        self.assertEqual(finalized.count("hal.import.bitcode = true"), 13)
        self.assertEqual(finalized.count('test.keep = "yes"'), 13)
        self.assertNotIn("ukernel_descriptor", finalized)
        context = REWRITER.ir.Context()
        module = REWRITER.ir.Module.parse(finalized, context=context)
        self.assertTrue(module.operation.verify())


def synthetic_mixed_model() -> str:
    return """module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>, tosa.target_env = #tosa.target_env<specification_version = "1.0", level = none, profiles = [pro_int, pro_fp], extensions = [dynamic, doubleround]>} {
  util.global private @__device_0 = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-unknown", {cpu = "", cpu_features = "", data_layout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "thumbv7em-unknown-unknown-eabi-elf"}>]> : !hal.device
  util.func public @main(%arg0: tensor<1x6x6x2xi8> {ml_program.identifier = "input"}) -> tensor<49x5xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-1_i8 = arith.constant -1 : i8
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %padded = tensor.pad %arg0 low[%c0, %c1, %c1, %c0] high[%c0, %c1, %c1, %c0] {
    ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
      tensor.yield %c-1_i8 : i8
    } : tensor<1x6x6x2xi8> to tensor<1x8x8x2xi8>
    %w = arith.constant dense<[[[[-2, 3], [1, 0]], [[-1, 2], [0, 1]]], [[[1, -1], [2, 1]], [[0, -2], [1, 1]]], [[[0, 1], [-1, 2]], [[1, 0], [2, -1]]]]> : tensor<3x2x2x2xi8>
    %empty0 = tensor.empty() : tensor<2x2x2x3xi8>
    %wt = linalg.transpose ins(%w : tensor<3x2x2x2xi8>) outs(%empty0 : tensor<2x2x2x3xi8>) permutation = [1, 2, 3, 0]
    %bias = arith.constant dense<[100, -50, 25]> : tensor<3xi32>
    %acc_empty = tensor.empty() : tensor<1x7x7x3xi32>
    %acc = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bias : tensor<3xi32>) outs(%acc_empty : tensor<1x7x7x3xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x7x7x3xi32>
    %conv = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %wt, %c-1_i32, %c0_i32 : tensor<1x8x8x2xi8>, tensor<2x2x2x3xi8>, i32, i32) outs(%acc : tensor<1x7x7x3xi32>) -> tensor<1x7x7x3xi32>
    %mult = arith.constant dense<[1073741824, 1610612736, 1879048192]> : tensor<3xi32>
    %shift = arith.constant dense<[30, 31, 32]> : tensor<3xi8>
    %out_empty = tensor.empty() : tensor<1x7x7x3xi8>
    %conv8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv, %mult, %shift : tensor<1x7x7x3xi32>, tensor<3xi32>, tensor<3xi8>) outs(%out_empty : tensor<1x7x7x3xi8>) {
    ^bb0(%v: i32, %m: i32, %s: i8, %o: i8):
      %277 = tosa.apply_scale %v, %m, %s {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %278 = arith.addi %277, %c2_i32 : i32
      %279 = arith.maxsi %278, %c-128_i32 : i32
      %280 = arith.minsi %279, %c127_i32 : i32
      %281 = arith.trunci %280 : i32 to i8
      linalg.yield %281 : i8
    } -> tensor<1x7x7x3xi8>
    %collapsed = tensor.collapse_shape %conv8 [[0, 1, 2], [3]] : tensor<1x7x7x3xi8> into tensor<49x3xi8>
    %fc_w = arith.constant dense<[[1, 2, 3], [-1, 0, 2], [3, -2, 1], [0, 1, -1], [2, 2, 2]]> : tensor<5x3xi8>
    %fc_empty = tensor.empty() : tensor<3x5xi8>
    %fc_wt = linalg.transpose ins(%fc_w : tensor<5x3xi8>) outs(%fc_empty : tensor<3x5xi8>) permutation = [1, 0]
    %fc_bias = arith.constant dense<[10, -20, 30, -40, 50]> : tensor<5xi32>
    %fc_acc_empty = tensor.empty() : tensor<49x5xi32>
    %fc_acc = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%fc_bias : tensor<5xi32>) outs(%fc_acc_empty : tensor<49x5xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<49x5xi32>
    %fc = linalg.quantized_matmul ins(%collapsed, %fc_wt, %c-1_i32, %c0_i32 : tensor<49x3xi8>, tensor<3x5xi8>, i32, i32) outs(%fc_acc : tensor<49x5xi32>) -> tensor<49x5xi32>
    %fc_mult = arith.constant dense<[1073741824, 1610612736, 1879048192, 805306368, 1073741824]> : tensor<5xi32>
    %fc_shift = arith.constant dense<[30, 31, 32, 29, 30]> : tensor<5xi8>
    %fc_out_empty = tensor.empty() : tensor<49x5xi8>
    %fc8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%fc, %fc_mult, %fc_shift : tensor<49x5xi32>, tensor<5xi32>, tensor<5xi8>) outs(%fc_out_empty : tensor<49x5xi8>) {
    ^bb0(%v: i32, %m: i32, %s: i8, %o: i8):
      %277 = tosa.apply_scale %v, %m, %s {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %278 = arith.addi %277, %c2_i32 : i32
      %279 = arith.maxsi %278, %c-128_i32 : i32
      %280 = arith.minsi %279, %c127_i32 : i32
      %281 = arith.trunci %280 : i32 to i8
      linalg.yield %281 : i8
    } -> tensor<49x5xi8>
    util.return %fc8 : tensor<49x5xi8>
  }
}
"""



def synthetic_conv_ib_model() -> str:
    """An inverted bottleneck whose pointwise stages are 1x1 hwcf convs
    (conv1x1 -> depthwise -> conv1x1), the TFLite-import form."""
    return """module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>, tosa.target_env = #tosa.target_env<specification_version = "1.0", level = none, profiles = [pro_int, pro_fp], extensions = [dynamic, doubleround]>} {
  util.global private @__device_0 = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-unknown", {cpu = "", cpu_features = "", data_layout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "thumbv7em-unknown-unknown-eabi-elf"}>]> : !hal.device
  util.func public @main(%arg0: tensor<1x6x6x2xi8> {ml_program.identifier = "input"}) -> tensor<1x6x6x2xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-128_i8 = arith.constant -128 : i8
    %c-128_i32 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %we = arith.constant dense<[[[[1, 2]]], [[[-1, 1]]], [[[2, 0]]]]> : tensor<3x1x1x2xi8>
    %we_empty = tensor.empty() : tensor<1x1x2x3xi8>
    %we_t = linalg.transpose ins(%we : tensor<3x1x1x2xi8>) outs(%we_empty : tensor<1x1x2x3xi8>) permutation = [1, 2, 3, 0]
    %be = arith.constant dense<[10, -20, 30]> : tensor<3xi32>
    %ae_empty = tensor.empty() : tensor<1x6x6x3xi32>
    %ae = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%be : tensor<3xi32>) outs(%ae_empty : tensor<1x6x6x3xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x6x6x3xi32>
    %exp = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %we_t, %c2_i32, %c0_i32 : tensor<1x6x6x2xi8>, tensor<1x1x2x3xi8>, i32, i32) outs(%ae : tensor<1x6x6x3xi32>) -> tensor<1x6x6x3xi32>
    %me = arith.constant dense<[1073741824, 1610612736, 1879048192]> : tensor<3xi32>
    %se = arith.constant dense<[30, 31, 32]> : tensor<3xi8>
    %oe_empty = tensor.empty() : tensor<1x6x6x3xi8>
    %exp8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%exp, %me, %se : tensor<1x6x6x3xi32>, tensor<3xi32>, tensor<3xi8>) outs(%oe_empty : tensor<1x6x6x3xi8>) {
    ^bb0(%v: i32, %m: i32, %s: i8, %o: i8):
      %277 = tosa.apply_scale %v, %m, %s {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %278 = arith.addi %277, %c-128_i32 : i32
      %279 = arith.maxsi %278, %c-128_i32 : i32
      %280 = arith.minsi %279, %c127_i32 : i32
      %281 = arith.trunci %280 : i32 to i8
      linalg.yield %281 : i8
    } -> tensor<1x6x6x3xi8>
    %padded = tensor.pad %exp8 low[%c0, %c1, %c1, %c0] high[%c0, %c1, %c1, %c0] {
    ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
      tensor.yield %c-128_i8 : i8
    } : tensor<1x6x6x3xi8> to tensor<1x8x8x3xi8>
    %wd = arith.constant dense<[[[[1], [2], [1]], [[0], [-1], [1]], [[1], [0], [2]]], [[[2], [1], [0]], [[1], [1], [2]], [[0], [2], [1]]], [[[1], [0], [1]], [[2], [2], [0]], [[1], [1], [2]]]]> : tensor<3x3x3x1xi8>
    %d_empty = tensor.empty() : tensor<1x6x6x3x1xi32>
    %fill = linalg.fill ins(%c0_i32 : i32) outs(%d_empty : tensor<1x6x6x3x1xi32>) -> tensor<1x6x6x3x1xi32>
    %dw = linalg.depthwise_conv_2d_nhwc_hwcm_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %wd, %c-128_i32, %c0_i32 : tensor<1x8x8x3xi8>, tensor<3x3x3x1xi8>, i32, i32) outs(%fill : tensor<1x6x6x3x1xi32>) -> tensor<1x6x6x3x1xi32>
    %collapsed = tensor.collapse_shape %dw [[0], [1], [2], [3, 4]] : tensor<1x6x6x3x1xi32> into tensor<1x6x6x3xi32>
    %bd = arith.constant dense<[5, -5, 5]> : tensor<3xi32>
    %add_empty = tensor.empty() : tensor<1x6x6x3xi32>
    %bias_d = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bd, %collapsed : tensor<3xi32>, tensor<1x6x6x3xi32>) outs(%add_empty : tensor<1x6x6x3xi32>) {
    ^bb0(%in: i32, %in_2: i32, %out: i32):
      %277 = arith.addi %in, %in_2 : i32
      linalg.yield %277 : i32
    } -> tensor<1x6x6x3xi32>
    %md = arith.constant dense<[1610612736, 1073741824, 1610612736]> : tensor<3xi32>
    %sd = arith.constant dense<[31, 30, 31]> : tensor<3xi8>
    %od_empty = tensor.empty() : tensor<1x6x6x3xi8>
    %dw8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bias_d, %md, %sd : tensor<1x6x6x3xi32>, tensor<3xi32>, tensor<3xi8>) outs(%od_empty : tensor<1x6x6x3xi8>) {
    ^bb0(%v: i32, %m: i32, %s: i8, %o: i8):
      %277 = tosa.apply_scale %v, %m, %s {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %278 = arith.addi %277, %c2_i32 : i32
      %279 = arith.maxsi %278, %c-128_i32 : i32
      %280 = arith.minsi %279, %c127_i32 : i32
      %281 = arith.trunci %280 : i32 to i8
      linalg.yield %281 : i8
    } -> tensor<1x6x6x3xi8>
    %wp = arith.constant dense<[[[[1, 0, 2]]], [[[-1, 1, 1]]]]> : tensor<2x1x1x3xi8>
    %wp_empty = tensor.empty() : tensor<1x1x3x2xi8>
    %wp_t = linalg.transpose ins(%wp : tensor<2x1x1x3xi8>) outs(%wp_empty : tensor<1x1x3x2xi8>) permutation = [1, 2, 3, 0]
    %bp = arith.constant dense<[100, -100]> : tensor<2xi32>
    %ap_empty = tensor.empty() : tensor<1x6x6x2xi32>
    %ap = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bp : tensor<2xi32>) outs(%ap_empty : tensor<1x6x6x2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x6x6x2xi32>
    %proj = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%dw8, %wp_t, %c2_i32, %c0_i32 : tensor<1x6x6x3xi8>, tensor<1x1x3x2xi8>, i32, i32) outs(%ap : tensor<1x6x6x2xi32>) -> tensor<1x6x6x2xi32>
    %mp = arith.constant dense<[1610612736, 1073741824]> : tensor<2xi32>
    %sp = arith.constant dense<[31, 30]> : tensor<2xi8>
    %op_empty = tensor.empty() : tensor<1x6x6x2xi8>
    %out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%proj, %mp, %sp : tensor<1x6x6x2xi32>, tensor<2xi32>, tensor<2xi8>) outs(%op_empty : tensor<1x6x6x2xi8>) {
    ^bb0(%v: i32, %m: i32, %s: i8, %o: i8):
      %277 = tosa.apply_scale %v, %m, %s {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %278 = arith.addi %277, %c-128_i32 : i32
      %279 = arith.maxsi %278, %c-128_i32 : i32
      %280 = arith.minsi %279, %c127_i32 : i32
      %281 = arith.trunci %280 : i32 to i8
      linalg.yield %281 : i8
    } -> tensor<1x6x6x2xi8>
    util.return %out : tensor<1x6x6x2xi8>
  }
}
"""


def synthetic_arith_rescale_model() -> str:
    """One conv2d whose rescale uses the expanded 64-bit DOUBLE_ROUND arith
    form (IREE's ApplyScaleGenericOpConverter lowering, as produced for
    TFLite-imported models like LeNet5)."""
    return """module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>, tosa.target_env = #tosa.target_env<specification_version = "1.0", level = none, profiles = [pro_int, pro_fp], extensions = [dynamic, doubleround]>} {
  util.global private @__device_0 = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-unknown", {cpu = "", cpu_features = "", data_layout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "thumbv7em-unknown-unknown-eabi-elf"}>]> : !hal.device
  util.func public @main(%arg0: tensor<1x6x6x2xi8> {ml_program.identifier = "input"}) -> tensor<49x2xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-1_i8 = arith.constant -1 : i8
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i64 = arith.constant 1 : i64
    %c31_i32 = arith.constant 31 : i32
    %c1073741824_i64 = arith.constant 1073741824 : i64
    %c-1073741824_i64 = arith.constant -1073741824 : i64
    %padded = tensor.pad %arg0 low[%c0, %c1, %c1, %c0] high[%c0, %c1, %c1, %c0] {
    ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
      tensor.yield %c-1_i8 : i8
    } : tensor<1x6x6x2xi8> to tensor<1x8x8x2xi8>
    %w = arith.constant dense<[[[[-2, 3], [1, 0]], [[-1, 2], [0, 1]]], [[[1, -1], [2, 1]], [[0, -2], [1, 1]]]]> : tensor<2x2x2x2xi8>
    %empty0 = tensor.empty() : tensor<2x2x2x2xi8>
    %wt = linalg.transpose ins(%w : tensor<2x2x2x2xi8>) outs(%empty0 : tensor<2x2x2x2xi8>) permutation = [1, 2, 3, 0]
    %bias = arith.constant dense<[100, -50]> : tensor<2xi32>
    %acc_empty = tensor.empty() : tensor<1x7x7x2xi32>
    %acc = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bias : tensor<2xi32>) outs(%acc_empty : tensor<1x7x7x2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x7x7x2xi32>
    %conv = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %wt, %c-1_i32, %c0_i32 : tensor<1x8x8x2xi8>, tensor<2x2x2x2xi8>, i32, i32) outs(%acc : tensor<1x7x7x2xi32>) -> tensor<1x7x7x2xi32>
    %mult = arith.constant dense<[1073741824, 1610612736]> : tensor<2xi32>
    %shift = arith.constant dense<[30, 31]> : tensor<2xi8>
    %out_empty = tensor.empty() : tensor<1x7x7x2xi8>
    %out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv, %mult, %shift : tensor<1x7x7x2xi32>, tensor<2xi32>, tensor<2xi8>) outs(%out_empty : tensor<1x7x7x2xi8>) {
    ^bb0(%in: i32, %in_27: i32, %in_28: i8, %out: i8):
      %51 = arith.extui %in_28 : i8 to i32
      %52 = arith.extsi %in : i32 to i64
      %53 = arith.extsi %in_27 : i32 to i64
      %54 = arith.muli %52, %53 : i64
      %55 = arith.extui %in_28 : i8 to i64
      %56 = arith.shli %c1_i64, %55 : i64
      %57 = arith.shrui %56, %c1_i64 : i64
      %58 = arith.addi %54, %57 : i64
      %59 = arith.cmpi sge, %in, %c0_i32 : i32
      %60 = arith.select %59, %c1073741824_i64, %c-1073741824_i64 : i64
      %61 = arith.addi %60, %58 : i64
      %62 = arith.cmpi sgt, %51, %c31_i32 : i32
      %63 = arith.select %62, %61, %58 : i64
      %64 = arith.shrsi %63, %55 : i64
      %65 = arith.trunci %64 : i64 to i32
      %66 = arith.addi %65, %c2_i32 : i32
      %67 = arith.maxsi %66, %c-128_i32 : i32
      %68 = arith.minsi %67, %c127_i32 : i32
      %69 = arith.trunci %68 : i32 to i8
      linalg.yield %69 : i8
    } -> tensor<1x7x7x2xi8>
    %final = tensor.collapse_shape %out [[0, 1, 2], [3]] : tensor<1x7x7x2xi8> into tensor<49x2xi8>
    util.return %final : tensor<49x2xi8>
  }
}
"""


class VmcuGenericRewriteTests(unittest.TestCase):
    def setUp(self):
        self.source = synthetic_mixed_model()
        self.matched = REWRITER.match_generic(self.source)

    def test_matches_conv_and_fc_modules(self):
        kinds = [module.kind for module in self.matched.modules]
        self.assertEqual(kinds, ["conv2d", "fc"])
        conv, fc = self.matched.modules
        self.assertEqual(conv.kernel_name, REWRITER.CONV2D_KERNEL_NAME)
        self.assertEqual(fc.kernel_name, REWRITER.FC_KERNEL_NAME)
        self.assertEqual(conv.scratch_bytes, 2 * 6 * 2)
        self.assertEqual(fc.scratch_bytes, 5)
        self.assertEqual(conv.config[2:10], (1, 6, 6, 2, 7, 7, 3, 2))
        self.assertEqual(fc.config[2:5], (49, 3, 5))
        self.assertEqual(conv.config[19:21], (-1, 2))
        self.assertEqual(fc.config[20:22], (-1, 2))
        self.assertEqual(conv.config[36], REWRITER.CONFIG_MAGIC)
        self.assertEqual(fc.config[36], REWRITER.CONFIG_MAGIC)

    def test_plan_reports_combined_footprint(self):
        plan = self.matched.plan
        self.assertEqual(plan.mode, "auto")
        self.assertEqual(plan.block_count, 2)
        self.assertEqual(plan.residual_block_count, 0)
        self.assertEqual(plan.max_segment_bytes, 24)
        self.assertEqual(plan.saved_intermediate_bytes, plan.standard_peak_intermediate_bytes - 24)

    def test_rewrite_leaves_unmatched_ops_and_replaces_matched(self):
        rewritten, plan = REWRITER.rewrite_generic(self.source)
        self.assertEqual(plan.block_count, 2)
        self.assertEqual(rewritten.count("iree_linalg_ext.custom_op"), 2)
        self.assertEqual(rewritten.count(REWRITER.CONV), 0)
        self.assertEqual(rewritten.count(REWRITER.MATMUL), 0)
        self.assertEqual(rewritten.count("linalg.transpose"), 1)
        self.assertEqual(rewritten.count(REWRITER.CONV2D_KERNEL_NAME), 1)
        self.assertEqual(rewritten.count(REWRITER.FC_KERNEL_NAME), 1)
        self.assertEqual(rewritten.count(REWRITER.GENERIC_BITCODE_PATH), 2)

    def test_rejects_out_of_range_shift_at_compile_time(self):
        source = synthetic_mixed_model().replace(
            "dense<[30, 31, 32]> : tensor<3xi8>",
            "dense<[30, 31, 0]> : tensor<3xi8>",
        )
        context = REWRITER.ir.Context()
        module = REWRITER.ir.Module.parse(source, context=context)
        direct = REWRITER.direct_operations([f for f in REWRITER.operations_named(module, "util.func")][0])
        conv = [candidate for candidate in direct if candidate.name == REWRITER.CONV][0]
        with self.assertRaisesRegex(ValueError, "shift must be in \[1, 62\]"):
            REWRITER.match_conv(conv, "conv2d")
        self.assertEqual(
            [module.kind for module in REWRITER.match_generic(source).modules],
            ["fc"],
        )

    def test_rejects_out_of_range_zero_point_at_compile_time(self):
        source = synthetic_mixed_model().replace(
            "%c2_i32 = arith.constant 2 : i32",
            "%c2_i32 = arith.constant 200 : i32",
        )
        context = REWRITER.ir.Context()
        module = REWRITER.ir.Module.parse(source, context=context)
        direct = REWRITER.direct_operations([f for f in REWRITER.operations_named(module, "util.func")][0])
        conv = [candidate for candidate in direct if candidate.name == REWRITER.CONV][0]
        with self.assertRaisesRegex(ValueError, "zero point must be in \[-128, 127\]"):
            REWRITER.match_conv(conv, "conv2d")
        with self.assertRaisesRegex(ValueError, "no vMCU-compatible subgraphs found"):
            REWRITER.match_generic(source)

    def test_rejects_model_without_any_pattern(self):
        empty = """module { util.func public @main(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi8>
  util.return %0 : tensor<4xi8>
} }"""
        with self.assertRaises(ValueError):
            REWRITER.match_generic(empty)

    def test_matches_conv_pointwise_inverted_bottleneck(self):
        """A pointwise 1x1 convolution (hwcf) can serve as the expansion and
        projection stages of an inverted bottleneck (TFLite-imported models
        lower pointwise convs to linalg.conv_2d_nhwc_hwcf_q with kernel 1)."""
        source = synthetic_conv_ib_model()
        context = REWRITER.ir.Context()
        module = REWRITER.ir.Module.parse(source, context=context)
        direct = REWRITER.direct_operations([f for f in REWRITER.operations_named(module, "util.func")][0])
        index = {c: i for i, c in enumerate(direct)}
        convs = [c for c in direct if c.name == REWRITER.CONV]
        dws = [c for c in direct if c.name == REWRITER.DEPTHWISE]
        dw = dws[0]
        before = [c for c in convs if index[c] < index[dw]]
        after = [c for c in convs if index[c] > index[dw]]
        block = REWRITER.match_ib_block(dw, before[-1], after[0], direct, index, 1)
        self.assertIsNone(block.residual)
        self.assertEqual(
            tuple(REWRITER.ir.RankedTensorType(block.expansion.weight.type).shape),
            (3, 2),
        )
        self.assertEqual(
            tuple(REWRITER.ir.RankedTensorType(block.projection.weight.type).shape),
            (2, 3),
        )

    def test_matches_expanded_arith_rescale_conv(self):
        matched = REWRITER.match_generic(synthetic_arith_rescale_model())
        self.assertEqual([module.kind for module in matched.modules], ["conv2d"])
        module = matched.modules[0]
        self.assertEqual(module.config[2:9], (1, 6, 6, 2, 7, 7, 2))
        rewritten, plan = REWRITER.rewrite_generic(synthetic_arith_rescale_model())
        self.assertEqual(plan.block_count, 1)
        self.assertEqual(rewritten.count("iree_linalg_ext.custom_op"), 1)
        self.assertEqual(rewritten.count(REWRITER.CONV2D_KERNEL_NAME), 1)


if __name__ == "__main__":
    unittest.main()
