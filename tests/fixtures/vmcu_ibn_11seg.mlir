module {
  util.func public @semantic_ibn(%input: tensor<1x4x4x2xi8>) -> tensor<1x4x4x2xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero_i32 = arith.constant 0 : i32
    %clamp_min = arith.constant -128 : i32
    %clamp_max = arith.constant 127 : i32

    // Expansion boundary: (-3 input zp, +2 weight zp) -> -5 output zp.
    %exp_input_zp = arith.constant -3 : i32
    %exp_weight_zp = arith.constant 2 : i32
    %exp_output_zp = arith.constant -5 : i32
    %exp_weight = arith.constant dense<1> : tensor<1x1x2x4xi8>
    %exp_bias = arith.constant dense<[7, -4, 3, 1]> : tensor<4xi32>
    %exp_multiplier = arith.constant dense<[1073741824, 1073741824, 1073741824, 1073741824]> : tensor<4xi32>
    %exp_shift = arith.constant dense<[31, 31, 31, 31]> : tensor<4xi8>
    %exp_acc_empty = tensor.empty() : tensor<1x4x4x4xi32>
    %exp_biased = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%exp_bias : tensor<4xi32>) outs(%exp_acc_empty : tensor<1x4x4x4xi32>) {
    ^bb0(%bias: i32, %unused: i32):
      linalg.yield %bias : i32
    } -> tensor<1x4x4x4xi32>
    %exp_acc = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %exp_weight, %exp_input_zp, %exp_weight_zp : tensor<1x4x4x2xi8>, tensor<1x1x2x4xi8>, i32, i32)
      outs(%exp_biased : tensor<1x4x4x4xi32>) -> tensor<1x4x4x4xi32>
    %exp_output_empty = tensor.empty() : tensor<1x4x4x4xi8>
    %exp_output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%exp_acc, %exp_multiplier, %exp_shift : tensor<1x4x4x4xi32>, tensor<4xi32>, tensor<4xi8>) outs(%exp_output_empty : tensor<1x4x4x4xi8>) {
    ^bb0(%value: i32, %multiplier: i32, %shift: i8, %unused: i8):
      %scaled = tosa.apply_scale %value, %multiplier, %shift {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %offset = arith.addi %scaled, %exp_output_zp : i32
      %lower = arith.maxsi %offset, %clamp_min : i32
      %upper = arith.minsi %lower, %clamp_max : i32
      %result = arith.trunci %upper : i32 to i8
      linalg.yield %result : i8
    } -> tensor<1x4x4x4xi8>

    // The 3x3 B patch is padded with expansion output zp (-5).
    %exp_output_zp_i8 = arith.constant -5 : i8
    %dw_padded = tensor.pad %exp_output low[%c0, %c1, %c1, %c0] high[%c0, %c1, %c1, %c0] {
    ^bb0(%n: index, %h: index, %w: index, %c: index):
      tensor.yield %exp_output_zp_i8 : i8
    } : tensor<1x4x4x4xi8> to tensor<1x6x6x4xi8>

    // Depthwise boundary: (-5 input zp, -2 weight zp) -> +6 output zp.
    %dw_input_zp = arith.constant -5 : i32
    %dw_weight_zp = arith.constant -2 : i32
    %dw_output_zp = arith.constant 6 : i32
    %dw_weight = arith.constant dense<2> : tensor<3x3x4x1xi8>
    %dw_bias = arith.constant dense<[13, -17, 19, -23]> : tensor<4xi32>
    %dw_multiplier = arith.constant dense<[1073741824, 1073741824, 1073741824, 1073741824]> : tensor<4xi32>
    %dw_shift = arith.constant dense<[31, 31, 31, 31]> : tensor<4xi8>
    %dw_rank5_empty = tensor.empty() : tensor<1x4x4x4x1xi32>
    %dw_rank5_zero = linalg.fill ins(%zero_i32 : i32) outs(%dw_rank5_empty : tensor<1x4x4x4x1xi32>) -> tensor<1x4x4x4x1xi32>
    %dw_rank5 = linalg.depthwise_conv_2d_nhwc_hwcm_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%dw_padded, %dw_weight, %dw_input_zp, %dw_weight_zp : tensor<1x6x6x4xi8>, tensor<3x3x4x1xi8>, i32, i32)
      outs(%dw_rank5_zero : tensor<1x4x4x4x1xi32>) -> tensor<1x4x4x4x1xi32>
    %dw_collapsed = tensor.collapse_shape %dw_rank5 [[0], [1], [2], [3, 4]] : tensor<1x4x4x4x1xi32> into tensor<1x4x4x4xi32>
    %dw_bias_empty = tensor.empty() : tensor<1x4x4x4xi32>
    %dw_biased = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%dw_bias, %dw_collapsed : tensor<4xi32>, tensor<1x4x4x4xi32>) outs(%dw_bias_empty : tensor<1x4x4x4xi32>) {
    ^bb0(%bias: i32, %value: i32, %unused: i32):
      %sum = arith.addi %bias, %value : i32
      linalg.yield %sum : i32
    } -> tensor<1x4x4x4xi32>
    %dw_output_empty = tensor.empty() : tensor<1x4x4x4xi8>
    %dw_output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%dw_biased, %dw_multiplier, %dw_shift : tensor<1x4x4x4xi32>, tensor<4xi32>, tensor<4xi8>) outs(%dw_output_empty : tensor<1x4x4x4xi8>) {
    ^bb0(%value: i32, %multiplier: i32, %shift: i8, %unused: i8):
      %scaled = tosa.apply_scale %value, %multiplier, %shift {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %offset = arith.addi %scaled, %dw_output_zp : i32
      %lower = arith.maxsi %offset, %clamp_min : i32
      %upper = arith.minsi %lower, %clamp_max : i32
      %result = arith.trunci %upper : i32 to i8
      linalg.yield %result : i8
    } -> tensor<1x4x4x4xi8>

    // Projection boundary: (+6 input zp, +3 weight zp) -> -3 output zp.
    %proj_input_zp = arith.constant 6 : i32
    %proj_weight_zp = arith.constant 3 : i32
    %proj_output_zp = arith.constant -3 : i32
    %proj_weight = arith.constant dense<-1> : tensor<1x1x4x2xi8>
    %proj_bias = arith.constant dense<[29, -31]> : tensor<2xi32>
    %proj_multiplier = arith.constant dense<[1073741824, 1073741824]> : tensor<2xi32>
    %proj_shift = arith.constant dense<[31, 31]> : tensor<2xi8>
    %proj_acc_empty = tensor.empty() : tensor<1x4x4x2xi32>
    %proj_biased = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%proj_bias : tensor<2xi32>) outs(%proj_acc_empty : tensor<1x4x4x2xi32>) {
    ^bb0(%bias: i32, %unused: i32):
      linalg.yield %bias : i32
    } -> tensor<1x4x4x2xi32>
    %proj_acc = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%dw_output, %proj_weight, %proj_input_zp, %proj_weight_zp : tensor<1x4x4x4xi8>, tensor<1x1x4x2xi8>, i32, i32)
      outs(%proj_biased : tensor<1x4x4x2xi32>) -> tensor<1x4x4x2xi32>
    %proj_output_empty = tensor.empty() : tensor<1x4x4x2xi8>
    %proj_output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%proj_acc, %proj_multiplier, %proj_shift : tensor<1x4x4x2xi32>, tensor<2xi32>, tensor<2xi8>) outs(%proj_output_empty : tensor<1x4x4x2xi8>) {
    ^bb0(%value: i32, %multiplier: i32, %shift: i8, %unused: i8):
      %scaled = tosa.apply_scale %value, %multiplier, %shift {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %offset = arith.addi %scaled, %proj_output_zp : i32
      %lower = arith.maxsi %offset, %clamp_min : i32
      %upper = arith.minsi %lower, %clamp_max : i32
      %result = arith.trunci %upper : i32 to i8
      linalg.yield %result : i8
    } -> tensor<1x4x4x2xi8>

    // Same-quantization residual keeps the projection boundary explicit.
    %residual_empty = tensor.empty() : tensor<1x4x4x2xi8>
    %residual = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%proj_output, %input : tensor<1x4x4x2xi8>, tensor<1x4x4x2xi8>) outs(%residual_empty : tensor<1x4x4x2xi8>) {
    ^bb0(%projection: i8, %skip: i8, %unused: i8):
      %p_i32 = arith.extsi %projection : i8 to i32
      %s_i32 = arith.extsi %skip : i8 to i32
      %p_center = arith.subi %p_i32, %proj_output_zp : i32
      %s_center = arith.subi %s_i32, %proj_output_zp : i32
      %sum = arith.addi %p_center, %s_center : i32
      %offset = arith.addi %sum, %proj_output_zp : i32
      %lower = arith.maxsi %offset, %clamp_min : i32
      %upper = arith.minsi %lower, %clamp_max : i32
      %result = arith.trunci %upper : i32 to i8
      linalg.yield %result : i8
    } -> tensor<1x4x4x2xi8>
    util.return %residual : tensor<1x4x4x2xi8>
  }
}
