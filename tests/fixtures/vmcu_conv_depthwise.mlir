module {
  util.func public @synthetic_conv(%input: tensor<1x5x5x2xi8>) -> tensor<1x3x3x3xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %input_zp_i8 = arith.constant -3 : i8
    %input_zp = arith.constant -3 : i32
    %weight_zp = arith.constant 2 : i32
    %output_zp = arith.constant 5 : i32
    %clamp_min = arith.constant -128 : i32
    %clamp_max = arith.constant 127 : i32
    %weight = arith.constant dense<1> : tensor<3x3x2x3xi8>
    %bias = arith.constant dense<[7, -4, 3]> : tensor<3xi32>
    %multiplier = arith.constant dense<[1073741824, 1073741824, 1073741824]> : tensor<3xi32>
    %shift = arith.constant dense<[31, 31, 31]> : tensor<3xi8>
    %padded = tensor.pad %input low[%c0, %c1, %c1, %c0] high[%c0, %c1, %c1, %c0] {
    ^bb0(%n: index, %h: index, %w: index, %c: index):
      tensor.yield %input_zp_i8 : i8
    } : tensor<1x5x5x2xi8> to tensor<1x7x7x2xi8>
    %acc_empty = tensor.empty() : tensor<1x3x3x3xi32>
    %biased = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%bias : tensor<3xi32>) outs(%acc_empty : tensor<1x3x3x3xi32>) {
    ^bb0(%b: i32, %unused: i32):
      linalg.yield %b : i32
    } -> tensor<1x3x3x3xi32>
    %acc = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%padded, %weight, %input_zp, %weight_zp : tensor<1x7x7x2xi8>, tensor<3x3x2x3xi8>, i32, i32)
      outs(%biased : tensor<1x3x3x3xi32>) -> tensor<1x3x3x3xi32>
    %output_empty = tensor.empty() : tensor<1x3x3x3xi8>
    %output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%acc, %multiplier, %shift : tensor<1x3x3x3xi32>, tensor<3xi32>, tensor<3xi8>) outs(%output_empty : tensor<1x3x3x3xi8>) {
    ^bb0(%value: i32, %m: i32, %s: i8, %unused: i8):
      %scaled = tosa.apply_scale %value, %m, %s {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %shifted = arith.addi %scaled, %output_zp : i32
      %lower = arith.maxsi %shifted, %clamp_min : i32
      %upper = arith.minsi %lower, %clamp_max : i32
      %result = arith.trunci %upper : i32 to i8
      linalg.yield %result : i8
    } -> tensor<1x3x3x3xi8>
    util.return %output : tensor<1x3x3x3xi8>
  }

  util.func public @unrelated_depthwise_network(%input: tensor<1x4x4x3xi8>) -> tensor<1x4x4x3xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    %input_zp_i8 = arith.constant -9 : i8
    %input_zp = arith.constant -9 : i32
    %weight_zp = arith.constant -2 : i32
    %output_zp = arith.constant 11 : i32
    %clamp_min = arith.constant -128 : i32
    %clamp_max = arith.constant 127 : i32
    %weight = arith.constant dense<2> : tensor<3x3x3x1xi8>
    %bias = arith.constant dense<[13, -17, 19]> : tensor<3xi32>
    %multiplier = arith.constant dense<[1073741824, 1073741824, 1073741824]> : tensor<3xi32>
    %shift = arith.constant dense<[31, 31, 31]> : tensor<3xi8>
    %padded = tensor.pad %input low[%c0, %c1, %c1, %c0] high[%c0, %c1, %c1, %c0] {
    ^bb0(%n: index, %h: index, %w: index, %c: index):
      tensor.yield %input_zp_i8 : i8
    } : tensor<1x4x4x3xi8> to tensor<1x6x6x3xi8>
    %rank5_empty = tensor.empty() : tensor<1x4x4x3x1xi32>
    %rank5_zero = linalg.fill ins(%zero : i32) outs(%rank5_empty : tensor<1x4x4x3x1xi32>) -> tensor<1x4x4x3x1xi32>
    %rank5 = linalg.depthwise_conv_2d_nhwc_hwcm_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%padded, %weight, %input_zp, %weight_zp : tensor<1x6x6x3xi8>, tensor<3x3x3x1xi8>, i32, i32)
      outs(%rank5_zero : tensor<1x4x4x3x1xi32>) -> tensor<1x4x4x3x1xi32>
    %collapsed = tensor.collapse_shape %rank5 [[0], [1], [2], [3, 4]] : tensor<1x4x4x3x1xi32> into tensor<1x4x4x3xi32>
    %biased_empty = tensor.empty() : tensor<1x4x4x3xi32>
    %biased = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%bias, %collapsed : tensor<3xi32>, tensor<1x4x4x3xi32>) outs(%biased_empty : tensor<1x4x4x3xi32>) {
    ^bb0(%b: i32, %value: i32, %unused: i32):
      %sum = arith.addi %b, %value : i32
      linalg.yield %sum : i32
    } -> tensor<1x4x4x3xi32>
    %output_empty = tensor.empty() : tensor<1x4x4x3xi8>
    %output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%biased, %multiplier, %shift : tensor<1x4x4x3xi32>, tensor<3xi32>, tensor<3xi8>) outs(%output_empty : tensor<1x4x4x3xi8>) {
    ^bb0(%value: i32, %m: i32, %s: i8, %unused: i8):
      %scaled = tosa.apply_scale %value, %m, %s {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %shifted = arith.addi %scaled, %output_zp : i32
      %lower = arith.maxsi %shifted, %clamp_min : i32
      %upper = arith.minsi %lower, %clamp_max : i32
      %result = arith.trunci %upper : i32 to i8
      linalg.yield %result : i8
    } -> tensor<1x4x4x3xi8>
    util.return %output : tensor<1x4x4x3xi8>
  }
}
