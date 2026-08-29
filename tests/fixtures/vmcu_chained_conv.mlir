module {
  util.func public @chained_conv(%input: tensor<1x2x2x2xi8>) -> tensor<1x2x2x2xi8> {
    %input_zp = arith.constant -3 : i32
    %first_weight_zp = arith.constant 2 : i32
    %first_output_zp = arith.constant 5 : i32
    %second_weight_zp = arith.constant -1 : i32
    %second_output_zp = arith.constant -7 : i32
    %clamp_min = arith.constant -128 : i32
    %clamp_max = arith.constant 127 : i32

    %first_weight = arith.constant dense<1> : tensor<1x1x2x3xi8>
    %first_bias = arith.constant dense<[7, -4, 3]> : tensor<3xi32>
    %first_multiplier = arith.constant dense<[1073741824, 1073741824, 1073741824]> : tensor<3xi32>
    %first_shift = arith.constant dense<[31, 31, 31]> : tensor<3xi8>
    %first_acc_empty = tensor.empty() : tensor<1x2x2x3xi32>
    %first_biased = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%first_bias : tensor<3xi32>) outs(%first_acc_empty : tensor<1x2x2x3xi32>) {
    ^bb0(%bias: i32, %unused: i32):
      linalg.yield %bias : i32
    } -> tensor<1x2x2x3xi32>
    %first_acc = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %first_weight, %input_zp, %first_weight_zp : tensor<1x2x2x2xi8>, tensor<1x1x2x3xi8>, i32, i32)
      outs(%first_biased : tensor<1x2x2x3xi32>) -> tensor<1x2x2x3xi32>
    %first_output_empty = tensor.empty() : tensor<1x2x2x3xi8>
    %first_output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%first_acc, %first_multiplier, %first_shift : tensor<1x2x2x3xi32>, tensor<3xi32>, tensor<3xi8>) outs(%first_output_empty : tensor<1x2x2x3xi8>) {
    ^bb0(%value: i32, %multiplier: i32, %shift: i8, %unused: i8):
      %scaled = tosa.apply_scale %value, %multiplier, %shift {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %offset = arith.addi %scaled, %first_output_zp : i32
      %lower = arith.maxsi %offset, %clamp_min : i32
      %upper = arith.minsi %lower, %clamp_max : i32
      %result = arith.trunci %upper : i32 to i8
      linalg.yield %result : i8
    } -> tensor<1x2x2x3xi8>

    %second_weight = arith.constant dense<-2> : tensor<1x1x3x2xi8>
    %second_bias = arith.constant dense<[11, -13]> : tensor<2xi32>
    %second_multiplier = arith.constant dense<[1073741824, 1073741824]> : tensor<2xi32>
    %second_shift = arith.constant dense<[31, 31]> : tensor<2xi8>
    %second_acc_empty = tensor.empty() : tensor<1x2x2x2xi32>
    %second_biased = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%second_bias : tensor<2xi32>) outs(%second_acc_empty : tensor<1x2x2x2xi32>) {
    ^bb0(%bias: i32, %unused: i32):
      linalg.yield %bias : i32
    } -> tensor<1x2x2x2xi32>
    %second_acc = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%first_output, %second_weight, %first_output_zp, %second_weight_zp : tensor<1x2x2x3xi8>, tensor<1x1x3x2xi8>, i32, i32)
      outs(%second_biased : tensor<1x2x2x2xi32>) -> tensor<1x2x2x2xi32>
    %second_output_empty = tensor.empty() : tensor<1x2x2x2xi8>
    %second_output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%second_acc, %second_multiplier, %second_shift : tensor<1x2x2x2xi32>, tensor<2xi32>, tensor<2xi8>) outs(%second_output_empty : tensor<1x2x2x2xi8>) {
    ^bb0(%value: i32, %multiplier: i32, %shift: i8, %unused: i8):
      %scaled = tosa.apply_scale %value, %multiplier, %shift {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %offset = arith.addi %scaled, %second_output_zp : i32
      %lower = arith.maxsi %offset, %clamp_min : i32
      %upper = arith.minsi %lower, %clamp_max : i32
      %result = arith.trunci %upper : i32 to i8
      linalg.yield %result : i8
    } -> tensor<1x2x2x2xi8>
    util.return %second_output : tensor<1x2x2x2xi8>
  }
}
