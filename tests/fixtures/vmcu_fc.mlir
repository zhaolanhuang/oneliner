module {
  util.func public @main(%input: tensor<2x3xi8> {ml_program.identifier = "input"}) -> (tensor<2x4xi8> {ml_program.identifier = "output"}) {
    // All three boundaries use non-zero affine offsets.  The rewritten dot
    // product must center both input and weight values.
    %input_zp = arith.constant -3 : i32
    %weight_zp = arith.constant 2 : i32
    %output_zp = arith.constant 5 : i32
    %clamp_min = arith.constant -128 : i32
    %clamp_max = arith.constant 127 : i32
    %weight = arith.constant dense<[[1, 2, 3], [-2, 1, 4], [3, -1, 2], [1, 1, -1]]> : tensor<4x3xi8>
    %bias = arith.constant dense<[7, -4, 3, 1]> : tensor<4xi32>
    %multiplier = arith.constant dense<[1073741824, 1073741824, 1073741824, 1073741824]> : tensor<4xi32>
    %shift = arith.constant dense<[8, 8, 8, 8]> : tensor<4xi8>
    // Frontends emit output-major weights followed by this [1, 0] transpose for
    // linalg.quantized_matmul's [Cin, Cout] right-hand-side convention.
    %weight_init = tensor.empty() : tensor<3x4xi8>
    %transposed = linalg.transpose ins(%weight : tensor<4x3xi8>) outs(%weight_init : tensor<3x4xi8>) permutation = [1, 0]
    %accumulator_init = tensor.empty() : tensor<2x4xi32>
    // Broadcast the channel bias into the full i32 destination tensor. The
    // rewriter proves this body yields the bias argument unchanged.
    %biased = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%bias : tensor<4xi32>) outs(%accumulator_init : tensor<2x4xi32>) {
    ^bb0(%bias_value: i32, %unused: i32):
      linalg.yield %bias_value : i32
    } -> tensor<2x4xi32>
    %accumulator = linalg.quantized_matmul
      ins(%input, %transposed, %input_zp, %weight_zp : tensor<2x3xi8>, tensor<3x4xi8>, i32, i32)
      outs(%biased : tensor<2x4xi32>) -> tensor<2x4xi32>
    %output_init = tensor.empty() : tensor<2x4xi8>
    // Canonical per-channel DOUBLE_ROUND requantization, output offset, and
    // signed-int8 clamp. Any change to this scalar dataflow must reject safely.
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%accumulator, %multiplier, %shift : tensor<2x4xi32>, tensor<4xi32>, tensor<4xi8>) outs(%output_init : tensor<2x4xi8>) {
    ^bb0(%value: i32, %multiplier_value: i32, %shift_value: i8, %unused: i8):
      %scaled = tosa.apply_scale %value, %multiplier_value, %shift_value {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
      %shifted = arith.addi %scaled, %output_zp : i32
      %lower = arith.maxsi %shifted, %clamp_min : i32
      %clamped = arith.minsi %lower, %clamp_max : i32
      %result = arith.trunci %clamped : i32 to i8
      linalg.yield %result : i8
    } -> tensor<2x4xi8>
    util.return %output : tensor<2x4xi8>
  }
}
