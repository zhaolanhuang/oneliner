module {
  func.func @main(%input: tensor<4x4xi8>) -> tensor<4x3xi8> {
    %c0_i32 = arith.constant 0 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %weight0 = arith.constant dense<[
      [1, 0, -1, 2, 1],
      [0, 1, 1, -1, 2],
      [1, 1, 0, 1, -1],
      [-1, 2, 1, 0, 1]
    ]> : tensor<4x5xi8>
    %weight1 = arith.constant dense<[
      [1, 0, -1],
      [0, 1, 1],
      [1, -1, 0],
      [-1, 1, 1],
      [1, 1, -1]
    ]> : tensor<5x3xi8>

    %acc0_empty = tensor.empty() : tensor<4x5xi32>
    %acc0_init = linalg.fill ins(%c0_i32 : i32) outs(%acc0_empty : tensor<4x5xi32>) -> tensor<4x5xi32>
    %acc0 = linalg.quantized_matmul ins(%input, %weight0, %c0_i32, %c0_i32 : tensor<4x4xi8>, tensor<4x5xi8>, i32, i32) outs(%acc0_init : tensor<4x5xi32>) -> tensor<4x5xi32>
    %mid_empty = tensor.empty() : tensor<4x5xi8>
    %mid = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%acc0 : tensor<4x5xi32>) outs(%mid_empty : tensor<4x5xi8>) {
    ^bb0(%value: i32, %unused: i8):
      %lowered = arith.maxsi %value, %c-128_i32 : i32
      %clamped = arith.minsi %lowered, %c127_i32 : i32
      %result = arith.trunci %clamped : i32 to i8
      linalg.yield %result : i8
    } -> tensor<4x5xi8>

    %acc1_empty = tensor.empty() : tensor<4x3xi32>
    %acc1_init = linalg.fill ins(%c0_i32 : i32) outs(%acc1_empty : tensor<4x3xi32>) -> tensor<4x3xi32>
    %acc1 = linalg.quantized_matmul ins(%mid, %weight1, %c0_i32, %c0_i32 : tensor<4x5xi8>, tensor<5x3xi8>, i32, i32) outs(%acc1_init : tensor<4x3xi32>) -> tensor<4x3xi32>
    %output_empty = tensor.empty() : tensor<4x3xi8>
    %output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%acc1 : tensor<4x3xi32>) outs(%output_empty : tensor<4x3xi8>) {
    ^bb0(%value: i32, %unused: i8):
      %lowered = arith.maxsi %value, %c-128_i32 : i32
      %clamped = arith.minsi %lowered, %c127_i32 : i32
      %result = arith.trunci %clamped : i32 to i8
      linalg.yield %result : i8
    } -> tensor<4x3xi8>
    return %output : tensor<4x3xi8>
  }
}
