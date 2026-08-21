module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-unknown", {cpu = "", cpu_features = "", data_layout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "thumbv7em-unknown-unknown-eabi-elf"}>]> : !hal.device
  util.func public @main(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @main(%input0: tensor<4x4xi8>) -> (%output0: tensor<4x3xi8>)"}} {
    %cst = arith.constant dense<[[1, 0, -1], [0, 1, 1], [1, -1, 0], [-1, 1, 1], [1, 1, -1]]> : tensor<5x3xi8>
    %cst_0 = arith.constant dense<[[1, 0, -1, 2, 1], [0, 1, 1, -1, 2], [1, 1, 0, 1, -1], [-1, 2, 1, 0, 1]]> : tensor<4x5xi8>
    %c127_i32 = arith.constant 127 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<4x4xi8>
    %1 = tensor.empty() : tensor<4x5xi32>
    %2 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<4x5xi32>) -> tensor<4x5xi32>
    %3 = linalg.quantized_matmul ins(%0, %cst_0, %c0_i32, %c0_i32 : tensor<4x4xi8>, tensor<4x5xi8>, i32, i32) outs(%2 : tensor<4x5xi32>) -> tensor<4x5xi32>
    %4 = tensor.empty() : tensor<4x5xi8>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<4x5xi32>) outs(%4 : tensor<4x5xi8>) {
    ^bb0(%in: i32, %out: i8):
      %12 = arith.maxsi %in, %c-128_i32 : i32
      %13 = arith.minsi %12, %c127_i32 : i32
      %14 = arith.trunci %13 : i32 to i8
      linalg.yield %14 : i8
    } -> tensor<4x5xi8>
    %6 = tensor.empty() : tensor<4x3xi32>
    %7 = linalg.fill ins(%c0_i32 : i32) outs(%6 : tensor<4x3xi32>) -> tensor<4x3xi32>
    %8 = linalg.quantized_matmul ins(%5, %cst, %c0_i32, %c0_i32 : tensor<4x5xi8>, tensor<5x3xi8>, i32, i32) outs(%7 : tensor<4x3xi32>) -> tensor<4x3xi32>
    %9 = tensor.empty() : tensor<4x3xi8>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<4x3xi32>) outs(%9 : tensor<4x3xi8>) {
    ^bb0(%in: i32, %out: i8):
      %12 = arith.maxsi %in, %c-128_i32 : i32
      %13 = arith.minsi %12, %c127_i32 : i32
      %14 = arith.trunci %13 : i32 to i8
      linalg.yield %14 : i8
    } -> tensor<4x3xi8>
    %11 = hal.tensor.export %10 "output0" : tensor<4x3xi8> -> !hal.buffer_view
    util.return %11 : !hal.buffer_view
  }
}
