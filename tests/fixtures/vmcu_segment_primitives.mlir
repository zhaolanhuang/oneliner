module {
  util.func public @segment_roundtrip(%input: tensor<10xi8>) -> tensor<10xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c10 = arith.constant 10 : index
    %c12 = arith.constant 12 : index
    // The padding value is the input tensor's affine zero-point, never integer 0.
    %input_zp = arith.constant -7 : i8
    %scratch_empty = tensor.empty() : tensor<3x4xi8>
    %scratch_init = linalg.fill ins(%input_zp : i8) outs(%scratch_empty : tensor<3x4xi8>) -> tensor<3x4xi8>
    // Carry the complete static segment state through the loop. The final two
    // logical lanes are masked and retain input_zp.
    %scratch = scf.for %i = %c0 to %c12 step %c1 iter_args(%state = %scratch_init) -> tensor<3x4xi8> {
      %valid = arith.cmpi ult, %i, %c10 : index
      %value = scf.if %valid -> i8 {
        %loaded = tensor.extract %input[%i] : tensor<10xi8>
        scf.yield %loaded : i8
      } else {
        scf.yield %input_zp : i8
      }
      %segment = arith.divui %i, %c4 : index
      // remui is the modulo-address primitive for a fixed four-lane segment.
      %lane = arith.remui %i, %c4 : index
      %updated = tensor.insert %value into %state[%segment, %lane] : tensor<3x4xi8>
      scf.yield %updated : tensor<3x4xi8>
    }
    %output_empty = tensor.empty() : tensor<10xi8>
    %output = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%output_empty : tensor<10xi8>) {
    ^bb0(%unused: i8):
      %i = linalg.index 0 : index
      %segment = arith.divui %i, %c4 : index
      %lane = arith.remui %i, %c4 : index
      %value = tensor.extract %scratch[%segment, %lane] : tensor<3x4xi8>
      linalg.yield %value : i8
    } -> tensor<10xi8>
    util.return %output : tensor<10xi8>
  }
}
