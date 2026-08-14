# LeNet5 + CMSIS-NN on Ariel OS

This Rust example follows [`examples/ariel-os-iree`](../ariel-os-iree/) and runs the quantized LeNet5 model on a Cortex-M4F target. When `cmsis_nn = true`, the `#[model]` macro rewrites seven supported int8 operations to CMSIS-NN ukernels: two Conv2D, two MaxPool, and three fully connected forms. Other operations remain on IREE's normal LLVMCPU pipelines.

```rust
#[model(
    "../models/lenet5_quantized.tflite",
    backend = "iree",
    arena = "shared",
    cmsis_nn = true
)]
struct LeNet5;
```

The default documented board is `nrf52840dk`, which uses `thumbv7em-none-eabihf` and has a Cortex-M4F processor.

## Requirements

- Rust and Laze as required by Ariel OS v0.5.0
- IREE compiler 3.12 or newer
- `tosa-converter-for-tflite`
- LLVM `clang`, `llvm-link`, and `llvm-nm`
- Arm newlib headers (`libnewlib-arm-none-eabi` on Debian/Ubuntu)
- the initialized CMSIS-NN submodule
- an nRF52840 DK for on-device execution

Initialize dependencies from the repository root:

```sh
git submodule update --init third_party/cmsis-nn
```

Keep the Python/IREE environment active during every Laze build.

## Build

From this directory:

```sh
laze build -b nrf52840dk
```

The Rust macro imports `../models/lenet5_quantized.tflite`, compiles it for `thumbv7em-none-eabihf`, builds CMSIS-NN bitcode, links the generated static model object, and builds the complete Ariel OS firmware.

Set `cmsis_nn = false` or omit the option to compile the same model with IREE's standard LLVMCPU pipeline for differential testing.

Generated Cargo artifacts are under:

```text
build/bin/nrf52840dk/cargo/thumbv7em-none-eabihf/
```

## Run

Connect an nRF52840 DK and run:

```sh
laze build -b nrf52840dk run --bin oneliner-lenet5-cmsis-nn
```

The firmware fills the statically allocated `1x28x28x1` input tensor with `7.0`, runs LeNet5, reports inference time, and compares the ten outputs against the desktop reference values.

The model uses `arena = "shared"` and `ConstStaticCell` input storage so the model workspace and input do not consume the Ariel thread stack.

## Verify CMSIS-NN Lowering

The macro's generated `*.10.executable-targets.mlir` contains calls to:

```text
llvm.call @oneliner_cmsis_nn_conv_s8
llvm.call @oneliner_cmsis_nn_max_pool_s8
llvm.call @oneliner_cmsis_nn_fully_connected_s8
```

Those seven calls correspond to LeNet5's two 5x5 int8 convolutions, two max-pooling layers, and three fully connected layers. Quantization support operations and output conversion retain their normal IREE lowering.

The generated model object can contain unresolved C/compiler-runtime helpers such as `memcpy`, `roundf`, or ARM soft-float routines. Ariel OS resolves these during the final firmware link.
