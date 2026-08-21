# LeNet5 vMCU auto mode QEMU validation (Cortex-M4 and Cortex-M7)

This example validates the vMCU `auto` mode on the bundled LeNet5 model
(`lenet5_quantized.tflite`): the two 5x5 convolutions run as single-layer
2D convolution ukernels, the three fully-connected layers (lowered to 1x1
convolutions by the TFLite importer) run as the same 2D convolution ukernel
with the folded per-tensor scales, and the max pooling stays on IREE codegen.

The model has float input/output boundaries; the internal int8 subgraphs are
what the vMCU lowering matches, demonstrating that matched subgraphs inside
a partially float graph are fused while everything else falls back to IREE.

Build and run the vMCU path from the repository root:

```sh
PATH="$PWD/.venv/bin:$PATH" \
  cargo build --manifest-path examples/qemu-lenet5-auto/Cargo.toml \
    --release --target thumbv7em-none-eabi

# Cortex-M4
qemu-system-arm -machine mps2-an386 -cpu cortex-m4 -nographic \
  -semihosting-config enable=on,target=native \
  -kernel examples/qemu-lenet5-auto/target/thumbv7em-none-eabi/release/qemu-lenet5-auto

# Cortex-M7
qemu-system-arm -machine mps2-an500 -cpu cortex-m7 -nographic \
  -semihosting-config enable=on,target=native \
  -kernel examples/qemu-lenet5-auto/target/thumbv7em-none-eabi/release/qemu-lenet5-auto
```

Both machines fill the `1x1x28x28` float input with `7.0` and require the
exact output `[0.11666615, 0.11666615, 0.13124943, 0.68541366, 0.0,
0.36458173, 0.0, 0.0, 1.2104113, 0.16041596]` (the standard IREE lowering
reference for this input).

The reported segment is logical dispatch scratch; measure linked data and
stack high-water marks before claiming physical peak RAM savings.