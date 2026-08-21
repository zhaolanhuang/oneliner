# vMCU auto mode QEMU validation (Cortex-M4 and Cortex-M7)

This example validates the vMCU `auto` mode over the full MCUNet graph: the
13 inverted bottlenecks run as fused pointwise-depthwise-pointwise ukernels,
the stem 3x3 convolution runs as a single-layer 2D convolution ukernel, and
the fully connected head stays on IREE codegen. Every subgraph that matches a
vMCU pattern is lowered; everything else falls back to the standard IREE
lowering.

Build and run the vMCU path from the repository root:

```sh
PATH="$PWD/.venv/bin:$PATH" \
  cargo build --manifest-path examples/qemu-vmcu-auto/Cargo.toml \
    --release --target thumbv7em-none-eabi

# Cortex-M4
qemu-system-arm -machine mps2-an386 -cpu cortex-m4 -nographic \
  -semihosting-config enable=on,target=native \
  -kernel examples/qemu-vmcu-auto/target/thumbv7em-none-eabi/release/qemu-vmcu-auto

# Cortex-M7
qemu-system-arm -machine mps2-an500 -cpu cortex-m7 -nographic \
  -semihosting-config enable=on,target=native \
  -kernel examples/qemu-vmcu-auto/target/thumbv7em-none-eabi/release/qemu-vmcu-auto
```

Both machines fill the `1x64x64x3` input with `7` and require the exact output
`[4, -5]` (the IREE x86 reference for this model).

The linker grants 512 KiB RAM so correctness is independent of the previous
artificial 128 KiB limit. The reported segment is logical dispatch scratch;
measure linked data and stack high-water marks before claiming physical peak
RAM savings.