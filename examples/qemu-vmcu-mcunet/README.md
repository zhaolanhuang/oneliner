# Full MCUNet vMCU QEMU validation

This example validates all 13 MCUNet inverted bottlenecks with streaming
pointwise-depthwise-pointwise ukernels. Expansion rows are cached in circular
halo buffers, and the seven residual blocks reuse their input allocation with
delayed row writeback.

Build and run the vMCU path from the repository root:

```sh
PATH="$PWD/.venv/bin:$PATH" \
  cargo build --manifest-path examples/qemu-vmcu-mcunet/Cargo.toml \
    --release --target thumbv7em-none-eabi

qemu-system-arm -machine mps2-an386 -cpu cortex-m4 -nographic \
  -semihosting-config enable=on,target=native \
  -kernel examples/qemu-vmcu-mcunet/target/thumbv7em-none-eabi/release/qemu-vmcu-mcunet
```

Build with `--features standard` for the normal IREE lowering. Both variants
fill the `1x64x64x3` input with `7` and require the exact output `[4, -5]`.

The linker grants 512 KiB RAM so correctness is independent of the previous
artificial 128 KiB limit. The reported segment is logical dispatch scratch;
measure linked data and stack high-water marks before claiming physical peak
RAM savings.
