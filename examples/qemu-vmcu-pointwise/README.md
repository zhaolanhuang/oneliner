# vMCU pointwise-pair QEMU MVP

This example validates the first vMCU-inspired Oneliner lowering on an emulated
Cortex-M4. It fuses two static int8 pointwise matrix multiplications and keeps
only one row-sized intermediate segment.

The fixture has four rows and five intermediate channels. Standard tensor
execution materializes 20 intermediate bytes; the fused ukernel declares a
5-byte segment, saving 15 logical intermediate bytes before alignment.

Build and run the vMCU lowering from the repository root:

```sh
PATH="$PWD/.venv/bin:$PATH" \
  cargo build --manifest-path examples/qemu-vmcu-pointwise/Cargo.toml \
    --release --target thumbv7em-none-eabi

qemu-system-arm -machine mps2-an386 -cpu cortex-m4 -nographic \
  -semihosting-config enable=on,target=native \
  -kernel examples/qemu-vmcu-pointwise/target/thumbv7em-none-eabi/release/qemu-vmcu-pointwise
```

Build with `--features standard` to run the same model through normal IREE
lowering. Both variants use the same input and exact expected output.

The runner uses QEMU's `mps2-an386` Cortex-M4 machine and semihosting. The
linker script limits RAM to 128 KiB even though the emulated board exposes more.

This MVP validates compilation, segment-sized intermediate storage, and exact
results. For this tiny fixture, standard IREE reports a 64-byte transient arena;
the vMCU dispatch has no transient arena binding and uses the planned 5-byte
segment in dispatch-local storage. Alignment means this fixture is a correctness
test, not a physical-RAM benchmark. It does not yet implement vMCU's general circular pool, input/output
overlap, ILP planner, convolution halo scheduling, or inverted bottlenecks.
QEMU timing is not representative of physical Cortex-M4 performance.
