# QEMU CMSIS-NN example

Validates the Oneliner CMSIS-NN ukernels end-to-end on QEMU's Cortex-M
machines, without real hardware.

## Prerequisites

- `qemu-system-arm`
- The Oneliner toolchain (`iree-compile`, `tosa-converter-for-tflite`)
- Rust targets: `thumbv7m-none-eabi` (M3); the M33/MVE builds use the
  built-in `thumbv8m.main-none-eabihf` target / a custom target JSON.

## Cortex-M3 (mps2-an385) — validated

Both models pass with `cmsis_nn = true`:

```sh
cargo build --release                          # LeNet5
qemu-system-arm -machine mps2-an385 -cpu cortex-m3 -nographic \
  -semihosting -kernel target/thumbv7m-none-eabi/release/qemu-cmsis-nn

cargo build --release --features mcunet        # MCUNet visual wake word
qemu-system-arm -machine mps2-an385 -cpu cortex-m3 -nographic \
  -semihosting -kernel target/thumbv7m-none-eabi/release/qemu-cmsis-nn
```

The M3 run exercises the non-DSP CMSIS-NN Conv2D, depthwise Conv2D, MaxPool,
average pool, and fully connected ukernels, plus the full macro -> IREE ->
prebuilt-bitcode toolchain. LeNet5 output matches the host reference exactly;
MCUNet outputs `[4, -5]` as validated on hardware.

## Cortex-M33 (mps2-an521)

Build with the built-in target: `cargo build --release --target
thumbv8m.main-none-eabihf --features m33`. This currently hard-faults in QEMU
during boot (MPS2 TrustZone SSE-200 setup); the binary itself links and loads.

## Cortex-M55/MVE (mps3-an547) — boots, but QEMU MVE results are untrustworthy

MVE requires a custom Rust target (no shipped `thumbv8m.1m` target) plus
`-Z build-std=core -Z json-target-spec`:

```sh
RUSTC_BOOTSTRAP=1 cargo -Z build-std=core -Z json-target-spec build --release \
  --target thumbv8m.main-none-eabihf.json --features mve
qemu-system-arm -machine mps3-an547 -nographic -semihosting \
  -kernel target/thumbv8m.main-none-eabihf/release/qemu-cmsis-nn
```

The MVE build boots, links the MVE prebuilt bitcode, and runs both models, but
the numerical results are wrong. The same wrong results occur with
`cmsis_nn = false`, i.e. with the standard IREE codegen and no ukernels at
all, so the divergence comes from the base IREE/LLVM MVE (Helium) compilation
or QEMU 8.2's MVE execution model — not from the CMSIS-NN ukernels. Validate
MVE numerics on real M55 hardware instead.
