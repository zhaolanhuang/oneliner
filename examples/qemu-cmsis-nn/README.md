# QEMU CMSIS-NN example

Validates the Oneliner CMSIS-NN ukernels end-to-end on QEMU's Cortex-M
machines, without real hardware.

## Prerequisites

- `qemu-system-arm`
- The Oneliner toolchain (`iree-compile`, `tosa-converter-for-tflite`)
- Rust targets: `thumbv6m-none-eabi`, `thumbv7m-none-eabi`,
  `thumbv7em-none-eabi[ hf]`, `thumbv8m.main-none-eabi[ hf]`. The MVE build
  uses a custom target JSON plus `-Z build-std`.

## Validated matrix

Each board/model combination below prints the inference output and exits
`0` on success (semihosting). "PASS" means the output matches the host /
hardware reference exactly.

| Core | Feature | Rust target | QEMU machine | LeNet5 | MCUNet |
|---|---|---|---|---|---|
| M0 | `m0` | `thumbv6m-none-eabi` | `microbit` | PASS | does not fit (256KB flash) |
| M3 | `m3` | `thumbv7m-none-eabi` | `mps2-an385` | PASS | PASS |
| M4 (soft) | `m4` | `thumbv7em-none-eabi` | `mps2-an386` | PASS | PASS |
| M4F (hard) | `m4f` | `thumbv7em-none-eabihf` | `olimex-stm32-h405` | PASS | PASS |
| M33 (soft) | `m33` | `thumbv8m.main-none-eabi` | `mps2-an505` | PASS | PASS |
| M33 (hard) | `m33` | `thumbv8m.main-none-eabihf` | `mps2-an505` | PASS | PASS |
| M55/MVE | `mve` | custom `thumbv8m.main-none-eabihf.json` (+mve) | `mps3-an547` | boots, qemu MVE numerics unreliable | same |

This exercises the Conv2D, depthwise Conv2D, MaxPool, average pool, and fully
connected ukernels (non-DSP on M0/M3, DSP on M4/M33, MVE on M55), plus the
full macro -> IREE -> prebuilt-bitcode toolchain.

## Commands

```sh
# M3 (default): LeNet5, then MCUNet
cargo build --release
cargo build --release --features mcunet
qemu-system-arm -machine mps2-an385 -cpu cortex-m3 -nographic -semihosting \
  -kernel target/thumbv7m-none-eabi/release/qemu-cmsis-nn

# Other cores: pick the feature and target, e.g.
cargo build --release --target thumbv7em-none-eabihf --features m4f
qemu-system-arm -machine olimex-stm32-h405 -nographic -semihosting \
  -kernel target/thumbv7em-none-eabihf/release/qemu-cmsis-nn

# M33 soft float
cargo build --release --target thumbv8m.main-none-eabi --features m33
qemu-system-arm -machine mps2-an505 -nographic -semihosting \
  -kernel target/thumbv8m.main-none-eabi/release/qemu-cmsis-nn

# MVE: custom target + build-std (boots, but see caveat below)
RUSTC_BOOTSTRAP=1 cargo -Z build-std=core -Z json-target-spec build --release \
  --target thumbv8m.main-none-eabihf.json --features mve
qemu-system-arm -machine mps3-an547 -nographic -semihosting \
  -kernel target/thumbv8m.main-none-eabihf/release/qemu-cmsis-nn
```

Add `--features mcunet` to any build to run the MCUNet visual-wake-word model.

## MVE caveat

The MVE build boots, links the MVE prebuilt bitcode, and runs both models on
`mps3-an547`, but the numerical results are wrong. The same wrong results
occur with `cmsis_nn = false` (standard IREE codegen, no ukernels at all), so
the divergence comes from the base IREE/LLVM MVE (Helium) compilation or
QEMU 8.2's MVE execution model — not from the CMSIS-NN ukernels. Validate MVE
numerics on real M55 hardware.
