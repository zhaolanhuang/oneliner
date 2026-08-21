# Oneliner + IREE + Embassy on QEMU

This example runs a Oneliner model in an Embassy executor on QEMU's Stellaris LM3S6965EVB machine. It provides a hardware-free `no_std` smoke test for Cortex-M model inference.

```rust
#[model("../models/lenet5_quantized.tflite", arena = "shared")]
struct Model;
```

## What This Example Shows

- Cross-compiling a TFLite model for `thumbv7m-none-eabi`
- Running inference from an Embassy async application
- Using a synchronized shared workspace and static input storage
- Reporting status and exiting QEMU through semihosting
- Validating inference against a known reference output

The LM3S6965 has no dedicated Embassy HAL. This example uses the hardware-independent `embassy-executor` with `cortex-m-rt`; no peripheral initialization or Embassy time driver is required.

## Prerequisites

- QEMU with `qemu-system-arm` and the `lm3s6965evb` machine
- Stable Rust with the `thumbv7m-none-eabi` target
- The Python/IREE toolchain from [docs/installation.md](../../docs/installation.md)

Keep the Python virtual environment active during the build.

## Build

```sh
cargo build --release
```

## Run

```sh
cargo run --release
```

The Cargo runner expands to:

```sh
qemu-system-arm \
  -cpu cortex-m3 \
  -machine lm3s6965evb \
  -display none \
  -serial none \
  -monitor none \
  -semihosting-config enable=on,target=native \
  -kernel target/thumbv7m-none-eabi/release/embassy-qemu-lm3s-minimal
```

A successful run prints model sizes and `Model IREE validation passed`, then exits with status zero. A panic or output mismatch exits nonzero.

`target=native` permits guest semihosting calls to access host services. Only run trusted firmware.

## Why LM3S6965EVB?

QEMU's RP2040 support is unavailable. Among the commonly emulated Cortex-M boards, LM3S6965EVB provides 64 KiB of RAM, enough for the LeNet5 workspace and static tensors. The QEMU `microbit` machine has only 16 KiB and `stm32vldiscovery` has only 8 KiB.
