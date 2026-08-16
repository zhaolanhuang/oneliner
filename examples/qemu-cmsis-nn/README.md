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

## MVE

Both models validate on `mps3-an547` (Cortex-M55) with `cmsis_nn = true` under
QEMU 11.1+ (the ARMv7M SysTick and the STM32 RCC are not emulated, so MVE
timing still needs real hardware). Two MVE bugs were found and fixed here:

- CMSIS-NN's MVE fully connected does not apply the input offset; the shim
  precomputes the kernel-sum buffer as `bias + input_offset * row_sum`.
- IREE 3.12 (LLVM 24) generates incorrect code for the non-ukernel MVE ops;
  the toolchain strips `+mve` from iree-compile's main codegen, keeping the
  prebuilt MVE ukernels.

## Benchmark (QEMU emulation, Cortex-M4F)

Measured on `olimex-stm32-h405` (STM32F405) with `-icount 1`, 128
inferences/iteration, host wall clock. QEMU has no working guest cycle
counter, so the latency is QEMU emulation throughput, not real hardware
(CMsis-NN's DSP instructions emulate slowly in QEMU). Flash/RAM come from the
ELF sections.

| Model | cmsis_nn | Flash | RAM | latency/iter (QEMU) |
|---|---|---|---|---|
| LeNet5 | on | 82.8 KB | 8.7 KB | ~44 ms |
| LeNet5 | off | 77.5 KB | 8.6 KB | ~6.6 ms |
| MCUNet | on | 478.8 KB | 95.6 KB | ~535 ms |
| MCUNet | off | 548.7 KB | 117.6 KB | ~135 ms |

MCUNet with CMSIS-NN saves ~70 KB Flash (13%) and ~22 KB RAM (19%); for tiny
models like LeNet5 the ukernel overhead can exceed the codegen savings. The
QEMU latency numbers overstate CMSIS-NN and must be re-measured on hardware.
