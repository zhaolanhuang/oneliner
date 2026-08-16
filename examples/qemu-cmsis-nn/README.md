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
  routes fully connected layers through `arm_nn_mat_mult_nt_t_s8`, which
  applies the input offset on both the DSP and MVE paths (and is also faster
  than `arm_fully_connected_s8` on Cortex-M4).
- IREE 3.12 (LLVM 24) generates incorrect code for the non-ukernel MVE ops;
  the toolchain strips `+mve` from iree-compile's main codegen, keeping the
  prebuilt MVE ukernels.

## Benchmark (QEMU emulation, Cortex-M4F)

Measured on `olimex-stm32-h405` (STM32F405) with `-icount 1`, 128
inferences/iteration, host wall clock. QEMU has no working guest cycle
counter, so the latency is QEMU emulation throughput, not real hardware
(CMSIS-NN's DSP instructions emulate slowly in QEMU). Flash/RAM come from the
ELF sections.

| Model | cmsis_nn | Flash | RAM | latency/iter (QEMU) |
|---|---|---|---|---|
| LeNet5 | on | 83.7 KB | ~0.4 KB | ~6.5 ms |
| LeNet5 | off | 79.0 KB | ~0.4 KB | ~7.3 ms |
| MCUNet | on | 488.0 KB | ~2.0 KB | ~67 ms |
| MCUNet | off | 560.2 KB | ~1.5 KB | ~103 ms |

CMSIS-NN is faster than the standard codegen for both models in QEMU. On
real hardware (nRF52840DK) the LeNet5 latency went from 254 ms to ~49 ms
(after the bitcode fix described below; the standard codegen takes ~41 ms).
MCUNet with CMSIS-NN saves ~72 KB Flash (13%) and ~-0.5 KB RAM vs the
standard codegen.

## Bitcode build notes

The ukernel bitcode is compiled with clang `-O3 -flto` for each Cortex-M
variant. Two things were essential for performance:

- The CMSIS-NN sources call `memcpy`/`memset` for unaligned accesses. The
  bitcode build ships a minimal `<string.h>` (see `include/`) so the
  prototypes exist, but `-ffreestanding` disables clang's builtin
  `memcpy`, leaving real `__aeabi_memcpy` calls in the innermost
  matmul loops. Dropping `-ffreestanding` lets clang inline them into
  single loads/stores; on nRF52840DK this took LeNet5 from ~254 ms to
  ~49 ms.
- `arm_nn_mat_mult_kernel_s8_s16` keeps its SMLAD blocks only when the
  accumulator packing is recognized; the remaining scalar byte mul-adds
  dominate the convolution kernels for small channel counts.
