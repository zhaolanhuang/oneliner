# CMSIS-NN ukernels

This directory contains the CMSIS-NN ukernel shims for the IREE backend and
the tooling that compiles them into target-specific LLVM bitcode. IREE links
this bitcode into the generated executables and calls the shims from the
rewritten dispatches (see `oneliner-macro/python/rewrite_cmsis_nn.py`).

## Layout

| Path | Purpose |
|---|---|
| `oneliner_cmsis_nn.c` | The ukernel shims (`oneliner_cmsis_nn_conv_s8`, `..._fully_connected_s8`, `..._max_pool_s8`, ...) that translate the IREE dispatch ABI (base pointer + offset per tensor, plus an int32 config) into CMSIS-NN API calls |
| `build_bitcode.py` | Builds the shim + the CMSIS-NN sources into one self-contained LLVM bitcode module per Cortex-M target |
| `include/string.h` | Minimal `<string.h>` stub for the freestanding build (see "Build flags" below) |
| `prebuilt/*.bc` | The precompiled bitcode variants shipped with the macro crate |
| `prebuilt/manifest.json` | Tool/version metadata for the prebuilt variants |

## Requirements

- `clang`, `llvm-link`, `llvm-nm` on `PATH`. The checked-in prebuilt variants
  were produced with clang 22 (Homebrew LLVM). The bitcode is forward
  compatible: the installed `iree-compile` must use LLVM >= the prebuilt's
  LLVM major (the manifest records it, and the toolchain refuses to use the
  prebuilt otherwise).
- A CMSIS-NN checkout with the sources listed in `SOURCES` in
  `build_bitcode.py` (the repo pins `third_party/cmsis-nn`).

## Building the bitcode

Regenerate all eight prebuilt variants (M0/M3/M4/M33/M55, soft and hard float):

```sh
python build_bitcode.py --build-all outdir \
  --cmsis-nn ../../third_party/cmsis-nn \
  --shim oneliner_cmsis_nn.c
```

This writes `outdir/<variant>.bc` plus `outdir/manifest.json`. To make the
macro crate use them, copy the files into `prebuilt/`:

```sh
cp outdir/*.bc outdir/manifest.json prebuilt/
```

Build a single variant on the fly (what the toolchain does when it compiles
at expansion time):

```sh
python build_bitcode.py \
  --cmsis-nn ../../third_party/cmsis-nn \
  --shim oneliner_cmsis_nn.c \
  --output /tmp/m4.bc \
  --target thumbv7em-none-eabihf \
  --cpu cortex-m4 \
  --features "+vfp4d16sp"
```

The single-variant mode computes the float ABI from the triple/features
(`eabihf` or `+vfp*` -> hard float). `--build-all` always uses the variants
table in the script (identical flags to the manifest).

### Cache

The build is cached in `<workspace>/target/cmsis-nn-bc`. The cache key is a
SHA-256 over the target, compile flags, tool versions, the shim, the CMSIS-NN
sources and headers, and the extra include dirs, so editing the shim or
bumping CMSIS-NN invalidates it automatically. Pass `--no-cache` to bypass.

### How the toolchain picks the bitcode

During a normal build the macro crate copies the matching
`prebuilt/<triple>__<cpu>.bc` (selected via `manifest.json`) into the artifact
directory. Set `ONELINER_CMSIS_NN_FORCE_BUILD=1` on the cargo invocation to
always compile the bitcode on the fly from the current shim instead (used for
iteration on the shim and for toolchains whose LLVM is older than the
prebuilt's).

## Build flags (read this before changing them)

The compile flags are `-O3 -flto -ffunction-sections -fdata-sections` with
`-mcpu=<cpu>` and the variant features. Two things are load-bearing:

- **Do not add `-ffreestanding`.** `-ffreestanding` disables clang's builtin
  `memcpy`/`memset`, and the CMSIS-NN kernels use `memcpy` for every unaligned
  access. Without the builtin knowledge those become real `__aeabi_memcpy`
  calls in the innermost matmul loops (6 per iteration in
  `arm_nn_mat_mult_kernel_s8_s16`), which was the cause of a ~5x slowdown on
  real hardware (LeNet5 on nRF52840DK: 254 ms -> ~49 ms after inlining them).
  The `<string.h>` stub in `include/` provides the prototypes so the builtins
  resolve and inline; the `memcpy`/`memset` fallback definitions in the shim
  keep the bitcode free of C library dependencies.
- The fully connected shim deliberately uses `arm_nn_mat_mult_nt_t_s8`
  instead of `arm_fully_connected_s8`: the latter's
  `arm_nn_vec_mat_mult_t_s8` compiles to scalar byte mul-adds on Cortex-M4,
  while `arm_nn_mat_mult_nt_t_s8` lowers to the SIMD kernels and applies the
  input offset itself on both the DSP and MVE paths.

## Verification

- `llvm-nm --undefined-only <out>.bc` must be empty (the build script already
  checks this).
- Inspect the generated IR for the hot kernels, e.g.:
  `llvm-dis <out>.bc -o - | awk '/define.*arm_nn_mat_mult_kernel_s8_s16/,/^}/'`
  — the inner loop should contain `smlad` and no `call ptr @memcpy`.
- End-to-end: run the `examples/qemu-cmsis-nn` matrix (all Cortex-M cores,
  LeNet5 + MCUNet) and, if available, the `examples/lenet5-cmsis-nn` example
  on real hardware.
