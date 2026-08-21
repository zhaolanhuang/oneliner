# Installing the host model toolchain

Oneliner compiles models on your host machine during `cargo build`, so you need a small set of Python tools installed before the `#[model(...)]` codegen can work.

## Prerequisites

- Python 3.12 or newer
- MSRV: 1.95
- Clang with Cortex-M4 support for the experimental `vmcu = "pointwise-pair"` lowering

The vMCU lowering uses `clang` from `PATH` by default. Set `CLANG` to an explicit compiler path when needed; its LLVM bitcode must be compatible with the installed IREE compiler.

## Install the compiler packages

The quickest way to get everything (compiler, TFLite/ONNX/PyTorch/TensorFlow support) is the pinned installation script, run from the repository root inside your active environment:

```sh
./install_all.sh
```

Or install the groups individually below.

```sh
pip install --pre --find-links https://iree.dev/pip-release-links.html iree-base-compiler[onnx]==3.12.0rc20260812
pip install tosa-converter-for-tflite==2026.2.0
```

This covers TFLite, ONNX, and MLIR models.

### TensorFlow SavedModels

To compile TensorFlow SavedModels, install TensorFlow and the matching IREE TensorFlow tools in the same environment:

```sh
pip install tensorflow==2.21.0 iree-tools-tf==20250718.1326
```

### PyTorch models

To compile PyTorch models, install the CPU build of PyTorch and IREE Turbine in the same environment. PyTorch, Turbine, and IREE must be mutually compatible, so pin a working combination together for reproducible builds:

```sh
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install iree-turbine==3.9.0
```

## Verify the installation

```sh
iree-compile --version
tosa-converter-for-tflite --version
iree-import-onnx --help
iree-import-tf --help
clang --version
python -c "import torch, iree.turbine.aot"
python -c "import tensorflow"
```

## Package reference

The following pinned combination is verified to work together:

| Package | Pinned version | Purpose |
| --- | --- | --- |
| `iree-base-compiler[onnx]` | `3.12.0rc` | The IREE compiler (plus ONNX import) used for every model |
| `tosa-converter-for-tflite` | `2026.2.0` | TFLite-to-TOSA import support |
| `torch` | `2.13.0+cpu` | Exports and loads PyTorch `ExportedProgram` models |
| `iree-turbine` | `3.9.0` | Imports PyTorch programs into IREE-compatible MLIR |
| `tensorflow` | `2.21.0` | Loads and inspects SavedModel signatures |
| `iree-tools-tf` | `20250718.1326` | Imports TensorFlow SavedModels into IREE-compatible MLIR |

If you only use MLIR input, `iree-base-compiler` is sufficient.
