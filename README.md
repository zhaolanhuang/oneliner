# Oneliner

[![Current Crates.io Version](https://img.shields.io/crates/v/oneliner.svg)](https://crates.io/crates/oneliner)
[![Minimum Supported Rust Version](https://img.shields.io/crates/msrv/oneliner)](https://crates.io/crates/oneliner)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)

> **TinyML model inference with one-line code. Support `no_std` embedded targets.**

Oneliner turns a model file into a callable Rust type with one attribute:

```rust
#[model("models/model.tflite")]
struct MyModel;
```

At build time, Oneliner imports the model, compiles it for the selected Rust target, generates the Rust binding, and links the native model code. At runtime, your application works with ordinary, strongly typed Rust tensors.

```rust
use oneliner::model;
use oneliner::runtime::ModelInference;

#[model("models/model.tflite")]
struct MyModel;

fn main() {
    let mut model = MyModel::new();
    let mut input = MyModel::create_input_tensor();
    input.fill(1);

    let output = model.run(&input);
    println!("{:?}", output.as_slice());
}
```

## Why Oneliner?

- **One-line model binding:** Replace conversion scripts, native linking setup, tensor declarations, and dispatch glue with `#[model(...)]`.
- **Typed inputs and outputs:** Tensor element types and shapes come from the model, so mismatches surface during the build instead of on the device.
- **Made for on-device inference:** The model is compiled into target-native code. Inference does not depend on a cloud service.
- **Embedded-ready:** The runtime supports `no_std` and is demonstrated with Ariel OS and Embassy on RP2040.
- **Memory-aware by design:** Choose independent per-instance workspaces or one synchronized shared workspace.

## Quick Start

### 1. Install the host model toolchain

Python 3.10 or newer is required. A virtual environment keeps the compiler tools isolated from the rest of your system:

```sh
pip install "iree-base-compiler[onnx]" tosa-converter-for-tflite
```

To compile TensorFlow SavedModels, install TensorFlow and the matching IREE
TensorFlow tools in the same environment:

```sh
pip install tensorflow iree-tools-tf
```

To compile PyTorch models, install the CPU build of PyTorch and IREE Turbine in
the same environment. PyTorch, Turbine, and IREE must be mutually compatible,
so pin a working combination together for reproducible builds:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install iree-turbine
```

Verify the installation:

```sh
iree-compile --version
tosa-converter-for-tflite --version
iree-import-onnx --help
iree-import-tf --help
python -c "import torch, iree.turbine.aot"
python -c "import tensorflow"
```

The packages provide:

- `iree-base-compiler`: the IREE compiler used for every model
- `iree-base-compiler[onnx]`: ONNX import support
- `tosa-converter-for-tflite`: TFLite-to-TOSA import support
- `torch`: exports and loads PyTorch `ExportedProgram` models
- `iree-turbine`: imports PyTorch programs into IREE-compatible MLIR
- `tensorflow`: loads and inspects SavedModel signatures
- `iree-tools-tf`: imports TensorFlow SavedModels into IREE-compatible MLIR

If you only use MLIR input, `iree-base-compiler` is sufficient.

### 2. Add Oneliner

Add the crate to your application's `Cargo.toml`:

```toml
[dependencies]
oneliner = "0.1.0"
```
### 3. Bind and run a model

Model paths are resolved relative to the application's `Cargo.toml`.

```rust
use oneliner::model;
use oneliner::runtime::ModelInference;

#[model("models/model.tflite")]
struct MyModel;

let mut model = MyModel::new();
let mut input = MyModel::create_input_tensor();
input.as_slice_mut().copy_from_slice(&input_data);

let output = model.run(&input);
let values = output.as_slice();
```

Oneliner generates the input and output tensor types directly from the model. The application does not need to repeat their data types or dimensions.

### PyTorch models

OneLiner accepts a PyTorch `ExportedProgram` saved with the conventional `.pt2`
extension. Export the inference model with fixed example input shapes:

```python
import torch

model = MyModel()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

example_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
exported = torch.export.export(model, (example_input,))
torch.export.save(exported, "model.pt2")
```

Then bind it like any other model:

```rust
#[model("models/model.pt2")]
struct MyModel;
```

`.pt` and `.pth` are checkpoint conventions rather than self-contained model
formats: a checkpoint may contain only a `state_dict`, with no forward graph or
input signature. Convert those checkpoints to `.pt2` before using them with
OneLiner. Only load models from trusted sources because PyTorch deserialization
uses pickle internally.

### TensorFlow SavedModels

OneLiner accepts TensorFlow SavedModel v2 directories. For the conventional
exported `main` method and `serving_default` signature, only the format is
needed:

```rust
#[model("models/my_saved_model", format = "tensorflow")]
struct MyModel;
```

The model must expose a `main` method with a `serving_default` signature, and
the directory must contain `saved_model.pb`. TensorFlow, `iree-tools-tf`, and
`iree-base-compiler` should be pinned to mutually compatible versions.

## Examples

Each example is an independent Cargo project. Run its commands from the example directory with the Python environment activated.

| Example | What it demonstrates | Active model |
| --- | --- | --- |
| [Desktop IREE](examples/std-iree/) | The shortest end-to-end validation path on a standard host | Quantized MCUNet visual wake word |
| [Ariel OS + IREE](examples/ariel-os-iree/) | `no_std`, Ariel OS threads, native-board validation, and inference timing | Quantized LeNet5 |
| [Embassy + IREE on Pico](examples/embassy-pico-iree/) | Bare-metal RP2040, shared model workspace, static input storage, and `defmt` logging | Quantized LeNet5 |
| [LeNet5 + CMSIS-NN](examples/lenet5-cmsis-nn/) | Cortex-M4 cross-compilation and verification of CMSIS-NN int8 ukernels | Quantized LeNet5 |

Start with the [desktop example](examples/std-iree/) to confirm the model toolchain, then move to the operating system or board example that matches your target.

## Supported Models

The built-in IREE backend currently accepts:

- TFLite
- ONNX
- PyTorch `ExportedProgram` (`.pt2`)
- TensorFlow SavedModel v2 directories
- MLIR accepted by IREE

The generated `ModelInference` API currently targets fixed-shape models with:

- exactly one input tensor
- exactly one output tensor
- up to four dimensions

Integer and floating-point tensor element types are inferred automatically.

## Memory Modes

The default `owned` mode gives each model instance an independent workspace:

```rust
#[model("models/model.tflite")]
struct MyModel;
```

This is the natural choice when model instances may run concurrently.

The `shared` mode keeps one synchronized static workspace for all instances of a model type:

```rust
#[model("models/model.tflite", arena = "shared")]
struct MyModel;
```

Use it when reducing duplicate RAM use matters more than concurrent inference. The Pico example demonstrates this configuration.

## CMSIS-NN

CMSIS-NN lowering is disabled by default. Enable it per model on a supported Cortex-M4 target:

```rust
#[model("models/model.tflite", cmsis_nn = true)]
struct MyModel;
```

Supported static int8 Conv2D, depthwise Conv2D, MaxPool, and fully connected forms use CMSIS-NN ukernels; other operations retain their normal IREE LLVMCPU lowering. Depthwise Conv2D currently requires batch size 1, channel multiplier 1, unit dilation, and symmetric weights. Set `cmsis_nn = false` or omit the option to build the standard IREE implementation for differential testing. See the [LeNet5 CMSIS-NN example](examples/lenet5-cmsis-nn/) for the complete Cortex-M4F setup.


## Project Status

Oneliner is currently at version `0.1.0`. The project focuses on making fixed-shape, single-input, single-output inference straightforward across desktop Rust and memory-constrained `no_std` targets.

The examples are intentionally small and explicit. They are designed to help you validate the toolchain, understand the memory trade-offs, and replace the bundled model with your own.

## Testing

With the host model toolchain active, run the std end-to-end test suite from
the repository root:

```sh
cargo test
```

This runs end-to-end inference for every model in `examples/models`, using both
the `owned` and `shared` arena modes.
