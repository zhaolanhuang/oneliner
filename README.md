# Oneliner

**TinyML model inference with one-line code. Focus on `no_std` embedded targets.**

For the verified host and embedded targets, see [Target support](docs/target_support.md).

[![Current Crates.io Version](https://img.shields.io/crates/v/oneliner.svg)](https://crates.io/crates/oneliner)
[![Minimum Supported Rust Version](https://img.shields.io/crates/msrv/oneliner)](https://crates.io/crates/oneliner)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)

## Why Oneliner?

- **One-line model deployment:** Replace conversion scripts, native linking setup, tensor declarations, and dispatch glue with `#[model(...)]`.
- **Embedded-ready:** The runtime supports `no_std` and is demonstrated with Ariel OS and Embassy on ARM Cortex-M targets.

Oneliner turns a model file into a callable Rust type with oneline code:

```rust
#[model("models/model.tflite")]
struct MyModel;
```
## Quick Start

1. [Install the host model compilation toolchain](docs/installation.md).
2. Add the crate to your `Cargo.toml`:

   ```toml
   [dependencies]
   oneliner = "0.2"
   ```

3. Bind and run a model:

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

## Supported Models

Oneliner accepts:

- TFLite
- ONNX
- PyTorch `ExportedProgram` (`.pt2`)
- TensorFlow SavedModel v2 directories
- MLIR accepted by IREE

See [Model formats](docs/model_formats.md) for per-format guides and [Memory model](docs/memory_model.md) for the `owned`/`shared` arena modes.

## The `#[model]` attribute

The attribute takes the model path as its first positional argument, followed by optional named parameters:

```rust
#[model("models/model.tflite", backend = "iree", arena = "shared", format = "tflite")]
struct MyModel;
```

| Parameter | Values | Default | Description |
| --- | --- | --- | --- |
| `"<path>"` (positional, required) | string | — | Model file or directory path, resolved relative to the application's `Cargo.toml`. |
| `backend` | `"iree"` | `"iree"` | Execution backend. Only IREE is currently available. |
| `arena` | `"owned"`, `"shared"` | `"owned"` | Workspace memory mode, see [Memory model](docs/memory_model.md). |
| `format` | `"mlir"`, `"onnx"`, `"pytorch"`/`"pt2"`, `"tensorflow"`/`"tf"`, `"tflite"` | auto-detected from the file extension | Explicit model format override. Required for TensorFlow SavedModel v2 directories (no extension). |
| `vmcu` | `"pointwise-pair"`, `"mcunet"`, `"auto"` | disabled | Experimental Cortex-M4/M7 streaming lowering. `mcunet` fuses the 13 MCUNet inverted bottlenecks with circular halo buffers and in-place residual outputs. `auto` applies vMCU to any straight-line int8 model whose subgraphs match the vMCU patterns (inverted bottleneck, pointwise pair, single 2D convolution, fully connected); all other operations fall back to IREE codegen. |

The format is normally inferred from the extension (`.mlir`, `.onnx`, `.pt2`, `.tflite`); pass `format` explicitly to override it, which is mandatory for TensorFlow SavedModel directories. Duplicate or unknown options are rejected at compile time.

The experimental vMCU plan reports logical intermediate and segment bytes. Dispatch-local stack allocation and alignment are not included in the generated model's arena size; measure total stack usage on the target before drawing peak-RAM conclusions.

## Profiling

Measure inference latency with the optional `oneliner-profiler` crate and get an automatic flash/RAM footprint report on every model build. See [Profiling and footprint reporting](docs/profiling.md) and the [benchmark numbers](docs/benchmark.md).

## Examples

Each example is an independent Cargo project. Run its commands from the example directory with the Python environment activated.

| Example | What it demonstrates | Active model |
| --- | --- | --- |
| [Desktop Std](examples/std-minimal/) | The shortest end-to-end validation path on a standard host | Quantized MCUNet visual wake word |
| [Ariel OS](examples/ariel-os-minimal/) | `no_std`, Ariel OS as environment | Quantized LeNet5 and MCUNet|
| [Embassy on Rasperry Pi Pico](examples/embassy-pico-minimal/) | Bare-metal RP2040, static input storage| Quantized LeNet5 |
| [Ariel OS + Profiler](examples/ariel-os-profiler/) | `no_std` latency profiling with `Profiler` | Quantized LeNet5 and MCUNet |
| [Embassy on Rasperry Pi Pico + Profiler](examples/embassy-pico-profiler/) | Bare-metal RP2040 latency profiling | Quantized LeNet5 |
| [vMCU pointwise pair on QEMU](examples/qemu-vmcu-pointwise/) | Experimental segment-buffer ukernel on an emulated Cortex-M4 | Two int8 pointwise layers |
| [vMCU auto on QEMU](examples/qemu-vmcu-auto/) | vMCU `auto` mode over the full MCUNet graph on emulated Cortex-M4 and Cortex-M7 | Quantized MCUNet visual wake word |
| [LeNet5 auto on QEMU](examples/qemu-lenet5-auto/) | vMCU `auto` on a float-boundary model with int8 conv/FC subgraphs, Cortex-M4 and Cortex-M7 | Quantized LeNet5 |

Start with the [desktop example](examples/std-minimal/) to confirm the model toolchain, then move to the operating system or board example that matches your target.

## Project Status

Oneliner is currently at version `0.2.0`. The project focuses on making fixed-shape, single-input, single-output inference straightforward across desktop Rust and memory-constrained `no_std` targets.

The examples are intentionally small and explicit. They are designed to help you validate the toolchain, understand the memory trade-offs, and replace the bundled model with your own.

### Acknowledgements

Oneliner evolved from [ariel-ml](https://github.com/ariel-os/ariel-ml): it keeps the mature IREE compiler for model compilation and optimization, while dropping the bulky IREE runtime that was difficult to integrate and compile with Rust. The interface design is inspired by [microflow-rs](https://github.com/matteocarnelos/microflow-rs). We thank all involved projects for their work.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you shall be dual-licensed as above, without any additional terms or conditions.

**Other languages:** [简体中文](README-zh-CN.md)
