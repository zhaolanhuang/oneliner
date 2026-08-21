# Oneliner

**一行代码完成 TinyML 模型推理，专注 `no_std` 嵌入式目标。**

已实机验证的宿主与嵌入式目标详见[目标支持](docs/target_support.md)。

[![Current Crates.io Version](https://img.shields.io/crates/v/oneliner.svg)](https://crates.io/crates/oneliner)
[![Minimum Supported Rust Version](https://img.shields.io/crates/msrv/oneliner)](https://crates.io/crates/oneliner)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#许可证)

## 为什么选择 Oneliner？

- **一行部署模型：** 用 `#[model(...)]` 取代模型转换脚本、原生链接配置、张量声明和调度胶水代码。
- **为嵌入式准备：** 运行时支持 `no_std`，并已在 ARM Cortex-M 目标上的 Ariel OS 和 Embassy 中验证。

Oneliner 只需要一行代码就能把一个模型文件变成一个可直接调用的 Rust 类型：

```rust
#[model("models/model.tflite")]
struct MyModel;
```

## 快速开始

1. [安装宿主端模型编译工具链](docs/installation.md)。
2. 在 `Cargo.toml` 中添加依赖：

   ```toml
   [dependencies]
   oneliner = "0.2"
   ```

3. 绑定并运行模型：

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

Oneliner 会直接从模型生成输入和输出张量类型，应用无需重复声明它们的数据类型和维度。

## 支持的模型

Oneliner 支持：

- TFLite
- ONNX
- PyTorch `ExportedProgram`（`.pt2`）
- TensorFlow SavedModel v2 目录
- IREE 接受的 MLIR

各格式的使用指南详见[模型格式说明](docs/model_formats.md)，`owned`/`shared` 工作区模式详见[内存模型](docs/memory_model.md)。

## `#[model]` 属性参数

属性第一个位置参数是模型路径，其后跟可选命名参数：

```rust
#[model("models/model.tflite", backend = "iree", arena = "shared", format = "tflite")]
struct MyModel;
```

| 参数 | 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `"<路径>"`（位置参数，必填） | 字符串 | — | 模型文件或目录路径，相对于应用的 `Cargo.toml` 解析。 |
| `backend` | `"iree"` | `"iree"` | 执行后端，目前仅支持 IREE。 |
| `arena` | `"owned"`、`"shared"` | `"owned"` | 工作区内存模式，详见[内存模型](docs/memory_model.md)。 |
| `format` | `"mlir"`、`"onnx"`、`"pytorch"`/`"pt2"`、`"tensorflow"`/`"tf"`、`"tflite"` | 按文件扩展名自动推断 | 显式指定模型格式。TensorFlow SavedModel v2 目录（无扩展名）必须显式指定。 |
| `vmcu` | `"pointwise-pair"`、`"mcunet"`、`"auto"` | 禁用 | 实验性 Cortex-M4/M7 流式 lowering。`mcunet` 使用循环 halo 缓冲和原地 residual 输出融合全部 13 个倒残差块。`auto` 对任意直线型 int8 模型中满足 vMCU pattern（倒残差块、pointwise pair、单层 2D 卷积、全连接）的子图应用 vMCU；其余算子回退 IREE codegen。 |

格式通常按扩展名（`.mlir`、`.onnx`、`.pt2`、`.tflite`）自动推断；需要覆盖时用 `format` 显式指定，TensorFlow SavedModel 目录必须指定。重复或未知选项会在编译期报错。

实验性 vMCU plan 报告的是中间张量和 segment 的逻辑字节数。生成模型的 arena 大小不包含 dispatch-local 栈分配及其对齐开销；评估峰值 RAM 前应在目标设备上测量总栈占用。

## 性能剖析

使用可选的 `oneliner-profiler` crate 测量推理延迟，每次模型构建还会自动输出 flash/RAM 占用报告。详见[性能剖析与占用报告](docs/profiling.md)及[基准测试数据](docs/benchmark.md)。

## 示例

每个示例都是独立的 Cargo 工程。请在示例目录下、并激活 Python 虚拟环境后运行相应命令。

| 示例 | 演示内容 | 使用的模型 |
| --- | --- | --- |
| [Desktop Std](examples/std-minimal/) | 在标准宿主上的最短端到端验证路径 | 量化 MCUNet 视觉唤醒词 |
| [Ariel OS](examples/ariel-os-minimal/) | `no_std`，以 Ariel OS 为运行环境 | 量化 LeNet5 和 MCUNet |
| [Embassy on Raspberry Pi Pico](examples/embassy-pico-minimal/) | 裸机 RP2040，静态输入存储 | 量化 LeNet5 |
| [Ariel OS + Profiler](examples/ariel-os-profiler/) | `no_std` 延迟剖析（`Profiler`） | 量化 LeNet5 和 MCUNet |
| [Embassy on Raspberry Pi Pico + Profiler](examples/embassy-pico-profiler/) | 裸机 RP2040 延迟剖析 | 量化 LeNet5 |
| [QEMU 上的 vMCU pointwise pair](examples/qemu-vmcu-pointwise/) | 在模拟 Cortex-M4 上验证实验性分段缓冲 ukernel | 两层 int8 pointwise |
| [QEMU 上的 vMCU auto](examples/qemu-vmcu-auto/) | 在模拟 Cortex-M4 与 Cortex-M7 上以 `auto` 模式运行完整 MCUNet 图 | 量化 MCUNet 视觉唤醒词 |
| [QEMU 上的 LeNet5 auto](examples/qemu-lenet5-auto/) | 在浮点边界的模型上以 `auto` 模式 vMCU 化 int8 卷积/全连接子图，Cortex-M4 与 Cortex-M7 | 量化 LeNet5 |

建议先运行[桌面示例](examples/std-minimal/)确认模型工具链，再选择与目标环境匹配的操作系统或开发板示例。

## 项目状态

Oneliner 当前版本为 `0.2.0`。项目专注于在桌面 Rust 和内存受限的 `no_std` 目标上，让固定形状、单输入单输出的推理变得简单直接。

示例刻意保持小而直观，目的是帮助你验证工具链、理解内存取舍，并把内置模型替换成你自己的模型。

### 致谢

Oneliner 由 [ariel-ml](https://github.com/ariel-os/ariel-ml) 演变而来：保留了成熟的 IREE compiler 用于模型编译与优化，同时去掉了臃肿、难以与 Rust 兼容编译的 IREE runtime。接口设计借鉴了 [microflow-rs](https://github.com/matteocarnelos/microflow-rs)。在此对上述项目表示感谢。

## 许可证

本仓库基于 [Apache License, Version 2.0](LICENSE-APACHE) 或 [MIT license](LICENSE-MIT) 二选一授权。

除非你明确声明，任何有意提交的贡献都将按上述双重许可授权，不附加任何额外条款或条件。

**其他语言：** [English](README.md)
