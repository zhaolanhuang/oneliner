use std::path::{Path, PathBuf};
use std::process::Command;

use proc_macro2::Span;

use super::super::llvm_target_info::llvm_target_info_from_rust_triple;
use crate::utils::{query_rustc_host, run_command, target_from_process_args};

#[derive(Debug)]
/// Target settings shared by both halves of a split IREE compilation.
struct IreeTarget {
    /// LLVM target triple derived from Cargo's Rust target.
    llvm_triple: String,
    /// Optional CPU name, omitted when Rust did not provide one.
    cpu: Option<String>,
    /// Optional LLVM CPU feature string forwarded without reinterpretation.
    features: Option<String>,
    /// Power-of-two compiler guard that permits post-object budget analysis.
    stack_allocation_limit: Option<usize>,
}

pub(super) struct Compiler {
    target: IreeTarget,
}

impl Compiler {
    pub(super) fn new(stack_allocation_limit: Option<usize>) -> syn::Result<Self> {
        Ok(Self {
            target: resolve_target(stack_allocation_limit)?,
        })
    }

    pub(super) fn compile_full(
        &self,
        input: &Path,
        vmfb: &Path,
        object: &Path,
        ir_dump_dir: &Path,
    ) -> syn::Result<()> {
        let mut command = Command::new("iree-compile");
        configure_target(&mut command, input, &self.target);
        configure_final_pipeline(&mut command, vmfb, object, ir_dump_dir);
        run_command(&mut command, "iree-compile")
    }

    pub(super) fn compile_preprocessing(
        &self,
        input: &Path,
        output: &Path,
        object: &Path,
    ) -> syn::Result<()> {
        let mut command = Command::new("iree-compile");
        configure_target(&mut command, input, &self.target);
        configure_optimization(&mut command);
        configure_static_link(&mut command, object);
        command
            .arg("--compile-to=preprocessing")
            .arg("--emit-mlir-bytecode=false")
            .arg("-o")
            .arg(output);
        run_command(&mut command, "IREE preprocessing")
    }

    pub(super) fn compile_from_preprocessing(
        &self,
        input: &Path,
        vmfb: &Path,
        object: &Path,
        ir_dump_dir: &Path,
    ) -> syn::Result<()> {
        let mut command = Command::new("iree-compile");
        configure_target(&mut command, input, &self.target);
        command.arg("--compile-from=preprocessing");
        configure_final_pipeline(&mut command, vmfb, object, ir_dump_dir);
        run_command(&mut command, "iree-compile from preprocessing IR")
    }
}

/// Resolves Cargo/rustc target information once for consistent split commands.
fn resolve_target(stack_allocation_limit: Option<usize>) -> syn::Result<IreeTarget> {
    // Cargo exposes cross-compilation targets through environment variables;
    // direct rustc invocations are covered by the process-argument/host fallbacks.
    let rust_target = if let Ok(target) =
        std::env::var("CARGO_BUILD_TARGET").or_else(|_| std::env::var("TARGET"))
    {
        target
    } else {
        target_from_process_args()
            .or_else(|| query_rustc_host().map(Into::into))
            .ok_or_else(|| syn::Error::new(Span::call_site(), "Rust target triple is unavailable"))?
            .into_string()
            .map_err(|_| syn::Error::new(Span::call_site(), "target argument is not valid UTF-8"))?
    };
    let target_info = llvm_target_info_from_rust_triple(&rust_target).map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("failed to get LLVM target info for Rust target {rust_target}: {error}"),
        )
    })?;
    Ok(IreeTarget {
        llvm_triple: target_info.llvm_triple,
        cpu: target_info.cpu.filter(|value| !value.is_empty()),
        features: target_info.features.filter(|value| !value.is_empty()),
        stack_allocation_limit,
    })
}

/// Adds the target device, LLVM triple, CPU, and features to an IREE command.
fn configure_target(command: &mut Command, input: &Path, target: &IreeTarget) {
    command
        .arg(input)
        .arg("--iree-hal-target-device=local")
        .arg("--iree-hal-local-target-device-backends=llvm-cpu")
        .arg(format!(
            "--iree-llvmcpu-target-triple={}",
            target.llvm_triple
        ));
    if let Some(cpu) = &target.cpu {
        command.arg(format!("--iree-llvmcpu-target-cpu={cpu}"));
    }
    if let Some(features) = &target.features {
        command.arg(format!("--iree-llvmcpu-target-cpu-features={features}"));
    }
    if let Some(limit) = target.stack_allocation_limit {
        command.arg(format!("--iree-llvmcpu-stack-allocation-limit={limit}"));
    }
}

/// Adds optimization flags shared by one-shot and resumed compilation.
fn configure_optimization(command: &mut Command) {
    command
        .arg("--iree-opt-level=O3")
        .arg("--iree-stream-partitioning-favor=min-peak-memory")
        .arg("--iree-dispatch-creation-enable-aggressive-fusion=true")
        .arg("--iree-dispatch-creation-fuse-multi-use=true")
        .arg("--iree-dispatch-creation-enable-fuse-padding-into-linalg-consumer-ops=false");
}

/// Configures the standalone object consumed by Oneliner's Rust linker.
fn configure_static_link(command: &mut Command, object: &Path) {
    command
        .arg("--iree-llvmcpu-link-embedded=false")
        .arg("--iree-llvmcpu-link-static")
        .arg(format!(
            "--iree-llvmcpu-static-library-output-path={}",
            object.display()
        ));
}

/// Adds all options needed by the final compiler invocation and phase dumps.
fn configure_final_pipeline(command: &mut Command, vmfb: &Path, object: &Path, ir_dump_dir: &Path) {
    configure_optimization(command);
    configure_static_link(command, object);
    command
        .arg(format!(
            "--dump-compilation-phases-to={}",
            ir_dump_dir.display()
        ))
        .arg("-o")
        .arg(vmfb);
}

pub(super) fn run_converter(
    input: &Path,
    rust_output: &Path,
    json_output: &Path,
) -> syn::Result<()> {
    let mut command = Command::new(python_executable());
    command
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("python")
                .join("oneliner_iree")
                .join("stream_flow_to_rust.py"),
        )
        .arg(input)
        .arg("--rust-output")
        .arg(rust_output)
        .arg("--json-output")
        .arg(json_output);
    run_command(&mut command, "IREE Stream/Flow converter")
}

/// Returns the configured Python interpreter for every build-time helper.
pub(super) fn python_executable() -> std::ffi::OsString {
    // PYTHON lets users bind all helper scripts to the same environment as the
    // pinned IREE package; the conventional executable remains the default.
    std::env::var_os("PYTHON").unwrap_or_else(|| "python".into())
}
