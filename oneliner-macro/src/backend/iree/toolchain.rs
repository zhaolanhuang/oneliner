use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use proc_macro2::Span;

use super::super::llvm_target_info::llvm_target_info_from_rust_triple;
use crate::utils::{query_rustc_host, run_command, target_from_process_args};

pub(super) fn run_iree_compile(
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    cmsis_nn: bool,
) -> syn::Result<()> {
    //TODO: hacky, shall be improved.
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
    let llvm_triple = target_info.llvm_triple;
    let target_cpu = target_info
        .cpu
        .filter(|value| !value.is_empty())
        .or_else(|| is_cortex_m4_target(&llvm_triple).then(|| "cortex-m4".to_owned()));
    let cpu_features = target_info.features.filter(|value| !value.is_empty());

    if cmsis_nn && is_cortex_m4_target(&llvm_triple) {
        return run_iree_compile_with_cmsis_nn(
            compile_input,
            vmfb,
            object,
            ir_dump_dir,
            &llvm_triple,
            target_cpu.as_deref().expect("Cortex-M4 CPU was inferred"),
            cpu_features.as_deref(),
        );
    }

    run_standard_iree_compile(
        compile_input,
        vmfb,
        object,
        ir_dump_dir,
        &llvm_triple,
        target_cpu.as_deref(),
        cpu_features.as_deref(),
    )
}

fn is_cortex_m4_target(llvm_triple: &str) -> bool {
    matches!(llvm_triple, "thumbv7em-none-eabi" | "thumbv7em-none-eabihf")
}

fn configure_iree_target(
    command: &mut Command,
    input: &Path,
    llvm_triple: &str,
    target_cpu: Option<&str>,
    cpu_features: Option<&str>,
) {
    command
        .arg(input)
        .arg("--iree-hal-target-device=local")
        .arg("--iree-hal-local-target-device-backends=llvm-cpu")
        .arg(format!("--iree-llvmcpu-target-triple={llvm_triple}"));
    if let Some(cpu) = target_cpu {
        command.arg(format!("--iree-llvmcpu-target-cpu={cpu}"));
    }
    if let Some(features) = cpu_features {
        command.arg(format!("--iree-llvmcpu-target-cpu-features={features}"));
    }
}

fn run_iree_compile_with_cmsis_nn(
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    llvm_triple: &str,
    target_cpu: &str,
    cpu_features: Option<&str>,
) -> syn::Result<()> {
    let artifact_dir = vmfb
        .parent()
        .ok_or_else(|| syn::Error::new(Span::call_site(), "VMFB output has no parent directory"))?;
    let preprocessing = artifact_dir.join("cmsis-nn.preprocessing.mlir");
    let rewritten = artifact_dir.join("cmsis-nn.rewritten.mlir");
    let configured = artifact_dir.join("cmsis-nn.configured.mlir");
    let lowered = artifact_dir.join("cmsis-nn.lowered.mlir");
    let finalized = artifact_dir.join("cmsis-nn.finalized.mlir");
    let bitcode = artifact_dir.join("oneliner_cmsis_nn.bc");
    let macro_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = macro_dir.parent().ok_or_else(|| {
        syn::Error::new(Span::call_site(), "oneliner-macro has no workspace parent")
    })?;
    let rewriter = macro_dir.join("python").join("rewrite_cmsis_nn.py");

    let mut preprocess = Command::new("iree-compile");
    configure_iree_target(
        &mut preprocess,
        compile_input,
        llvm_triple,
        Some(target_cpu),
        cpu_features,
    );
    preprocess
        .arg("--compile-to=preprocessing")
        .arg("--emit-mlir-bytecode=false")
        .arg("--iree-llvmcpu-link-embedded=false")
        .arg("--iree-llvmcpu-link-static")
        .arg(format!(
            "--iree-llvmcpu-static-library-output-path={}",
            object.display()
        ))
        .arg("-o")
        .arg(&preprocessing);
    run_command(&mut preprocess, "IREE preprocessing")?;

    run_python_filter(&rewriter, &preprocessing, &rewritten, &[])?;
    let rewritten_text = std::fs::read_to_string(&rewritten)
        .map_err(|error| syn::Error::new(Span::call_site(), error))?;
    if !rewritten_text.contains("oneliner_cmsis_nn_") {
        return run_standard_iree_compile(
            compile_input,
            vmfb,
            object,
            ir_dump_dir,
            llvm_triple,
            Some(target_cpu),
            cpu_features,
        );
    }

    let mut configure = Command::new("iree-compile");
    configure_iree_target(
        &mut configure,
        &rewritten,
        llvm_triple,
        Some(target_cpu),
        cpu_features,
    );
    configure
        .arg("--compile-from=preprocessing")
        .arg("--compile-to=executable-configurations")
        .arg("--emit-mlir-bytecode=false")
        .arg("--iree-llvmcpu-link-embedded=false")
        .arg("--iree-llvmcpu-link-static")
        .arg(format!(
            "--iree-llvmcpu-static-library-output-path={}",
            object.display()
        ))
        .arg("-o")
        .arg(&configured);
    run_command(&mut configure, "IREE CMSIS-NN dispatch configuration")?;

    let mut lower = Command::new("iree-opt");
    lower
        .arg(&configured)
        .arg("--pass-pipeline=builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-lower-bitcode-ukernels)),iree-hal-hoist-executable-objects)))")
        .arg("-o")
        .arg(&lowered);
    run_command(&mut lower, "IREE CMSIS-NN ukernel lowering")?;

    run_python_filter(&rewriter, &lowered, &finalized, &["--finalize-configured"])?;

    let mut build_bitcode = Command::new("python");
    build_bitcode
        .arg(macro_dir.join("cmsis_nn").join("build_bitcode.py"))
        .arg("--cmsis-nn")
        .arg(workspace_dir.join("third_party").join("cmsis-nn"))
        .arg("--shim")
        .arg(macro_dir.join("cmsis_nn").join("oneliner_cmsis_nn.c"))
        .arg("--output")
        .arg(&bitcode)
        .arg("--target")
        .arg(llvm_triple)
        .arg("--cpu")
        .arg(target_cpu)
        .arg("--features")
        .arg(cpu_features.unwrap_or_default());
    run_command(&mut build_bitcode, "CMSIS-NN bitcode builder")?;

    let mut compile = Command::new("iree-compile");
    configure_iree_target(
        &mut compile,
        &finalized,
        llvm_triple,
        Some(target_cpu),
        cpu_features,
    );
    compile
        .arg("--compile-from=executable-configurations")
        .arg("--iree-stream-partitioning-favor=min-peak-memory")
        .arg("--iree-llvmcpu-link-embedded=false")
        .arg("--iree-llvmcpu-link-static")
        .arg(format!(
            "--iree-hal-executable-object-search-path={}",
            artifact_dir.display()
        ))
        .arg(format!(
            "--iree-llvmcpu-static-library-output-path={}",
            object.display()
        ))
        .arg(format!(
            "--dump-compilation-phases-to={}",
            ir_dump_dir.display()
        ))
        .arg("-o")
        .arg(vmfb);
    run_command(&mut compile, "iree-compile with CMSIS-NN")
}

fn run_standard_iree_compile(
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    llvm_triple: &str,
    target_cpu: Option<&str>,
    cpu_features: Option<&str>,
) -> syn::Result<()> {
    let mut command = Command::new("iree-compile");
    configure_iree_target(
        &mut command,
        compile_input,
        llvm_triple,
        target_cpu,
        cpu_features,
    );
    command
        .arg("--iree-stream-partitioning-favor=min-peak-memory")
        .arg("--iree-llvmcpu-link-embedded=false")
        .arg("--iree-llvmcpu-link-static")
        .arg(format!(
            "--iree-llvmcpu-static-library-output-path={}",
            object.display()
        ))
        .arg(format!(
            "--dump-compilation-phases-to={}",
            ir_dump_dir.display()
        ))
        .arg("-o")
        .arg(vmfb);
    run_command(&mut command, "iree-compile")
}

fn run_python_filter(script: &Path, input: &Path, output: &Path, args: &[&str]) -> syn::Result<()> {
    let stdin = File::open(input).map_err(|error| syn::Error::new(Span::call_site(), error))?;
    let stdout = File::create(output).map_err(|error| syn::Error::new(Span::call_site(), error))?;
    let mut command = Command::new("python");
    command
        .arg(script)
        .args(args)
        .stdin(Stdio::from(stdin))
        .stdout(Stdio::from(stdout));
    run_command(&mut command, "CMSIS-NN MLIR rewriter")
}

pub(super) fn run_converter(
    input: &Path,
    rust_output: &Path,
    json_output: &Path,
) -> syn::Result<()> {
    let mut command = Command::new("python");
    command
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("python")
                .join("iree_stream_flow_to_rust_using_re.py"),
        )
        .arg(input)
        .arg("--rust-output")
        .arg(rust_output)
        .arg("--json-output")
        .arg(json_output);
    run_command(&mut command, "IREE Stream/Flow converter")
}
