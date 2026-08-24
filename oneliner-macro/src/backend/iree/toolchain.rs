use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use proc_macro2::Span;

use super::super::llvm_target_info::llvm_target_info_from_rust_triple;
use crate::args::VmcuArg;
use crate::utils::{query_rustc_host, run_command, target_from_process_args};

pub(super) fn run_iree_compile(
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    vmcu: Option<VmcuArg>,
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
    let target_cpu = target_info.cpu.filter(|value| !value.is_empty());
    let cpu_features = target_info.features.filter(|value| !value.is_empty());

    if let Some(mode) = vmcu {
        return run_vmcu_compile(
            compile_input,
            vmfb,
            object,
            ir_dump_dir,
            &llvm_triple,
            target_cpu.as_deref(),
            cpu_features.as_deref(),
            mode,
        );
    }

    let mut command = Command::new("iree-compile");
    command
        .arg(compile_input)
        .arg("--iree-hal-target-device=local")
        .arg("--iree-hal-local-target-device-backends=llvm-cpu")
        .arg("--iree-llvmcpu-target-float-abi=soft")
        .arg(format!("--iree-llvmcpu-target-triple={llvm_triple}"));
    if let Some(cpu) = target_cpu {
        command.arg(format!("--iree-llvmcpu-target-cpu={cpu}"));
    }
    if let Some(features) = cpu_features {
        command.arg(format!("--iree-llvmcpu-target-cpu-features={features}"));
    }
    command
        .arg("--iree-opt-level=O3")
        .arg("--iree-stream-partitioning-favor=min-peak-memory")
        .arg("--iree-dispatch-creation-enable-aggressive-fusion=true")
        .arg("--iree-dispatch-creation-fuse-multi-use=true")
        .arg("--iree-dispatch-creation-enable-fuse-padding-into-linalg-consumer-ops=false")
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

fn configure_vmcu_target(
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
    if !llvm_triple.starts_with("riscv") {
        if let Some(cpu) = target_cpu {
            command.arg(format!("--iree-llvmcpu-target-cpu={cpu}"));
        }
    }
    if let Some(features) = cpu_features {
        command.arg(format!("--iree-llvmcpu-target-cpu-features={features}"));
    }
}

fn add_static_link_options(command: &mut Command, object: &Path) {
    command
        .arg("--iree-llvmcpu-link-embedded=false")
        .arg("--iree-llvmcpu-link-static")
        .arg(format!(
            "--iree-llvmcpu-static-library-output-path={}",
            object.display()
        ));
}

fn run_vmcu_compile(
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    llvm_triple: &str,
    target_cpu: Option<&str>,
    cpu_features: Option<&str>,
    mode: VmcuArg,
) -> syn::Result<()> {
    let artifact_dir = vmfb
        .parent()
        .ok_or_else(|| syn::Error::new(Span::call_site(), "VMFB output has no parent directory"))?;
    let (stem, rewriter_name, source_name, bitcode_name) = match mode {
        VmcuArg::Auto => (
            "vmcu-generic",
            "rewrite_vmcu_mcunet.py",
            "oneliner_vmcu_generic.c",
            "oneliner_vmcu_generic.bc",
        ),
    };
    let cpu = target_cpu;
    let preprocessing = artifact_dir.join(format!("{stem}.preprocessing.mlir"));
    let rewritten = artifact_dir.join(format!("{stem}.rewritten.mlir"));
    let configured = artifact_dir.join(format!("{stem}.configured.mlir"));
    let lowered = artifact_dir.join(format!("{stem}.lowered.mlir"));
    let finalized = artifact_dir.join(format!("{stem}.finalized.mlir"));
    let plan = artifact_dir.join(format!("{stem}.plan.json"));
    let bitcode = artifact_dir.join(bitcode_name);
    let macro_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let rewriter = macro_dir.join("python").join(rewriter_name);

    let mut preprocess = Command::new("iree-compile");
    configure_vmcu_target(&mut preprocess, compile_input, llvm_triple, cpu.as_deref(), cpu_features);
    preprocess
        .arg("--compile-to=preprocessing")
        .arg("--emit-mlir-bytecode=false")
        .arg("--iree-llvmcpu-stack-allocation-limit=65536");
    add_static_link_options(&mut preprocess, object);
    preprocess.arg("-o").arg(&preprocessing);
    run_command(&mut preprocess, "IREE vMCU preprocessing")?;

    let plan_arg = plan.to_string_lossy().into_owned();
    run_python_filter(
        &rewriter,
        &preprocessing,
        &rewritten,
        &["--plan-output", &plan_arg],
        "vMCU pointwise-pair rewriter",
    )?;

    let mut configure = Command::new("iree-compile");
    configure_vmcu_target(&mut configure, &rewritten, llvm_triple, cpu.as_deref(), cpu_features);
    configure
        .arg("--compile-from=preprocessing")
        .arg("--compile-to=executable-configurations")
        .arg("--emit-mlir-bytecode=false")
        .arg("--iree-opt-level=O3")
        .arg("--iree-dispatch-creation-enable-aggressive-fusion=true")
        .arg("--iree-dispatch-creation-fuse-multi-use=true")
        .arg("--iree-dispatch-creation-enable-fuse-padding-into-linalg-consumer-ops=false");
    add_static_link_options(&mut configure, object);
    configure.arg("-o").arg(&configured);
    run_command(&mut configure, "IREE vMCU dispatch configuration")?;

    let mut lower = Command::new("iree-opt");
    lower
        .arg(&configured)
        .arg("--pass-pipeline=builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-lower-bitcode-ukernels)),iree-hal-hoist-executable-objects)))")
        .arg("-o")
        .arg(&lowered);
    run_command(&mut lower, "IREE vMCU ukernel lowering")?;

    run_python_filter(
        &rewriter,
        &lowered,
        &finalized,
        &["--finalize-configured"],
        "vMCU configured ukernel finalizer",
    )?;

    let clang_path = std::env::var_os("CLANG").unwrap_or_else(|| "clang".into());
    let mut clang = Command::new(clang_path);
    clang.arg(format!("--target={llvm_triple}"));
    if llvm_triple.starts_with("riscv") {
        clang.arg("-march=rv32imac");
    } else {
        clang.arg("-mthumb");
        if let Some(cpu) = cpu {
            clang.arg(format!("-mcpu={cpu}"));
        }
    }
    // Keep clang's target configuration aligned with the iree-compile
    // invocation: the rustc target features (e.g. riscv32imac "+m,+a,+c",
    // or a future "-dsp" for thumbv7em) are forwarded to clang as well so
    // the bitcode carries the same function-level target-features.
    if let Some(features) = cpu_features {
        for feature in features.split(',') {
            if !feature.is_empty() {
                clang.arg("-Xclang").arg("-target-feature").arg("-Xclang").arg(feature);
            }
        }
    }
    clang
        .arg("-ffreestanding")
        .arg("-fno-builtin")
        .arg("-O3")
        .arg("-emit-llvm")
        .arg("-c")
        .arg(macro_dir.join("vmcu").join(source_name))
        .arg("-o")
        .arg(&bitcode);
    run_command(&mut clang, "vMCU pointwise bitcode builder")?;

    let mut compile = Command::new("iree-compile");
    configure_vmcu_target(&mut compile, &finalized, llvm_triple, cpu.as_deref(), cpu_features);
    compile
        .arg("--compile-from=executable-configurations")
        .arg("--iree-opt-level=O3")
        .arg("--iree-stream-partitioning-favor=min-peak-memory")
        .arg(format!(
            "--iree-hal-executable-object-search-path={}",
            artifact_dir.display()
        ))
        .arg(format!(
            "--dump-compilation-phases-to={}",
            ir_dump_dir.display()
        ))
        .arg("--iree-llvmcpu-stack-allocation-limit=65536");
    add_static_link_options(&mut compile, object);
    compile.arg("-o").arg(vmfb);
    run_command(&mut compile, "iree-compile with vMCU pointwise ukernel")
}

fn run_python_filter(
    script: &Path,
    input: &Path,
    output: &Path,
    args: &[&str],
    label: &str,
) -> syn::Result<()> {
    let stdin = File::open(input).map_err(|error| syn::Error::new(Span::call_site(), error))?;
    let stdout = File::create(output).map_err(|error| syn::Error::new(Span::call_site(), error))?;
    let mut command = Command::new("python");
    command
        .arg(script)
        .args(args)
        .stdin(Stdio::from(stdin))
        .stdout(Stdio::from(stdout));
    run_command(&mut command, label)
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
                .join("iree_stream_flow_to_rust.py"),
        )
        .arg(input)
        .arg("--rust-output")
        .arg(rust_output)
        .arg("--json-output")
        .arg(json_output);
    run_command(&mut command, "IREE Stream/Flow converter")
}
