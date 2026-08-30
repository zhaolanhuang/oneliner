use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use proc_macro2::Span;

use super::super::llvm_target_info::llvm_target_info_from_rust_triple;
use crate::args::{VmcuArg, VmcuScheduleArg};
use crate::utils::{query_rustc_host, run_command, rust_ident, target_from_process_args};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Outcome of post-lowering SRAM accounting.
enum ResourceStatus {
    WithinBudget,
    ExceedsBudget,
}

/// Compiles a frontend-normalized model and returns the final IR dump stem.
///
/// `Off` keeps the original one-shot path. `Auto` and `Strict` pause after
/// preprocessing, invoke the checked-in Python rewriter, and resume the same
/// IREE pipeline from the rewritten textual MLIR.
pub(super) fn run_iree_compile(
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    artifact_dir: &Path,
    ir_dump_stem: &str,
    vmcu: VmcuArg,
    vmcu_sram: Option<usize>,
    vmcu_schedule: VmcuScheduleArg,
    vmcu_search_states: usize,
) -> syn::Result<String> {
    let target = resolve_target(vmcu_sram)?;
    match vmcu {
        VmcuArg::Off => {
            let mut command = Command::new("iree-compile");
            configure_target(&mut command, compile_input, &target);
            configure_final_pipeline(&mut command, vmfb, object, ir_dump_dir);
            run_command(&mut command, "iree-compile")?;
            Ok(ir_dump_stem.to_owned())
        }
        VmcuArg::Auto | VmcuArg::Strict => run_vmcu_compile(
            compile_input,
            vmfb,
            object,
            ir_dump_dir,
            artifact_dir,
            &target,
            vmcu,
            vmcu_sram,
            vmcu_schedule,
            vmcu_search_states,
        ),
    }
}

/// Resolves Cargo/rustc target information once for consistent split commands.
fn resolve_target(vmcu_sram: Option<usize>) -> syn::Result<IreeTarget> {
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
        stack_allocation_limit: vmcu_sram.map(ceiling_power_of_two),
    })
}

/// Rounds up so IREE does not reject before the exact object-level gate runs.
fn ceiling_power_of_two(value: usize) -> usize {
    debug_assert!(value > 0);
    value
        .checked_next_power_of_two()
        .unwrap_or(1usize << (usize::BITS - 1))
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

/// Runs preprocessing, Python graph rewriting, and resumed IREE compilation.
///
/// The three intermediate files are intentionally retained under `artifact_dir`
/// so users can audit the compiler boundary and every accepted/rejected match.
fn run_vmcu_compile(
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    artifact_dir: &Path,
    target: &IreeTarget,
    mode: VmcuArg,
    vmcu_sram: Option<usize>,
    vmcu_schedule: VmcuScheduleArg,
    vmcu_search_states: usize,
) -> syn::Result<String> {
    // Fixed names are safe because each model type receives its own artifact
    // directory. They also make build diagnostics predictable.
    let preprocessing = artifact_dir.join("vmcu.preprocessing.mlir");
    let rewritten = artifact_dir.join("vmcu.rewritten.mlir");
    let plan = artifact_dir.join("vmcu.plan.json");

    let mut preprocess = Command::new("iree-compile");
    configure_target(&mut preprocess, compile_input, target);
    // IREE materializes the executable target during preprocessing. Static
    // linking options must therefore be present in this first invocation as
    // well as in the resumed pipeline, or the final VMFB is produced without
    // the object file consumed by Oneliner's Rust linker.
    configure_optimization(&mut preprocess);
    configure_static_link(&mut preprocess, object);
    preprocess
        .arg("--compile-to=preprocessing")
        .arg("--emit-mlir-bytecode=false")
        .arg("-o")
        .arg(&preprocessing);
    run_command(&mut preprocess, "IREE preprocessing for vMCU")?;

    // Resolve relative to the macro crate, not the consuming application's
    // working directory, so Cargo can invoke the checked-in implementation.
    let rewriter = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("rewrite_vmcu.py");
    let mode_name = match mode {
        VmcuArg::Auto => "auto",
        VmcuArg::Strict => "strict",
        VmcuArg::Off => unreachable!("off mode does not enter the vMCU pipeline"),
    };
    let mut rewrite = Command::new(python_executable());
    rewrite
        .arg(rewriter)
        .arg(&preprocessing)
        .arg("-o")
        .arg(&rewritten)
        .arg("--plan-output")
        .arg(&plan)
        .arg("--mode")
        .arg(mode_name)
        .arg("--schedule-search")
        .arg(vmcu_schedule.as_str())
        // The Python bindings parse the phase boundary while this executable
        // resumes it.  Exact version equality is therefore a correctness
        // condition rather than an informational dependency check.
        .arg("--iree-compile")
        .arg("iree-compile");
    if vmcu_schedule == VmcuScheduleArg::Bounded {
        rewrite
            .arg("--search-state-limit")
            .arg(vmcu_search_states.to_string());
    }
    if let Some(bytes) = vmcu_sram {
        rewrite.arg("--sram-budget").arg(bytes.to_string());
    }
    run_command(&mut rewrite, "Python vMCU graph rewriter")?;

    // Resume at the exact phase boundary used by the first invocation. The
    // Python output contains only standard dialects understood by stock IREE.
    let mut compile = Command::new("iree-compile");
    configure_target(&mut compile, &rewritten, target);
    compile.arg("--compile-from=preprocessing");
    configure_final_pipeline(&mut compile, vmfb, object, ir_dump_dir);
    run_command(&mut compile, "iree-compile from rewritten preprocessing IR")?;

    let rewritten_stem = rewritten
        .file_stem()
        .and_then(OsStr::to_str)
        .map(rust_ident)
        .unwrap_or_else(|| "vmcu_rewritten".to_owned());
    let resource_status =
        run_resource_report(&plan, object, ir_dump_dir, &rewritten_stem, "rewritten")?;
    if resource_status == ResourceStatus::ExceedsBudget {
        match mode {
            VmcuArg::Strict => {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "rewritten object exceeds vmcu_sram; see vmcu.plan.json",
                ));
            }
            VmcuArg::Auto => {
                // Auto mode must not deploy an over-budget object. Recompile
                // the immutable preprocessing input and account for it again.
                let mut fallback = Command::new("iree-compile");
                configure_target(&mut fallback, &preprocessing, target);
                fallback.arg("--compile-from=preprocessing");
                configure_final_pipeline(&mut fallback, vmfb, object, ir_dump_dir);
                run_command(&mut fallback, "IREE baseline fallback for vmcu_sram")?;
                let fallback_stem = preprocessing
                    .file_stem()
                    .and_then(OsStr::to_str)
                    .map(rust_ident)
                    .unwrap_or_else(|| "vmcu_preprocessing".to_owned());
                if run_resource_report(
                    &plan,
                    object,
                    ir_dump_dir,
                    &fallback_stem,
                    "baseline-fallback",
                )? == ResourceStatus::ExceedsBudget
                {
                    return Err(syn::Error::new(
                        Span::call_site(),
                        "baseline fallback also exceeds vmcu_sram; see vmcu.plan.json",
                    ));
                }
                eprintln!("[oneliner] vMCU auto fallback selected baseline for SRAM budget");
                return Ok(fallback_stem);
            }
            VmcuArg::Off => unreachable!("off mode does not enter resource reporting"),
        }
    }

    eprintln!("[oneliner] vMCU rewrite report: {}", plan.display());
    // IREE derives dump filenames from the resumed input filename. Return its
    // normalized stem so artifact discovery opens the correct phase-10 file.
    Ok(rewritten_stem)
}

/// Runs the final-object resource analyzer and preserves its budget status.
fn run_resource_report(
    plan: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    dump_stem: &str,
    deployment: &str,
) -> syn::Result<ResourceStatus> {
    let reporter = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("report_vmcu_resources.py");
    let stream = ir_dump_dir.join(format!("{dump_stem}.7.stream.mlir"));
    let executable = ir_dump_dir.join(format!("{dump_stem}.10.executable-targets.mlir"));
    let mut command = Command::new(python_executable());
    command
        .arg(reporter)
        .arg("--plan")
        .arg(plan)
        .arg("--stream")
        .arg(stream)
        .arg("--executable")
        .arg(executable)
        .arg("--object")
        .arg(object)
        .arg("--deployment")
        .arg(deployment);
    // Cross builds normally provide a target-aware objdump through this
    // conventional environment variable; native builds use the script default.
    if let Some(objdump) = std::env::var_os("OBJDUMP") {
        command.arg("--objdump").arg(objdump);
    }
    let status = command.status().map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("failed to start vMCU resource analyzer: {error}"),
        )
    })?;
    match status.code() {
        Some(0) => Ok(ResourceStatus::WithinBudget),
        Some(3) => Ok(ResourceStatus::ExceedsBudget),
        code => Err(syn::Error::new(
            Span::call_site(),
            format!("vMCU resource analyzer failed with status {code:?}"),
        )),
    }
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
                .join("iree_stream_flow_to_rust.py"),
        )
        .arg(input)
        .arg("--rust-output")
        .arg(rust_output)
        .arg("--json-output")
        .arg(json_output);
    run_command(&mut command, "IREE Stream/Flow converter")
}

/// Returns the configured Python interpreter for every build-time helper.
fn python_executable() -> std::ffi::OsString {
    // PYTHON lets users bind all helper scripts to the same environment as the
    // pinned IREE package; the conventional executable remains the default.
    std::env::var_os("PYTHON").unwrap_or_else(|| "python".into())
}

#[cfg(test)]
mod tests {
    use super::ceiling_power_of_two;

    #[test]
    fn stack_guard_rounds_up_without_rejecting_exact_budget_analysis() {
        // The final object report, not this compiler guard, decides whether
        // non-power-of-two vmcu_sram values are deployable.
        assert_eq!(ceiling_power_of_two(1), 1);
        assert_eq!(ceiling_power_of_two(500), 512);
        assert_eq!(ceiling_power_of_two(65_536), 65_536);
    }
}
