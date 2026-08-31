use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use proc_macro2::Span;

use super::options::{EnabledOptions, Mode, Search};
use super::plan::{load_compact_io, load_resource_usage, CompactIo, ResourceUsage};
use crate::backend::iree::toolchain::{python_executable, Compiler};
use crate::utils::{run_command, rust_ident};

pub(crate) struct Deployment {
    pub(crate) dump_stem: String,
    pub(crate) compact_io: Option<CompactIo>,
    pub(crate) resources: ResourceUsage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResourceStatus {
    WithinBudget,
    ExceedsBudget,
}

pub(crate) fn compile(
    options: EnabledOptions,
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    artifact_dir: &Path,
) -> syn::Result<Deployment> {
    let compiler = Compiler::new(options.sram_budget.map(ceiling_power_of_two))?;
    let preprocessing = artifact_dir.join("vmcu.preprocessing.mlir");
    let rewritten = artifact_dir.join("vmcu.rewritten.mlir");
    let plan = artifact_dir.join("vmcu.plan.json");

    compiler.compile_preprocessing(compile_input, &preprocessing, object)?;
    run_rewriter(&options, &preprocessing, &rewritten, &plan)?;
    compiler.compile_from_preprocessing(&rewritten, vmfb, object, ir_dump_dir)?;

    let rewritten_stem = file_stem(&rewritten, "vmcu_rewritten");
    let rewritten_resources =
        run_resource_report(&plan, object, ir_dump_dir, &rewritten_stem, "rewritten")?;
    if rewritten_resources.status == ResourceStatus::ExceedsBudget {
        match options.mode {
            Mode::Strict => {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "rewritten object exceeds vmcu_sram; see vmcu.plan.json",
                ));
            }
            Mode::Auto => {
                compiler.compile_from_preprocessing(&preprocessing, vmfb, object, ir_dump_dir)?;
                let fallback_stem = file_stem(&preprocessing, "vmcu_preprocessing");
                let fallback_resources = run_resource_report(
                    &plan,
                    object,
                    ir_dump_dir,
                    &fallback_stem,
                    "baseline-fallback",
                )?;
                if fallback_resources.status == ResourceStatus::ExceedsBudget {
                    return Err(syn::Error::new(
                        Span::call_site(),
                        "baseline fallback also exceeds vmcu_sram; see vmcu.plan.json",
                    ));
                }
                eprintln!("[oneliner] vMCU auto fallback selected baseline for SRAM budget");
                return Ok(Deployment {
                    dump_stem: fallback_stem,
                    compact_io: None,
                    resources: fallback_resources.usage,
                });
            }
        }
    }

    eprintln!("[oneliner] vMCU rewrite report: {}", plan.display());
    Ok(Deployment {
        dump_stem: rewritten_stem,
        compact_io: load_compact_io(&plan)?,
        resources: rewritten_resources.usage,
    })
}

struct ResourceReport {
    status: ResourceStatus,
    usage: ResourceUsage,
}

fn run_rewriter(
    options: &EnabledOptions,
    preprocessing: &Path,
    rewritten: &Path,
    plan: &Path,
) -> syn::Result<()> {
    let rewriter = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("rewrite_vmcu.py");
    let mut command = Command::new(python_executable());
    command
        .arg(rewriter)
        .arg(preprocessing)
        .arg("-o")
        .arg(rewritten)
        .arg("--plan-output")
        .arg(plan)
        .arg("--mode")
        .arg(options.mode.as_str())
        .arg("--schedule-search")
        .arg(options.search.as_str())
        .arg("--iree-compile")
        .arg("iree-compile");
    if let Search::Bounded { state_limit } = options.search {
        command
            .arg("--search-state-limit")
            .arg(state_limit.to_string());
    }
    if let Some(bytes) = options.sram_budget {
        command.arg("--sram-budget").arg(bytes.to_string());
    }
    run_command(&mut command, "Python vMCU graph rewriter")
}

fn run_resource_report(
    plan: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    dump_stem: &str,
    deployment: &str,
) -> syn::Result<ResourceReport> {
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
    let status = command.status().map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("failed to start vMCU resource analyzer: {error}"),
        )
    })?;
    let status = match status.code() {
        Some(0) => ResourceStatus::WithinBudget,
        Some(3) => ResourceStatus::ExceedsBudget,
        code => Err(syn::Error::new(
            Span::call_site(),
            format!("vMCU resource analyzer failed with status {code:?}"),
        ))?,
    };
    Ok(ResourceReport {
        status,
        usage: load_resource_usage(plan)?,
    })
}

fn file_stem(path: &Path, fallback: &str) -> String {
    path.file_stem()
        .and_then(OsStr::to_str)
        .map(rust_ident)
        .unwrap_or_else(|| fallback.to_owned())
}

fn ceiling_power_of_two(value: usize) -> usize {
    debug_assert!(value > 0);
    value
        .checked_next_power_of_two()
        .unwrap_or(1usize << (usize::BITS - 1))
}

#[cfg(test)]
mod tests {
    use super::ceiling_power_of_two;

    #[test]
    fn stack_guard_rounds_up_without_rejecting_exact_budget_analysis() {
        assert_eq!(ceiling_power_of_two(1), 1);
        assert_eq!(ceiling_power_of_two(500), 512);
        assert_eq!(ceiling_power_of_two(65_536), 65_536);
    }
}
