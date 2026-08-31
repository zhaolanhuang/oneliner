use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use proc_macro2::Span;

use super::options::EnabledOptions;
use super::plan::{load_compact_io, load_resource_usage, CompactIo, ResourceUsage};
use crate::backend::iree::toolchain::{python_executable, Compiler};
use crate::utils::{run_command, rust_ident};

pub(crate) struct Deployment {
    pub(crate) dump_stem: String,
    pub(crate) compact_io: Option<CompactIo>,
    pub(crate) resources: ResourceUsage,
}

pub(crate) fn compile(
    options: EnabledOptions,
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    artifact_dir: &Path,
) -> syn::Result<Deployment> {
    let compiler = Compiler::new()?;
    let preprocessing = artifact_dir.join("vmcu.preprocessing.mlir");
    let rewritten = artifact_dir.join("vmcu.rewritten.mlir");
    let plan = artifact_dir.join("vmcu.plan.json");

    compiler.compile_preprocessing(compile_input, &preprocessing, object)?;
    run_rewriter(&options, &preprocessing, &rewritten, &plan)?;
    compiler.compile_from_preprocessing(&rewritten, vmfb, object, ir_dump_dir)?;

    let rewritten_stem = file_stem(&rewritten, "vmcu_rewritten");
    let resources = run_resource_report(&plan, ir_dump_dir, &rewritten_stem)?;

    eprintln!("[oneliner] vMCU rewrite report: {}", plan.display());
    Ok(Deployment {
        dump_stem: rewritten_stem,
        compact_io: load_compact_io(&plan)?,
        resources,
    })
}

fn run_rewriter(
    options: &EnabledOptions,
    preprocessing: &Path,
    rewritten: &Path,
    plan: &Path,
) -> syn::Result<()> {
    let rewriter = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("oneliner_vmcu")
        .join("cli.py");
    let mut command = Command::new(python_executable());
    command
        .arg(rewriter)
        .arg(preprocessing)
        .arg("-o")
        .arg(rewritten)
        .arg("--plan-output")
        .arg(plan)
        .arg("--search-mode")
        .arg(options.search.as_str())
        .arg("--iree-compile")
        .arg("iree-compile");
    if let Some(budget) = options.search.budget() {
        command.arg("--search-budget").arg(budget.to_string());
    }
    run_command(&mut command, "Python vMCU graph rewriter")
}

fn run_resource_report(
    plan: &Path,
    ir_dump_dir: &Path,
    dump_stem: &str,
) -> syn::Result<ResourceUsage> {
    let reporter = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("oneliner_vmcu")
        .join("resource_cli.py");
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
        .arg(executable);
    let status = command.status().map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("failed to start vMCU resource analyzer: {error}"),
        )
    })?;
    if !status.success() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "vMCU resource analyzer failed with status {:?}",
                status.code()
            ),
        ));
    }
    load_resource_usage(plan)
}

fn file_stem(path: &Path, fallback: &str) -> String {
    path.file_stem()
        .and_then(OsStr::to_str)
        .map(rust_ident)
        .unwrap_or_else(|| fallback.to_owned())
}
