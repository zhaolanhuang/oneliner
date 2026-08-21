use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use proc_macro2::Span;
use syn::Ident;

use super::discovery::parse_query_function;
use super::metadata::load_metadata;
use super::object_size::measure_object;
use super::toolchain::{run_converter, run_iree_compile};
use super::{ArtifactPaths, BindingArtifact, IreeArtifacts, VmcuPlanArtifact};
use crate::args::VmcuArg;
use crate::frontend::{Model, TensorInfo};
use crate::utils::{required_path_env, rust_ident};

pub(super) fn build(
    struct_ident: &Ident,
    model: Model,
    vmcu: Option<VmcuArg>,
) -> syn::Result<IreeArtifacts> {
    let Model {
        source_path: model_path,
        compile_input_path,
        ir_dump_stem,
        model_io,
    } = model;
    let struct_name = rust_ident(&struct_ident.to_string());
    let manifest_dir = required_path_env("CARGO_MANIFEST_DIR")?;
    let out_root = std::env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("target").join("oneliner"));
    let model_stem = if model_path.is_dir() {
        model_path.file_name()
    } else {
        model_path.file_stem()
    }
    .and_then(OsStr::to_str)
    .map(rust_ident)
    .unwrap_or_else(|| struct_name.clone());
    let artifact_dir = out_root.join(format!("{struct_name}_iree_{model_stem}"));
    let ir_dump_dir = artifact_dir.join("iree-ir-dumps");
    if ir_dump_dir.exists() {
        fs::remove_dir_all(&ir_dump_dir)
            .map_err(|error| syn::Error::new(Span::call_site(), error))?;
    }
    fs::create_dir_all(&ir_dump_dir).map_err(|error| syn::Error::new(Span::call_site(), error))?;

    let vmfb_path = artifact_dir.join(format!("{model_stem}.vmfb"));
    let object_path = artifact_dir.join(format!("{model_stem}.o"));

    run_iree_compile(
        &compile_input_path,
        &vmfb_path,
        &object_path,
        &ir_dump_dir,
        vmcu,
    )?;
    let (query_fn, query_link_name) = parse_query_function(&object_path)?;
    let footprint = measure_object(&object_path)?;
    let vmcu_plan = match vmcu {
        Some(VmcuArg::PointwisePair) => Some(load_vmcu_plan(
            &artifact_dir.join("vmcu-pointwise.plan.json"),
        )?),
        None => None,
    };

    let ir_path = executable_targets_path(&ir_dump_dir, &ir_dump_stem)?;
    let flow_rs = artifact_dir.join(format!("{model_stem}.flow.rs"));
    let metadata_json = artifact_dir.join(format!("{model_stem}.flow.json"));
    run_converter(&ir_path, &flow_rs, &metadata_json)?;

    let metadata = load_metadata(&metadata_json)?;
    let input = metadata.input.ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            "IREE metadata does not contain an input binding",
        )
    })?;
    validate_tensor_size("input", &input, &model_io.input)?;
    validate_tensor_size("output", &metadata.output, &model_io.output)?;

    Ok(IreeArtifacts {
        paths: ArtifactPaths {
            model: model_path,
            compile_input: compile_input_path,
            object: object_path,
            ir: ir_path,
            flow_rs,
            metadata_json,
        },
        query_fn,
        query_link_name,
        execute_fns: metadata.execute_fns,
        input,
        output: metadata.output,
        input_tensor: model_io.input,
        output_tensor: model_io.output,
        params_size: metadata.params_size,
        code_size: footprint.code_size,
        rodata_size: footprint.rodata_size,
        ram_size: metadata.ram_size,
        vmcu_plan,
    })
}

#[derive(serde::Deserialize)]
struct VmcuPlanMetadata {
    schema_version: u32,
    full_intermediate_bytes: usize,
    segment_bytes: usize,
    saved_intermediate_bytes: usize,
}

fn load_vmcu_plan(path: &Path) -> syn::Result<VmcuPlanArtifact> {
    let plan: VmcuPlanMetadata = serde_json::from_str(
        &fs::read_to_string(path).map_err(|error| syn::Error::new(Span::call_site(), error))?,
    )
    .map_err(|error| syn::Error::new(Span::call_site(), error))?;
    if plan.schema_version != 1 {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("unsupported vMCU plan schema {}", plan.schema_version),
        ));
    }
    Ok(VmcuPlanArtifact {
        full_intermediate_bytes: plan.full_intermediate_bytes,
        segment_bytes: plan.segment_bytes,
        saved_intermediate_bytes: plan.saved_intermediate_bytes,
    })
}

fn executable_targets_path(ir_dump_dir: &Path, preferred_stem: &str) -> syn::Result<PathBuf> {
    let preferred = ir_dump_dir.join(format!("{preferred_stem}.10.executable-targets.mlir"));
    if preferred.is_file() {
        return Ok(preferred);
    }

    let entries = fs::read_dir(ir_dump_dir)
        .map_err(|error| syn::Error::new(Span::call_site(), error))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| syn::Error::new(Span::call_site(), error))?;
    let mut candidates = entries
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.ends_with(".10.executable-targets.mlir"))
        })
        .collect::<Vec<_>>();
    candidates.sort();
    match candidates.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err(syn::Error::new(
            Span::call_site(),
            format!(
                "IREE did not emit an executable-targets artifact in {}",
                ir_dump_dir.display()
            ),
        )),
        _ => Err(syn::Error::new(
            Span::call_site(),
            format!(
                "IREE emitted multiple executable-targets artifacts in {}",
                ir_dump_dir.display()
            ),
        )),
    }
}

fn validate_tensor_size(
    label: &str,
    binding: &BindingArtifact,
    tensor: &TensorInfo,
) -> syn::Result<()> {
    let tensor_size = tensor
        .byte_len()
        .expect("frontend validated tensor byte size");
    if tensor_size != binding.size {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "{label} tensor {:?} with element width {} occupies {} bytes, but the IREE binding occupies {} bytes",
                tensor.shape,
                tensor.element_type.byte_width(),
                tensor_size,
                binding.size,
            ),
        ));
    }
    Ok(())
}
