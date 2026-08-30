use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;

use proc_macro2::Span;
use syn::Ident;

use super::discovery::parse_query_function;
use super::metadata::load_metadata;
use super::object_size::measure_object;
use super::toolchain::{run_converter, Compiler};
use super::vmcu;
use super::{ArtifactPaths, IreeArtifacts};
use crate::frontend::{Model, TensorInfo};
use crate::utils::{required_path_env, rust_ident};

pub(super) fn build(
    struct_ident: &Ident,
    model: Model,
    vmcu_options: vmcu::Options,
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
    fs::create_dir_all(&ir_dump_dir).map_err(|error| syn::Error::new(Span::call_site(), error))?;

    let vmfb_path = artifact_dir.join(format!("{model_stem}.vmfb"));
    let object_path = artifact_dir.join(format!("{model_stem}.o"));

    let (final_dump_stem, compact_io) = match vmcu_options {
        vmcu::Options::Disabled => {
            Compiler::new(None)?.compile_full(
                &compile_input_path,
                &vmfb_path,
                &object_path,
                &ir_dump_dir,
            )?;
            (ir_dump_stem, None)
        }
        vmcu::Options::Enabled(options) => {
            let deployment = vmcu::compile(
                options,
                &compile_input_path,
                &vmfb_path,
                &object_path,
                &ir_dump_dir,
                &artifact_dir,
            )?;
            (deployment.dump_stem, deployment.compact_io)
        }
    };
    let (query_fn, query_link_name) = parse_query_function(&object_path)?;
    let footprint = measure_object(&object_path)?;

    let ir_path = ir_dump_dir.join(format!("{final_dump_stem}.10.executable-targets.mlir"));
    let flow_rs = artifact_dir.join(format!("{model_stem}.flow.rs"));
    let metadata_json = artifact_dir.join(format!("{model_stem}.flow.json"));
    run_converter(&ir_path, &flow_rs, &metadata_json)?;

    let metadata = load_metadata(&metadata_json)?;
    let io = vmcu::resolve_io(metadata.io, compact_io)?;
    validate_tensor_size("input", io.input_size(), &model_io.input)?;
    validate_tensor_size("output", io.output_size(), &model_io.output)?;

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
        io,
        input_tensor: model_io.input,
        output_tensor: model_io.output,
        params_size: metadata.params_size,
        code_size: footprint.code_size,
        rodata_size: footprint.rodata_size,
        ram_size: metadata.ram_size,
    })
}

fn validate_tensor_size(label: &str, binding_size: usize, tensor: &TensorInfo) -> syn::Result<()> {
    let tensor_size = tensor
        .byte_len()
        .expect("frontend validated tensor byte size");
    if tensor_size != binding_size {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "{label} tensor {:?} with element width {} occupies {} bytes, but the IREE binding occupies {} bytes",
                tensor.shape,
                tensor.element_type.byte_width(),
                tensor_size,
                binding_size,
            ),
        ));
    }
    Ok(())
}
