mod artifacts;
mod codegen;
mod discovery;
mod metadata;
mod toolchain;

use std::path::PathBuf;

use proc_macro2::TokenStream;
use syn::Ident;

use crate::args::ArenaArg;
use crate::frontend::{Model, TensorInfo};

#[derive(Debug)]
struct ArtifactPaths {
    model: PathBuf,
    compile_input: PathBuf,
    object: PathBuf,
    ir: PathBuf,
    flow_rs: PathBuf,
    metadata_json: PathBuf,
}

#[derive(Debug, Clone)]
struct BindingArtifact {
    size: usize,
}

#[derive(Debug)]
struct IreeArtifacts {
    paths: ArtifactPaths,
    query_fn: Ident,
    query_link_name: String,
    execute_fns: Vec<Ident>,
    input: BindingArtifact,
    output: BindingArtifact,
    input_tensor: TensorInfo,
    output_tensor: TensorInfo,
}

pub fn expand(
    input_struct: syn::ItemStruct,
    model: Model,
    arena: ArenaArg,
    cmsis_nn: bool,
) -> syn::Result<TokenStream> {
    let artifacts = artifacts::build(&input_struct.ident, model, cmsis_nn)?;

    let expanded = codegen::expand(input_struct, artifacts, arena);
    // eprintln!("generated:\n{expanded}");
    Ok(expanded.into())
}
