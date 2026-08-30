mod artifacts;
mod codegen;
mod discovery;
mod metadata;
mod object_size;
mod toolchain;
mod vmcu;

use std::path::PathBuf;

use proc_macro2::TokenStream;
use syn::{Ident, NestedMeta};

use crate::args::ArenaArg;
use crate::frontend::{Model, TensorInfo};

pub(crate) struct Options {
    vmcu: vmcu::Options,
}

impl Options {
    pub(super) fn parse(args: Vec<NestedMeta>) -> syn::Result<Self> {
        Ok(Self {
            vmcu: vmcu::Options::parse(args)?,
        })
    }
}

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

#[derive(Debug, Clone, Copy)]
struct IoView {
    offset: usize,
    size: usize,
}

#[derive(Debug)]
enum IoLayout {
    Separate {
        input_size: usize,
        output_size: usize,
    },
    InPlace {
        storage_size: usize,
        input: IoView,
        output: IoView,
    },
}

impl IoLayout {
    const fn input_size(&self) -> usize {
        match self {
            Self::Separate { input_size, .. } => *input_size,
            Self::InPlace { input, .. } => input.size,
        }
    }

    const fn output_size(&self) -> usize {
        match self {
            Self::Separate { output_size, .. } => *output_size,
            Self::InPlace { output, .. } => output.size,
        }
    }
}

#[derive(Debug)]
struct IreeArtifacts {
    paths: ArtifactPaths,
    query_fn: Ident,
    query_link_name: String,
    execute_fns: Vec<Ident>,
    io: IoLayout,
    input_tensor: TensorInfo,
    output_tensor: TensorInfo,
    params_size: usize,
    code_size: usize,
    rodata_size: usize,
    ram_size: usize,
}

/// Compiles IREE artifacts and emits the user-facing model implementation.
///
/// vMCU modes generate an in-place pool ABI; off mode keeps owned tensors.
pub fn expand(
    input_struct: syn::ItemStruct,
    model: Model,
    arena: ArenaArg,
    options: Options,
) -> syn::Result<TokenStream> {
    let artifacts = artifacts::build(&input_struct.ident, model, options.vmcu)?;

    let params_size = artifacts.params_size;
    let code_size = artifacts.code_size;
    let rodata_size = artifacts.rodata_size;
    let total_flash_size = params_size + code_size + rodata_size;
    let ram_size = artifacts.ram_size;
    let input_size = artifacts.io.input_size();
    let output_size = artifacts.io.output_size();

    eprintln!(
        "[oneliner-profiler] {} memory footprint:",
        input_struct.ident
    );
    eprintln!(
        "  Flash Usage: params = {} B ({} KiB), text(code) = {} B ({} KiB), rodata = {} B ({} KiB), total = {} B ({} KiB)",
        params_size,
        params_size / 1024,
        code_size,
        code_size / 1024,
        rodata_size,
        rodata_size / 1024,
        total_flash_size,
        total_flash_size / 1024,
    );
    match &artifacts.io {
        IoLayout::InPlace { storage_size, .. } => eprintln!(
            "  RAM Usage: io_pool = {} B ({} KiB), transient arena = {} B ({} KiB), input = {} B ({} KiB), output = {} B ({} KiB)",
            storage_size,
            storage_size / 1024,
            ram_size,
            ram_size / 1024,
            input_size,
            input_size / 1024,
            output_size,
            output_size / 1024,
        ),
        IoLayout::Separate { .. } => eprintln!(
            "  RAM Usage: arena = {} B ({} KiB), input = {} B ({} KiB), output = {} B ({} KiB)",
            ram_size,
            ram_size / 1024,
            input_size,
            input_size / 1024,
            output_size,
            output_size / 1024,
        ),
    }

    let expanded = codegen::expand(input_struct, artifacts, arena);
    // eprintln!("generated:\n{expanded}");
    Ok(expanded.into())
}
