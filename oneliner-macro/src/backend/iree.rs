mod artifacts;
mod codegen;
mod discovery;
mod metadata;
mod object_size;
mod toolchain;

use std::path::PathBuf;

use proc_macro2::TokenStream;
use syn::Ident;

use crate::args::{ArenaArg, VmcuArg, VmcuScheduleArg};
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
    io_pool: Option<metadata::CompactIoMetadata>,
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
    vmcu: VmcuArg,
    vmcu_sram: Option<usize>,
    vmcu_schedule: VmcuScheduleArg,
    vmcu_search_states: usize,
) -> syn::Result<TokenStream> {
    let artifacts = artifacts::build(
        &input_struct.ident,
        model,
        vmcu,
        vmcu_sram,
        vmcu_schedule,
        vmcu_search_states,
    )?;

    let params_size = artifacts.params_size;
    let code_size = artifacts.code_size;
    let rodata_size = artifacts.rodata_size;
    let total_flash_size = params_size + code_size + rodata_size;
    let ram_size = artifacts.ram_size;
    let input_size = artifacts.input.size;
    let output_size = artifacts.output.size;
    let io_pool_size = artifacts
        .io_pool
        .as_ref()
        .map_or(0, |pool| pool.allocated_bytes);

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
    if io_pool_size != 0 {
        eprintln!(
            "  RAM Usage: io_pool = {} B ({} KiB), transient arena = {} B ({} KiB)",
            io_pool_size,
            io_pool_size / 1024,
            ram_size,
            ram_size / 1024,
        );
        eprintln!(
            "  Logical pool views: input = {} B, output = {} B (not additional RAM)",
            input_size, output_size,
        );
    } else {
        eprintln!(
            "  RAM Usage: arena = {} B ({} KiB), input = {} B ({} KiB), output = {} B ({} KiB)",
            ram_size,
            ram_size / 1024,
            input_size,
            input_size / 1024,
            output_size,
            output_size / 1024,
        );
    }

    let expanded = codegen::expand(input_struct, artifacts, arena);
    // eprintln!("generated:\n{expanded}");
    Ok(expanded.into())
}
