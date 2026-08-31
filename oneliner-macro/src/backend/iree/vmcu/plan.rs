use std::{fs, path::Path};

use proc_macro2::Span;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub(crate) struct CompactIo {
    pub(super) storage_size: usize,
    pub(super) input_offset: usize,
    pub(super) input_size: usize,
    pub(super) output_offset: usize,
    pub(super) output_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResourceUsage {
    pub(crate) io_pool_size: usize,
    pub(crate) transient_size: usize,
    pub(crate) stack_size: usize,
    pub(crate) total_size: usize,
}

#[derive(Deserialize)]
struct VmcuPlan {
    schema_version: usize,
    applied: bool,
    compact_graph: serde_json::Value,
}

#[derive(Deserialize)]
struct ResourcePlan {
    resources: Resources,
}

#[derive(Deserialize)]
struct Resources {
    arena_bytes: usize,
    io_pool_allocated_bytes: usize,
    stack_bytes: usize,
    total_sram_bytes: usize,
}

#[derive(Deserialize)]
struct CompactGraph {
    allocated_pool_bytes: usize,
    tensors: Vec<CompactTensor>,
    placements: Vec<CompactPlacement>,
}

#[derive(Deserialize)]
struct CompactTensor {
    name: String,
    size_bytes: usize,
    #[serde(default)]
    is_graph_input: bool,
    #[serde(default)]
    is_graph_output: bool,
}

#[derive(Deserialize)]
struct CompactPlacement {
    tensor: String,
    base: usize,
}

pub(super) fn load_compact_io(path: &Path) -> syn::Result<Option<CompactIo>> {
    let plan: VmcuPlan = serde_json::from_str(
        &fs::read_to_string(path).map_err(|error| syn::Error::new(Span::call_site(), error))?,
    )
    .map_err(|error| syn::Error::new(Span::call_site(), error))?;
    if plan.schema_version != 4 {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "unsupported vMCU plan schema {}; expected schema 4",
                plan.schema_version
            ),
        ));
    }
    if !plan.applied {
        return Ok(None);
    }

    let graph: CompactGraph = serde_json::from_value(plan.compact_graph)
        .map_err(|error| syn::Error::new(Span::call_site(), error))?;
    if graph.allocated_pool_bytes == 0 {
        return Err(syn::Error::new(
            Span::call_site(),
            "applied vMCU plan has an empty I/O pool",
        ));
    }
    let input = unique_graph_tensor(&graph.tensors, true)?;
    let output = unique_graph_tensor(&graph.tensors, false)?;
    let input_offset = placement_base(&graph.placements, &input.name)?;
    let output_offset = placement_base(&graph.placements, &output.name)?;
    validate_view(
        "input",
        input_offset,
        input.size_bytes,
        graph.allocated_pool_bytes,
    )?;
    validate_view(
        "output",
        output_offset,
        output.size_bytes,
        graph.allocated_pool_bytes,
    )?;

    Ok(Some(CompactIo {
        storage_size: graph.allocated_pool_bytes,
        input_offset,
        input_size: input.size_bytes,
        output_offset,
        output_size: output.size_bytes,
    }))
}

pub(super) fn load_resource_usage(path: &Path) -> syn::Result<ResourceUsage> {
    let plan: ResourcePlan = serde_json::from_str(
        &fs::read_to_string(path).map_err(|error| syn::Error::new(Span::call_site(), error))?,
    )
    .map_err(|error| syn::Error::new(Span::call_site(), error))?;
    Ok(ResourceUsage {
        io_pool_size: plan.resources.io_pool_allocated_bytes,
        transient_size: plan.resources.arena_bytes,
        stack_size: plan.resources.stack_bytes,
        total_size: plan.resources.total_sram_bytes,
    })
}

fn unique_graph_tensor(tensors: &[CompactTensor], input: bool) -> syn::Result<&CompactTensor> {
    let mut matches = tensors.iter().filter(|tensor| {
        if input {
            tensor.is_graph_input
        } else {
            tensor.is_graph_output
        }
    });
    let tensor = matches.next().ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            format!(
                "vMCU plan has no graph {} tensor",
                if input { "input" } else { "output" }
            ),
        )
    })?;
    if matches.next().is_some() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "vMCU plan has multiple graph {} tensors",
                if input { "input" } else { "output" }
            ),
        ));
    }
    Ok(tensor)
}

fn placement_base(placements: &[CompactPlacement], tensor: &str) -> syn::Result<usize> {
    placements
        .iter()
        .find(|placement| placement.tensor == tensor)
        .map(|placement| placement.base)
        .ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                format!("vMCU plan has no placement for {tensor}"),
            )
        })
}

fn validate_view(label: &str, offset: usize, size: usize, pool_size: usize) -> syn::Result<()> {
    let end = offset.checked_add(size);
    if size == 0 || end.is_none() || end.unwrap() > pool_size {
        return Err(syn::Error::new(
            Span::call_site(),
            format!("vMCU {label} view is outside the allocated I/O pool"),
        ));
    }
    Ok(())
}
