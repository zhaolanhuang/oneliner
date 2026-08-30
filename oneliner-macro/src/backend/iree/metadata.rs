use std::{fs, path::Path};

use proc_macro2::Span;
use serde::Deserialize;
use syn::Ident;

use super::BindingArtifact;
use crate::utils::parse_ident;

#[derive(Debug)]
pub(super) struct FlowMetadata {
    pub execute_fns: Vec<Ident>,
    pub input: Option<BindingArtifact>,
    pub output: BindingArtifact,
    pub io_pool: Option<CompactIoMetadata>,
    /// Deduplicated constant/weight bytes placed in flash.
    pub params_size: usize,
    /// Deduplicated transient workspace bytes held in RAM.
    pub ram_size: usize,
}

#[derive(Debug, Clone)]
pub(super) struct CompactIoMetadata {
    pub allocated_bytes: usize,
    pub input_offset: usize,
    pub input_size: usize,
    pub output_offset: usize,
    pub output_size: usize,
}

#[derive(Deserialize)]
struct Metadata {
    cmd_executes: Vec<Execute>,
}

#[derive(Deserialize)]
struct VmcuPlan {
    schema_version: usize,
    applied: bool,
    compact_graph: serde_json::Value,
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

#[derive(Deserialize)]
struct Execute {
    name: String,
    resources: Vec<Resource>,
}

#[derive(Deserialize)]
struct Resource {
    static_ident: Option<String>,
    size: Option<usize>,
    role: Role,
}

#[derive(Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
enum Role {
    Input,
    Output,
    Constant,
    Temporary,
    Inout,
    #[serde(other)]
    Other,
}

pub(super) fn load_compact_metadata(path: &Path) -> syn::Result<Option<CompactIoMetadata>> {
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

    Ok(Some(CompactIoMetadata {
        allocated_bytes: graph.allocated_pool_bytes,
        input_offset,
        input_size: input.size_bytes,
        output_offset,
        output_size: output.size_bytes,
    }))
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

pub(super) fn load_metadata(
    path: &Path,
    compact: Option<CompactIoMetadata>,
) -> syn::Result<FlowMetadata> {
    let metadata: Metadata = serde_json::from_str(
        &fs::read_to_string(path).map_err(|error| syn::Error::new(Span::call_site(), error))?,
    )
    .map_err(|error| syn::Error::new(Span::call_site(), error))?;

    let execute_fns = metadata
        .cmd_executes
        .iter()
        .map(|execute| parse_ident(&execute.name, "generated IREE execute function"))
        .collect::<syn::Result<_>>()?;
    let resources: Vec<&Resource> = metadata
        .cmd_executes
        .iter()
        .flat_map(|execute| execute.resources.iter())
        .collect();
    let (input, output, io_pool) = if let Some(pool) = compact {
        let inout_resources: Vec<_> = resources
            .iter()
            .filter(|resource| resource.role == Role::Inout)
            .collect();
        if inout_resources.is_empty() {
            return Err(syn::Error::new(
                Span::call_site(),
                "applied vMCU plan requires an external read/write Flow resource",
            ));
        }
        if resources
            .iter()
            .any(|resource| matches!(resource.role, Role::Input | Role::Output))
        {
            return Err(syn::Error::new(
                Span::call_site(),
                "applied vMCU Flow must not expose separate input or output resources",
            ));
        }
        if inout_resources
            .iter()
            .any(|resource| resource.size != Some(pool.allocated_bytes))
        {
            return Err(syn::Error::new(
                Span::call_site(),
                "vMCU plan and Flow inout pool sizes differ",
            ));
        }
        (
            Some(BindingArtifact {
                size: pool.input_size,
            }),
            BindingArtifact {
                size: pool.output_size,
            },
            Some(pool),
        )
    } else {
        if resources
            .iter()
            .any(|resource| resource.role == Role::Inout)
        {
            return Err(syn::Error::new(
                Span::call_site(),
                "Flow exposes an inout resource without an applied compact plan",
            ));
        }
        let input = resources
            .iter()
            .find(|resource| resource.role == Role::Input)
            .and_then(|resource| resource.size)
            .map(|size| BindingArtifact { size });
        let output = resources
            .iter()
            .find(|resource| resource.role == Role::Output)
            .and_then(|resource| resource.size)
            .map(|size| BindingArtifact { size })
            .ok_or_else(|| syn::Error::new(Span::call_site(), "missing output binding"))?;
        (input, output, None)
    };

    let (params_size, ram_size) = footprint_sizes(resources.iter().copied());

    Ok(FlowMetadata {
        execute_fns,
        input,
        output,
        io_pool,
        params_size,
        ram_size,
    })
}

/// Sums constant and transient resource sizes, counting each storage blob once.
///
/// A constant or temporary shared by several `cmd_execute` blocks is rendered
/// as a single Rust static, so it must contribute to the footprint only once.
fn footprint_sizes<'a>(resources: impl Iterator<Item = &'a Resource>) -> (usize, usize) {
    let mut params_size = 0usize;
    let mut ram_size = 0usize;
    let mut seen_flash = std::collections::HashSet::new();
    let mut seen_ram = std::collections::HashSet::new();
    for resource in resources {
        match (&resource.role, resource.size) {
            (Role::Constant, Some(size)) => {
                if seen_flash.insert(resource.static_ident.as_deref()) {
                    params_size += size;
                }
            }
            (Role::Temporary, Some(size)) => {
                if seen_ram.insert(resource.static_ident.as_deref()) {
                    ram_size += size;
                }
            }
            _ => {}
        }
    }
    (params_size, ram_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_footprint_deduplicating_shared_resources() {
        let resources = [
            Resource {
                static_ident: Some("CONST_ARG2".into()),
                size: Some(100),
                role: Role::Constant,
            },
            Resource {
                static_ident: Some("CONST_ARG2".into()),
                size: Some(100),
                role: Role::Constant,
            },
            Resource {
                static_ident: Some("CONST_ARG5".into()),
                size: Some(7),
                role: Role::Constant,
            },
            Resource {
                static_ident: Some("TEMP_ARG4".into()),
                size: Some(50),
                role: Role::Temporary,
            },
            Resource {
                static_ident: Some("TEMP_ARG9".into()),
                size: Some(50),
                role: Role::Temporary,
            },
            Resource {
                static_ident: Some("INPUT_ARG1".into()),
                size: Some(16),
                role: Role::Input,
            },
        ];
        let (flash, ram) = footprint_sizes(resources.iter());
        assert_eq!(flash, 107);
        assert_eq!(ram, 100);
    }

    #[test]
    fn deserializes_constant_and_temporary_roles() {
        let value: Metadata = serde_json::from_str(
            r#"{
                "cmd_executes": [{
                    "name": "cmd_execute_0",
                    "resources": [
                        {"static_ident": "CONST_ARG2", "kind": "constant", "size": 100, "role": "constant"},
                        {"static_ident": "TEMP_ARG4", "kind": "transient", "size": 50, "role": "temporary"}
                    ]
                }]
            }"#,
        )
        .unwrap();
        assert_eq!(value.cmd_executes[0].resources[0].role, Role::Constant);
        assert_eq!(value.cmd_executes[0].resources[1].role, Role::Temporary);
    }
}
