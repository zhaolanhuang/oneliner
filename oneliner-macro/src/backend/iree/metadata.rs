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
    pub output_offset: usize,
}

#[derive(Deserialize)]
struct Metadata {
    cmd_executes: Vec<Execute>,
    io_pool: Option<IoPool>,
}

#[derive(Deserialize)]
struct IoPool {
    allocated_bytes: usize,
    input_view: IoView,
    output_view: IoView,
}

#[derive(Deserialize)]
struct IoView {
    offset: usize,
    size: usize,
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

pub(super) fn load_metadata(path: &Path) -> syn::Result<FlowMetadata> {
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
    let (input, output, io_pool) = if let Some(pool) = metadata.io_pool {
        let binding_size = resources
            .iter()
            .find(|resource| resource.role == Role::Inout)
            .and_then(|resource| resource.size)
            .ok_or_else(|| syn::Error::new(Span::call_site(), "missing vMCU inout pool"))?;
        if binding_size != pool.allocated_bytes {
            return Err(syn::Error::new(
                Span::call_site(),
                "vMCU plan and Flow inout pool sizes differ",
            ));
        }
        (
            Some(BindingArtifact {
                size: pool.input_view.size,
            }),
            BindingArtifact {
                size: pool.output_view.size,
            },
            Some(CompactIoMetadata {
                allocated_bytes: pool.allocated_bytes,
                input_offset: pool.input_view.offset,
                output_offset: pool.output_view.offset,
            }),
        )
    } else {
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
