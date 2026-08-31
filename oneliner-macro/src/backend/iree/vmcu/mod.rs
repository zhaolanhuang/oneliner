//! IREE-specific vMCU integration boundary.
//!
//! Configuration, split compilation, schema-v4 interpretation, and compact
//! Rust ABI generation stay behind this facade. The surrounding IREE backend
//! consumes only a deployment result and the neutral `IoLayout` it resolves.

mod codegen;
mod options;
mod pipeline;
mod plan;

pub(super) use codegen::compact_fragments;
pub(super) use options::Options;
pub(super) use pipeline::compile;
pub(super) use plan::{CompactIo, ResourceUsage};

use proc_macro2::Span;

use super::metadata::FlowIo;
use super::{IoLayout, IoView};

pub(super) fn resolve_io(flow: FlowIo, compact: Option<CompactIo>) -> syn::Result<IoLayout> {
    match (flow, compact) {
        (FlowIo::Separate { input, output }, None) => Ok(IoLayout::Separate {
            input_size: input.size,
            output_size: output.size,
        }),
        (FlowIo::InPlace { size }, Some(compact)) => {
            if size != compact.storage_size {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "vMCU plan and Flow inout pool sizes differ",
                ));
            }
            Ok(IoLayout::InPlace {
                storage_size: compact.storage_size,
                input: IoView {
                    offset: compact.input_offset,
                    size: compact.input_size,
                },
                output: IoView {
                    offset: compact.output_offset,
                    size: compact.output_size,
                },
            })
        }
        (FlowIo::Separate { .. }, Some(_)) => Err(syn::Error::new(
            Span::call_site(),
            "applied vMCU plan requires an external read/write Flow resource",
        )),
        (FlowIo::InPlace { .. }, None) => Err(syn::Error::new(
            Span::call_site(),
            "Flow exposes an inout resource without an applied compact plan",
        )),
    }
}
