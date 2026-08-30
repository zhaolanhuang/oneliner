mod iree;
mod llvm_target_info;

use syn::ItemStruct;
use syn::NestedMeta;

use crate::args::{ArenaArg, BackendArg};
use crate::frontend::Model;

pub(crate) enum Options {
    Iree(iree::Options),
}

pub(crate) fn parse_options(backend: BackendArg, args: Vec<NestedMeta>) -> syn::Result<Options> {
    match backend {
        BackendArg::Iree => iree::Options::parse(args).map(Options::Iree),
    }
}

/// Delegates a frontend-validated model to the selected backend.
///
/// Input: backend, arena and vMCU options, annotated struct, and normalized model.
/// Output: generated Rust tokens or a `syn::Error` for invalid input/backend failure.
pub fn expand(
    arena: ArenaArg,
    options: Options,
    input_struct: ItemStruct,
    model: Model,
) -> syn::Result<proc_macro2::TokenStream> {
    match options {
        Options::Iree(options) => iree::expand(input_struct, model, arena, options),
    }
}
