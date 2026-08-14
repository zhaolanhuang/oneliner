mod iree;
mod llvm_target_info;

use syn::ItemStruct;

use crate::args::{ArenaArg, BackendArg};
use crate::frontend::Model;

/// Delegates a frontend-validated model to the selected backend.
///
/// Input: backend options, annotated struct, and normalized model description.
/// Output: generated Rust tokens or a `syn::Error` for invalid input/backend failure.
pub fn expand(
    backend: BackendArg,
    arena: ArenaArg,
    cmsis_nn: bool,
    input_struct: ItemStruct,
    model: Model,
) -> syn::Result<proc_macro2::TokenStream> {
    match backend {
        BackendArg::Iree => iree::expand(input_struct, model, arena, cmsis_nn),
    }
}
