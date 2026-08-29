mod iree;
mod llvm_target_info;

use syn::ItemStruct;

use crate::args::{ArenaArg, BackendArg, VmcuArg, VmcuScheduleArg};
use crate::frontend::Model;

/// Delegates a frontend-validated model to the selected backend.
///
/// Input: backend, arena and vMCU options, annotated struct, and normalized model.
/// Output: generated Rust tokens or a `syn::Error` for invalid input/backend failure.
pub fn expand(
    backend: BackendArg,
    arena: ArenaArg,
    vmcu: VmcuArg,
    vmcu_sram: Option<usize>,
    vmcu_schedule: VmcuScheduleArg,
    vmcu_search_states: usize,
    input_struct: ItemStruct,
    model: Model,
) -> syn::Result<proc_macro2::TokenStream> {
    match backend {
        BackendArg::Iree => iree::expand(
            input_struct,
            model,
            arena,
            vmcu,
            vmcu_sram,
            vmcu_schedule,
            vmcu_search_states,
        ),
    }
}
