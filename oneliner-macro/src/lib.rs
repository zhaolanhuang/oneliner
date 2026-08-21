mod args;
mod backend;
mod frontend;
mod utils;

use proc_macro::TokenStream;
use syn::{AttributeArgs, ItemStruct, parse_macro_input};

/// Expands `#[model(...)]` on a unit struct into backend-specific model bindings.
///
/// Input: attribute arguments and the annotated struct item.
/// Output: generated Rust tokens or `compile_error!` tokens on failure. IREE models
/// accept `arena = "owned"` (the default) or `arena = "shared"`, and optionally
/// `vmcu = "pointwise-pair"`.
#[proc_macro_attribute]
pub fn model(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as AttributeArgs);
    let input_struct = parse_macro_input!(item as ItemStruct);

    let expanded = args::ModelArgs::parse(attr).and_then(|args| {
        let model = frontend::prepare(&args, &input_struct)?;
        backend::expand(args.backend, args.arena, args.vmcu, input_struct, model)
    });
    match expanded {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}
