use syn::spanned::Spanned;
use syn::{AttributeArgs, Lit, LitStr, Meta, NestedMeta};

pub struct ModelArgs {
    pub model_path: LitStr,
    pub backend: BackendArg,
    pub arena: ArenaArg,
    pub format: Option<ModelFormat>,
    pub vmcu: Option<VmcuArg>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendArg {
    Iree,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaArg {
    Owned,
    Shared,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmcuArg {
    PointwisePair,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFormat {
    Mlir,
    Onnx,
    Pytorch,
    Tensorflow,
    Tflite,
}

impl ModelArgs {
    /// Parses `#[model("path", backend = "...", arena = "...", vmcu = "...", ...)]` arguments.
    ///
    /// Input: `syn::AttributeArgs` from the procedural macro entry point.
    /// Output: model path literal, backend selector, and backend options.
    pub fn parse(args: AttributeArgs) -> syn::Result<Self> {
        let mut args = args.into_iter();
        let model_path = match args.next() {
            Some(NestedMeta::Lit(Lit::Str(path))) => path,
            Some(arg) => return Err(syn::Error::new(arg.span(), "expected model path string")),
            None => {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "missing model path",
                ));
            }
        };

        let mut backend = None;
        let mut arena = None;
        let mut format = None;
        let mut vmcu = None;
        for arg in args {
            match arg {
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("backend") => {
                    if backend.is_some() {
                        return Err(syn::Error::new(meta.span(), "duplicate backend option"));
                    }
                    backend = Some(parse_backend(meta.lit)?);
                }
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("arena") => {
                    if arena.is_some() {
                        return Err(syn::Error::new(meta.span(), "duplicate arena option"));
                    }
                    arena = Some(parse_arena(meta.lit)?);
                }
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("format") => {
                    if format.is_some() {
                        return Err(syn::Error::new(meta.span(), "duplicate format option"));
                    }
                    format = Some(parse_format(meta.lit)?);
                }
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("vmcu") => {
                    if vmcu.is_some() {
                        return Err(syn::Error::new(meta.span(), "duplicate vmcu option"));
                    }
                    vmcu = Some(parse_vmcu(meta.lit)?);
                }
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "unknown #[model] option; expected backend, arena, format, or vmcu",
                    ));
                }
            }
        }

        Ok(Self {
            model_path,
            backend: backend.unwrap_or(BackendArg::Iree),
            arena: arena.unwrap_or(ArenaArg::Owned),
            format,
            vmcu,
        })
    }
}

fn parse_vmcu(lit: Lit) -> syn::Result<VmcuArg> {
    let Lit::Str(value) = lit else {
        return Err(syn::Error::new(
            lit.span(),
            "vmcu must be a string literal, for example vmcu = \"pointwise-pair\"",
        ));
    };

    match value.value().as_str() {
        "pointwise-pair" => Ok(VmcuArg::PointwisePair),
        other => Err(syn::Error::new(
            value.span(),
            format!("unknown vmcu mode '{other}', expected 'pointwise-pair'"),
        )),
    }
}

fn parse_format(lit: Lit) -> syn::Result<ModelFormat> {
    let Lit::Str(value) = lit else {
        return Err(syn::Error::new(
            lit.span(),
            "format must be a string literal, for example format = \"tensorflow\"",
        ));
    };
    match value.value().trim().to_ascii_lowercase().as_str() {
        "mlir" => Ok(ModelFormat::Mlir),
        "onnx" => Ok(ModelFormat::Onnx),
        "pytorch" | "pt2" => Ok(ModelFormat::Pytorch),
        "tensorflow" | "tf" => Ok(ModelFormat::Tensorflow),
        "tflite" => Ok(ModelFormat::Tflite),
        other => Err(syn::Error::new(
            value.span(),
            format!(
                "unknown model format '{other}', expected 'mlir', 'onnx', 'pytorch', 'tensorflow', or 'tflite'"
            ),
        )),
    }
}

/// Parses a backend string literal into a known backend selector.
///
/// Input: `backend = "..."` literal.
/// Output: `Ok(())` for IREE or a `syn::Error` for unsupported names.
fn parse_backend(lit: Lit) -> syn::Result<BackendArg> {
    let value = match lit {
        Lit::Str(value) => value,
        other => {
            return Err(syn::Error::new(
                other.span(),
                "backend must be a string literal, for example backend = \"iree\"",
            ));
        }
    };

    match value.value().trim().to_ascii_lowercase().as_str() {
        "iree" => Ok(BackendArg::Iree),
        other => Err(syn::Error::new(
            value.span(),
            format!("unknown backend '{other}', expected 'iree'"),
        )),
    }
}

fn parse_arena(lit: Lit) -> syn::Result<ArenaArg> {
    let value = match lit {
        Lit::Str(value) => value,
        other => {
            return Err(syn::Error::new(
                other.span(),
                "arena must be a string literal, for example arena = \"shared\"",
            ));
        }
    };

    match value.value().trim().to_ascii_lowercase().as_str() {
        "owned" => Ok(ArenaArg::Owned),
        "shared" => Ok(ArenaArg::Shared),
        other => Err(syn::Error::new(
            value.span(),
            format!("unknown arena '{other}', expected 'owned' or 'shared'"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_backend_options() {
        let args: AttributeArgs = vec![
            syn::parse_quote!("model.tflite"),
            syn::parse_quote!(backend = "iree"),
            syn::parse_quote!(backend = "iree"),
        ];

        assert!(ModelArgs::parse(args).is_err());
    }

    #[test]
    fn defaults_to_iree() {
        let args: AttributeArgs = vec![syn::parse_quote!("model.tflite")];

        assert!(ModelArgs::parse(args).is_ok());
    }

    #[test]
    fn parses_tensorflow_format() {
        let args: AttributeArgs = vec![
            syn::parse_quote!("saved_model"),
            syn::parse_quote!(format = "tensorflow"),
        ];

        let args = ModelArgs::parse(args).unwrap();
        assert_eq!(args.format, Some(ModelFormat::Tensorflow));
    }

    #[test]
    fn vmcu_defaults_to_disabled() {
        let args: AttributeArgs = vec![syn::parse_quote!("model.tflite")];

        let args = ModelArgs::parse(args).unwrap();
        assert_eq!(args.vmcu, None);
    }

    #[test]
    fn parses_pointwise_pair_vmcu() {
        let args: AttributeArgs = vec![
            syn::parse_quote!("model.tflite"),
            syn::parse_quote!(vmcu = "pointwise-pair"),
        ];

        let args = ModelArgs::parse(args).unwrap();
        assert_eq!(args.vmcu, Some(VmcuArg::PointwisePair));
    }

    #[test]
    fn rejects_duplicate_vmcu_options() {
        let args: AttributeArgs = vec![
            syn::parse_quote!("model.tflite"),
            syn::parse_quote!(vmcu = "pointwise-pair"),
            syn::parse_quote!(vmcu = "pointwise-pair"),
        ];

        let error = ModelArgs::parse(args)
            .err()
            .expect("duplicate vmcu rejected");
        assert_eq!(error.to_string(), "duplicate vmcu option");
    }

    #[test]
    fn rejects_non_string_vmcu() {
        let args: AttributeArgs = vec![
            syn::parse_quote!("model.tflite"),
            syn::parse_quote!(vmcu = true),
        ];

        let error = ModelArgs::parse(args)
            .err()
            .expect("non-string vmcu rejected");
        assert_eq!(
            error.to_string(),
            "vmcu must be a string literal, for example vmcu = \"pointwise-pair\""
        );
    }

    #[test]
    fn rejects_unknown_vmcu_mode() {
        let args: AttributeArgs = vec![
            syn::parse_quote!("model.tflite"),
            syn::parse_quote!(vmcu = "unsupported"),
        ];

        let error = ModelArgs::parse(args).err().expect("unknown vmcu rejected");
        assert_eq!(
            error.to_string(),
            "unknown vmcu mode 'unsupported', expected 'pointwise-pair'"
        );
    }
}
