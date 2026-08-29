use syn::spanned::Spanned;
use syn::{AttributeArgs, Lit, LitStr, Meta, NestedMeta};

pub struct ModelArgs {
    pub model_path: LitStr,
    pub backend: BackendArg,
    pub arena: ArenaArg,
    pub format: Option<ModelFormat>,
    /// Selects the optional post-preprocessing Python graph rewrite pipeline.
    pub vmcu: VmcuArg,
    /// Optional total deployable SRAM cap in bytes for vMCU builds.
    pub vmcu_sram: Option<usize>,
    /// Compact-DAG topology/base search policy.
    pub vmcu_schedule: VmcuScheduleArg,
    /// Deterministic explored-state cap for bounded scheduling.
    pub vmcu_search_states: usize,
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
/// Controls whether and how the experimental vMCU rewriter is invoked.
pub enum VmcuArg {
    /// Preserve the original one-shot IREE compilation pipeline.
    Off,
    /// Rewrite proven patterns and continue unchanged when no pattern matches.
    Auto,
    /// Rewrite proven patterns and fail the build when no pattern matches.
    Strict,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmcuScheduleArg {
    Bounded,
    Optimal,
    Greedy,
}

impl VmcuScheduleArg {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bounded => "bounded",
            Self::Optimal => "optimal",
            Self::Greedy => "greedy",
        }
    }
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
    /// Parses `#[model("path", backend = "...", arena = "...", ...)]` arguments.
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
        let mut vmcu_sram = None;
        let mut vmcu_schedule = None;
        let mut vmcu_search_states = None;
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
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("vmcu_sram") => {
                    if vmcu_sram.is_some() {
                        return Err(syn::Error::new(meta.span(), "duplicate vmcu_sram option"));
                    }
                    vmcu_sram = Some(parse_vmcu_sram(meta.lit)?);
                }
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("vmcu_schedule") => {
                    if vmcu_schedule.is_some() {
                        return Err(syn::Error::new(
                            meta.span(),
                            "duplicate vmcu_schedule option",
                        ));
                    }
                    vmcu_schedule = Some(parse_vmcu_schedule(meta.lit)?);
                }
                NestedMeta::Meta(Meta::NameValue(meta))
                    if meta.path.is_ident("vmcu_search_states") =>
                {
                    if vmcu_search_states.is_some() {
                        return Err(syn::Error::new(
                            meta.span(),
                            "duplicate vmcu_search_states option",
                        ));
                    }
                    vmcu_search_states =
                        Some(parse_positive_usize(meta.lit, "vmcu_search_states")?);
                }
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "unknown #[model] option; expected backend, arena, format, vmcu, vmcu_sram, vmcu_schedule, or vmcu_search_states",
                    ));
                }
            }
        }

        let vmcu = vmcu.unwrap_or(VmcuArg::Off);
        if vmcu_sram.is_some() && vmcu == VmcuArg::Off {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "vmcu_sram requires vmcu = \"auto\" or vmcu = \"strict\"",
            ));
        }
        if (vmcu_schedule.is_some() || vmcu_search_states.is_some()) && vmcu == VmcuArg::Off {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "vmcu_schedule and vmcu_search_states require vmcu = \"auto\" or vmcu = \"strict\"",
            ));
        }
        let vmcu_schedule = vmcu_schedule.unwrap_or(VmcuScheduleArg::Bounded);
        if vmcu_search_states.is_some() && vmcu_schedule != VmcuScheduleArg::Bounded {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "vmcu_search_states is valid only with vmcu_schedule = \"bounded\"",
            ));
        }
        Ok(Self {
            model_path,
            backend: backend.unwrap_or(BackendArg::Iree),
            arena: arena.unwrap_or(ArenaArg::Owned),
            format,
            vmcu,
            vmcu_sram,
            vmcu_schedule,
            vmcu_search_states: vmcu_search_states.unwrap_or(1_000_000),
        })
    }
}

fn parse_positive_usize(lit: Lit, name: &str) -> syn::Result<usize> {
    let Lit::Int(value) = lit else {
        return Err(syn::Error::new(
            lit.span(),
            format!("{name} must be a positive integer"),
        ));
    };
    let parsed = value.base10_parse::<usize>()?;
    if parsed == 0 {
        return Err(syn::Error::new(
            value.span(),
            format!("{name} must be greater than zero"),
        ));
    }
    Ok(parsed)
}

fn parse_vmcu_schedule(lit: Lit) -> syn::Result<VmcuScheduleArg> {
    let Lit::Str(value) = lit else {
        return Err(syn::Error::new(
            lit.span(),
            "vmcu_schedule must be \"bounded\", \"optimal\", or \"greedy\"",
        ));
    };
    match value.value().trim().to_ascii_lowercase().as_str() {
        "bounded" => Ok(VmcuScheduleArg::Bounded),
        "optimal" => Ok(VmcuScheduleArg::Optimal),
        "greedy" => Ok(VmcuScheduleArg::Greedy),
        other => Err(syn::Error::new(
            value.span(),
            format!("unknown vmcu_schedule '{other}', expected 'bounded', 'optimal', or 'greedy'"),
        )),
    }
}

/// Parses a positive byte count without accepting strings or negative values.
fn parse_vmcu_sram(lit: Lit) -> syn::Result<usize> {
    let Lit::Int(value) = lit else {
        return Err(syn::Error::new(
            lit.span(),
            "vmcu_sram must be a positive integer byte count",
        ));
    };
    let bytes = value.base10_parse::<usize>()?;
    if bytes == 0 {
        return Err(syn::Error::new(
            value.span(),
            "vmcu_sram must be greater than zero",
        ));
    }
    Ok(bytes)
}

/// Parses the user-facing `vmcu = "..."` model attribute.
fn parse_vmcu(lit: Lit) -> syn::Result<VmcuArg> {
    let Lit::Str(value) = lit else {
        return Err(syn::Error::new(
            lit.span(),
            "vmcu must be a string literal, for example vmcu = \"auto\"",
        ));
    };
    match value.value().trim().to_ascii_lowercase().as_str() {
        "off" => Ok(VmcuArg::Off),
        "auto" => Ok(VmcuArg::Auto),
        "strict" => Ok(VmcuArg::Strict),
        other => Err(syn::Error::new(
            value.span(),
            format!("unknown vmcu mode '{other}', expected 'off', 'auto', or 'strict'"),
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
    fn parses_vmcu_modes_and_defaults_to_off() {
        // Keeping vMCU opt-in avoids changing existing users' compiler output.
        let default_args: AttributeArgs = vec![syn::parse_quote!("model.mlir")];
        assert_eq!(ModelArgs::parse(default_args).unwrap().vmcu, VmcuArg::Off);

        for (value, expected) in [
            ("off", VmcuArg::Off),
            ("auto", VmcuArg::Auto),
            ("strict", VmcuArg::Strict),
        ] {
            let literal = LitStr::new(value, proc_macro2::Span::call_site());
            let args: AttributeArgs = vec![
                syn::parse_quote!("model.mlir"),
                syn::parse_quote!(vmcu = #literal),
            ];
            assert_eq!(ModelArgs::parse(args).unwrap().vmcu, expected);
        }
    }

    #[test]
    fn rejects_duplicate_and_unknown_vmcu_modes() {
        // Ambiguous or misspelled modes must fail during macro parsing, before
        // any external compiler command is started.
        let duplicate: AttributeArgs = vec![
            syn::parse_quote!("model.mlir"),
            syn::parse_quote!(vmcu = "auto"),
            syn::parse_quote!(vmcu = "strict"),
        ];
        assert!(ModelArgs::parse(duplicate).is_err());

        let unknown: AttributeArgs = vec![
            syn::parse_quote!("model.mlir"),
            syn::parse_quote!(vmcu = "fast"),
        ];
        assert!(ModelArgs::parse(unknown).is_err());
    }

    #[test]
    fn parses_and_validates_vmcu_sram() {
        // The cap is numeric so unit ambiguity cannot enter the build plan.
        let args: AttributeArgs = vec![
            syn::parse_quote!("model.mlir"),
            syn::parse_quote!(vmcu = "auto"),
            syn::parse_quote!(vmcu_sram = 65536),
        ];
        assert_eq!(ModelArgs::parse(args).unwrap().vmcu_sram, Some(65536));

        let disabled: AttributeArgs = vec![
            syn::parse_quote!("model.mlir"),
            syn::parse_quote!(vmcu_sram = 1024),
        ];
        assert!(ModelArgs::parse(disabled).is_err());

        let zero: AttributeArgs = vec![
            syn::parse_quote!("model.mlir"),
            syn::parse_quote!(vmcu = "strict"),
            syn::parse_quote!(vmcu_sram = 0),
        ];
        assert!(ModelArgs::parse(zero).is_err());
    }
}
