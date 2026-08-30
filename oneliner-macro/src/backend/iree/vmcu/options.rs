use proc_macro2::Span;
use syn::spanned::Spanned;
use syn::{Lit, Meta, NestedMeta};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Options {
    Disabled,
    Enabled(EnabledOptions),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EnabledOptions {
    pub(crate) mode: Mode,
    pub(crate) search: Search,
    pub(crate) sram_budget: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Auto,
    Strict,
}

impl Mode {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Strict => "strict",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Search {
    Bounded { state_limit: usize },
    Optimal,
    Greedy,
}

impl Search {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Bounded { .. } => "bounded",
            Self::Optimal => "optimal",
            Self::Greedy => "greedy",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParsedMode {
    Off,
    Auto,
    Strict,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParsedSearch {
    Bounded,
    Optimal,
    Greedy,
}

impl Options {
    pub(crate) fn parse(args: Vec<NestedMeta>) -> syn::Result<Self> {
        let mut mode = None;
        let mut sram_budget = None;
        let mut schedule = None;
        let mut search_states = None;

        for arg in args {
            match arg {
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("vmcu") => {
                    if mode.is_some() {
                        return Err(syn::Error::new(meta.span(), "duplicate vmcu option"));
                    }
                    mode = Some(parse_mode(meta.lit)?);
                }
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("vmcu_sram") => {
                    if sram_budget.is_some() {
                        return Err(syn::Error::new(meta.span(), "duplicate vmcu_sram option"));
                    }
                    sram_budget = Some(parse_positive_usize(meta.lit, "vmcu_sram")?);
                }
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("vmcu_schedule") => {
                    if schedule.is_some() {
                        return Err(syn::Error::new(
                            meta.span(),
                            "duplicate vmcu_schedule option",
                        ));
                    }
                    schedule = Some(parse_schedule(meta.lit)?);
                }
                NestedMeta::Meta(Meta::NameValue(meta))
                    if meta.path.is_ident("vmcu_search_states") =>
                {
                    if search_states.is_some() {
                        return Err(syn::Error::new(
                            meta.span(),
                            "duplicate vmcu_search_states option",
                        ));
                    }
                    search_states = Some(parse_positive_usize(meta.lit, "vmcu_search_states")?);
                }
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "unknown #[model] option; expected backend, arena, format, vmcu, vmcu_sram, vmcu_schedule, or vmcu_search_states",
                    ));
                }
            }
        }

        let mode = mode.unwrap_or(ParsedMode::Off);
        if mode == ParsedMode::Off {
            if sram_budget.is_some() {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "vmcu_sram requires vmcu = \"auto\" or vmcu = \"strict\"",
                ));
            }
            if schedule.is_some() || search_states.is_some() {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "vmcu_schedule and vmcu_search_states require vmcu = \"auto\" or vmcu = \"strict\"",
                ));
            }
            return Ok(Self::Disabled);
        }

        let schedule = schedule.unwrap_or(ParsedSearch::Bounded);
        if search_states.is_some() && schedule != ParsedSearch::Bounded {
            return Err(syn::Error::new(
                Span::call_site(),
                "vmcu_search_states is valid only with vmcu_schedule = \"bounded\"",
            ));
        }
        let search = match schedule {
            ParsedSearch::Bounded => Search::Bounded {
                state_limit: search_states.unwrap_or(1_000_000),
            },
            ParsedSearch::Optimal => Search::Optimal,
            ParsedSearch::Greedy => Search::Greedy,
        };
        let mode = match mode {
            ParsedMode::Auto => Mode::Auto,
            ParsedMode::Strict => Mode::Strict,
            ParsedMode::Off => unreachable!(),
        };
        Ok(Self::Enabled(EnabledOptions {
            mode,
            search,
            sram_budget,
        }))
    }
}

fn parse_mode(lit: Lit) -> syn::Result<ParsedMode> {
    let Lit::Str(value) = lit else {
        return Err(syn::Error::new(
            lit.span(),
            "vmcu must be a string literal, for example vmcu = \"auto\"",
        ));
    };
    match value.value().trim().to_ascii_lowercase().as_str() {
        "off" => Ok(ParsedMode::Off),
        "auto" => Ok(ParsedMode::Auto),
        "strict" => Ok(ParsedMode::Strict),
        other => Err(syn::Error::new(
            value.span(),
            format!("unknown vmcu mode '{other}', expected 'off', 'auto', or 'strict'"),
        )),
    }
}

fn parse_schedule(lit: Lit) -> syn::Result<ParsedSearch> {
    let Lit::Str(value) = lit else {
        return Err(syn::Error::new(
            lit.span(),
            "vmcu_schedule must be \"bounded\", \"optimal\", or \"greedy\"",
        ));
    };
    match value.value().trim().to_ascii_lowercase().as_str() {
        "bounded" => Ok(ParsedSearch::Bounded),
        "optimal" => Ok(ParsedSearch::Optimal),
        "greedy" => Ok(ParsedSearch::Greedy),
        other => Err(syn::Error::new(
            value.span(),
            format!("unknown vmcu_schedule '{other}', expected 'bounded', 'optimal', or 'greedy'"),
        )),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_modes_and_defaults_to_disabled() {
        assert_eq!(Options::parse(vec![]).unwrap(), Options::Disabled);
        assert!(matches!(
            Options::parse(vec![syn::parse_quote!(vmcu = "auto")]).unwrap(),
            Options::Enabled(EnabledOptions {
                mode: Mode::Auto,
                ..
            })
        ));
        assert!(matches!(
            Options::parse(vec![syn::parse_quote!(vmcu = "strict")]).unwrap(),
            Options::Enabled(EnabledOptions {
                mode: Mode::Strict,
                ..
            })
        ));
    }

    #[test]
    fn validates_budget_and_search_options() {
        let options = Options::parse(vec![
            syn::parse_quote!(vmcu = "auto"),
            syn::parse_quote!(vmcu_sram = 65536),
            syn::parse_quote!(vmcu_schedule = "bounded"),
            syn::parse_quote!(vmcu_search_states = 123),
        ])
        .unwrap();
        assert!(matches!(
            options,
            Options::Enabled(EnabledOptions {
                sram_budget: Some(65_536),
                search: Search::Bounded { state_limit: 123 },
                ..
            })
        ));

        assert!(Options::parse(vec![syn::parse_quote!(vmcu_sram = 1024)]).is_err());
        assert!(Options::parse(vec![
            syn::parse_quote!(vmcu = "strict"),
            syn::parse_quote!(vmcu_schedule = "greedy"),
            syn::parse_quote!(vmcu_search_states = 1),
        ])
        .is_err());
    }

    #[test]
    fn rejects_duplicate_and_unknown_options() {
        assert!(Options::parse(vec![
            syn::parse_quote!(vmcu = "auto"),
            syn::parse_quote!(vmcu = "strict"),
        ])
        .is_err());
        assert!(Options::parse(vec![syn::parse_quote!(vmcu = "fast")]).is_err());
        assert!(Options::parse(vec![syn::parse_quote!(unknown = true)]).is_err());
    }
}
