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
    pub(crate) search: Search,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Search {
    Greedy,
    Optimal { budget: Option<usize> },
}

impl Search {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Optimal { .. } => "optimal",
        }
    }

    pub(super) const fn budget(self) -> Option<usize> {
        match self {
            Self::Greedy | Self::Optimal { budget: None } => None,
            Self::Optimal {
                budget: Some(budget),
            } => Some(budget),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParsedSearch {
    Greedy,
    Optimal,
}

impl Options {
    pub(crate) fn parse(args: Vec<NestedMeta>) -> syn::Result<Self> {
        let mut enabled = None;
        let mut search_mode = None;
        let mut search_budget = None;

        for arg in args {
            match arg {
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("vmcu") => {
                    if enabled.is_some() {
                        return Err(syn::Error::new(meta.span(), "duplicate vmcu option"));
                    }
                    enabled = Some(parse_enabled(meta.lit)?);
                }
                NestedMeta::Meta(Meta::NameValue(meta))
                    if meta.path.is_ident("vmcu_search_mode") =>
                {
                    if search_mode.is_some() {
                        return Err(syn::Error::new(
                            meta.span(),
                            "duplicate vmcu_search_mode option",
                        ));
                    }
                    search_mode = Some(parse_search_mode(meta.lit)?);
                }
                NestedMeta::Meta(Meta::NameValue(meta))
                    if meta.path.is_ident("vmcu_search_budget") =>
                {
                    if search_budget.is_some() {
                        return Err(syn::Error::new(
                            meta.span(),
                            "duplicate vmcu_search_budget option",
                        ));
                    }
                    search_budget = Some(parse_positive_usize(meta.lit, "vmcu_search_budget")?);
                }
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "unknown #[model] option; expected backend, arena, format, vmcu, vmcu_search_mode, or vmcu_search_budget",
                    ));
                }
            }
        }

        if !enabled.unwrap_or(false) {
            if search_mode.is_some() || search_budget.is_some() {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "vmcu_search_mode and vmcu_search_budget require vmcu = \"on\"",
                ));
            }
            return Ok(Self::Disabled);
        }

        let search_mode = search_mode.unwrap_or(ParsedSearch::Greedy);
        if search_budget.is_some() && search_mode != ParsedSearch::Optimal {
            return Err(syn::Error::new(
                Span::call_site(),
                "vmcu_search_budget is valid only with vmcu_search_mode = \"optimal\"",
            ));
        }
        let search = match search_mode {
            ParsedSearch::Greedy => Search::Greedy,
            ParsedSearch::Optimal => Search::Optimal {
                budget: search_budget,
            },
        };
        Ok(Self::Enabled(EnabledOptions { search }))
    }
}

fn parse_enabled(lit: Lit) -> syn::Result<bool> {
    let Lit::Str(value) = lit else {
        return Err(syn::Error::new(
            lit.span(),
            "vmcu must be a string literal, for example vmcu = \"on\"",
        ));
    };
    match value.value().trim().to_ascii_lowercase().as_str() {
        "off" => Ok(false),
        "on" => Ok(true),
        other => Err(syn::Error::new(
            value.span(),
            format!("unknown vmcu mode '{other}', expected 'off' or 'on'"),
        )),
    }
}

fn parse_search_mode(lit: Lit) -> syn::Result<ParsedSearch> {
    let Lit::Str(value) = lit else {
        return Err(syn::Error::new(
            lit.span(),
            "vmcu_search_mode must be \"greedy\" or \"optimal\"",
        ));
    };
    match value.value().trim().to_ascii_lowercase().as_str() {
        "greedy" => Ok(ParsedSearch::Greedy),
        "optimal" => Ok(ParsedSearch::Optimal),
        other => Err(syn::Error::new(
            value.span(),
            format!("unknown vmcu_search_mode '{other}', expected 'greedy' or 'optimal'"),
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
    fn parses_activation_and_defaults_to_disabled() {
        assert_eq!(Options::parse(vec![]).unwrap(), Options::Disabled);
        assert_eq!(
            Options::parse(vec![syn::parse_quote!(vmcu = "on")]).unwrap(),
            Options::Enabled(EnabledOptions {
                search: Search::Greedy,
            })
        );
    }

    #[test]
    fn validates_search_options() {
        let options = Options::parse(vec![
            syn::parse_quote!(vmcu = "on"),
            syn::parse_quote!(vmcu_search_mode = "optimal"),
            syn::parse_quote!(vmcu_search_budget = 123),
        ])
        .unwrap();
        assert_eq!(
            options,
            Options::Enabled(EnabledOptions {
                search: Search::Optimal { budget: Some(123) },
            })
        );

        assert_eq!(
            Options::parse(vec![
                syn::parse_quote!(vmcu = "on"),
                syn::parse_quote!(vmcu_search_mode = "optimal"),
            ])
            .unwrap(),
            Options::Enabled(EnabledOptions {
                search: Search::Optimal { budget: None },
            })
        );

        assert!(Options::parse(vec![
            syn::parse_quote!(vmcu = "on"),
            syn::parse_quote!(vmcu_search_mode = "greedy"),
            syn::parse_quote!(vmcu_search_budget = 1),
        ])
        .is_err());
    }

    #[test]
    fn rejects_duplicate_and_unknown_options() {
        assert!(Options::parse(vec![
            syn::parse_quote!(vmcu = "on"),
            syn::parse_quote!(vmcu = "off"),
        ])
        .is_err());
        assert!(Options::parse(vec![syn::parse_quote!(vmcu = "auto")]).is_err());
        assert!(Options::parse(vec![syn::parse_quote!(vmcu = "fast")]).is_err());
        assert!(Options::parse(vec![syn::parse_quote!(unknown = true)]).is_err());
    }
}
