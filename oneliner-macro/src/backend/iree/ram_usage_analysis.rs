use std::path::{Path, PathBuf};
use std::process::Command;

use proc_macro2::Span;
use serde::Deserialize;

use super::toolchain::python_executable;

#[derive(Debug, Deserialize)]
pub(super) struct LoweringRamUsage {
    pub(super) transient_size: usize,
    pub(super) stack_size: usize,
}

pub(super) fn analyze(stream: &Path, executable: &Path) -> syn::Result<LoweringRamUsage> {
    let reporter = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("analyze_iree_ram_usage.py");
    let output = Command::new(python_executable())
        .arg(reporter)
        .arg("--stream")
        .arg(stream)
        .arg("--executable")
        .arg(executable)
        .output()
        .map_err(|error| {
            syn::Error::new(
                Span::call_site(),
                format!("failed to start IREE resource analyzer: {error}"),
            )
        })?;
    if !output.status.success() {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "IREE resource analyzer failed with status {:?}: {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| syn::Error::new(Span::call_site(), error))
}
