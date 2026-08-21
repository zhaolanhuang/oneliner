use std::fs;
use std::path::Path;

use object::{Object, ObjectSection, SectionKind};
use proc_macro2::Span;

/// Flash footprint of the compiled model object file.
#[derive(Debug, Clone, Copy)]
pub(super) struct ObjectFootprint {
    /// Executable machine code (`.text` sections) placed in flash.
    pub code_size: usize,
    /// Read-only data embedded in the object (`.rodata`, `.data.rel.ro`,
    /// `.ARM.exidx`) placed in flash.
    pub rodata_size: usize,
}

/// Measures the machine code and read-only data bytes of the IREE static
/// library object produced for a model.
///
/// The object file is a relocatable object in the host or target format
/// (ELF on Linux/embedded, Mach-O on macOS, COFF on Windows). Sections are
/// classified by their normalized [`SectionKind`]; ELF-specific `.data.rel.ro`
/// library tables and `.ARM.exidx` unwind tables are counted as read-only data
/// because they are emitted into flash alongside the parameters.
pub(super) fn measure_object(object_path: &Path) -> syn::Result<ObjectFootprint> {
    let bytes = fs::read(object_path)
        .map_err(|error| syn::Error::new(Span::call_site(), error))?;
    let file = object::File::parse(&*bytes)
        .map_err(|error| syn::Error::new(Span::call_site(), error))?;

    let mut code_size = 0u64;
    let mut rodata_size = 0u64;
    for section in file.sections() {
        match section.kind() {
            SectionKind::Text => code_size += section.size(),
            SectionKind::ReadOnlyData
            | SectionKind::ReadOnlyDataWithRel
            | SectionKind::ReadOnlyString => rodata_size += section.size(),
            _ => {
                let name = section.name().unwrap_or_default();
                if name.starts_with(".ARM.exidx") || name.starts_with(".data.rel.ro") {
                    rodata_size += section.size();
                }
            }
        }
    }

    Ok(ObjectFootprint {
        code_size: usize::try_from(code_size).unwrap_or(usize::MAX),
        rodata_size: usize::try_from(rodata_size).unwrap_or(usize::MAX),
    })
}
