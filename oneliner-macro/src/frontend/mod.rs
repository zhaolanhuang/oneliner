mod generate_input_mlir;
mod model_io;

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use proc_macro2::Span;
use syn::ItemStruct;

use crate::args::{ModelArgs, ModelFormat};
use crate::utils::rust_ident;

pub(crate) use model_io::{ModelIo, TensorInfo};

#[derive(Debug)]
/// A frontend-prepared model ready for backend compilation.
pub(crate) struct Model {
    /// Original model path supplied to `#[model]`.
    pub(crate) source_path: PathBuf,
    /// IREE-compatible model input consumed by the backend compiler.
    pub(crate) compile_input_path: PathBuf,
    /// File stem of `compile_input_path`.
    pub(crate) ir_dump_stem: String,
    /// Validated model input and output tensor metadata.
    pub(crate) model_io: ModelIo,
}

pub(crate) fn prepare(args: &ModelArgs, input_struct: &ItemStruct) -> syn::Result<Model> {
    let model_path = &args.model_path;
    let caller_manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| {
            syn::Error::new(
                model_path.span(),
                "CARGO_MANIFEST_DIR is not set; Cargo must expand #[model] inside a package build",
            )
        })?;
    let path = PathBuf::from(model_path.value());
    let path = if path.is_absolute() {
        path
    } else {
        caller_manifest_dir.join(path)
    };
    let struct_ident = &input_struct.ident;
    let format = args.format.map_or_else(|| detect_format(&path), Ok)?;
    let expects_directory = format == ModelFormat::Tensorflow;
    let struct_name = rust_ident(&struct_ident.to_string());
    let model_stem = if expects_directory {
        path.file_name()
    } else {
        path.file_stem()
    }
    .and_then(OsStr::to_str)
    .map(rust_ident)
    .unwrap_or_else(|| struct_name.clone());

    let (compile_input_path, ir_dump_stem, model_io) = match format {
        ModelFormat::Mlir => {
            let model_io = model_io::load_mlir(&path)?;
            (path.clone(), model_stem.clone(), model_io)
        }
        ModelFormat::Onnx => {
            let model_io = model_io::load_onnx(&path)?;
            let output = normalized_path(&struct_name, &model_stem, "tosa.mlir")?;
            generate_input_mlir::from_onnx(&path, &output)?;
            normalize_dynamic_dims(&output)?;
            let ir_dump_stem = rust_ident(output.file_stem().and_then(OsStr::to_str).unwrap());
            (output, ir_dump_stem, model_io)
        }
        ModelFormat::Pytorch => {
            let output = normalized_path(&struct_name, &model_stem, "torch.mlir")?;
            generate_input_mlir::from_pytorch(&path, &output, &struct_name)?;
            normalize_dynamic_dims(&output)?;
            let model_io = model_io::load_mlir(&output)?;
            (output, struct_name, model_io)
        }
        // Dynamic shapes are rejected by `oneliner-macro/python/inspect_tensorflow_saved_model.py`.
        ModelFormat::Tensorflow => {
            let output = normalized_path(&struct_name, &model_stem, "tensorflow.mlir")?;
            let io_output = normalized_path(&struct_name, &model_stem, "tensorflow.json")?;
            generate_input_mlir::inspect_tensorflow(&path, &io_output)?;
            let model_io = model_io::load_tensorflow_metadata(&io_output)?;
            generate_input_mlir::from_tensorflow(&path, &output)?;
            (output, "_".to_owned(), model_io)
        }
        ModelFormat::Tflite => {
            let output = normalized_path(&struct_name, &model_stem, "tosa.mlir")?;
            generate_input_mlir::from_tflite(&path, &output)?;
            normalize_dynamic_dims(&output)?;
            let model_io = model_io::load_mlir(&output)?;
            let ir_dump_stem = rust_ident(output.file_stem().and_then(OsStr::to_str).unwrap());
            (output, ir_dump_stem, model_io)
        }
    };
    model_io.validate()?;

    Ok(Model {
        source_path: path,
        compile_input_path,
        ir_dump_stem,
        model_io,
    })
}

fn normalize_dynamic_dims(path: &Path) -> syn::Result<()> {
    let text = fs::read_to_string(path).map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!(
                "failed to read generated MLIR from {}: {error}",
                path.display()
            ),
        )
    })?;
    let (normalized, count) = replace_dynamic_dims(&text);
    if count == 0 {
        return Ok(());
    }
    fs::write(path, normalized).map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!(
                "failed to rewrite generated MLIR at {}: {error}",
                path.display()
            ),
        )
    })?;
    eprintln!(
        "[oneliner] warning: {} contains {count} dynamic dimension(s) <?>; replacing each with 1",
        path.display()
    );
    Ok(())
}

fn replace_dynamic_dims(text: &str) -> (String, usize) {
    let mut normalized = String::with_capacity(text.len());
    let mut count = 0usize;
    let mut angle_depth = 0usize;
    let mut square_depth = 0usize;
    let mut chars = text.char_indices().peekable();
    while let Some((_, character)) = chars.next() {
        match character {
            '<' => {
                angle_depth += 1;
                normalized.push('<');
            }
            '>' => {
                angle_depth = angle_depth.saturating_sub(1);
                normalized.push('>');
            }
            '[' => {
                square_depth += 1;
                normalized.push('[');
            }
            ']' => {
                square_depth = square_depth.saturating_sub(1);
                normalized.push(']');
            }
            '?' if angle_depth > 0 || square_depth > 0 => {
                normalized.push('1');
                count += 1;
            }
            '/' if angle_depth == 0 && square_depth == 0 => {
                if chars.peek().is_some_and(|(_, next)| *next == '/') {
                    normalized.push_str("//");
                    chars.next();
                    for (_, comment_char) in chars.by_ref() {
                        normalized.push(comment_char);
                        if comment_char == '\n' {
                            break;
                        }
                    }
                } else {
                    normalized.push('/');
                }
            }
            '"' if angle_depth == 0 && square_depth == 0 => {
                normalized.push('"');
                let iter = chars.by_ref();
                while let Some((_, string_char)) = iter.next() {
                    normalized.push(string_char);
                    if string_char == '\\' {
                        if let Some((_, escaped)) = iter.next() {
                            normalized.push(escaped);
                        }
                        continue;
                    }
                    if string_char == '"' {
                        break;
                    }
                }
            }
            _ => normalized.push(character),
        }
    }
    (normalized, count)
}

fn normalized_path(struct_name: &str, model_stem: &str, suffix: &str) -> syn::Result<PathBuf> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                "CARGO_MANIFEST_DIR is not set; Cargo must expand #[model] inside a package build",
            )
        })?;
    let out_root = std::env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("target").join("oneliner"));
    let output_dir = out_root
        .join("frontend")
        .join(format!("{struct_name}_{model_stem}"));
    fs::create_dir_all(&output_dir).map_err(|error| syn::Error::new(Span::call_site(), error))?;
    Ok(output_dir.join(format!("{model_stem}.{suffix}")))
}

fn detect_format(path: &Path) -> syn::Result<ModelFormat> {
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                format!(
                    "model path has no supported extension: {}; expected .mlir, .onnx, .pt2, or .tflite",
                    path.display()
                ),
            )
        })?;

    match extension.as_str() {
        "mlir" => Ok(ModelFormat::Mlir),
        "onnx" => Ok(ModelFormat::Onnx),
        "pt2" => Ok(ModelFormat::Pytorch),
        "tflite" => Ok(ModelFormat::Tflite),
        "pt" | "pth" => Err(syn::Error::new(
            Span::call_site(),
            format!(
                "PyTorch checkpoint '.{extension}' at {} is not a self-contained export; use torch.export.save(..., \"model.pt2\") and pass the .pt2 file to #[model]",
                path.display()
            ),
        )),
        _ => Err(syn::Error::new(
            Span::call_site(),
            format!(
                "unsupported model format '.{extension}' at {}; expected .mlir, .onnx, .pt2, or .tflite",
                path.display()
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_supported_model_formats_case_insensitively() {
        assert_eq!(
            detect_format(Path::new("model.mlir")).unwrap(),
            ModelFormat::Mlir
        );
        assert_eq!(
            detect_format(Path::new("model.ONNX")).unwrap(),
            ModelFormat::Onnx
        );
        assert_eq!(
            detect_format(Path::new("model.pt2")).unwrap(),
            ModelFormat::Pytorch
        );
        assert_eq!(
            detect_format(Path::new("model.TFLITE")).unwrap(),
            ModelFormat::Tflite
        );
    }

    #[test]
    fn rejects_ambiguous_pytorch_checkpoint_extensions() {
        let error = detect_format(Path::new("model.pth")).unwrap_err();
        assert!(error.to_string().contains("use torch.export.save"));
    }
}
