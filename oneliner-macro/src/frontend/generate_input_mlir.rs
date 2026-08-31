use std::path::{Path, PathBuf};
use std::process::Command;

use crate::utils::run_command;

pub(super) fn from_tflite(input: &Path, output: &Path) -> syn::Result<()> {
    let mut command = Command::new("tosa-converter-for-tflite");
    command.arg(input).arg("--text").arg("-o").arg(output);
    run_command(&mut command, "tosa-converter-for-tflite")
}

pub(super) fn from_onnx(input: &Path, output: &Path) -> syn::Result<()> {
    let mut command = Command::new("iree-import-onnx");
    command.arg("-o").arg(output).arg(input);
    run_command(&mut command, "iree-import-onnx")
}

pub(super) fn from_pytorch(input: &Path, output: &Path, module_name: &str) -> syn::Result<()> {
    let mut command = Command::new("python");
    command
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("python")
                .join("oneliner_iree")
                .join("pytorch_import.py"),
        )
        .arg(input)
        .arg("--output")
        .arg(output)
        .arg("--module-name")
        .arg(module_name);
    if let Some(model_dir) = input.parent() {
        command.current_dir(model_dir);
    }
    run_command(&mut command, "PyTorch ExportedProgram importer")
}

pub(super) fn from_tensorflow(input: &Path, output: &Path) -> syn::Result<()> {
    let mut command = Command::new("iree-import-tf");
    command
        .arg("--tf-import-type=savedmodel_v2")
        .arg("--tf-savedmodel-exported-names=main");
    command.arg(input).arg("-o").arg(output);
    run_command(&mut command, "iree-import-tf")
}

pub(super) fn inspect_tensorflow(input: &Path, output: &Path) -> syn::Result<()> {
    let mut command = Command::new("python");
    command
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("python")
                .join("inspect_tensorflow_saved_model.py"),
        )
        .arg(input)
        .arg("--output")
        .arg(output);
    run_command(&mut command, "TensorFlow SavedModel inspector")
}
