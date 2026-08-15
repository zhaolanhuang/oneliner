use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use proc_macro2::Span;

use super::super::llvm_target_info::llvm_target_info_from_rust_triple;
use crate::utils::{query_rustc_host, run_command, target_from_process_args};

pub(super) fn run_iree_compile(
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    cmsis_nn: bool,
) -> syn::Result<()> {
    //TODO: hacky, shall be improved.
    let rust_target = if let Ok(target) =
        std::env::var("CARGO_BUILD_TARGET").or_else(|_| std::env::var("TARGET"))
    {
        target
    } else {
        target_from_process_args()
            .or_else(|| query_rustc_host().map(Into::into))
            .ok_or_else(|| syn::Error::new(Span::call_site(), "Rust target triple is unavailable"))?
            .into_string()
            .map_err(|_| syn::Error::new(Span::call_site(), "target argument is not valid UTF-8"))?
    };
    let target_info = llvm_target_info_from_rust_triple(&rust_target).map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("failed to get LLVM target info for Rust target {rust_target}: {error}"),
        )
    })?;
    let llvm_triple = target_info.llvm_triple;
    let cpu_features = target_info.features.filter(|value| !value.is_empty());
    let target_cpu = target_info
        .cpu
        .filter(|value| !value.is_empty())
        .or_else(|| cortex_m_cpu(&llvm_triple, cpu_features.as_deref()).map(str::to_owned));

    if cmsis_nn && cortex_m_cpu(&llvm_triple, cpu_features.as_deref()).is_some() {
        return run_iree_compile_with_cmsis_nn(
            compile_input,
            vmfb,
            object,
            ir_dump_dir,
            &llvm_triple,
            target_cpu.as_deref().expect("Cortex-M CPU was inferred"),
            cpu_features.as_deref(),
        );
    }

    run_standard_iree_compile(
        compile_input,
        vmfb,
        object,
        ir_dump_dir,
        &llvm_triple,
        target_cpu.as_deref(),
        cpu_features.as_deref(),
    )
}

/// Maps an LLVM triple to the Cortex-M CPU that the precompiled CMSIS-NN
/// bitcode was built for. Returns `None` for targets without a prebuilt
/// bitcode (CMSIS-NN lowering stays disabled for those). The `thumbv8m.main`
/// triples cover both Cortex-M33 and Cortex-M55; the `+mve` feature selects
/// the MVE target.
fn cortex_m_cpu(llvm_triple: &str, features: Option<&str>) -> Option<&'static str> {
    let has_mve = features.is_some_and(|value| value.contains("+mve"));
    match llvm_triple {
        "thumbv6m-none-eabi" => Some("cortex-m0"),
        "thumbv7m-none-eabi" => Some("cortex-m3"),
        "thumbv7em-none-eabi" | "thumbv7em-none-eabihf" => Some("cortex-m4"),
        "thumbv8m.main-none-eabi" | "thumbv8m.main-none-eabihf" => {
            Some(if has_mve { "cortex-m55" } else { "cortex-m33" })
        }
        "thumbv8.1m.main-none-eabi" | "thumbv8.1m.main-none-eabihf" => Some("cortex-m55"),
        _ => None,
    }
}

/// CMSIS-NN kernel family for a Cortex-M CPU: MVE cores use different scratch
/// buffer layouts and require MVE scratch sizes in the rewriter.
fn kernel_class_for_cpu(cpu: &str) -> &'static str {
    if matches!(cpu, "cortex-m55" | "cortex-m85") {
        "mve"
    } else {
        "dsp"
    }
}

#[cfg(test)]
mod tests {
    use super::cortex_m_cpu;

    #[test]
    fn maps_cortex_m_triples_to_cpus() {
        assert_eq!(cortex_m_cpu("thumbv6m-none-eabi", None), Some("cortex-m0"));
        assert_eq!(cortex_m_cpu("thumbv7m-none-eabi", None), Some("cortex-m3"));
        assert_eq!(cortex_m_cpu("thumbv7em-none-eabi", None), Some("cortex-m4"));
        assert_eq!(cortex_m_cpu("thumbv7em-none-eabihf", None), Some("cortex-m4"));
        assert_eq!(
            cortex_m_cpu("thumbv8m.main-none-eabi", None),
            Some("cortex-m33")
        );
        assert_eq!(
            cortex_m_cpu("thumbv8m.main-none-eabihf", None),
            Some("cortex-m33")
        );
        assert_eq!(
            cortex_m_cpu("thumbv8m.main-none-eabi", Some("+mve")),
            Some("cortex-m55")
        );
        assert_eq!(
            cortex_m_cpu("thumbv8m.main-none-eabihf", Some("+mve.fp")),
            Some("cortex-m55")
        );
        assert_eq!(
            cortex_m_cpu("thumbv8.1m.main-none-eabi", None),
            Some("cortex-m55")
        );
    }

    #[test]
    fn leaves_non_cortex_m_targets_unchanged() {
        assert_eq!(cortex_m_cpu("aarch64-unknown-none", None), None);
        assert_eq!(cortex_m_cpu("x86_64-unknown-linux-gnu", None), None);
    }
}

fn configure_iree_target(
    command: &mut Command,
    input: &Path,
    llvm_triple: &str,
    target_cpu: Option<&str>,
    cpu_features: Option<&str>,
) {
    command
        .arg(input)
        .arg("--iree-hal-target-device=local")
        .arg("--iree-hal-local-target-device-backends=llvm-cpu")
        .arg(format!("--iree-llvmcpu-target-triple={llvm_triple}"));
    if let Some(cpu) = target_cpu {
        // Newer IREE builds (LLVM >= 24) cannot resolve an explicit -mcpu to
        // CPU features for bare-metal ARM triples; pass only the features in
        // that case.
        if iree_cpu_flag_supported() {
            command.arg(format!("--iree-llvmcpu-target-cpu={cpu}"));
        }
    }
    if let Some(features) = cpu_features {
        command.arg(format!("--iree-llvmcpu-target-cpu-features={features}"));
    }
}

fn run_iree_compile_with_cmsis_nn(
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    llvm_triple: &str,
    target_cpu: &str,
    cpu_features: Option<&str>,
) -> syn::Result<()> {
    let artifact_dir = vmfb
        .parent()
        .ok_or_else(|| syn::Error::new(Span::call_site(), "VMFB output has no parent directory"))?;
    let preprocessing = artifact_dir.join("cmsis-nn.preprocessing.mlir");
    let rewritten = artifact_dir.join("cmsis-nn.rewritten.mlir");
    let configured = artifact_dir.join("cmsis-nn.configured.mlir");
    let lowered = artifact_dir.join("cmsis-nn.lowered.mlir");
    let finalized = artifact_dir.join("cmsis-nn.finalized.mlir");
    let bitcode = artifact_dir.join("oneliner_cmsis_nn.bc");
    let macro_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = macro_dir.parent().ok_or_else(|| {
        syn::Error::new(Span::call_site(), "oneliner-macro has no workspace parent")
    })?;
    let rewriter = macro_dir.join("python").join("rewrite_cmsis_nn.py");

    let mut preprocess = Command::new("iree-compile");
    configure_iree_target(
        &mut preprocess,
        compile_input,
        llvm_triple,
        Some(target_cpu),
        cpu_features,
    );
    preprocess
        .arg("--compile-to=preprocessing")
        .arg("--emit-mlir-bytecode=false")
        .arg("--iree-llvmcpu-link-embedded=false")
        .arg("--iree-llvmcpu-link-static")
        .arg(format!(
            "--iree-llvmcpu-static-library-output-path={}",
            object.display()
        ))
        .arg("--iree-llvmcpu-stack-allocation-limit=65536")
        .arg("-o")
        .arg(&preprocessing);
    run_command(&mut preprocess, "IREE preprocessing")?;

    run_python_filter(
        &rewriter,
        &preprocessing,
        &rewritten,
        &["--kernel-class", kernel_class_for_cpu(target_cpu)],
    )?;
    let rewritten_text = std::fs::read_to_string(&rewritten)
        .map_err(|error| syn::Error::new(Span::call_site(), error))?;
    if !rewritten_text.contains("oneliner_cmsis_nn_") {
        return run_standard_iree_compile(
            compile_input,
            vmfb,
            object,
            ir_dump_dir,
            llvm_triple,
            Some(target_cpu),
            cpu_features,
        );
    }

    let mut configure = Command::new("iree-compile");
    configure_iree_target(
        &mut configure,
        &rewritten,
        llvm_triple,
        Some(target_cpu),
        cpu_features,
    );
    configure
        .arg("--compile-from=preprocessing")
        .arg("--compile-to=executable-configurations")
        .arg("--emit-mlir-bytecode=false")
        .arg("--iree-llvmcpu-link-embedded=false")
        .arg("--iree-llvmcpu-link-static")
        .arg(format!(
            "--iree-llvmcpu-static-library-output-path={}",
            object.display()
        ))
        .arg("-o")
        .arg(&configured);
    run_command(&mut configure, "IREE CMSIS-NN dispatch configuration")?;

    let mut lower = Command::new("iree-opt");
    lower
        .arg(&configured)
        .arg("--pass-pipeline=builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-lower-bitcode-ukernels)),iree-hal-hoist-executable-objects)))")
        .arg("-o")
        .arg(&lowered);
    run_command(&mut lower, "IREE CMSIS-NN ukernel lowering")?;

    run_python_filter(&rewriter, &lowered, &finalized, &["--finalize-configured"])?;

    let mut build_bitcode = Command::new("python");
    build_bitcode
        .arg(macro_dir.join("cmsis_nn").join("build_bitcode.py"))
        .arg("--cmsis-nn")
        .arg(workspace_dir.join("third_party").join("cmsis-nn"))
        .arg("--shim")
        .arg(macro_dir.join("cmsis_nn").join("oneliner_cmsis_nn.c"))
        .arg("--output")
        .arg(&bitcode)
        .arg("--target")
        .arg(llvm_triple)
        .arg("--cpu")
        .arg(target_cpu)
        .arg("--features")
        .arg(cpu_features.unwrap_or_default());
    if !resolve_prebuilt_bitcode(
        &macro_dir,
        llvm_triple,
        target_cpu,
        &bitcode,
    )? {
        run_command(&mut build_bitcode, "CMSIS-NN bitcode builder")?;
    }

    let mut compile = Command::new("iree-compile");
    configure_iree_target(
        &mut compile,
        &finalized,
        llvm_triple,
        Some(target_cpu),
        cpu_features,
    );
    compile
        .arg("--compile-from=executable-configurations")
        .arg("--iree-stream-partitioning-favor=min-peak-memory")
        .arg("--iree-llvmcpu-link-embedded=false")
        .arg("--iree-llvmcpu-link-static")
        .arg(format!(
            "--iree-hal-executable-object-search-path={}",
            artifact_dir.display()
        ))
        .arg(format!(
            "--iree-llvmcpu-static-library-output-path={}",
            object.display()
        ))
        .arg(format!(
            "--dump-compilation-phases-to={}",
            ir_dump_dir.display()
        ))
        .arg("--iree-llvmcpu-stack-allocation-limit=65536")
        .arg("-o")
        .arg(vmfb);
    run_command(&mut compile, "iree-compile with CMSIS-NN")
}

/// Queries the LLVM major version used by the installed `iree-compile`,
/// cached after the first call. Returns `None` when it cannot be determined
/// (callers treat that as "compatible").
static IREE_LLVM_MAJOR: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();

fn iree_compiler_llvm_major() -> Option<u64> {
    *IREE_LLVM_MAJOR.get_or_init(|| {
        let output = Command::new("iree-compile").arg("--version").output().ok()?;
        let text = String::from_utf8_lossy(&output.stdout);
        text.lines()
            .find_map(|line| line.strip_prefix("  LLVM version "))
            .and_then(|version| version.split('.').next())
            .and_then(|major| major.parse().ok())
    })
}

/// Whether the installed `iree-compile` accepts an explicit
/// `--iree-llvmcpu-target-cpu` for bare-metal ARM triples. IREE builds on
/// LLVM 24+ reject it ("Resolution of CPU to CPU-features is not
/// implemented"); older builds accept it.
fn iree_cpu_flag_supported() -> bool {
    iree_compiler_llvm_major().is_none_or(|major| major < 24)
}

/// Selects a precompiled CMSIS-NN bitcode variant for the target and copies
/// it over `bitcode`. Returns `Ok(true)` when a prebuilt bitcode was used, or
/// `Ok(false)` when the caller must build the bitcode on the fly.
///
/// Precompiled bitcode is tied to the LLVM major version it was built with:
/// a mismatch with the installed `iree-compile` is a hard error (set
/// `ONELINER_CMSIS_NN_FORCE_BUILD=1` to always compile on the fly).
fn resolve_prebuilt_bitcode(
    macro_dir: &Path,
    llvm_triple: &str,
    target_cpu: &str,
    bitcode: &Path,
) -> syn::Result<bool> {
    if std::env::var_os("ONELINER_CMSIS_NN_FORCE_BUILD").is_some() {
        return Ok(false);
    }
    let prebuilt_dir = macro_dir.join("cmsis_nn").join("prebuilt");
    let manifest_text = std::fs::read_to_string(prebuilt_dir.join("manifest.json"))
        .map_err(|error| {
            syn::Error::new(
                Span::call_site(),
                format!(
                    "CMSIS-NN prebuilt manifest is missing or unreadable ({}: {error}); \
                     regenerate it with \
                     `python {}/cmsis_nn/build_bitcode.py --build-all ...` or set \
                     ONELINER_CMSIS_NN_FORCE_BUILD=1 to compile on the fly",
                    prebuilt_dir.display(),
                    macro_dir.display()
                ),
            )
        })?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text)
        .map_err(|error| syn::Error::new(Span::call_site(), format!("invalid CMSIS-NN prebuilt manifest: {error}")))?;

    let prebuilt_llvm_major = manifest.get("llvm_major").and_then(|value| value.as_u64());
    let installed_llvm_major = iree_compiler_llvm_major();
    // LLVM bitcode is forward-compatible: an IREE using a newer LLVM can read
    // bitcode built with an older clang, but not the other way around. Refuse
    // to use the prebuilt bitcode when the installed IREE is older than the
    // LLVM the prebuilt was generated with.
    if prebuilt_llvm_major.is_some() && installed_llvm_major.is_some()
        && installed_llvm_major < prebuilt_llvm_major
    {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "CMSIS-NN prebuilt bitcode was built with LLVM {} but the installed \
                 iree-compile uses LLVM {}, which cannot read it. Align the toolchain \
                 version and regenerate the prebuilt bitcode \
                 (`build_bitcode.py --build-all`), or set \
                 ONELINER_CMSIS_NN_FORCE_BUILD=1 to compile on the fly",
                prebuilt_llvm_major.unwrap_or_default(),
                installed_llvm_major.unwrap_or_default(),
            ),
        ));
    }

    let float_abi = if llvm_triple.ends_with("eabihf") { "hard" } else { "soft" };
    let variants = manifest
        .get("variants")
        .and_then(|value| value.as_array())
        .ok_or_else(|| syn::Error::new(Span::call_site(), "CMSIS-NN prebuilt manifest has no variants"))?;
    let matching = variants.iter().find(|variant| {
        variant.get("triple").and_then(|value| value.as_str()) == Some(llvm_triple)
            && variant.get("cpu").and_then(|value| value.as_str()) == Some(target_cpu)
            && variant.get("float_abi").and_then(|value| value.as_str()) == Some(float_abi)
    });
    let Some(variant) = matching else {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "no precompiled CMSIS-NN bitcode for triple {llvm_triple}, cpu {target_cpu}, \
                 {float_abi} float ABI; regenerate it with `build_bitcode.py --build-all` or \
                 set ONELINER_CMSIS_NN_FORCE_BUILD=1 to compile on the fly"
            ),
        ));
    };
    let file = variant
        .get("file")
        .and_then(|value| value.as_str())
        .ok_or_else(|| syn::Error::new(Span::call_site(), "CMSIS-NN prebuilt variant has no file"))?;
    std::fs::copy(prebuilt_dir.join(file), bitcode).map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("failed to copy CMSIS-NN prebuilt bitcode {}: {error}", prebuilt_dir.join(file).display()),
        )
    })?;
    Ok(true)
}

fn run_standard_iree_compile(
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
    llvm_triple: &str,
    target_cpu: Option<&str>,
    cpu_features: Option<&str>,
) -> syn::Result<()> {
    let mut command = Command::new("iree-compile");
    configure_iree_target(
        &mut command,
        compile_input,
        llvm_triple,
        target_cpu,
        cpu_features,
    );
    command
        .arg("--iree-stream-partitioning-favor=min-peak-memory")
        .arg("--iree-llvmcpu-link-embedded=false")
        .arg("--iree-llvmcpu-link-static")
        .arg(format!(
            "--iree-llvmcpu-static-library-output-path={}",
            object.display()
        ))
        .arg(format!(
            "--dump-compilation-phases-to={}",
            ir_dump_dir.display()
        ))
        .arg("-o")
        .arg(vmfb);
    run_command(&mut command, "iree-compile")
}

fn run_python_filter(script: &Path, input: &Path, output: &Path, args: &[&str]) -> syn::Result<()> {
    let stdin = File::open(input).map_err(|error| syn::Error::new(Span::call_site(), error))?;
    let stdout = File::create(output).map_err(|error| syn::Error::new(Span::call_site(), error))?;
    let mut command = Command::new("python");
    command
        .arg(script)
        .args(args)
        .stdin(Stdio::from(stdin))
        .stdout(Stdio::from(stdout));
    run_command(&mut command, "CMSIS-NN MLIR rewriter")
}

pub(super) fn run_converter(
    input: &Path,
    rust_output: &Path,
    json_output: &Path,
) -> syn::Result<()> {
    let mut command = Command::new("python");
    command
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("python")
                .join("iree_stream_flow_to_rust_using_re.py"),
        )
        .arg(input)
        .arg("--rust-output")
        .arg(rust_output)
        .arg("--json-output")
        .arg(json_output);
    run_command(&mut command, "IREE Stream/Flow converter")
}
