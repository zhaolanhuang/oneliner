use std::path::Path;

use oneliner::runtime::{ModelArtifacts, ModelSource};

pub fn assert_artifacts<M: ModelSource>(model_name: &str) {
    let ModelArtifacts {
        backend,
        expansion,
        model_path,
        compile_input_path,
        object_path,
        link_path,
        ir_path,
        flow_rs_path,
        metadata_json_path,
        input_size,
        output_size,
        io_pool_size,
        input_offset,
        output_offset,
        params_size,
        code_size,
        rodata_size,
        total_flash_size,
        ram_size,
    } = M::ARTIFACTS;

    assert_eq!(backend, "iree", "{model_name}: unexpected backend");
    assert_eq!(
        expansion, "static-flow",
        "{model_name}: unexpected expansion"
    );
    assert!(input_size > 0, "{model_name}: empty input binding");
    assert!(output_size > 0, "{model_name}: empty output binding");
    // Footprints may be zero for models without weights or scratch buffers
    // (for example a trivial identity/abs graph); they must still be reported.
    assert!(
        code_size > 0,
        "{model_name}: compiled model object must contain machine code"
    );
    assert_eq!(
        total_flash_size,
        params_size + code_size + rodata_size,
        "{model_name}: total flash must equal params + code + rodata"
    );
    let _ = ram_size;
    let _ = (io_pool_size, input_offset, output_offset);

    assert!(
        Path::new(model_path).exists(),
        "{model_name}: model does not exist: {model_path}"
    );

    for (label, path) in [
        ("compile input", compile_input_path),
        ("object", object_path),
        ("link", link_path),
        ("stream/flow IR", ir_path),
        ("generated Rust flow", flow_rs_path),
        ("metadata", metadata_json_path),
    ] {
        assert!(
            Path::new(path).is_file(),
            "{model_name}: {label} artifact does not exist: {path}"
        );
    }
}

#[allow(dead_code)]
pub fn assert_f32_slice_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "output length differs");
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual_value - expected_value).abs() <= tolerance,
            "output[{index}] differs: expected {expected_value}, got {actual_value} (tolerance {tolerance}); \
             expected output: {expected_slice:?}; actual output: {actual_slice:?}",
            expected_slice = expected,
            actual_slice = actual,
        );
    }
}
