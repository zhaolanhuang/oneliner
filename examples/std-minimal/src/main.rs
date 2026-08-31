use log::{error, info};

use oneliner::model;
use oneliner::runtime::{ModelInference, ModelSource};
use oneliner_profiler::Profiler;

#[model("../models/mcunet-10fps_vww.tflite")]
struct Model;
const INPUT_LEN: usize = 64 * 64 * 3;
const EXPECTED: [i8; 2] = [4, -5];

fn main() {
    let artifacts = <Model as ModelSource>::ARTIFACTS;
    env_logger::init();
    info!(
        "Flash usage: params={} code={} rodata={} total={} | RAM usage: arena={} stack={} total={} input={} output={}",
        artifacts.params_size,
        artifacts.code_size,
        artifacts.rodata_size,
        artifacts.total_flash_size,
        artifacts.ram_size,
        artifacts.stack_size,
        artifacts.total_ram_size,
        artifacts.input_size,
        artifacts.output_size
    );

    let mut model = Model::new();
    let mut profiler = Profiler::new();
    let mut input = Model::create_input_tensor();
    input.fill(7);

    let output = profiler.profile(|| model.run(&input));
    let actual = output.as_slice();
    if actual == EXPECTED {
        info!("Model IREE validation passed");
    } else {
        error!(
            "Model validation failed: expected {} output elements, received {} elements with different values",
            EXPECTED.len(),
            actual.len()
        );
        error!(
            "EXPECTED: [{}, {}], received: [{}, {}]",
            EXPECTED[0], EXPECTED[1], actual[0], actual[1]
        );
    }
    info!("Profiled inference stats: {}", profiler.stats());
}
