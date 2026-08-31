#![no_main]
#![no_std]

use ariel_os::debug::{exit, ExitCode};
use ariel_os::log::{error, info};

use oneliner::model;
use oneliner::runtime::{ModelInference, ModelSource};
use oneliner_profiler::Profiler;

use static_cell::ConstStaticCell;

// #[model(
//     "../models/mcunet-10fps_vww.tflite",
//     arena = "shared"
// )]
// struct Model;
// const INPUT_LEN: usize = 64 * 64 * 3;
// const EXPECTED: [i8; 2] = [4, -5];

#[model("../models/lenet5_quantized.tflite")]
struct Model;
const INPUT_LEN: usize = 28 * 28 * 1;
const OUTPUT_LEN: usize = 10;
const EXPECTED: [f32; OUTPUT_LEN] = [
    0.11666615, 0.11666615, 0.13124943, 0.68541366, 0.0, 0.36458173, 0.0, 0.0, 1.2104113,
    0.16041596,
];
// static INPUT_CELL: ConstStaticCell<<Model as ModelInference>::InputTensor> = ConstStaticCell::new(<Model as ModelInference>::InputTensor::new(0));

#[ariel_os::thread(autostart, priority = 1, stacksize = 20480)]
// #[ariel_os::thread(autostart, priority = 1)]
fn main() {
    let artifacts = <Model as ModelSource>::ARTIFACTS;
    info!(
        "Oneliner IREE profiler example running on {}",
        ariel_os::buildinfo::BOARD
    );
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
    // let mut input = INPUT_CELL.take();
    input.fill(7.0);
    let output = profiler.profile(|| model.run(&input));
    info!("Profiled inference stats: {}", profiler.stats());

    let actual = output.as_slice();
    if actual == EXPECTED {
        info!("Model IREE validation passed");
        exit(ExitCode::SUCCESS);
    }
    error!(
        "Model validation failed: expected {} output elements, received {} elements with different values",
        EXPECTED.len(),
        actual.len()
    );
    error!(
        "EXPECTED: [{}, {}], received: [{}, {}]",
        EXPECTED[0], EXPECTED[1], actual[0], actual[1]
    );
    exit(ExitCode::FAILURE);
}
