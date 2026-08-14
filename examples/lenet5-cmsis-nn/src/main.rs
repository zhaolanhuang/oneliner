#![no_main]
#![no_std]

use ariel_os::debug::{exit, ExitCode};
use ariel_os::log::{error, info};
use ariel_os::time;

use oneliner::model;
use oneliner::runtime::{ModelInference, ModelSource};
use static_cell::ConstStaticCell;

#[model(
    "../models/lenet5_quantized.tflite",
    backend = "iree",
    arena = "shared",
    cmsis_nn = true
)]
struct LeNet5;

const OUTPUT_LEN: usize = 10;
const EXPECTED: [f32; OUTPUT_LEN] = [
    0.11666615, 0.11666615, 0.13124943, 0.68541366, 0.0, 0.36458173, 0.0, 0.0, 1.2104113,
    0.16041596,
];
static INPUT: ConstStaticCell<<LeNet5 as ModelInference>::InputTensor> =
    ConstStaticCell::new(<LeNet5 as ModelInference>::InputTensor::new(0.0));

#[ariel_os::thread(autostart, priority = 1, stacksize = 20480)]
fn main() {
    let artifacts = <LeNet5 as ModelSource>::ARTIFACTS;
    info!(
        "LeNet5 CMSIS-NN example running on {}",
        ariel_os::buildinfo::BOARD
    );
    info!(
        "Model artifact sizes: input={} output={}",
        artifacts.input_size, artifacts.output_size
    );

    let mut model = LeNet5::new();
    let input = INPUT.take();
    input.fill(7.0);

    let start = time::Instant::now().as_micros();
    let output = model.run(input);
    let elapsed = time::Instant::now().as_micros() - start;
    info!("Model inference time: {:?} us", elapsed);

    let actual = output.as_slice();
    let matches = actual
        .iter()
        .zip(EXPECTED)
        .all(|(actual, expected)| (*actual - expected).abs() <= 1.0e-5);
    if matches {
        info!("LeNet5 CMSIS-NN validation passed");
        exit(ExitCode::SUCCESS);
    }

    error!(
        "LeNet5 validation failed: expected [{}, {}], received [{}, {}]",
        EXPECTED[0], EXPECTED[1], actual[0], actual[1]
    );
    exit(ExitCode::FAILURE);
}
