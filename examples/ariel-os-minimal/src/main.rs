#![no_main]
#![no_std]

use ariel_os::debug::{exit, ExitCode};
use ariel_os::log::{error, info};
use ariel_os::time;

use oneliner::model;
use oneliner::runtime::{ModelInference, ModelSource};

use static_cell::ConstStaticCell;

 #[model(
     "../models/mcunet-10fps_vww.tflite",
     arena = "shared",
     vmcu = "auto"
 )]
 struct Model;
 const INPUT_LEN: usize = 64 * 64 * 3;
 const EXPECTED: [i8; 2] = [4, -5];

//#[model("../models/lenet5_quantized.tflite")]
//struct Model;
//const INPUT_LEN: usize = 28 * 28 * 1;
//const OUTPUT_LEN: usize = 10;
//const EXPECTED: [f32; OUTPUT_LEN] = [
//    0.11666615, 0.11666615, 0.13124943, 0.68541366, 0.0, 0.36458173, 0.0, 0.0, 1.2104113,
//    0.16041596,
//];
// static INPUT_CELL: ConstStaticCell<<Model as ModelInference>::InputTensor> = ConstStaticCell::new(<Model as ModelInference>::InputTensor::new(0));

#[ariel_os::thread(autostart, priority = 1, stacksize=20480)]
// #[ariel_os::thread(autostart, priority = 1)]
fn main() {
    let artifacts = <Model as ModelSource>::ARTIFACTS;
    info!(
        "Oneliner IREE example running on {}",
        ariel_os::buildinfo::BOARD
    );
    info!(
        "Model artifact sizes: input={} output={}",
        artifacts.input_size, artifacts.output_size
    );

    let mut model = Model::new();
    let mut input = Model::create_input_tensor();
    // let mut input = INPUT_CELL.take();
    input.fill(7);
    let time_begin_us = time::Instant::now().as_micros();
    let output = model.run(&input);
    let time_end_us = time::Instant::now().as_micros();
    info!("Model inference time: {:?} us", time_end_us - time_begin_us);

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
