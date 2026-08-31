#![no_std]
#![no_main]

use defmt::{error, info};
use embassy_executor::Spawner;
use {defmt_rtt as _, panic_probe as _};

use oneliner::model;
use oneliner::runtime::{ModelInference, ModelSource};
use oneliner_profiler::Profiler;
use static_cell::ConstStaticCell;

#[model("../models/lenet5_quantized.tflite", arena = "shared")]
struct Model;
const INPUT_LEN: usize = 28 * 28 * 1;
const OUTPUT_LEN: usize = 10;
const EXPECTED: [f32; OUTPUT_LEN] = [0.0; OUTPUT_LEN];
static INPUT_CELL: ConstStaticCell<<Model as ModelInference>::InputTensor> =
    ConstStaticCell::new(<Model as ModelInference>::InputTensor::new(0.0));

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let _p = embassy_rp::init(Default::default());
    let artifacts = <Model as ModelSource>::ARTIFACTS;

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
    // let mut input = Model::create_input_tensor();
    let mut input = INPUT_CELL.take();
    input.fill(7.0);

    let mut profiler = Profiler::new();
    let output = profiler.profile(|| model.run(&input));
    info!("Profiled inference stats: {}", profiler.stats());

    let actual = output.as_slice();
    if actual == EXPECTED {
        info!("Model IREE validation passed");
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
}
