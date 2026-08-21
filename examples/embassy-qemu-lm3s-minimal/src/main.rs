#![no_main]
#![no_std]

use core::panic::PanicInfo;

use cortex_m_semihosting::{
    debug::{self, EXIT_FAILURE, EXIT_SUCCESS},
    hprintln,
};
use embassy_executor::Spawner;
use oneliner::model;
use oneliner::runtime::{ModelInference, ModelSource};
use static_cell::ConstStaticCell;

#[model("../models/lenet5_quantized.tflite", arena = "shared")]
struct Model;

const EXPECTED: [f32; 10] = [
    0.11666615, 0.11666615, 0.13124943, 0.68541366, 0.0, 0.36458173, 0.0, 0.0, 1.2104113,
    0.16041596,
];

static INPUT: ConstStaticCell<<Model as ModelInference>::InputTensor> =
    ConstStaticCell::new(<Model as ModelInference>::InputTensor::new(0.0));

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let artifacts = <Model as ModelSource>::ARTIFACTS;
    let _ = hprintln!(
        "Model sizes: flash={} arena={} input={} output={}",
        artifacts.total_flash_size,
        artifacts.ram_size,
        artifacts.input_size,
        artifacts.output_size
    );

    let input = INPUT.take();
    input.fill(7.0);

    let mut model = Model::new();
    let output = model.run(input);
    if output.as_slice() == EXPECTED {
        let _ = hprintln!("Model IREE validation passed");
        debug::exit(EXIT_SUCCESS);
    }

    let _ = hprintln!("Model IREE validation failed");
    debug::exit(EXIT_FAILURE);
    loop {
        cortex_m::asm::wfi();
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    let _ = hprintln!("panic");
    debug::exit(EXIT_FAILURE);
    loop {
        cortex_m::asm::wfi();
    }
}
