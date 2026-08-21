#![no_main]
#![no_std]

use core::panic::PanicInfo;

use cortex_m_rt::entry;
use cortex_m_semihosting::{
    debug::{self, EXIT_FAILURE, EXIT_SUCCESS},
    hprintln,
};
use oneliner::model;
use oneliner::runtime::{ModelInference, ModelSource};
use static_cell::ConstStaticCell;

#[cfg(not(feature = "standard"))]
#[model("../models/lenet5_quantized.tflite", arena = "shared", vmcu = "auto")]
struct Model;

#[cfg(feature = "standard")]
#[model("../models/lenet5_quantized.tflite", arena = "shared")]
struct Model;

const OUTPUT_LEN: usize = 10;
const EXPECTED: [f32; OUTPUT_LEN] = [
    0.11666615, 0.11666615, 0.13124943, 0.68541366, 0.0, 0.36458173, 0.0, 0.0, 1.2104113,
    0.16041596,
];

static INPUT: ConstStaticCell<<Model as ModelInference>::InputTensor> =
    ConstStaticCell::new(<Model as ModelInference>::InputTensor::new(7.0));

#[entry]
fn main() -> ! {
    let artifacts = <Model as ModelSource>::ARTIFACTS;
    let _ = hprintln!(
        "auto: vmcu={} arena={} input={} output={}",
        !cfg!(feature = "standard"),
        artifacts.ram_size,
        artifacts.input_size,
        artifacts.output_size
    );

    let output = Model::new().run(INPUT.take());
    let actual = output.as_slice();
    let _ = hprintln!("lenet5 output: {:?}", actual);
    if actual == EXPECTED {
        let _ = hprintln!("LeNet5 QEMU validation PASSED");
        debug::exit(EXIT_SUCCESS);
    } else {
        let _ = hprintln!("LeNet5 QEMU validation FAILED: expected {:?}", EXPECTED);
        debug::exit(EXIT_FAILURE);
    }

    loop {
        cortex_m::asm::wfi();
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    let _ = hprintln!("LeNet5 QEMU panic");
    debug::exit(EXIT_FAILURE);
    loop {
        cortex_m::asm::wfi();
    }
}
