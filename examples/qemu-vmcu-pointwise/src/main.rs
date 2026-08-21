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
#[model(
    "../models/vmcu_pointwise_pair.mlir",
    arena = "shared",
    vmcu = "auto"
)]
struct Model;

#[cfg(feature = "standard")]
#[model("../models/vmcu_pointwise_pair.mlir", arena = "shared")]
struct Model;

const INPUT_VALUES: [i8; 16] = [1, 2, 3, 4, -1, 0, 1, 2, 5, -3, 2, 1, 127, -128, 64, -64];
const EXPECTED: [i8; 12] = [8, 17, 10, 2, 1, 6, -18, 21, 12, -128, -1, 0];

static INPUT: ConstStaticCell<<Model as ModelInference>::InputTensor> =
    ConstStaticCell::new(<Model as ModelInference>::InputTensor::new(0));

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

    let input = INPUT.take();
    input.as_slice_mut().copy_from_slice(&INPUT_VALUES);
    let output = Model::new().run(input);
    let actual = output.as_slice();
    let _ = hprintln!("pointwise-pair output: {:?}", actual);

    if actual == EXPECTED {
        let _ = hprintln!("pointwise-pair QEMU validation PASSED");
        debug::exit(EXIT_SUCCESS);
    } else {
        let _ = hprintln!(
            "pointwise-pair QEMU validation FAILED: expected {:?}",
            EXPECTED
        );
        debug::exit(EXIT_FAILURE);
    }

    loop {
        cortex_m::asm::wfi();
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    let _ = hprintln!("vMCU QEMU panic");
    debug::exit(EXIT_FAILURE);
    loop {
        cortex_m::asm::wfi();
    }
}
