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
#[model("../models/mcunet_10fps_vww.mlir", arena = "shared", vmcu = "auto")]
struct Model;

#[cfg(feature = "standard")]
#[model("../models/mcunet_10fps_vww.mlir", arena = "shared")]
struct Model;

const EXPECTED: [i8; 2] = [4, -5];

static INPUT: ConstStaticCell<<Model as ModelInference>::InputTensor> =
    ConstStaticCell::new(<Model as ModelInference>::InputTensor::new(7));

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
    let _ = hprintln!("mcunet output: {:?}", actual);
    if actual == EXPECTED {
        let _ = hprintln!("MCUNet QEMU validation PASSED");
        debug::exit(EXIT_SUCCESS);
    } else {
        let _ = hprintln!("MCUNet QEMU validation FAILED: expected {:?}", EXPECTED);
        debug::exit(EXIT_FAILURE);
    }

    loop {
        cortex_m::asm::wfi();
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    let _ = hprintln!("MCUNet QEMU panic");
    debug::exit(EXIT_FAILURE);
    loop {
        cortex_m::asm::wfi();
    }
}
