#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::debug::{exit, EXIT_FAILURE, EXIT_SUCCESS};
use cortex_m_semihosting::hprintln;
use panic_halt as _;

use oneliner::model;
use oneliner::runtime::{ModelInference, ModelSource};
use static_cell::ConstStaticCell;

#[cfg(not(feature = "mcunet"))]
#[model(
    "../models/lenet5_quantized.tflite",
    backend = "iree",
    arena = "shared",
    cmsis_nn = true
)]
struct Model;
#[cfg(not(feature = "mcunet"))]
const EXPECTED: [f32; 10] = [
    0.11666615, 0.11666615, 0.13124943, 0.68541366, 0.0, 0.36458173, 0.0, 0.0, 1.2104113,
    0.16041596,
];

#[cfg(feature = "mcunet")]
#[model(
    "../models/mcunet-10fps_vww.tflite",
    backend = "iree",
    arena = "shared",
    cmsis_nn = true
)]
struct Model;
#[cfg(feature = "mcunet")]
const EXPECTED: [i8; 2] = [4, -5];

#[cfg(not(feature = "mcunet"))]
static INPUT: ConstStaticCell<<Model as ModelInference>::InputTensor> =
    ConstStaticCell::new(<Model as ModelInference>::InputTensor::new(0.0));
#[cfg(feature = "mcunet")]
static INPUT: ConstStaticCell<<Model as ModelInference>::InputTensor> =
    ConstStaticCell::new(<Model as ModelInference>::InputTensor::new(0));

#[entry]
fn main() -> ! {
    let artifacts = <Model as ModelSource>::ARTIFACTS;
    hprintln!(
        "QEMU CMSIS-NN: input={} output={}",
        artifacts.input_size,
        artifacts.output_size
    );

    let mut model = Model::new();
    let input = INPUT.take();
    #[cfg(not(feature = "mcunet"))]
    input.fill(7.0);
    #[cfg(feature = "mcunet")]
    input.fill(7);

    let output = model.run(input);
    let actual = output.as_slice();
    hprintln!("QEMU CMSIS-NN output: {:?}", actual);

    #[cfg(not(feature = "mcunet"))]
    let matches = {
        let mut all = actual.len() == EXPECTED.len();
        for (actual, expected) in actual.iter().zip(EXPECTED) {
            if (*actual - expected).abs() > 1.0e-5 {
                all = false;
            }
        }
        all
    };
    #[cfg(feature = "mcunet")]
    let matches = actual == EXPECTED;

    if matches {
        hprintln!("QEMU CMSIS-NN validation PASSED");
        exit(EXIT_SUCCESS);
    }
    hprintln!(
        "QEMU CMSIS-NN validation FAILED: expected {:?}, received {:?}",
        EXPECTED,
        actual
    );
    exit(EXIT_FAILURE);
    loop {}
}
