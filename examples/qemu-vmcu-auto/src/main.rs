#![no_main]
#![no_std]

use core::panic::PanicInfo;

use cortex_m::peripheral::syst::SystClkSource;
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
const ITERATIONS: u32 = 30;

unsafe extern "C" {
    static _stack_start: u8;
    static _stack_end: u8;
}

const STACK_SENTINEL_MARGIN: usize = 1024;

fn fill_stack_sentinel() {
    let bottom = &raw const _stack_end as usize;
    let mut sp: usize = 0;
    unsafe {
        core::arch::asm!("mov {0}, sp", out(reg) sp);
    }
    if sp > bottom + STACK_SENTINEL_MARGIN {
        unsafe {
            core::ptr::write_bytes(
                bottom as *mut u8,
                0xAA,
                sp - bottom - STACK_SENTINEL_MARGIN,
            );
        }
    }
}

fn stack_high_water_bytes() -> usize {
    let top = &raw const _stack_start as usize;
    let bottom = &raw const _stack_end as usize;
    let mut p = top;
    let mut deepest = top;
    unsafe {
        while p > bottom {
            p -= 1;
            if *((p) as *const u8) != 0xAA {
                deepest = p;
            }
        }
    }
    top - deepest
}

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

    fill_stack_sentinel();

    let mut peripherals = cortex_m::Peripherals::take().unwrap();
    let syst = &mut peripherals.SYST;
    syst.set_clock_source(SystClkSource::Core);
    syst.set_reload(0x00FF_FFFF);
    syst.clear_current();
    syst.enable_counter();

    let mut best = u32::MAX;
    let mut total = 0u64;
    let mut valid = 0u32;
    let mut output = None;
    for iteration in 0..ITERATIONS {
        let input = <Model as ModelInference>::InputTensor::new(7);
        let start = cortex_m::peripheral::SYST::get_current();
        let result = Model::new().run(&input);
        let end = cortex_m::peripheral::SYST::get_current();
        output = Some(result);
        if end > start {
            continue;
        }
        let ticks = start - end;
        best = best.min(ticks);
        total += u64::from(ticks);
        valid += 1;
        let _ = hprintln!("iter {}: {} ticks", iteration, ticks);
    }
    let last_output = output.unwrap();
    let actual = last_output.as_slice();
    let _ = hprintln!("mcunet output: {:?}", actual);
    if valid > 0 {
        let _ = hprintln!(
            "latency: best={} ticks avg={} ticks ({} valid iterations)",
            best,
            total / u64::from(valid),
            valid
        );
    } else {
        let _ = hprintln!("latency: all iterations wrapped the 24-bit SysTick counter");
    }
    let _ = hprintln!("stack high-water: {} bytes", stack_high_water_bytes());
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