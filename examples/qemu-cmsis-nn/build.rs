//! Copies the selected `memory.x` (M3 vs MVE) into the linker search path and
//! applies the `cortex-m-rt` link script.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let memory: &[u8] = if env::var("CARGO_FEATURE_MVE").is_ok() {
        include_bytes!("memory-mve.x")
    } else if env::var("CARGO_FEATURE_M33").is_ok() {
        include_bytes!("memory-m33.x")
    } else if env::var("CARGO_FEATURE_M4F").is_ok() {
        include_bytes!("memory-m4f.x")
    } else if env::var("CARGO_FEATURE_M4").is_ok() {
        include_bytes!("memory-m4.x")
    } else if env::var("CARGO_FEATURE_M0").is_ok() {
        include_bytes!("memory-m0.x")
    } else {
        include_bytes!("memory-m3.x")
    };
    let out = &PathBuf::from(env::var_os("OUT_DIR").unwrap());
    File::create(out.join("memory.x"))
        .unwrap()
        .write_all(memory)
        .unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    for name in ["m0", "m3", "m4", "m4f", "m33", "mve"] {
        println!("cargo:rerun-if-changed=memory-{name}.x");
    }

    println!("cargo:rustc-link-arg-bins=--nmagic");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
}
