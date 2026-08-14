#!/usr/bin/env python3
"""Builds the CMSIS-NN ukernel shims as target-specific LLVM bitcode."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SOURCES = [
    "Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c",
    "Source/ConvolutionFunctions/arm_convolve_1x1_s8.c",
    "Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c",
    "Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c",
    "Source/ConvolutionFunctions/arm_convolve_s8.c",
    "Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c",
    "Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c",
    "Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.c",
    "Source/ConvolutionFunctions/arm_depthwise_conv_s8.c",
    "Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c",
    "Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c",
    "Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.c",
    "Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c",
    "Source/FullyConnectedFunctions/arm_fully_connected_s8.c",
    "Source/PoolingFunctions/arm_max_pool_s8.c",
    "Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c",
    "Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c",
    "Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c",
    "Source/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.c",
]


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmsis-nn", type=Path, required=True)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--features", default="")
    args = parser.parse_args()

    clang = shutil.which("clang")
    llvm_link = shutil.which("llvm-link")
    llvm_nm = shutil.which("llvm-nm")
    if clang is None or llvm_link is None or llvm_nm is None:
        parser.error("clang, llvm-link, and llvm-nm must be available in PATH")

    cmsis_nn = args.cmsis_nn.resolve()
    shim = args.shim.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    object_dir = output.parent / "cmsis-nn-bc"
    object_dir.mkdir(parents=True, exist_ok=True)

    compile_flags = [
        clang,
        f"--target={args.target}",
        f"-mcpu={args.cpu}",
        "-mthumb",
        "-O3",
        "-flto",
        "-emit-llvm",
        "-c",
        "-ffunction-sections",
        "-fdata-sections",
        "-I",
        str(cmsis_nn / "Include"),
        "-I",
        "/usr/include/newlib",
    ]
    if "+vfp" in args.features or args.target.endswith("eabihf"):
        compile_flags.append("-mfloat-abi=hard")

    inputs = [shim, *(cmsis_nn / source for source in SOURCES)]
    bitcode_files: list[Path] = []
    for index, source in enumerate(inputs):
        bitcode = object_dir / f"{index:02d}-{source.stem}.bc"
        run([*compile_flags, str(source), "-o", str(bitcode)])
        bitcode_files.append(bitcode)

    run([llvm_link, *(str(path) for path in bitcode_files), "-o", str(output)])
    undefined = subprocess.check_output(
        [llvm_nm, "--undefined-only", str(output)], text=True
    )
    if undefined:
        print("linked CMSIS-NN bitcode has undefined symbols:", file=sys.stderr)
        print(undefined, end="", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(f"CMSIS-NN bitcode build failed: {error}", file=sys.stderr)
        raise SystemExit(error.returncode)
