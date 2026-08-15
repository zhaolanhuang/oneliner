#!/usr/bin/env python3
"""Builds the CMSIS-NN ukernel shims as target-specific LLVM bitcode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
    "Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.c",
    "Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c",
    "Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c",
    "Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c",
    "Source/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.c",
    "Source/TransposeFunctions/arm_transpose_s8.c",
]

# Precompiled Cortex-M variants shipped with the macro crate. The id doubles
# as the file name (without the .bc suffix) and as the lookup key in
# manifest.json; it must be unique per (triple, cpu) since e.g. the
# thumbv8m.main triples cover both Cortex-M33 and Cortex-M55.
VARIANTS = [
    {
        "id": "thumbv6m-none-eabi__cortex-m0",
        "triple": "thumbv6m-none-eabi",
        "cpu": "cortex-m0",
        "features": "+strict-align,+atomics-32",
        "float_abi": "soft",
    },
    {
        "id": "thumbv7m-none-eabi__cortex-m3",
        "triple": "thumbv7m-none-eabi",
        "cpu": "cortex-m3",
        "features": "",
        "float_abi": "soft",
    },
    {
        "id": "thumbv7em-none-eabi__cortex-m4",
        "triple": "thumbv7em-none-eabi",
        "cpu": "cortex-m4",
        "features": "",
        "float_abi": "soft",
    },
    {
        "id": "thumbv7em-none-eabihf__cortex-m4",
        "triple": "thumbv7em-none-eabihf",
        "cpu": "cortex-m4",
        "features": "+vfp4d16sp",
        "float_abi": "hard",
    },
    {
        "id": "thumbv8m.main-none-eabi__cortex-m33",
        "triple": "thumbv8m.main-none-eabi",
        "cpu": "cortex-m33",
        "features": "",
        "float_abi": "soft",
    },
    {
        "id": "thumbv8m.main-none-eabihf__cortex-m33",
        "triple": "thumbv8m.main-none-eabihf",
        "cpu": "cortex-m33",
        "features": "+fp-armv8d16sp",
        "float_abi": "hard",
    },
    {
        "id": "thumbv8m.main-none-eabi__cortex-m55",
        "triple": "thumbv8m.main-none-eabi",
        "cpu": "cortex-m55",
        "features": "+mve",
        "float_abi": "soft",
    },
    {
        "id": "thumbv8m.main-none-eabihf__cortex-m55",
        "triple": "thumbv8m.main-none-eabihf",
        "cpu": "cortex-m55",
        "features": "+mve.fp",
        "float_abi": "hard",
    },
]


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def tool_version(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        return f"{name}:missing"
    try:
        version = subprocess.check_output(
            [path, "--version"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError):
        version = ""
    return f"{path}:{version}"


def llvm_major(name: str) -> int | None:
    match = re.search(r"clang version (\d+)\.", tool_version(name))
    return int(match.group(1)) if match else None


def compute_cache_key(
    cmsis_nn: Path,
    shim: Path,
    include_dirs: list[Path],
    target: str,
    cpu: str,
    features: str,
    compile_flags: list[str],
    tool_versions: dict[str, str],
) -> str:
    digest = hashlib.sha256()
    for component in (target, cpu, features):
        digest.update(component.encode())
        digest.update(b"\0")
    for flag in compile_flags:
        digest.update(flag.encode())
        digest.update(b"\0")
    for name, version in tool_versions.items():
        digest.update(name.encode())
        digest.update(version.encode())
        digest.update(b"\0")
    digest.update(shim.read_bytes())
    for include_dir in include_dirs:
        for header in sorted(include_dir.rglob("*")):
            if header.is_file():
                digest.update(header.relative_to(include_dir).as_posix().encode())
                digest.update(b"\0")
                digest.update(header.read_bytes())
    for source in SOURCES:
        digest.update((cmsis_nn / source).read_bytes())
    for header in sorted((cmsis_nn / "Include").rglob("*")):
        if header.is_file():
            digest.update(header.relative_to(cmsis_nn).as_posix().encode())
            digest.update(b"\0")
            digest.update(header.read_bytes())
    return digest.hexdigest()


def check_no_undefined_symbols(llvm_nm: str, bitcode: Path) -> bool:
    undefined = subprocess.check_output(
        [llvm_nm, "--undefined-only", str(bitcode)], text=True
    )
    if undefined:
        print("linked CMSIS-NN bitcode has undefined symbols:", file=sys.stderr)
        print(undefined, end="", file=sys.stderr)
        return False
    return True


def build_one(
    cmsis_nn: Path,
    shim: Path,
    output: Path,
    triple: str,
    cpu: str,
    features: str,
    float_abi: str,
    include_dirs: list[Path],
    cache_dir: Path | None,
    no_cache: bool,
    clang: str,
    llvm_link: str,
    llvm_nm: str,
) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)

    compile_flags = [
        clang,
        f"--target={triple}",
        f"-mcpu={cpu}",
        "-mthumb",
        "-O3",
        "-flto",
        "-emit-llvm",
        "-c",
        "-ffreestanding",
        "-ffunction-sections",
        "-fdata-sections",
        "-I",
        str(cmsis_nn / "Include"),
    ]
    for include_dir in include_dirs:
        compile_flags.extend(["-I", str(include_dir)])
    if float_abi == "hard":
        compile_flags.append("-mfloat-abi=hard")
        # A hard-float ABI on an ARMv8-M core requires an explicit FPU.
        if cpu in ("cortex-m33", "cortex-m55"):
            compile_flags.append("-mfpu=fpv5-d16")

    cached: Path | None = None
    if not no_cache and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tool_versions = {
            "clang": tool_version("clang"),
            "llvm-link": tool_version("llvm-link"),
            "llvm-nm": tool_version("llvm-nm"),
        }
        key = compute_cache_key(
            cmsis_nn, shim, include_dirs, triple, cpu, features, compile_flags,
            tool_versions,
        )
        cached = cache_dir / f"{key}.bc"
        if cached.exists():
            shutil.copyfile(cached, output)
            if not check_no_undefined_symbols(llvm_nm, output):
                return False
            return True

    object_dir = output.parent / "cmsis-nn-bc"
    object_dir.mkdir(parents=True, exist_ok=True)

    inputs = [shim, *(cmsis_nn / source for source in SOURCES)]
    bitcode_files: list[Path] = []
    for index, source in enumerate(inputs):
        bitcode = object_dir / f"{index:02d}-{source.stem}.bc"
        run([*compile_flags, str(source), "-o", str(bitcode)])
        bitcode_files.append(bitcode)

    run([llvm_link, *(str(path) for path in bitcode_files), "-o", str(output)])
    if not check_no_undefined_symbols(llvm_nm, output):
        return False

    if not no_cache and cache_dir is not None and cached is not None:
        fd, temporary = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
        with os.fdopen(fd, "wb") as handle:
            handle.write(output.read_bytes())
        os.replace(temporary, cached)

    return True


def write_manifest(prebuilt_dir: Path, llvm_major_value: int | None) -> None:
    manifest = {
        "llvm_major": llvm_major_value,
        "variants": [
            {
                "id": variant["id"],
                "triple": variant["triple"],
                "cpu": variant["cpu"],
                "features": variant["features"],
                "float_abi": variant["float_abi"],
                "file": f"{variant['id']}.bc",
            }
            for variant in VARIANTS
        ],
    }
    (prebuilt_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmsis-nn", type=Path, required=True)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--cpu")
    parser.add_argument("--features", default="")
    parser.add_argument(
        "--build-all",
        type=Path,
        default=None,
        metavar="DIR",
        help="build all precompiled Cortex-M variants into DIR and write "
        "manifest.json (ignores --output/--target/--cpu/--features)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="bitcode cache directory (default: <workspace>/target/cmsis-nn-bc)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="disable the bitcode cache"
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="DIR",
        help="extra include directory for the C library headers (repeatable)",
    )
    args = parser.parse_args()

    clang = shutil.which("clang")
    llvm_link = shutil.which("llvm-link")
    llvm_nm = shutil.which("llvm-nm")
    if clang is None or llvm_link is None or llvm_nm is None:
        parser.error("clang, llvm-link, and llvm-nm must be available in PATH")

    cmsis_nn = args.cmsis_nn.resolve()
    shim = args.shim.resolve()

    # Freestanding include directory shipping a minimal <string.h>; the rest
    # of the standard headers (stdint, stddef, stdbool, limits) come from
    # clang's builtin headers.
    include_dirs = [Path(__file__).resolve().parent / "include"]
    include_dirs += [Path(path).resolve() for path in args.include]

    if args.build_all is not None:
        prebuilt_dir = args.build_all.resolve()
        prebuilt_dir.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            output = prebuilt_dir / f"{variant['id']}.bc"
            print(f"building {variant['id']} ...", file=sys.stderr)
            if not build_one(
                cmsis_nn, shim, output, variant["triple"], variant["cpu"],
                variant["features"], variant["float_abi"], include_dirs,
                None, args.no_cache, clang, llvm_link, llvm_nm,
            ):
                return 1
        write_manifest(prebuilt_dir, llvm_major("clang"))
        return 0

    if args.output is None or args.target is None or args.cpu is None:
        parser.error(
            "--output, --target, and --cpu are required unless --build-all is used"
        )

    output = args.output.resolve()
    cache_dir: Path | None = None
    if not args.no_cache:
        cache_dir = args.cache
        if cache_dir is None:
            cache_dir = cmsis_nn.parent.parent / "target" / "cmsis-nn-bc"
        cache_dir = cache_dir.resolve()

    float_abi = "hard" if (args.target.endswith("eabihf") or "+vfp" in args.features) else "soft"
    if not build_one(
        cmsis_nn, shim, output, args.target, args.cpu, args.features,
        float_abi, include_dirs, cache_dir, args.no_cache, clang, llvm_link,
        llvm_nm,
    ):
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(f"CMSIS-NN bitcode build failed: {error}", file=sys.stderr)
        raise SystemExit(error.returncode)
