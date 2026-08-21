import ctypes
import os
import random
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "oneliner-macro" / "vmcu" / "oneliner_vmcu_generic.c"
GENERIC_SOURCE = ROOT / "oneliner-macro" / "vmcu" / "oneliner_vmcu_generic.c"
MAGIC = 0x564D4355
I8 = ctypes.c_int8
I32 = ctypes.c_int32
SIZE = ctypes.c_size_t
P_I8 = ctypes.POINTER(I8)
P_I32 = ctypes.POINTER(I32)


def s32(value):
    value &= 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


def scale(value, multiplier, shift, double_round):
    rounding = 1 << (shift - 1)
    if double_round and shift > 31:
        rounding += (1 << 30) if value * multiplier >= 0 else -(1 << 30)
    return s32((value * multiplier + rounding) // (1 << shift))


def requant(value, multiplier, shift, zero_point):
    shifted = s32(scale(value, multiplier, shift, True) + zero_point)
    return max(-128, min(127, shifted))


def accumulate(accumulator, activation, weight, zero_point):
    return s32(accumulator + (activation - zero_point) * weight)


@dataclass
class Block:
    config: list
    input_values: list
    w_expand: list
    w_depthwise: list
    w_project: list
    b_expand: list
    b_depthwise: list
    b_project: list
    m_expand: list
    m_depthwise: list
    m_project: list
    s_expand: list
    s_depthwise: list
    s_project: list


def oracle(block):
    cfg = block.config
    n_count, ih, iw, cin = cfg[2:6]
    oh, ow, cexp, cout = cfg[6:10]
    kh, kw, sh, sw, dh, dw = cfg[10:16]
    pt, pl = cfg[16:18]
    izp, ezp, dzp, pzp, ozp = cfg[20:25]
    expanded = [0] * (n_count * ih * iw * cexp)
    depthwise = [0] * (n_count * oh * ow * cexp)
    projected = [0] * (n_count * oh * ow * cout)

    for n in range(n_count):
        for y in range(ih):
            for x in range(iw):
                for ec in range(cexp):
                    acc = block.b_expand[ec]
                    for ic in range(cin):
                        inp = block.input_values[
                            ((n * ih + y) * iw + x) * cin + ic
                        ]
                        acc = accumulate(
                            acc, inp, block.w_expand[ec * cin + ic], izp
                        )
                    expanded[((n * ih + y) * iw + x) * cexp + ec] = requant(
                        acc, block.m_expand[ec], block.s_expand[ec], ezp
                    )

    for n in range(n_count):
        for oy in range(oh):
            for ox in range(ow):
                for ec in range(cexp):
                    acc = block.b_depthwise[ec]
                    for ky in range(kh):
                        iy = oy * sh + ky * dh - pt
                        if not 0 <= iy < ih:
                            continue
                        for kx in range(kw):
                            ix = ox * sw + kx * dw - pl
                            if not 0 <= ix < iw:
                                continue
                            activation = expanded[
                                ((n * ih + iy) * iw + ix) * cexp + ec
                            ]
                            weight = block.w_depthwise[
                                (ky * kw + kx) * cexp + ec
                            ]
                            acc = accumulate(acc, activation, weight, ezp)
                    depthwise[
                        ((n * oh + oy) * ow + ox) * cexp + ec
                    ] = requant(
                        acc, block.m_depthwise[ec], block.s_depthwise[ec], dzp
                    )

    for n in range(n_count):
        for oy in range(oh):
            for ox in range(ow):
                for oc in range(cout):
                    acc = block.b_project[oc]
                    for ec in range(cexp):
                        activation = depthwise[
                            ((n * oh + oy) * ow + ox) * cexp + ec
                        ]
                        acc = accumulate(
                            acc, activation, block.w_project[oc * cexp + ec], dzp
                        )
                    value = requant(
                        acc, block.m_project[oc], block.s_project[oc], pzp
                    )
                    if cfg[1] & 1:
                        new_value = scale(value - pzp, cfg[25], cfg[26], False)
                        if cfg[27]:
                            new_value = scale(new_value, cfg[27], cfg[28], True)
                        skip = block.input_values[
                            ((n * ih + oy) * iw + ox) * cin + oc
                        ] - izp
                        skip = scale(skip, cfg[29], cfg[30], False)
                        if cfg[31]:
                            skip = scale(skip, cfg[31], cfg[32], True)
                        value = requant(
                            s32(new_value + skip), cfg[33], cfg[34], ozp
                        )
                    projected[((n * oh + oy) * ow + ox) * cout + oc] = value
    return projected


def make_block(
    seed,
    *,
    ih,
    iw,
    cin,
    cexp,
    cout,
    kernel,
    stride,
    pads,
    residual,
    in_place_permitted=False,
    batches=1,
    dilation=1,
    edge_input=False,
):
    rng = random.Random(seed)
    pt, pl, pb, pr = pads
    effective = (kernel - 1) * dilation + 1
    oh = (ih + pt + pb - effective) // stride + 1
    ow = (iw + pl + pr - effective) // stride + 1
    flags = int(residual) | (2 if in_place_permitted else 0)
    izp, ezp, dzp, pzp, ozp = -11, 7, -5, 13, -9
    config = [
        1,
        flags,
        batches,
        ih,
        iw,
        cin,
        oh,
        ow,
        cexp,
        cout,
        kernel,
        kernel,
        stride,
        stride,
        dilation,
        dilation,
        pt,
        pl,
        pb,
        pr,
        izp,
        ezp,
        dzp,
        pzp,
        ozp,
        1073741824,
        30,
        1879048192 if residual else 0,
        31 if residual else 0,
        1073741824,
        30,
        0,
        0,
        1610612736,
        30,
        0,
        MAGIC,
    ]
    if not residual:
        config[25:35] = [0] * 10
    cache = kernel * iw * cexp
    depth_row = ow * cexp
    delayed = (pb + 1) * ow * cout if in_place_permitted else 0
    config[35] = cache + depth_row + delayed

    input_count = batches * ih * iw * cin
    if edge_input:
        edges = [-128, 127, izp, -1, 0, 1, 126, -127]
        input_values = [edges[i % len(edges)] for i in range(input_count)]
    else:
        input_values = [rng.randint(-128, 127) for _ in range(input_count)]

    def random_i8(count):
        return [rng.randint(-128, 127) for _ in range(count)]

    def random_bias(count):
        edges = [-(1 << 31), (1 << 31) - 1, -100000, 0, 100000]
        return [edges[i % len(edges)] if i < len(edges) else rng.randint(-50000, 50000)
                for i in range(count)]

    def multipliers(count):
        values = [1073741824, 1610612736, 1879048192, 805306368]
        return [values[i % len(values)] for i in range(count)]

    def shifts(count):
        values = [30, 31, 32, 29]
        return [values[i % len(values)] for i in range(count)]

    return Block(
        config,
        input_values,
        random_i8(cexp * cin),
        random_i8(kernel * kernel * cexp),
        random_i8(cout * cexp),
        random_bias(cexp),
        random_bias(cexp),
        random_bias(cout),
        multipliers(cexp),
        list(reversed(multipliers(cexp))),
        multipliers(cout),
        shifts(cexp),
        list(reversed(shifts(cexp))),
        shifts(cout),
    )


def padded_array(ctype, values, prefix_elements):
    suffix_elements = 5
    array = (ctype * (prefix_elements + len(values) + suffix_elements))()
    sentinel = -91 if ctype is I8 else -123456789
    for index in range(len(array)):
        array[index] = sentinel
    for index, value in enumerate(values):
        array[prefix_elements + index] = value
    return array, prefix_elements


class NativeKernel:
    def __init__(self, library):
        self.library = library
        self.function = library.oneliner_vmcu_ibn_s8
        pointer_types = [
            P_I8,
            P_I8,
            P_I8,
            P_I8,
            P_I32,
            P_I32,
            P_I32,
            P_I32,
            P_I32,
            P_I32,
            P_I8,
            P_I8,
            P_I8,
            P_I32,
            P_I8,
            P_I8,
        ]
        self.function.argtypes = [item for pointer in pointer_types for item in (pointer, SIZE)]
        self.function.restype = None

    def run(self, block, *, alias=False, config_override=None):
        config = block.config if config_override is None else config_override
        arrays = []
        values_and_types = [
            (block.input_values, I8),
            (block.w_expand, I8),
            (block.w_depthwise, I8),
            (block.w_project, I8),
            (block.b_expand, I32),
            (block.b_depthwise, I32),
            (block.b_project, I32),
            (block.m_expand, I32),
            (block.m_depthwise, I32),
            (block.m_project, I32),
            (block.s_expand, I8),
            (block.s_depthwise, I8),
            (block.s_project, I8),
            (config, I32),
        ]
        for index, (values, ctype) in enumerate(values_and_types):
            array, offset = padded_array(ctype, values, 2 + index % 3)
            arrays.append((array, offset))

        output_count = config[2] * config[6] * config[7] * config[9]
        if alias:
            output_array, output_offset = arrays[0]
        else:
            output_array, output_offset = padded_array(I8, [77] * output_count, 4)
        scratch_array, scratch_offset = padded_array(I8, [66] * max(config[35], 1), 3)
        arguments = []
        for array, offset in arrays:
            arguments.extend((array, offset))
        arguments.extend((output_array, output_offset, scratch_array, scratch_offset))
        before_scratch = list(scratch_array)
        self.function(*arguments)
        output_start = output_offset
        output_values = list(output_array)[
            output_start : output_start + output_count
        ]
        return output_values, list(scratch_array), before_scratch


class VmcuMcunetKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.kernels = []
        compilers = [os.environ.get("CC", "cc")]
        for compiler in dict.fromkeys(compilers):
            normal = Path(cls.temporary_directory.name) / "mcunet.so"
            command = [
                compiler,
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wconversion",
                "-Wsign-conversion",
                "-Werror",
                "-pedantic",
                "-fPIC",
                "-shared",
                str(SOURCE),
                "-o",
                str(normal),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            cls.kernels.append(("strict", NativeKernel(ctypes.CDLL(str(normal)))))

            sanitized = Path(cls.temporary_directory.name) / "mcunet_ubsan.so"
            sanitize_command = command[:-2] + [
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
                "-o",
                str(sanitized),
            ]
            try:
                subprocess.run(
                    sanitize_command, check=True, capture_output=True, text=True
                )
                os.environ.setdefault("UBSAN_OPTIONS", "halt_on_error=1")
                cls.kernels.append(
                    ("ubsan", NativeKernel(ctypes.CDLL(str(sanitized))))
                )
            except (subprocess.CalledProcessError, OSError):
                pass

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def assert_block(self, block):
        expected = oracle(block)
        for build, kernel in self.kernels:
            with self.subTest(build=build):
                actual, _, _ = kernel.run(block)
                self.assertEqual(actual, expected)

    def test_stride_one_residual_in_place_matches_out_of_place(self):
        block = make_block(
            0x51DE,
            ih=6,
            iw=5,
            cin=3,
            cexp=7,
            cout=3,
            kernel=3,
            stride=1,
            pads=(1, 1, 1, 1),
            residual=True,
            in_place_permitted=True,
            batches=2,
            edge_input=True,
        )
        expected = oracle(block)
        for build, kernel in self.kernels:
            with self.subTest(build=build):
                out_of_place, _, _ = kernel.run(block)
                in_place, _, _ = kernel.run(block, alias=True)
                self.assertEqual(out_of_place, expected)
                self.assertEqual(in_place, expected)
                self.assertEqual(in_place, out_of_place)

    def test_stride_two_non_residual_random(self):
        self.assert_block(
            make_block(
                0x5721,
                ih=7,
                iw=8,
                cin=4,
                cexp=6,
                cout=5,
                kernel=3,
                stride=2,
                pads=(1, 0, 1, 1),
                residual=False,
            )
        )

    def test_kernel_five_and_seven_edges(self):
        for kernel, size in ((5, 6), (7, 5)):
            with self.subTest(kernel=kernel):
                self.assert_block(
                    make_block(
                        0x7000 + kernel,
                        ih=size,
                        iw=size + 1,
                        cin=2,
                        cexp=4,
                        cout=3,
                        kernel=kernel,
                        stride=1,
                        pads=(kernel // 2,) * 4,
                        residual=False,
                        edge_input=True,
                    )
                )

    def test_dilated_kernel_exercises_non_rotating_cache(self):
        self.assert_block(
            make_block(
                0xD11A,
                ih=7,
                iw=6,
                cin=2,
                cexp=5,
                cout=4,
                kernel=3,
                stride=1,
                pads=(2, 2, 2, 2),
                residual=False,
                dilation=2,
            )
        )

    def test_double_round_uses_sign_of_value_at_large_shift(self):
        # TOSA/IREE apply_scale with DOUBLE_ROUND adds +/- (1 << 30) based on
        # the sign of the value when shift > 31 (MLIR ApplyScaleGenericOp
        # lowering). TFLite-style kernels instead always add +(1 << 30); the
        # two disagree at rounding boundaries. Regression for the MCUNet
        # block-1 projection boundary: acc=-2824, multiplier=1216494916,
        # shift=38 (IREE gives -35 before +zero-point, TFLite gives -34).
        block = make_block(
            0xD0B0,
            ih=2,
            iw=2,
            cin=2,
            cexp=3,
            cout=2,
            kernel=3,
            stride=1,
            pads=(1, 1, 1, 1),
            residual=False,
        )
        cout = block.config[9]
        cexp = block.config[8]
        pzp = block.config[23]
        multiplier = 1216494916
        shift = 38
        block.m_project = [multiplier] * cout
        block.s_project = [shift] * cout
        block.b_project = [-2824] + [0] * (cout - 1)
        block.w_project = [0] * (cout * cexp)
        expected = []
        for oc in range(cout):
            expected.append(
                max(-128, min(127, s32(scale(block.b_project[oc], multiplier, shift, True) + pzp)))
            )
        expected *= block.config[6] * block.config[7]
        positive_only = s32(
            (block.b_project[0] * multiplier + (1 << (shift - 1)) + (1 << 30))
            // (1 << shift) + pzp
        )
        self.assertNotEqual(expected[0], positive_only)
        for build, kernel in self.kernels:
            with self.subTest(build=build):
                actual, _, _ = kernel.run(block)
                self.assertEqual(actual, expected)

GENERIC_SOURCE = ROOT / "oneliner-macro" / "vmcu" / "oneliner_vmcu_generic.c"


def s32(value):
    value &= 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


def floor_div_pow2(value, shift):
    divisor = 1 << shift
    return value // divisor if value >= 0 else -1 - ((-1 - value) // divisor)


def scale(value, multiplier, shift, double_round):
    rounding = 1 << (shift - 1)
    if double_round and shift > 31:
        rounding += (1 << 30) if value * multiplier >= 0 else -(1 << 30)
    return s32((value * multiplier + rounding) // (1 << shift))


def requant(value, multiplier, shift, zero_point):
    shifted = s32(scale(value, multiplier, shift, True) + zero_point)
    return max(-128, min(127, shifted))


def accumulate(accumulator, activation, weight, zero_point):
    return s32(accumulator + (activation - zero_point) * weight)


def conv2d_oracle(input_values, weight_hwcf, bias, multiplier, shift, izp, ozp, cfg):
    n, ih, iw, cin = cfg[2:6]
    oh, ow, cout = cfg[6:9]
    kh, kw = cfg[9:11]
    sh, sw, dh, dw = cfg[11:15]
    pt, pl, pb, pr = cfg[15:19]
    output = []
    for b in range(n):
        for oy in range(oh):
            for ox in range(ow):
                for oc in range(cout):
                    acc = bias[oc]
                    for ky in range(kh):
                        iy = oy * sh + ky * dh - pt
                        if not 0 <= iy < ih:
                            continue
                        for kx in range(kw):
                            ix = ox * sw + kx * dw - pl
                            if not 0 <= ix < iw:
                                continue
                            for ic in range(cin):
                                activation = input_values[((b * ih + iy) * iw + ix) * cin + ic]
                                weight = weight_hwcf[((ky * kw + kx) * cin + ic) * cout + oc]
                                acc = accumulate(acc, activation, weight, izp)
                    output.append(requant(acc, multiplier[oc], shift[oc], ozp))
    return output


def fc_oracle(input_values, weight, bias, multiplier, shift, izp, ozp, cfg):
    rows, k, n_out = cfg[2:5]
    output = []
    for m in range(rows):
        for oc in range(n_out):
            acc = bias[oc]
            for ic in range(k):
                acc = accumulate(acc, input_values[m * k + ic], weight[oc * k + ic], izp)
            output.append(requant(acc, multiplier[oc], shift[oc], ozp))
    return output


def make_conv2d_block(seed, *, ih, iw, cin, cout, kernel, stride, pads, batches=1, dilation=1):
    rng = random.Random(seed)
    pt, pl, pb, pr = pads
    effective = (kernel - 1) * dilation + 1
    oh = (ih + pt + pb - effective) // stride + 1
    ow = (iw + pl + pr - effective) // stride + 1
    izp, ozp = -1, 2
    config = [
        1, 0, batches, ih, iw, cin, oh, ow, cout, kernel, kernel,
        stride, stride, dilation, dilation, pt, pl, pb, pr, izp, ozp,
        *(0,) * 14, kernel * iw * cin, MAGIC,
    ]
    input_values = [rng.randint(-128, 127) for _ in range(batches * ih * iw * cin)]
    weight_fhwc = [rng.randint(-128, 127) for _ in range(cout * kernel * kernel * cin)]
    weight_hwcf = [
        weight_fhwc[((f * kernel + ky) * kernel + kx) * cin + ic]
        for ky in range(kernel)
        for kx in range(kernel)
        for ic in range(cin)
        for f in range(cout)
    ]
    bias = [rng.randint(-50000, 50000) for _ in range(cout)]
    multiplier = [1073741824, 1610612736, 1879048192, 805306368][:cout]
    shift = [30, 31, 32, 29][:cout]
    return config, input_values, weight_hwcf, bias, multiplier, shift, izp, ozp


def make_fc_block(seed, *, rows, k, n_out):
    rng = random.Random(seed)
    izp, ozp = -11, 13
    config = [
        1, 0, rows, k, n_out,
        *(0,) * 15, izp, ozp,
        *(0,) * 13, n_out, MAGIC,
    ]
    input_values = [rng.randint(-128, 127) for _ in range(rows * k)]
    weight = [rng.randint(-128, 127) for _ in range(n_out * k)]
    bias = [1000, -2000, 3000][:n_out]
    multiplier = [1073741824, 1610612736, 1879048192, 805306368][:n_out]
    shift = [30, 31, 32, 29][:n_out]
    return config, input_values, weight, bias, multiplier, shift, izp, ozp



class GenericKernel:
    def __init__(self, library):
        self.library = library
        self.function = library.oneliner_vmcu_conv2d_s8
        self.pointer_types = [
            P_I8,
            P_I8,
            P_I32,
            P_I32,
            P_I8,
            P_I32,
            P_I8,
            P_I8,
        ]
        self.function.argtypes = [item for pointer in self.pointer_types for item in (pointer, SIZE)]
        self.function.restype = None

    def run(self, function_name, values_and_types, config, output_count, scratch_bytes):
        function = getattr(self.library, function_name)
        function.argtypes = [item for pointer in self.pointer_types for item in (pointer, SIZE)]
        function.restype = None
        arrays = []
        for values, ctype in values_and_types:
            array, offset = padded_array(ctype, values, 1 + len(arrays) % 3)
            arrays.append((array, offset))
        config_array, config_offset = padded_array(I32, config, 2)
        arrays.append((config_array, config_offset))
        output_array, output_offset = padded_array(I8, [77] * output_count, 3)
        scratch_array, scratch_offset = padded_array(I8, [66] * scratch_bytes, 4)
        arguments = []
        for array, offset in arrays:
            arguments.extend((array, offset))
        arguments.extend((output_array, output_offset, scratch_array, scratch_offset))
        function(*arguments)
        return list(output_array)[output_offset : output_offset + output_count]


class VmcuGenericKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.kernels = []
        library = Path(cls.temporary_directory.name) / "generic.so"
        command = [
            os.environ.get("CC", "cc"),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wconversion",
            "-Wsign-conversion",
            "-Werror",
            "-pedantic",
            "-fPIC",
            "-shared",
            str(GENERIC_SOURCE),
            "-o",
            str(library),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        cls.kernels.append(GenericKernel(ctypes.CDLL(str(library))))
        sanitized = Path(cls.temporary_directory.name) / "generic_ubsan.so"
        sanitize_command = command[:-2] + [
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-o",
            str(sanitized),
        ]
        try:
            subprocess.run(sanitize_command, check=True, capture_output=True, text=True)
            os.environ.setdefault("UBSAN_OPTIONS", "halt_on_error=1")
            cls.kernels.append(GenericKernel(ctypes.CDLL(str(sanitized))))
        except (subprocess.CalledProcessError, OSError):
            pass

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_conv2d_matches_oracle(self):
        block = make_conv2d_block(0xC0B0, ih=6, iw=5, cin=3, cout=4, kernel=3, stride=1, pads=(1, 1, 1, 1))
        config, input_values, weight, bias, multiplier, shift, izp, ozp = block
        expected = conv2d_oracle(input_values, weight, bias, multiplier, shift, izp, ozp, config)
        output_count = config[2] * config[6] * config[7] * config[8]
        for kernel in self.kernels:
            
                actual = kernel.run(
                    "oneliner_vmcu_conv2d_s8",
                    [(input_values, I8), (weight, I8), (bias, I32), (multiplier, I32), (shift, I8)],
                    config,
                    output_count,
                    config[35],
                )
                self.assertEqual(actual, expected)

    def test_conv2d_stride_two_dilated_padding(self):
        block = make_conv2d_block(0xD2A1, ih=7, iw=6, cin=2, cout=3, kernel=3, stride=2, pads=(2, 2, 2, 2), dilation=2)
        config, input_values, weight, bias, multiplier, shift, izp, ozp = block
        expected = conv2d_oracle(input_values, weight, bias, multiplier, shift, izp, ozp, config)
        output_count = config[2] * config[6] * config[7] * config[8]
        for kernel in self.kernels:
            
                actual = kernel.run(
                    "oneliner_vmcu_conv2d_s8",
                    [(input_values, I8), (weight, I8), (bias, I32), (multiplier, I32), (shift, I8)],
                    config,
                    output_count,
                    config[35],
                )
                self.assertEqual(actual, expected)

    def test_conv2d_without_padding(self):
        block = make_conv2d_block(0x0000, ih=8, iw=8, cin=2, cout=3, kernel=3, stride=1, pads=(0, 0, 0, 0))
        config, input_values, weight, bias, multiplier, shift, izp, ozp = block
        expected = conv2d_oracle(input_values, weight, bias, multiplier, shift, izp, ozp, config)
        output_count = config[2] * config[6] * config[7] * config[8]
        for kernel in self.kernels:
            actual = kernel.run(
                "oneliner_vmcu_conv2d_s8",
                [(input_values, I8), (weight, I8), (bias, I32), (multiplier, I32), (shift, I8)],
                config,
                output_count,
                config[35],
            )
            self.assertEqual(actual, expected)

    def test_fc_matches_oracle(self):
        block = make_fc_block(0xF0C0, rows=5, k=4, n_out=3)
        config, input_values, weight, bias, multiplier, shift, izp, ozp = block
        expected = fc_oracle(input_values, weight, bias, multiplier, shift, izp, ozp, config)
        output_count = config[2] * config[4]
        for kernel in self.kernels:
            
                actual = kernel.run(
                    "oneliner_vmcu_fc_s8",
                    [(input_values, I8), (weight, I8), (bias, I32), (multiplier, I32), (shift, I8)],
                    config,
                    output_count,
                    config[35],
                )
                self.assertEqual(actual, expected)

    def test_pair_matches_reference(self):
        rng = random.Random(0xABCD)
        rows, cin, mid, cout = 4, 3, 2, 5
        input_values = [rng.randint(-128, 127) for _ in range(rows * cin)]
        w0 = [rng.randint(-128, 127) for _ in range(cin * mid)]
        w1 = [rng.randint(-128, 127) for _ in range(mid * cout)]
        config = [rows, cin, mid, cout]
        expected = []
        for row in range(rows):
            segment = [max(-128, min(127, sum(input_values[row * cin + i] * w0[i * mid + m] for i in range(cin)))) for m in range(mid)]
            expected.extend(
                max(-128, min(127, sum(segment[m] * w1[m * cout + o] for m in range(mid))))
                for o in range(cout)
            )
        pair_types = [P_I8, P_I8, P_I8, P_I32, P_I8, P_I8]
        for kernel in self.kernels:
            
                function = kernel.library.oneliner_vmcu_pointwise_pair_s8
                function.argtypes = [item for pointer in pair_types for item in (pointer, SIZE)]
                function.restype = None
                arrays = []
                for values, ctype in [(input_values, I8), (w0, I8), (w1, I8)]:
                    array, offset = padded_array(ctype, values, 1 + len(arrays) % 3)
                    arrays.append((array, offset))
                config_array, config_offset = padded_array(I32, config, 2)
                arrays.append((config_array, config_offset))
                output_array, output_offset = padded_array(I8, [77] * (rows * cout), 3)
                segment_array, segment_offset = padded_array(I8, [66] * mid, 4)
                arguments = []
                for array, offset in arrays:
                    arguments.extend((array, offset))
                arguments.extend((output_array, output_offset, segment_array, segment_offset))
                function(*arguments)
                actual = list(output_array)[output_offset : output_offset + rows * cout]
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
