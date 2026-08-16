#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#define CONST_TENSOR(type, base, byte_offset)                                  \
  ((const type *)((const uint8_t *)(base) + (byte_offset)))
#define TENSOR(type, base, byte_offset)                                        \
  ((type *)((uint8_t *)(base) + (byte_offset)))

__attribute__((noinline, optnone)) void
oneliner_aeabi_memcpy(void *dest, const void *src,
                      size_t size) __asm__("__aeabi_memcpy");
void oneliner_aeabi_memcpy(void *dest, const void *src, size_t size) {
  volatile uint8_t *dest_bytes = dest;
  const volatile uint8_t *src_bytes = src;
  while (size-- != 0) {
    *dest_bytes++ = *src_bytes++;
  }
}

__attribute__((noinline, optnone)) void
oneliner_aeabi_memset(void *dest, size_t size,
                      int value) __asm__("__aeabi_memset");
void oneliner_aeabi_memset(void *dest, size_t size, int value) {
  volatile uint8_t *dest_bytes = dest;
  while (size-- != 0) {
    *dest_bytes++ = (uint8_t)value;
  }
}

// The ukernel bitcode is freestanding: CMSIS-NN's memcpy/memset calls are
// inlined by clang (see build_bitcode.py), but keep plain symbols around so
// the bitcode never depends on a C library.
void *memset(void *dest, int value, size_t size) {
  oneliner_aeabi_memset(dest, size, value);
  return dest;
}
void *memcpy(void *dest, const void *src, size_t size) {
  oneliner_aeabi_memcpy(dest, src, size);
  return dest;
}

// CMSIS-NN failures must not go unnoticed: zero-fill the output tensor so the
// result is deterministic (and fails any validation) instead of leaving
// garbage in the output buffer.
static void oneliner_fail_fill(int8_t *output, int32_t elements) {
  for (int32_t i = 0; i < elements; i++) {
    output[i] = 0;
  }
}

// Config: N, IH, IW, IC, OH, OW, OC, KH, KW, SH, SW, DH, DW, PH, PW,
// INPUT_OFFSET, OUTPUT_OFFSET, ACT_MIN, ACT_MAX, SCRATCH_SIZE.
void oneliner_cmsis_nn_conv_s8(const int8_t *input_base, size_t input_offset,
                               const int8_t *filter_base, size_t filter_offset,
                               const int32_t *bias_base, size_t bias_offset,
                               const int32_t *multiplier_base,
                               size_t multiplier_offset,
                               const int32_t *shift_base, size_t shift_offset,
                               int8_t *scratch_base, size_t scratch_offset,
                               const int32_t *config_base, size_t config_offset,
                               int8_t *output_base, size_t output_offset) {
  const int8_t *input = CONST_TENSOR(int8_t, input_base, input_offset);
  const int8_t *filter = CONST_TENSOR(int8_t, filter_base, filter_offset);
  const int32_t *bias = bias_base + bias_offset;
  int8_t *scratch = TENSOR(int8_t, scratch_base, scratch_offset);
  const int32_t *config = config_base + config_offset;
  int8_t *output = TENSOR(int8_t, output_base, output_offset);
  cmsis_nn_context context = {
      .buf = scratch,
      .size = config[19],
  };
  cmsis_nn_conv_params params = {
      .input_offset = config[15],
      .output_offset = config[16],
      .stride = {.w = config[10], .h = config[9]},
      .padding = {.w = config[14], .h = config[13]},
      .dilation = {.w = config[12], .h = config[11]},
      .activation = {.min = config[17], .max = config[18]},
  };
  cmsis_nn_per_channel_quant_params quant = {
      .multiplier = (int32_t *)multiplier_base + multiplier_offset,
      .shift = (int32_t *)shift_base + shift_offset,
  };
  cmsis_nn_dims input_dims = {
      .n = config[0], .h = config[1], .w = config[2], .c = config[3]};
  cmsis_nn_dims filter_dims = {
      .n = config[6], .h = config[7], .w = config[8], .c = config[3]};
  cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = config[6]};
  cmsis_nn_dims output_dims = {
      .n = config[0], .h = config[4], .w = config[5], .c = config[6]};

  arm_cmsis_nn_status status =
      arm_convolve_wrapper_s8(&context, &params, &quant, &input_dims, input,
                              &filter_dims, filter, &bias_dims, bias,
                              &output_dims, output);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    oneliner_fail_fill(output, config[0] * config[4] * config[5] * config[6]);
  }
}

// Config: N, IH, IW, C, OH, OW, KH, KW, SH, SW, PH, PW, ACT_MIN,
// ACT_MAX.
void oneliner_cmsis_nn_max_pool_s8(const int8_t *input_base,
                                   size_t input_offset,
                                   const int32_t *config_base,
                                   size_t config_offset, int8_t *output_base,
                                   size_t output_offset) {
  const int8_t *input = CONST_TENSOR(int8_t, input_base, input_offset);
  const int32_t *config = CONST_TENSOR(int32_t, config_base, config_offset);
  int8_t *output = TENSOR(int8_t, output_base, output_offset);
  cmsis_nn_context context = {0};
  cmsis_nn_pool_params params = {
      .stride = {.w = config[9], .h = config[8]},
      .padding = {.w = config[11], .h = config[10]},
      .activation = {.min = config[12], .max = config[13]},
  };
  cmsis_nn_dims input_dims = {
      .n = config[0], .h = config[1], .w = config[2], .c = config[3]};
  cmsis_nn_dims filter_dims = {.n = 1, .h = config[6], .w = config[7], .c = 1};
  cmsis_nn_dims output_dims = {
      .n = config[0], .h = config[4], .w = config[5], .c = config[3]};

  arm_cmsis_nn_status status =
      arm_max_pool_s8(&context, &params, &input_dims, input, &filter_dims,
                      &output_dims, output);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    oneliner_fail_fill(output, config[0] * config[4] * config[5] * config[3]);
  }
}

// Config: N, IH, IW, C, OH, OW, KH, KW, SH, SW, PH, PW, ACT_MIN, ACT_MAX,
// MULTIPLIER.
//
// IREE lowers a quantized average pool to linalg.pooling_nhwc_sum followed by
// a requant generic computing round(sum * MULTIPLIER / 2^32). This shim
// reproduces that exact arithmetic (instead of arm_avgpool_s8, whose
// `(sum +/- count/2) / count` rounding can differ by one from IREE for some
// window sizes), so results stay bit-identical to the standard codegen path.
void oneliner_cmsis_nn_avg_pool_s8(const int8_t *input_base,
                                   size_t input_offset,
                                   const int32_t *config_base,
                                   size_t config_offset, int8_t *output_base,
                                   size_t output_offset) {
  const int8_t *input = CONST_TENSOR(int8_t, input_base, input_offset);
  const int32_t *config = CONST_TENSOR(int32_t, config_base, config_offset);
  int8_t *output = TENSOR(int8_t, output_base, output_offset);
  const int32_t batch = config[0];
  const int32_t input_h = config[1];
  const int32_t input_w = config[2];
  const int32_t channels = config[3];
  const int32_t output_h = config[4];
  const int32_t output_w = config[5];
  const int32_t kernel_h = config[6];
  const int32_t kernel_w = config[7];
  const int32_t stride_h = config[8];
  const int32_t stride_w = config[9];
  const int32_t pad_h = config[10];
  const int32_t pad_w = config[11];
  const int32_t act_min = config[12];
  const int32_t act_max = config[13];
  const int64_t multiplier = config[14];

  for (int32_t b = 0; b < batch; b++) {
    for (int32_t oy = 0; oy < output_h; oy++) {
      for (int32_t ox = 0; ox < output_w; ox++) {
        for (int32_t c = 0; c < channels; c++) {
          int32_t sum = 0;
          for (int32_t ky = 0; ky < kernel_h; ky++) {
            const int32_t iy = oy * stride_h - pad_h + ky;
            if (iy < 0 || iy >= input_h) {
              continue;
            }
            for (int32_t kx = 0; kx < kernel_w; kx++) {
              const int32_t ix = ox * stride_w - pad_w + kx;
              if (ix < 0 || ix >= input_w) {
                continue;
              }
              sum += input[(iy * input_w + ix) * channels + c];
            }
          }
          const int64_t scaled = (int64_t)sum * multiplier + ((int64_t)1 << 31);
          int32_t value = (int32_t)((uint64_t)scaled >> 32);
          value = value < act_min ? act_min : value;
          value = value > act_max ? act_max : value;
          output[(oy * output_w + ox) * channels + c] = (int8_t)value;
        }
      }
    }
    input += input_h * input_w * channels;
    output += output_h * output_w * channels;
  }
}

// Config: N, ACCUM_DEPTH, OUTPUT_DEPTH, INPUT_OFFSET, FILTER_OFFSET,
// OUTPUT_OFFSET, MULTIPLIER, SHIFT, ACT_MIN, ACT_MAX, SCRATCH_SIZE.
void oneliner_cmsis_nn_fully_connected_s8(
    const int8_t *input_base, size_t input_offset, const int8_t *filter_base,
    size_t filter_offset, const int32_t *bias_base, size_t bias_offset,
    int8_t *scratch_base, size_t scratch_offset, const int32_t *config_base,
    size_t config_offset, int8_t *output_base, size_t output_offset) {
  const int8_t *input = CONST_TENSOR(int8_t, input_base, input_offset);
  const int8_t *filter = CONST_TENSOR(int8_t, filter_base, filter_offset);
  const int32_t *bias = bias_base + bias_offset;
  int8_t *scratch = TENSOR(int8_t, scratch_base, scratch_offset);
  (void)scratch;
  const int32_t *config = CONST_TENSOR(int32_t, config_base, config_offset);
  int8_t *output = TENSOR(int8_t, output_base, output_offset);
  // A fully connected layer is a 1x1 convolution: use the fast matmul kernel
  // instead of arm_fully_connected_s8. arm_nn_vec_mat_mult_t_s8 is several
  // times slower than the standard codegen for these sizes on Cortex-M4 (its
  // DSP path is compiled to scalar byte mul-adds by clang), while
  // arm_nn_mat_mult_nt_t_s8 lowers to the SIMD kernels and applies the input
  // offset itself.
  const int32_t input_offset_val = config[3];
  const int32_t output_offset_val = config[5];
  const int32_t multiplier = config[6];
  const int32_t shift = config[7];
  const int32_t activation_min = config[8];
  const int32_t activation_max = config[9];
  // arm_nn_mat_mult_nt_t_s8 indexes the multiplier/shift arrays per output
  // row, so materialize the per-tensor quant params as per-channel arrays.
  int32_t multipliers[config[2]];
  int32_t shifts[config[2]];
  for (int32_t i = 0; i < config[2]; i++) {
    multipliers[i] = multiplier;
    shifts[i] = shift;
  }
  arm_cmsis_nn_status status = arm_nn_mat_mult_nt_t_s8(
      input, filter, bias, output, multipliers, shifts, 1, config[2],
      config[1], input_offset_val, output_offset_val, activation_min,
      activation_max, config[2], config[1]);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    oneliner_fail_fill(output, config[0] * config[2]);
  }
}

// Config: N, IH, IW, IC, OH, OW, OC, KH, KW, SH, SW, DH, DW, PH, PW,
// INPUT_OFFSET, OUTPUT_OFFSET, ACT_MIN, ACT_MAX, CHANNEL_MULTIPLIER,
// SCRATCH_SIZE.
void oneliner_cmsis_nn_depthwise_conv_s8(
    const int8_t *input_base, size_t input_offset, const int8_t *filter_base,
    size_t filter_offset, const int32_t *bias_base, size_t bias_offset,
    const int32_t *multiplier_base, size_t multiplier_offset,
    const int32_t *shift_base, size_t shift_offset, int8_t *scratch_base,
    size_t scratch_offset, const int32_t *config_base, size_t config_offset,
    int8_t *output_base, size_t output_offset) {
  const int8_t *input = CONST_TENSOR(int8_t, input_base, input_offset);
  const int8_t *filter = CONST_TENSOR(int8_t, filter_base, filter_offset);
  const int32_t *bias = bias_base + bias_offset;
  int32_t *multiplier = (int32_t *)multiplier_base + multiplier_offset;
  int32_t *shift = (int32_t *)shift_base + shift_offset;
  int8_t *scratch = TENSOR(int8_t, scratch_base, scratch_offset);
  const int32_t *config = CONST_TENSOR(int32_t, config_base, config_offset);
  int8_t *output = TENSOR(int8_t, output_base, output_offset);
  cmsis_nn_context context = {.buf = scratch, .size = config[20]};
  cmsis_nn_dw_conv_params params = {
      .input_offset = config[15],
      .output_offset = config[16],
      .ch_mult = config[19],
      .stride = {.w = config[10], .h = config[9]},
      .padding = {.w = config[14], .h = config[13]},
      .dilation = {.w = config[12], .h = config[11]},
      .activation = {.min = config[17], .max = config[18]},
  };
  cmsis_nn_per_channel_quant_params quant = {
      .multiplier = multiplier,
      .shift = shift,
  };
  cmsis_nn_dims input_dims = {
      .n = config[0], .h = config[1], .w = config[2], .c = config[3]};
  cmsis_nn_dims filter_dims = {
      .n = 1, .h = config[7], .w = config[8], .c = config[6]};
  cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = config[6]};
  cmsis_nn_dims output_dims = {
      .n = config[0], .h = config[4], .w = config[5], .c = config[6]};

  arm_cmsis_nn_status status = arm_depthwise_conv_wrapper_s8(
      &context, &params, &quant, &input_dims, input, &filter_dims, filter,
      &bias_dims, bias, &output_dims, output);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    oneliner_fail_fill(output, config[0] * config[4] * config[5] * config[6]);
  }
}
