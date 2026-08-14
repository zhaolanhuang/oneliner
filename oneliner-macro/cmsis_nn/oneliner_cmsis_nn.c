#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"

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

  (void)arm_convolve_wrapper_s8(&context, &params, &quant, &input_dims, input,
                                &filter_dims, filter, &bias_dims, bias,
                                &output_dims, output);
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

  (void)arm_max_pool_s8(&context, &params, &input_dims, input, &filter_dims,
                        &output_dims, output);
}

// Config: N, ACCUM_DEPTH, OUTPUT_DEPTH, INPUT_OFFSET, FILTER_OFFSET,
// OUTPUT_OFFSET, MULTIPLIER, SHIFT, ACT_MIN, ACT_MAX, BIAS_BYTE_OFFSET.
void oneliner_cmsis_nn_fully_connected_s8(
    const int8_t *input_base, size_t input_offset, const int8_t *params_base,
    size_t params_offset, int8_t *scratch_base, size_t scratch_offset,
    const int32_t *config_base, size_t config_offset, int8_t *output_base,
    size_t output_offset) {
  const int8_t *input = CONST_TENSOR(int8_t, input_base, input_offset);
  const int8_t *params_data = CONST_TENSOR(int8_t, params_base, params_offset);
  int8_t *scratch = TENSOR(int8_t, scratch_base, scratch_offset);
  const int32_t *config = CONST_TENSOR(int32_t, config_base, config_offset);
  const int8_t *filter = params_data;
  const int32_t *bias = CONST_TENSOR(int32_t, params_data, config[10]);
  int8_t *output = TENSOR(int8_t, output_base, output_offset);
  cmsis_nn_context context = {.buf = scratch, .size = 0};
  cmsis_nn_fc_params params = {
      .input_offset = config[3],
      .filter_offset = config[4],
      .output_offset = config[5],
      .activation = {.min = config[8], .max = config[9]},
  };
  cmsis_nn_per_tensor_quant_params quant = {
      .multiplier = config[6],
      .shift = config[7],
  };
  cmsis_nn_dims input_dims = {.n = config[0], .h = 1, .w = 1, .c = config[1]};
  cmsis_nn_dims filter_dims = {.n = config[1], .h = 1, .w = 1, .c = config[2]};
  cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = config[2]};
  cmsis_nn_dims output_dims = {.n = config[0], .h = 1, .w = 1, .c = config[2]};

  (void)arm_fully_connected_s8(&context, &params, &quant, &input_dims, input,
                               &filter_dims, filter, &bias_dims, bias,
                               &output_dims, output);
}

// Config: N, IH, IW, IC, OH, OW, OC, KH, KW, SH, SW, DH, DW, PH, PW,
// INPUT_OFFSET, OUTPUT_OFFSET, ACT_MIN, ACT_MAX, CHANNEL_MULTIPLIER,
// BIAS_BYTE_OFFSET, MULTIPLIER_BYTE_OFFSET, SHIFT_BYTE_OFFSET, SCRATCH_SIZE.
void oneliner_cmsis_nn_depthwise_conv_s8(
    const int8_t *input_base, size_t input_offset, const int8_t *params_base,
    size_t params_offset, int8_t *scratch_base, size_t scratch_offset,
    const int32_t *config_base, size_t config_offset, int8_t *output_base,
    size_t output_offset) {
  const int8_t *input = CONST_TENSOR(int8_t, input_base, input_offset);
  const int8_t *params_data = CONST_TENSOR(int8_t, params_base, params_offset);
  int8_t *scratch = TENSOR(int8_t, scratch_base, scratch_offset);
  const int32_t *config = CONST_TENSOR(int32_t, config_base, config_offset);
  const int8_t *filter = params_data;
  const int32_t *bias = CONST_TENSOR(int32_t, params_data, config[20]);
  int32_t *multiplier = TENSOR(int32_t, params_data, config[21]);
  int32_t *shift = TENSOR(int32_t, params_data, config[22]);
  int8_t *output = TENSOR(int8_t, output_base, output_offset);
  cmsis_nn_context context = {.buf = scratch, .size = config[23]};
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

  (void)arm_depthwise_conv_wrapper_s8(&context, &params, &quant, &input_dims,
                                      input, &filter_dims, filter, &bias_dims,
                                      bias, &output_dims, output);
}
