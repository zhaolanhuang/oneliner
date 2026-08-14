#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"

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

void oneliner_cmsis_nn_conv_s8(const int8_t *input_base, size_t input_offset,
                               const int8_t *filter_base, size_t filter_offset,
                               const int32_t *bias_base, size_t bias_offset,
                               const int32_t *multiplier_base,
                               size_t multiplier_offset,
                               const int32_t *shift_base, size_t shift_offset,
                               int8_t *scratch_base, size_t scratch_offset,
                               const int32_t *config_base, size_t config_offset,
                               int8_t *output_base, size_t output_offset) {
  const int32_t *config = config_base + config_offset;
  cmsis_nn_context context = {
      .buf = scratch_base + scratch_offset,
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

  (void)arm_convolve_wrapper_s8(
      &context, &params, &quant, &input_dims, input_base + input_offset,
      &filter_dims, filter_base + filter_offset, &bias_dims,
      bias_base + bias_offset, &output_dims, output_base + output_offset);
}
