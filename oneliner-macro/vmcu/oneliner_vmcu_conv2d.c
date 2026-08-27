/* Single 2D convolution segment kernel (any kernel/pad/stride/dilation).
 *
 * The same source is compiled two ways:
 *  - generic: VMCU_SPECIALIZED undefined; every shape/zero point is read at
 *    runtime from the config tensor (compiled into oneliner_vmcu_generic.bc);
 *  - specialized: a generated wrapper defines VMCU_SPECIALIZED plus the
 *    VMCU_N/IH/IW/CIN/... macros; the kernel is exported under
 *    VMCU_ENTRY_NAME without the config argument (compile-time shapes).
 */
#include "oneliner_vmcu_common.h"

#ifndef VMCU_C2_ENTRY
#define VMCU_C2_ENTRY oneliner_vmcu_conv2d_s8
#endif

#define VMCU_C2_IDX_N 2
#define VMCU_C2_IDX_IH 3
#define VMCU_C2_IDX_IW 4
#define VMCU_C2_IDX_CIN 5
#define VMCU_C2_IDX_OH 6
#define VMCU_C2_IDX_OW 7
#define VMCU_C2_IDX_COUT 8
#define VMCU_C2_IDX_KH 9
#define VMCU_C2_IDX_KW 10
#define VMCU_C2_IDX_SH 11
#define VMCU_C2_IDX_SW 12
#define VMCU_C2_IDX_DH 13
#define VMCU_C2_IDX_DW 14
#define VMCU_C2_IDX_PT 15
#define VMCU_C2_IDX_PL 16
#define VMCU_C2_IDX_INPUT_ZP 19
#define VMCU_C2_IDX_OUTPUT_ZP 20

#ifdef VMCU_SPECIALIZED
#define VMCU_C2_DIM(field) ((size_t)VMCU_##field)
#define VMCU_C2_ZP(field) VMCU_##field
#else
#define VMCU_C2_DIM(field) ((size_t)config[VMCU_C2_IDX_##field])
#define VMCU_C2_ZP(field) config[VMCU_C2_IDX_##field]
#endif

static int oneliner_vmcu_conv2d_kernel(const int8_t *input, const int8_t *weight,
                         const int32_t *bias, const int32_t *multiplier,
                         const int8_t *shift,
#ifdef VMCU_SPECIALIZED
                         int8_t *output, int8_t *scratch) {
#else
                         const int32_t *config, int8_t *output,
                         int8_t *scratch) {
#endif
  size_t n;
  size_t input_count;
  size_t output_count;

  size_t batches;
  size_t input_height;
  size_t input_width;
  size_t input_channels;
  size_t output_height;
  size_t output_width;
  size_t output_channels;
  size_t kernel_height;
  size_t kernel_width;
  size_t stride_height;
  size_t stride_width;
  size_t dilation_height;
  size_t dilation_width;
  size_t pad_top;
  size_t pad_left;
  int32_t input_zero_point;
  int32_t output_zero_point;

  if (input == NULL || weight == NULL || bias == NULL || multiplier == NULL ||
      shift == NULL ||
#ifndef VMCU_SPECIALIZED
      config == NULL ||
#endif
      output == NULL || scratch == NULL) {
    return 0;
  }
  if (((uintptr_t)bias % _Alignof(int32_t)) != 0U ||
      ((uintptr_t)multiplier % _Alignof(int32_t)) != 0U) {
    return 0;
  }
  batches = VMCU_C2_DIM(N);
  input_height = VMCU_C2_DIM(IH);
  input_width = VMCU_C2_DIM(IW);
  input_channels = VMCU_C2_DIM(CIN);
  output_height = VMCU_C2_DIM(OH);
  output_width = VMCU_C2_DIM(OW);
  output_channels = VMCU_C2_DIM(COUT);
  kernel_height = VMCU_C2_DIM(KH);
  kernel_width = VMCU_C2_DIM(KW);
  stride_height = VMCU_C2_DIM(SH);
  stride_width = VMCU_C2_DIM(SW);
  dilation_height = VMCU_C2_DIM(DH);
  dilation_width = VMCU_C2_DIM(DW);
  pad_top = VMCU_C2_DIM(PT);
  pad_left = VMCU_C2_DIM(PL);
  input_zero_point = VMCU_C2_ZP(INPUT_ZP);
  output_zero_point = VMCU_C2_ZP(OUTPUT_ZP);

  input_count = batches * input_height * input_width * input_channels;
  output_count = batches * output_height * output_width * output_channels;

  for (n = 0U; n < batches; ++n) {
    int8_t *const cache = scratch;
    size_t ring_start = 0U;
    size_t oy;
    const int8_t *batch_input = input + n * input_count / batches;
    int8_t *batch_output = output + n * output_count / batches;

    for (oy = 0U; oy < output_height; ++oy) {
      size_t first_new_ky = 0U;
      size_t last_new_ky = kernel_height;
      size_t ky;
      size_t ox;
      size_t ring_index;

      if (oy != 0U && stride_height % dilation_height == 0U) {
        const size_t rotation = stride_height / dilation_height;
        if (rotation < kernel_height) {
          ring_start = (ring_start + rotation) % kernel_height;
          first_new_ky = kernel_height - rotation;
        } else {
          ring_start = 0U;
        }
      } else if (oy != 0U) {
        ring_start = 0U;
      }

      for (ky = first_new_ky; ky < last_new_ky; ++ky) {
        const size_t padded_y = oy * stride_height + ky * dilation_height;
        size_t iy;
        size_t ring_index = ring_start + ky;
        if (padded_y < pad_top ||
            (iy = padded_y - pad_top) >= input_height) {
          continue;
        }
        if (ring_index >= kernel_height) {
          ring_index -= kernel_height;
        }
        copy_bytes(cache + ring_index * input_width * input_channels,
                   batch_input + iy * input_width * input_channels,
                   input_width * input_channels);
      }

      for (ox = 0U; ox < output_width; ++ox) {
        size_t oc;
        for (oc = 0U; oc < output_channels; ++oc) {
          uint32_t accumulator = (uint32_t)bias[oc];
          for (ky = 0U; ky < kernel_height; ++ky) {
            const size_t padded_y = oy * stride_height + ky * dilation_height;
            size_t iy;
            size_t kx;
            const int8_t *cache_row;
            if (padded_y < pad_top ||
                (iy = padded_y - pad_top) >= input_height) {
              continue;
            }
            (void)iy;
            ring_index = ring_start + ky;
            if (ring_index >= kernel_height) {
              ring_index -= kernel_height;
            }
            cache_row = cache + ring_index * input_width * input_channels;
            for (kx = 0U; kx < kernel_width; ++kx) {
              const size_t padded_x = ox * stride_width + kx * dilation_width;
              size_t ix;
              size_t ic;
              if (padded_x < pad_left ||
                  (ix = padded_x - pad_left) >= input_width) {
                continue;
              }
              for (ic = 0U; ic < input_channels; ++ic) {
                accumulator = accumulate_product(
                    accumulator, cache_row[ix * input_channels + ic],
                    weight[((ky * kernel_width + kx) * input_channels + ic) *
                               output_channels +
                           oc],
                    input_zero_point);
              }
            }
          }
          batch_output[(oy * output_width + ox) * output_channels + oc] =
              requantize_i8(i32_from_bits(accumulator), multiplier[oc],
                            (uint8_t)shift[oc], output_zero_point);
        }
      }
    }
  }
  return 1;
}

#ifdef VMCU_SPECIALIZED
void VMCU_C2_ENTRY(VMCU_SINGLE_ARGUMENTS_NO_CONFIG) {
  if (!VMCU_SINGLE_BASES_VALID_NO_CONFIG) {
    return;
  }
  (void)oneliner_vmcu_conv2d_kernel(VMCU_SINGLE_CALL_NO_CONFIG);
}
#else
void VMCU_C2_ENTRY(VMCU_SINGLE_ARGUMENTS) {
  if (!VMCU_SINGLE_BASES_VALID) {
    return;
  }
  (void)oneliner_vmcu_conv2d_kernel(VMCU_SINGLE_CALL);
}
#endif