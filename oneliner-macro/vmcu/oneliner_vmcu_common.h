#ifndef ONELINER_VMCU_COMMON_H
#define ONELINER_VMCU_COMMON_H
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

_Static_assert(CHAR_BIT == 8, "requires 8-bit bytes");
_Static_assert(INT8_MIN == -128 && INT8_MAX == 127, "requires exact int8_t");
_Static_assert(INT32_MAX == 2147483647, "requires exact int32_t");
_Static_assert(UINT32_MAX == 4294967295U, "requires exact uint32_t");
_Static_assert(INT64_MAX == INT64_C(9223372036854775807),
               "requires exact int64_t");

#define CONST_TENSOR(type, base, element_offset)                               \
  ((const type *)(base) + (element_offset))
#define TENSOR(type, base, element_offset)                                     \
  ((type *)(base) + (element_offset))

enum {
  VMCU_FLAG_RESIDUAL = 1,
  VMCU_FLAG_IN_PLACE = 2,
  VMCU_CONFIG_MAGIC = 0x564d4355
};

static inline int checked_add_size(size_t a, size_t b, size_t *result) {
  if (a > SIZE_MAX - b) {
    return 0;
  }
  *result = a + b;
  return 1;
}

static inline int checked_mul_size(size_t a, size_t b, size_t *result) {
  if (a != 0U && b > SIZE_MAX / a) {
    return 0;
  }
  *result = a * b;
  return 1;
}

static inline int32_t i32_from_bits(uint32_t bits) {
  if (bits <= (uint32_t)INT32_MAX) {
    return (int32_t)bits;
  }
  return -1 - (int32_t)(UINT32_MAX - bits);
}

static inline int32_t add_wrap_i32(int32_t a, int32_t b) {
  return i32_from_bits((uint32_t)a + (uint32_t)b);
}

static inline int64_t floor_div_pow2_i64(int64_t value, uint8_t shift) {
  const int64_t divisor = INT64_C(1) << shift;
  if (value >= 0) {
    return value / divisor;
  }
  return -1 - ((-1 - value) / divisor);
}

static inline int32_t scale_single(int32_t value, int32_t multiplier, uint8_t shift) {
  const int64_t product = (int64_t)value * (int64_t)multiplier;
  const int64_t rounding = INT64_C(1) << (shift - 1U);
  return i32_from_bits(
      (uint32_t)floor_div_pow2_i64(product + rounding, shift));
}

static inline int32_t scale_double(int32_t value, int32_t multiplier, uint8_t shift) {
  const int64_t product = (int64_t)value * (int64_t)multiplier;
  int64_t rounding = INT64_C(1) << (shift - 1U);
  if (shift > 31U) {
    rounding += product >= 0 ? INT64_C(1073741824) : -INT64_C(1073741824);
  }
  return i32_from_bits(
      (uint32_t)floor_div_pow2_i64(product + rounding, shift));
}

static inline int8_t requantize_i8(int32_t value, int32_t multiplier, uint8_t shift,
                            int32_t zero_point) {
  const int32_t shifted =
      add_wrap_i32(scale_double(value, multiplier, shift), zero_point);
  if (shifted < INT8_MIN) {
    return INT8_MIN;
  }
  if (shifted > INT8_MAX) {
    return INT8_MAX;
  }
  return (int8_t)shifted;
}

static inline uint32_t accumulate_product(uint32_t accumulator, int8_t activation,
                                   int8_t weight,
                                   int32_t activation_zero_point) {
  const int32_t activation_delta =
      (int32_t)activation - activation_zero_point;
  return accumulator + (uint32_t)(activation_delta * (int32_t)weight);
}

static inline int valid_shift(int8_t shift) {
  return shift >= 1 && shift <= 62;
}

static inline int valid_zero_point(int32_t zero_point) {
  return zero_point >= INT8_MIN && zero_point <= INT8_MAX;
}

static inline void copy_bytes(int8_t *destination, const int8_t *source,
                       size_t count) {
  if ((((uintptr_t)source | (uintptr_t)destination) & 3U) == 0U) {
    const uint32_t *source32 = (const uint32_t *)(const void *)source;
    uint32_t *destination32 = (uint32_t *)(void *)destination;
    size_t words = count >> 2;
    size_t i;
    for (i = 0U; i < words; ++i) {
      destination32[i] = source32[i];
    }
    count &= 3U;
    source += words << 2;
    destination += words << 2;
  }
  while (count != 0U) {
    *destination = *source;
    ++destination;
    ++source;
    --count;
  }
}

/* Argument-list patterns for the single-configuration kernels (conv2d, fc):
 * input, weight, bias, multiplier, shift, config, output, scratch. */
#define VMCU_SINGLE_ARGUMENTS                                                  \
  const int8_t *input_base, size_t input_offset,                               \
      const int8_t *weight_base, size_t weight_offset,                         \
      const int32_t *bias_base, size_t bias_offset,                            \
      const int32_t *multiplier_base, size_t multiplier_offset,                \
      const int8_t *shift_base, size_t shift_offset,                           \
      const int32_t *config_base, size_t config_offset,                        \
      int8_t *output_base, size_t output_offset, int8_t *scratch_base,         \
      size_t scratch_offset

#define VMCU_SINGLE_CALL                                                       \
  CONST_TENSOR(int8_t, input_base, input_offset),                              \
      CONST_TENSOR(int8_t, weight_base, weight_offset),                        \
      CONST_TENSOR(int32_t, bias_base, bias_offset),                           \
      CONST_TENSOR(int32_t, multiplier_base, multiplier_offset),               \
      CONST_TENSOR(int8_t, shift_base, shift_offset),                          \
      CONST_TENSOR(int32_t, config_base, config_offset),                       \
      TENSOR(int8_t, output_base, output_offset),                              \
      TENSOR(int8_t, scratch_base, scratch_offset)

#define VMCU_SINGLE_BASES_VALID                                                \
  (input_base != NULL && weight_base != NULL && bias_base != NULL &&           \
   multiplier_base != NULL && shift_base != NULL && config_base != NULL &&     \
   output_base != NULL && scratch_base != NULL)

/* Shape-specialized variant of the single-configuration ABI: the config
 * tensor is dropped (shapes and zero points become compile-time macros) and
 * the exported function is named by VMCU_ENTRY_NAME. */
#define VMCU_SINGLE_ARGUMENTS_NO_CONFIG                                        \
  const int8_t *input_base, size_t input_offset,                               \
      const int8_t *weight_base, size_t weight_offset,                         \
      const int32_t *bias_base, size_t bias_offset,                            \
      const int32_t *multiplier_base, size_t multiplier_offset,                \
      const int8_t *shift_base, size_t shift_offset,                           \
      int8_t *output_base, size_t output_offset, int8_t *scratch_base,         \
      size_t scratch_offset

#define VMCU_SINGLE_CALL_NO_CONFIG                                             \
  CONST_TENSOR(int8_t, input_base, input_offset),                              \
      CONST_TENSOR(int8_t, weight_base, weight_offset),                        \
      CONST_TENSOR(int32_t, bias_base, bias_offset),                           \
      CONST_TENSOR(int32_t, multiplier_base, multiplier_offset),               \
      CONST_TENSOR(int8_t, shift_base, shift_offset),                          \
      TENSOR(int8_t, output_base, output_offset),                              \
      TENSOR(int8_t, scratch_base, scratch_offset)

#define VMCU_SINGLE_BASES_VALID_NO_CONFIG                                      \
  (input_base != NULL && weight_base != NULL && bias_base != NULL &&           \
   multiplier_base != NULL && shift_base != NULL &&                            \
   output_base != NULL && scratch_base != NULL)

#endif /* ONELINER_VMCU_COMMON_H */
