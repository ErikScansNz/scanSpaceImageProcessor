#pragma once

#include "pra32-u2-constants.h"
#include <cstddef>

using boolean = bool;

#define INLINE inline __attribute__((always_inline))

static INLINE uint8_t low_byte(uint16_t x) {
  return static_cast<uint8_t>(x & 0x00FFu);
}

static INLINE uint8_t high_byte(uint16_t x) {
  return static_cast<uint8_t>((x >> 8) & 0x00FFu);
}

static INLINE int32_t multiply_shift_right(int32_t x, int32_t y, uint8_t z) {
  return static_cast<int32_t>((static_cast<int64_t>(x) * y) >> z);
}

static INLINE int32_t minimum(int32_t value_0, int32_t value_1) {
  volatile int32_t result =
      (value_0 <  value_1) * value_0
    + (value_0 >= value_1) * value_1;
  return result;
}

static INLINE int32_t maximum(int32_t value_0, int32_t value_1) {
  volatile int32_t result =
      (value_0 >  value_1) * value_0
    + (value_0 <= value_1) * value_1;
  return result;
}

static INLINE int32_t clamp(int32_t value, int32_t minimum_value, int32_t maximum_value) {
  volatile int32_t result =
       (value >  maximum_value)                              * maximum_value
    +                              (value <  minimum_value)  * minimum_value
    + ((value <= maximum_value) && (value >= minimum_value)) * value;
  return result;
}
