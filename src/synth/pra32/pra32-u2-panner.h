#pragma once

#include <cmath>
#include "pra32-u2-common.h"

#ifndef PI
#define PI (3.1415926535897932384626433832795)
#endif

class PRA32_U2_Panner {
  static const uint8_t OSC_PAN_TABLE_LENGTH = 129;

  int32_t m_pan_table[OSC_PAN_TABLE_LENGTH];
  int16_t m_pan_control;
  int16_t m_pan_control_effective;
  int32_t m_gain_linear_l;
  int32_t m_gain_linear_r;

public:
PRA32_U2_Panner()
  : m_pan_table()
  , m_pan_control(64)
  , m_pan_control_effective(64)
  , m_gain_linear_l(16384 << 2)
  , m_gain_linear_r(16384 << 2)
  {
    for (uint8_t i = 1; i < OSC_PAN_TABLE_LENGTH - 1; ++i) {
      m_pan_table[i] = static_cast<int16_t>(std::sqrt(2.0) * std::sin((PI * (i - 1)) / (2 * (OSC_PAN_TABLE_LENGTH - 1))) * (1 << 14)) << 2;
    }

    m_pan_table[0]                        = m_pan_table[1];
    m_pan_table[OSC_PAN_TABLE_LENGTH - 1] = m_pan_table[OSC_PAN_TABLE_LENGTH - 2];
  }

  INLINE void set_pan(uint8_t controller_value) {
    m_pan_control = controller_value;
  }

  INLINE void process_at_low_rate() {
    update_gain_effective();
  }

  INLINE int32_t process(int32_t audio_input_int24, int32_t& audio_output_r_int24) {
    int32_t audio_output_l_int24 = 0;

    audio_output_l_int24 = multiply_shift_right(audio_input_int24, m_gain_linear_l, 16);
    audio_output_r_int24 = multiply_shift_right(audio_input_int24, m_gain_linear_r, 16);

    return audio_output_l_int24;
  }

private:
  INLINE void update_gain_effective() {
    m_pan_control_effective += (m_pan_control_effective < m_pan_control);
    m_pan_control_effective -= (m_pan_control_effective > m_pan_control);

    m_gain_linear_l = m_pan_table[128 - m_pan_control_effective];
    m_gain_linear_r = m_pan_table[m_pan_control_effective];
  }
};
