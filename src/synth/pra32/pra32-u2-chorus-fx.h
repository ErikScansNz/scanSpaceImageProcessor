#pragma once

#include "pra32-u2-common.h"
#include "pra32-u2-lfo-table.h"

class PRA32_U2_ChorusFx {
  static const uint16_t DELAY_BUFF_SIZE = 512;

  int32_t  m_delay_buff[2][DELAY_BUFF_SIZE];
  uint16_t m_delay_wp[2];

  uint16_t m_chorus_level_control;
  uint16_t m_chorus_level_control_effective;
  uint16_t m_chorus_depth_control;
  uint16_t m_chorus_depth_control_effective;
  uint32_t m_chorus_rate_control;
  uint16_t m_chorus_delay_time_control;
  uint16_t m_chorus_delay_time_control_effective;
  uint8_t  m_chorus_depth_control_actual;
  uint32_t m_chorus_lfo_phase;
  uint16_t m_chorus_delay_time[2];

  int32_t  m_prev_sample_to_push_0;
  int32_t  m_prev_sample_to_push_1;

public:
  PRA32_U2_ChorusFx()
  : m_delay_buff()
  , m_delay_wp()

  , m_chorus_level_control()
  , m_chorus_level_control_effective()
  , m_chorus_depth_control()
  , m_chorus_depth_control_effective()
  , m_chorus_rate_control()
  , m_chorus_delay_time_control()
  , m_chorus_delay_time_control_effective()
  , m_chorus_lfo_phase()
  , m_chorus_delay_time()

  , m_prev_sample_to_push_0()
  , m_prev_sample_to_push_1()
  {
    m_delay_wp[0] = DELAY_BUFF_SIZE - 1;
    m_delay_wp[1] = DELAY_BUFF_SIZE - 1;

    set_chorus_depth     (64 );
    set_chorus_rate      (64 );
    set_chorus_delay_time(64 );

    m_chorus_depth_control_effective = 64 << 6;
    m_chorus_delay_time_control_effective = 64 << 6;
  }

  INLINE void set_chorus_depth(uint8_t controller_value) {
    if (controller_value < 126) {
      m_chorus_depth_control = controller_value << 6;
    } else {
      m_chorus_depth_control = 126 << 6;
    }
  }

  INLINE void set_chorus_rate(uint8_t controller_value) {
    m_chorus_rate_control = g_chorus_rate_table[controller_value];
  }

  INLINE void set_chorus_delay_time(uint8_t controller_value) {
    m_chorus_delay_time_control = controller_value << 6;
  }

  INLINE void set_chorus_level(uint8_t controller_value) {
    m_chorus_level_control = (controller_value + 1) >> 1;
  }

  template <uint8_t N>
  INLINE uint16_t get_chorus_delay_time() {
    return m_chorus_delay_time[N];
  }

  INLINE void process_at_low_rate(uint8_t count) {
#if 1
    static_cast<void>(count);

    m_chorus_level_control_effective += (m_chorus_level_control_effective < m_chorus_level_control);
    m_chorus_level_control_effective -= (m_chorus_level_control_effective > m_chorus_level_control);

    m_chorus_depth_control_effective += (m_chorus_depth_control_effective < m_chorus_depth_control);
    m_chorus_depth_control_effective -= (m_chorus_depth_control_effective > m_chorus_depth_control);

    m_chorus_delay_time_control_effective += (m_chorus_delay_time_control_effective < m_chorus_delay_time_control);
    m_chorus_delay_time_control_effective -= (m_chorus_delay_time_control_effective > m_chorus_delay_time_control);

    uint16_t chorus_depth_control_effective_limited;
    if (m_chorus_delay_time_control_effective < (64 << 6)) {
      if (m_chorus_depth_control_effective > (m_chorus_delay_time_control_effective << 1)) {
        chorus_depth_control_effective_limited = (m_chorus_delay_time_control_effective << 1);
      } else {
        chorus_depth_control_effective_limited = m_chorus_depth_control_effective;
      }
    } else {
      if (m_chorus_depth_control_effective > (((127 << 6) - m_chorus_delay_time_control_effective) << 1)) {
        chorus_depth_control_effective_limited = (((127 << 6) - m_chorus_delay_time_control_effective) << 1);
      } else {
        chorus_depth_control_effective_limited = m_chorus_depth_control_effective;
      }
    }

    m_chorus_lfo_phase += m_chorus_rate_control;
    m_chorus_lfo_phase &= 0x00FFFFFF;

    int16_t chorus_lfo_wave_level = get_chorus_lfo_wave_level(m_chorus_lfo_phase);

    int16_t chorus_lfo_level = (chorus_lfo_wave_level * chorus_depth_control_effective_limited) >> 14;

    m_chorus_delay_time[0] = m_chorus_delay_time_control_effective - chorus_lfo_level;
    m_chorus_delay_time[1] = m_chorus_delay_time_control_effective + chorus_lfo_level;
#endif
  }

  INLINE int32_t process(int32_t left_input_int24, int32_t right_input_int24, int32_t& right_output_int24) {
    int32_t eff_sample_0 = delay_buff_get(0, get_chorus_delay_time<0>());
    int32_t eff_sample_1 = delay_buff_get(1, get_chorus_delay_time<1>());

    int32_t curr_sample_to_push_0 = (left_input_int24  * m_chorus_level_control_effective) >> 6;
    int32_t curr_sample_to_push_1 = (right_input_int24 * m_chorus_level_control_effective) >> 6;

#if 0
    // Do not apply LPF to the delay component
    m_prev_sample_to_push_0 = curr_sample_to_push_0;
    m_prev_sample_to_push_1 = curr_sample_to_push_1;
#endif

    delay_buff_push(0, (curr_sample_to_push_0 + m_prev_sample_to_push_0) >> 1);
    delay_buff_push(1, (curr_sample_to_push_1 + m_prev_sample_to_push_1) >> 1);

    m_prev_sample_to_push_0 = curr_sample_to_push_0;
    m_prev_sample_to_push_1 = curr_sample_to_push_1;

    right_output_int24 = right_input_int24 + eff_sample_1;
    return               left_input_int24  + eff_sample_0;
  }

private:
  INLINE void delay_buff_push(uint32_t lr, int32_t audio_input_int24) {
    m_delay_wp[lr] = (m_delay_wp[lr] + 1) & (DELAY_BUFF_SIZE - 1);
    m_delay_buff[lr][m_delay_wp[lr]] = audio_input_int24;
  }

  INLINE int32_t delay_buff_get(uint32_t lr, uint16_t sample_delay) {
    uint16_t curr_index  = (m_delay_wp[lr] - (sample_delay >> 4)) & (DELAY_BUFF_SIZE - 1);
    uint16_t next_index  = (curr_index - 1) & (DELAY_BUFF_SIZE - 1);
    uint16_t next_weight = (sample_delay & 0xF);
    int32_t  curr_data   = m_delay_buff[lr][curr_index];
    int32_t  next_data   = m_delay_buff[lr][next_index];

    // lerp
    int32_t result = curr_data + multiply_shift_right(next_data - curr_data, next_weight << 12, 16);

    return result;
  }

  INLINE void delay_buff_attenuate() {
    for (uint16_t i = 0; i < DELAY_BUFF_SIZE; ++i) {
      m_delay_buff[0][i] = m_delay_buff[0][i] >> 1;
      m_delay_buff[1][i] = m_delay_buff[1][i] >> 1;
    }
  }

  INLINE int16_t get_chorus_lfo_wave_level(uint32_t phase) {
    int16_t triangle_wave_level = 0;
    phase = (phase >> 9);

    if (phase < 0x00004000) {
      triangle_wave_level = phase - (512 << 4);
    } else {
      triangle_wave_level = (1535 << 4) - phase;
    }

    return triangle_wave_level;
  }
};
