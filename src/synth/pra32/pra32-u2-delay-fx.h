#pragma once

#include "pra32-u2-common.h"

class PRA32_U2_DelayFx {
  static const uint16_t DELAY_BUFF_SIZE = 16384;

  int32_t  m_delay_buff[2][DELAY_BUFF_SIZE];
  uint16_t m_delay_wp[2];

  uint16_t m_delay_level;
  uint16_t m_delay_level_effective;
  uint8_t  m_delay_feedback;
  uint8_t  m_delay_feedback_effective;
  uint16_t m_delay_time;
  uint16_t m_delay_time_effective;
  uint8_t  m_delay_mode;

  int32_t  m_prev_sample_to_push_0;
  int32_t  m_prev_sample_to_push_1;

public:
  PRA32_U2_DelayFx()
  : m_delay_buff()
  , m_delay_wp()

  , m_delay_level()
  , m_delay_level_effective()
  , m_delay_feedback()
  , m_delay_feedback_effective()
  , m_delay_time()
  , m_delay_time_effective()
  , m_delay_mode()

  , m_prev_sample_to_push_0()
  , m_prev_sample_to_push_1()
  {
    m_delay_wp[0] = DELAY_BUFF_SIZE - 1;
    m_delay_wp[1] = DELAY_BUFF_SIZE - 1;

    set_delay_level   (0  );
    set_delay_feedback(64 );
    set_delay_time    (93 );

    m_delay_time_effective = m_delay_time;
  }

  INLINE void set_delay_level(uint8_t controller_value) {
    m_delay_level = ((controller_value + 1) >> 1) << 2;
  }

  INLINE void set_delay_feedback(uint8_t controller_value) {
    m_delay_feedback = controller_value;
  }

  INLINE void set_delay_time(uint8_t controller_value) {
    static uint16_t delay_time_table[128] = {
         48,    96,   144,   192,   240,   288,   384,   480,
        576,   672,   768,   864,   960,  1056,  1152,  1248,
       1344,  1440,  1536,  1632,  1728,  1824,  1920,  2016,
       2112,  2208,  2304,  2400,  2560,  2720,  2880,  3040,
       3200,  3360,  3520,  3680,  3840,  4000,  4160,  4320,
       4480,  4640,  4800,  4960,  5120,  5280,  5440,  5600,
       5760,  5920,  6080,  6240,  6400,  6560,  6720,  6880,
       7040,  7200,  7360,  7520,  7680,  7840,  8000,  8160,
       8320,  8480,  8640,  8800,  8960,  9120,  9280,  9440,
       9600,  9760,  9920, 10080, 10240, 10400, 10560, 10720,
      10880, 11040, 11200, 11360, 11520, 11680, 11840, 12000,
      12160, 12320, 12480, 12640, 12800, 12960, 13120, 13280,
      13440, 13600, 13760, 13920, 14080, 14240, 14400, 14560,
      14720, 14880, 15040, 15200, 15360, 15520, 15680, 15840,
      16160, 16320, 16320, 16320, 16320, 16320, 16320, 16320,
      16320, 16320, 16320, 16320, 16320, 16320, 16320, 16320,
    };

    m_delay_time = delay_time_table[controller_value];
  }

  INLINE void set_delay_mode(uint8_t controller_value) {
    m_delay_mode = controller_value;
  }

  INLINE void process_at_low_rate(uint8_t count) {
    m_delay_level_effective += (m_delay_level_effective < m_delay_level);
    m_delay_level_effective -= (m_delay_level_effective > m_delay_level);

    m_delay_feedback_effective += (m_delay_feedback_effective < m_delay_feedback);
    m_delay_feedback_effective -= (m_delay_feedback_effective > m_delay_feedback);

    if ((count & 0x01) == 0) {
      m_delay_time_effective += (m_delay_time_effective < m_delay_time);
      m_delay_time_effective -= (m_delay_time_effective > m_delay_time);
    }
  }

  INLINE int32_t process(int32_t left_input_int24, int32_t right_input_int24, int32_t& right_output_int24) {
    int32_t left_delay   = delay_buff_get<0>(m_delay_time_effective);
    int32_t right_delay  = delay_buff_get<1>(m_delay_time_effective);

    int32_t left_feedback;
    int32_t right_feedback;

    int32_t left_send  = multiply_shift_right(left_input_int24,  m_delay_level_effective, 8);
    int32_t right_send = multiply_shift_right(right_input_int24, m_delay_level_effective, 8);

    if (m_delay_mode >= 64) {
      // Ping Pong Delay
      left_feedback  = multiply_shift_right((((left_send + right_send) >> 1) + right_delay), (m_delay_feedback_effective << 8), 16);
      right_feedback = multiply_shift_right((                                  left_delay ), (m_delay_feedback_effective << 8), 16);
    } else {
      // Stereo Delay
      left_feedback  = multiply_shift_right((left_send  + left_delay ), (m_delay_feedback_effective << 8), 16);
      right_feedback = multiply_shift_right((right_send + right_delay), (m_delay_feedback_effective << 8), 16);
    }

    int32_t curr_sample_to_push_0 = left_feedback;
    int32_t curr_sample_to_push_1 = right_feedback;

#if 0
    // Do not apply LPF to the delay component
    m_prev_sample_to_push_0 = curr_sample_to_push_0;
    m_prev_sample_to_push_1 = curr_sample_to_push_1;
#endif

    delay_buff_push<0>((curr_sample_to_push_0 + m_prev_sample_to_push_0) >> 1);
    delay_buff_push<1>((curr_sample_to_push_1 + m_prev_sample_to_push_1) >> 1);

    m_prev_sample_to_push_0 = curr_sample_to_push_0;
    m_prev_sample_to_push_1 = curr_sample_to_push_1;

    right_output_int24 = right_input_int24 + right_delay;
    return               left_input_int24  + left_delay;
  }

private:
  template <uint8_t N>
  INLINE void delay_buff_push(int32_t audio_input) {
    m_delay_wp[N] = (m_delay_wp[N] + 1) & (DELAY_BUFF_SIZE - 1);
    m_delay_buff[N][m_delay_wp[N]] = audio_input;
  }

  template <uint8_t N>
  INLINE int32_t delay_buff_get(uint16_t sample_delay) {
    uint16_t delay_rp = (m_delay_wp[N] - sample_delay) & (DELAY_BUFF_SIZE - 1);
    return m_delay_buff[N][delay_rp];
  }
};
