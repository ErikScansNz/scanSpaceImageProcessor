#pragma once

// refs https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt
// refs https://jatinchowdhury18.medium.com/complex-nonlinearities-episode-4-nonlinear-biquad-filters-ae6b3f23cb0e

#include "pra32-u2-common.h"
#include "pra32-u2-filter-table.h"

static const uint8_t FILTER_CALC_SCALING_BITS = 4;

static INLINE int32_t soft_clip(int32_t value) {
    // Note: Without anti-aliasing (oversampling)
    int32_t one       = (1 << 23) << FILTER_CALC_SCALING_BITS;
    int32_t two_three = one * 2 / 3;
    volatile int32_t clamped =
         (value >  (+one))                       * (+two_three)
      +                       (value <  (-one))  * (-two_three)
      + ((value <= (+one)) && (value >= (-one))) *
        (value - (multiply_shift_right(multiply_shift_right(
                  value << (5 - FILTER_CALC_SCALING_BITS), value << 4, 32)
                        << (5 - FILTER_CALC_SCALING_BITS), value << 4, 32) / 3));
    return clamped;
}

class PRA32_U2_Filter {
  int32_t         m_b_2_over_a_0;
  int32_t         m_a_1_over_a_0;
  int32_t         m_a_2_over_a_0;
  int32_t         m_z_1;
  int32_t         m_z_2;
  uint8_t         m_resonance_target;
  uint8_t         m_resonance_current;
  int16_t         m_cutoff_current;
  int16_t         m_cutoff_control;
  int16_t         m_cutoff_control_effective;
  int16_t         m_cutoff_eg_amt[2];
  int16_t         m_cutoff_lfo_amt[2];
  int16_t         m_cutoff_pitch_amt;
  uint8_t         m_filter_mode;
  int16_t         m_cutoff_breath_amt;
  int16_t         m_breath_controller;

public:
  PRA32_U2_Filter()
  : m_b_2_over_a_0()
  , m_a_1_over_a_0()
  , m_a_2_over_a_0()
  , m_z_1()
  , m_z_2()
  , m_resonance_target()
  , m_resonance_current()
  , m_cutoff_current()
  , m_cutoff_control()
  , m_cutoff_control_effective()
  , m_cutoff_eg_amt()
  , m_cutoff_lfo_amt()
  , m_cutoff_pitch_amt()
  , m_filter_mode()
  , m_cutoff_breath_amt()
  , m_breath_controller()
  {
    m_cutoff_current = 254;

    set_cutoff(127);
    set_resonance(0);
    set_cutoff_eg_amt(0, 64);
    set_cutoff_eg_amt(1, 64);
    set_cutoff_lfo_amt(0, 64);
    set_cutoff_lfo_amt(1, 64);
    set_cutoff_pitch_amt(0);

    update_coefs(0, 0, 60 << 8);
  }

  INLINE void set_cutoff(uint8_t controller_value) {
    m_cutoff_control = controller_value;
  }

  INLINE void set_resonance(uint8_t controller_value) {
    m_resonance_target = controller_value;
  }

  INLINE int8_t get_cutoff_mod_amt(uint8_t controller_value) {
    static int16_t cutoff_mod_amt_table[128] = {
      -120, -120, -120, -120, -120, -118, -116, -114,
      -112, -110, -108, -106, -104, -102, -100,  -98,
       -96,  -94,  -92,  -90,  -88,  -86,  -84,  -82,
       -80,  -78,  -76,  -74,  -72,  -70,  -68,  -66,
       -64,  -62,  -60,  -58,  -56,  -54,  -52,  -50,
       -48,  -46,  -44,  -42,  -40,  -38,  -36,  -34,
       -32,  -30,  -28,  -26,  -24,  -22,  -20,  -18,
       -16,  -14,  -12,  -10,   -8,   -6,   -4,   -2,
        +0,   +2,   +4,   +6,   +8,  +10,  +12,  +14,
       +16,  +18,  +20,  +22,  +24,  +26,  +28,  +30,
       +32,  +34,  +36,  +38,  +40,  +42,  +44,  +46,
       +48,  +50,  +52,  +54,  +56,  +58,  +60,  +62,
       +64,  +66,  +68,  +70,  +72,  +74,  +76,  +78,
       +80,  +82,  +84,  +86,  +88,  +90,  +92,  +94,
       +96,  +98, +100, +102, +104, +106, +108, +110,
      +112, +114, +116, +118, +120, +120, +120, +120,
    };

    return cutoff_mod_amt_table[controller_value];
  }

  INLINE void set_cutoff_eg_amt(uint8_t index, uint8_t controller_value) {
    m_cutoff_eg_amt[index] = get_cutoff_mod_amt(controller_value) << 1;
  }

  INLINE void set_cutoff_lfo_amt(uint8_t index, uint8_t controller_value) {
    m_cutoff_lfo_amt[index] = get_cutoff_mod_amt(controller_value) << 1;
  }

  INLINE void set_cutoff_pitch_amt(uint8_t controller_value) {
    m_cutoff_pitch_amt = ((controller_value - 64) + 4) >> 3;
  }

  INLINE void set_filter_mode(uint8_t controller_value) {
    m_filter_mode = controller_value;
  }

  INLINE void set_cutoff_breath_amt(uint8_t controller_value) {;
    m_cutoff_breath_amt = get_cutoff_mod_amt(controller_value) << 1;
  }

  INLINE void set_breath_controller(uint8_t controller_value) {
    m_breath_controller = (controller_value * 16384) / 127;
  }

  INLINE void process_at_low_rate(uint8_t count, int16_t eg_input, int16_t lfo_input, uint16_t osc_pitch) {
    update_cutoff_control_effective();
    update_coefs(eg_input, lfo_input, osc_pitch);
  }

  INLINE void reset_state() {
    m_z_1 = 0;
    m_z_2 = 0;
  }

  INLINE int32_t process(int32_t audio_input_int24) {
#if 1
    // Nonlinear Biquad Filter, Transposed Direct Form-II
    int32_t x_0 = audio_input_int24 << FILTER_CALC_SCALING_BITS;
    int32_t y_0 =           m_z_1 + (multiply_shift_right(m_b_2_over_a_0, x_0,      32) << (32 - FILTER_TABLE_FRACTION_BITS));
    m_z_1       = soft_clip(m_z_2 + (multiply_shift_right(m_b_2_over_a_0, x_0 << 1, 32) << (32 - FILTER_TABLE_FRACTION_BITS))
                                  - (multiply_shift_right(m_a_1_over_a_0, y_0,      32) << (32 - FILTER_TABLE_FRACTION_BITS)));
    m_z_2       = soft_clip(        (multiply_shift_right(m_b_2_over_a_0, x_0,      32) << (32 - FILTER_TABLE_FRACTION_BITS))
                                  - (multiply_shift_right(m_a_2_over_a_0, y_0,      32) << (32 - FILTER_TABLE_FRACTION_BITS)));

    if (m_filter_mode >= 64) {
      // high pass
      y_0 = x_0 - y_0;
    }
#else
    volatile int32_t y_0 = audio_input_int24 << FILTER_CALC_SCALING_BITS;
#endif

    return y_0 >> FILTER_CALC_SCALING_BITS;
  }

private:
  INLINE void update_cutoff_control_effective() {
#if 1
    m_cutoff_control_effective = m_cutoff_control;
#else
    m_cutoff_control_effective += (m_cutoff_control_effective < m_cutoff_control);
    m_cutoff_control_effective -= (m_cutoff_control_effective > m_cutoff_control);
#endif
  }

  INLINE void update_coefs(int16_t eg_input, int16_t lfo_input, uint16_t osc_pitch) {
    int16_t cutoff_candidate = m_cutoff_control_effective << (1 + 2);
    cutoff_candidate += (m_cutoff_eg_amt[0] * eg_input) >> (14 - 2);
    cutoff_candidate += (m_cutoff_eg_amt[1] * eg_input) >> (14 - 2);

    cutoff_candidate += (lfo_input * m_cutoff_lfo_amt[0]) >> (14 - 2);
    cutoff_candidate += (lfo_input * m_cutoff_lfo_amt[1]) >> (14 - 2);
    cutoff_candidate += (((osc_pitch - (60 << 8)) * m_cutoff_pitch_amt) + (1 << ((10 - 1) - 2))) >> (10 - 2);
    cutoff_candidate += (m_breath_controller * m_cutoff_breath_amt) >> (14 - 2);

    volatile int16_t cutoff_target = clamp(cutoff_candidate, 0, ((254 << 2) + 1));

    for (uint32_t i = 0; i < (4 * 2); ++i) {
      m_cutoff_current += (m_cutoff_current < cutoff_target);
      m_cutoff_current -= (m_cutoff_current > cutoff_target);
    }

    m_resonance_current += (m_resonance_current < m_resonance_target);
    m_resonance_current -= (m_resonance_current > m_resonance_target);

    uint8_t resonance_index = (m_resonance_current + ((1 << (3 - FILTER_TABLE_RESO_EXT_BITS)) >> 1)) >> (3 - FILTER_TABLE_RESO_EXT_BITS);
    const int32_t* filter_table = g_filter_tables[resonance_index];
#if defined(ARDUINO_ARCH_RP2040)
    // Uncached, untranslated XIP access -- bypass QMI address translation
    filter_table = reinterpret_cast<const int32_t*>(reinterpret_cast<uintptr_t>(filter_table) | 0x1c000000u);
#endif
    size_t index = ((m_cutoff_current + ((1 << (2 - FILTER_TABLE_CUTOFF_EXT_BITS)) >> 1)) >> (2 - FILTER_TABLE_CUTOFF_EXT_BITS)) * 3;
    m_b_2_over_a_0 = filter_table[index + 0];
    m_a_1_over_a_0 = filter_table[index + 1];
    m_a_2_over_a_0 = filter_table[index + 2];
  }
};
