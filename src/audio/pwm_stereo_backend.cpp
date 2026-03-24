#include "audio/pwm_stereo_backend.h"

#include "hardware/pwm.h"
#include "pico/stdlib.h"

#include "pico2w_synth/config.h"

namespace pico2w_synth {

namespace {

void InitStereoPwmPair(uint8_t pin_l,
                       uint8_t pin_r,
                       uint32_t wrap,
                       uint32_t* slice_l,
                       uint32_t* slice_r,
                       uint32_t* chan_l,
                       uint32_t* chan_r) {
  gpio_set_function(pin_l, GPIO_FUNC_PWM);
  gpio_set_function(pin_r, GPIO_FUNC_PWM);

  *slice_l = pwm_gpio_to_slice_num(pin_l);
  *slice_r = pwm_gpio_to_slice_num(pin_r);
  *chan_l = pwm_gpio_to_channel(pin_l);
  *chan_r = pwm_gpio_to_channel(pin_r);

  pwm_set_clkdiv_int_frac(*slice_l, 1, 0);
  pwm_set_clkdiv_int_frac(*slice_r, 1, 0);
  pwm_set_wrap(*slice_l, static_cast<uint16_t>(wrap));
  pwm_set_wrap(*slice_r, static_cast<uint16_t>(wrap));

  const uint16_t midpoint = static_cast<uint16_t>(wrap / 2u);
  pwm_set_chan_level(*slice_l, *chan_l, midpoint);
  pwm_set_chan_level(*slice_r, *chan_r, midpoint);

  pwm_set_enabled(*slice_l, true);
  pwm_set_enabled(*slice_r, true);
}

void WriteStereoToPair(uint32_t slice_l,
                       uint32_t slice_r,
                       uint32_t chan_l,
                       uint32_t chan_r,
                       int16_t left,
                       int16_t right) {
  uint32_t wrap_l = static_cast<uint32_t>(pwm_hw->slice[slice_l].top);
  uint32_t wrap_r = static_cast<uint32_t>(pwm_hw->slice[slice_r].top);

  uint32_t uleft = static_cast<uint16_t>(left + 32768);
  uint32_t uright = static_cast<uint16_t>(right + 32768);

  uint32_t level_l = (uleft * wrap_l) / 65535u;
  uint32_t level_r = (uright * wrap_r) / 65535u;

  if (level_l > 65535u) {
    level_l = 65535u;
  }
  if (level_r > 65535u) {
    level_r = 65535u;
  }

  pwm_set_chan_level(slice_l, chan_l, static_cast<uint16_t>(level_l));
  pwm_set_chan_level(slice_r, chan_r, static_cast<uint16_t>(level_r));
}

}  // namespace

bool PwmStereoBackend::Start(uint32_t sample_rate_hz) {
  // 125 MHz / (wrap+1) == sample_rate if clkdiv = 1.
  uint32_t wrap = (125000000u / sample_rate_hz);
  if (wrap == 0) {
    return false;
  }
  wrap -= 1u;
  if (wrap > 65535u) {
    wrap = 65535u;
  }

  InitStereoPwmPair(kPinPwmLeft, kPinPwmRight, wrap, &slice_l_, &slice_r_, &chan_l_, &chan_r_);
  mirror_started_ = false;
  if (kEnablePwmMirrorLegacyPins &&
      (kPinPwmLegacyLeft != kPinPwmLeft || kPinPwmLegacyRight != kPinPwmRight)) {
    InitStereoPwmPair(kPinPwmLegacyLeft,
                      kPinPwmLegacyRight,
                      wrap,
                      &mirror_slice_l_,
                      &mirror_slice_r_,
                      &mirror_chan_l_,
                      &mirror_chan_r_);
    mirror_started_ = true;
  }

  started_ = true;
  return true;
}

void PwmStereoBackend::Stop() {
  if (!started_) {
    return;
  }
  pwm_set_enabled(slice_l_, false);
  pwm_set_enabled(slice_r_, false);
  if (mirror_started_) {
    pwm_set_enabled(mirror_slice_l_, false);
    pwm_set_enabled(mirror_slice_r_, false);
    mirror_started_ = false;
  }
  started_ = false;
}

void PwmStereoBackend::WriteStereo(int16_t left, int16_t right) {
  if (!started_) {
    return;
  }

  WriteStereoToPair(slice_l_, slice_r_, chan_l_, chan_r_, left, right);
  if (mirror_started_) {
    WriteStereoToPair(mirror_slice_l_, mirror_slice_r_, mirror_chan_l_, mirror_chan_r_, left, right);
  }
}

}  // namespace pico2w_synth
