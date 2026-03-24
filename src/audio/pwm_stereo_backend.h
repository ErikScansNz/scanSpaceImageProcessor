#pragma once

#include <cstdint>

#include "audio/audio_backend.h"

namespace pico2w_synth {

class PwmStereoBackend : public AudioBackend {
 public:
  bool Start(uint32_t sample_rate_hz) override;
  void Stop() override;
  void WriteStereo(int16_t left, int16_t right) override;
  const char* Name() const override { return "pwm_stereo"; }

 private:
  uint32_t slice_l_ = 0;
  uint32_t slice_r_ = 0;
  uint32_t chan_l_ = 0;
  uint32_t chan_r_ = 0;
  uint32_t mirror_slice_l_ = 0;
  uint32_t mirror_slice_r_ = 0;
  uint32_t mirror_chan_l_ = 0;
  uint32_t mirror_chan_r_ = 0;
  bool mirror_started_ = false;
  bool started_ = false;
};

}  // namespace pico2w_synth
