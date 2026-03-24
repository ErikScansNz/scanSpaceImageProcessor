#pragma once

#include <cstdint>

namespace pico2w_synth {

class AudioBackend {
 public:
  virtual ~AudioBackend() = default;
  virtual bool Start(uint32_t sample_rate_hz) = 0;
  virtual void Stop() = 0;
  virtual void WriteStereo(int16_t left, int16_t right) = 0;
  virtual const char* Name() const = 0;
};

}  // namespace pico2w_synth
