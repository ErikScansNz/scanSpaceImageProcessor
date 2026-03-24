#include "audio/audio_backend.h"

namespace pico2w_synth {

class I2sBackendStub final : public AudioBackend {
 public:
  bool Start(uint32_t sample_rate_hz) override {
    static_cast<void>(sample_rate_hz);
    return false;
  }

  void Stop() override {}
  void WriteStereo(int16_t left, int16_t right) override {
    static_cast<void>(left);
    static_cast<void>(right);
  }

  const char* Name() const override { return "i2s_stub"; }
};

}  // namespace pico2w_synth
