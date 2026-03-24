#pragma once

#include <array>
#include <cstdint>

#include "pico/time.h"

#include "audio/pwm_stereo_backend.h"
#include "params/param_schema.h"
#include "util/event_queue.h"

namespace pico2w_synth {

class AudioWorker {
 public:
  bool Start(SynthEventQueue* queue);
  void Stop();

  uint32_t Heartbeat() const;
  uint32_t Overruns() const;
  uint32_t Frames() const;
  uint32_t CallbackLastUs() const;
  uint32_t CallbackMaxUs() const;
  uint32_t CallbackOverBudget() const;

 private:
  static void Core1Entry();
  void RunCore1();

  static bool AudioTimerThunk(struct repeating_timer* timer);
  bool OnAudioTimer();

  void HandleEvent(const Event& ev);
  void UpdateSmoothedParams();
  void ApplyOutputSafety(int16_t* left, int16_t* right);

  static AudioWorker* g_instance_;

  SynthEventQueue* queue_ = nullptr;
  PwmStereoBackend audio_backend_;
  repeating_timer audio_timer_{};

  std::array<uint8_t, static_cast<size_t>(ParamId::ParamCount)> smoothed_values_{};
  std::array<uint8_t, static_cast<size_t>(ParamId::ParamCount)> target_values_{};
  std::array<uint8_t, static_cast<size_t>(ParamId::ParamCount)> active_param_ids_{};
  std::array<uint8_t, static_cast<size_t>(ParamId::ParamCount)> active_param_present_{};
  uint8_t active_param_count_ = 0;

  volatile bool core1_ready_ = false;
  volatile bool running_ = false;
  volatile bool stop_requested_ = false;
  volatile bool audio_busy_ = false;

  volatile uint32_t heartbeat_ = 0;
  volatile uint32_t overrun_count_ = 0;
  volatile uint32_t frame_count_ = 0;
  volatile uint32_t callback_last_us_ = 0;
  volatile uint32_t callback_max_us_ = 0;
  volatile uint32_t callback_over_budget_ = 0;
  uint32_t sample_count_ = 0;

  bool dither_enabled_ = false;
  bool noise_shaping_enabled_ = false;
  uint32_t dither_lfsr_ = 0x12345678u;
  int32_t ns_error_l_ = 0;
  int32_t ns_error_r_ = 0;
};

}  // namespace pico2w_synth
