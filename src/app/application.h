#pragma once

#include <cstdint>

#include "app/runtime_diagnostics.h"
#include "audio/audio_worker.h"
#include "control/control_input.h"
#include "midi/midi_input.h"
#include "params/param_registry.h"
#include "serial/serial_control.h"
#include "util/event_queue.h"

namespace pico2w_synth {

class Application {
 public:
  bool Init();
  void Run();

 private:
  void InitStatusLed();
  void SetStatusLed(bool on);
  void UpdateStatusLed();
  void RefreshRuntimeState();
  void QueueDefaultParams();

  ParamRegistry registry_;
  RuntimeDiagnostics diagnostics_{};
  SynthEventQueue event_queue_;

  MidiInput midi_input_;
  SerialControl serial_control_;
  ControlInput control_input_;
  AudioWorker audio_worker_;

  bool status_led_use_gpio_ = false;
  bool status_led_on_ = false;

  uint32_t last_core1_heartbeat_ = 0;
  uint32_t last_core1_change_ms_ = 0;

  uint32_t last_watchdog_core0_heartbeat_ = 0;
  uint32_t last_watchdog_core1_heartbeat_ = 0;
};

}  // namespace pico2w_synth
