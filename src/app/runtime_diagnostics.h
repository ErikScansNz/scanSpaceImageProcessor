#pragma once

#include <cstdint>

namespace pico2w_synth {

enum class RuntimeStatus : uint8_t {
  Booting = 0,
  Running,
  QueuePressure,
  Core1Stall,
};

enum class InitErrorCode : uint8_t {
  None = 0,
  MidiInit,
  ControlInit,
  SerialInit,
  AudioWorkerInit,
};

struct RuntimeDiagnostics {
  volatile uint32_t core0_heartbeat = 0;
  volatile uint32_t core1_heartbeat = 0;
  volatile uint32_t audio_overruns = 0;
  volatile uint32_t audio_frames = 0;
  volatile uint32_t audio_callback_last_us = 0;
  volatile uint32_t audio_callback_max_us = 0;
  volatile uint32_t audio_callback_over_budget = 0;
  volatile uint32_t core0_loop_last_us = 0;
  volatile uint32_t core0_loop_max_us = 0;
  volatile uint32_t core0_midi_poll_last_us = 0;
  volatile uint32_t core0_midi_poll_max_us = 0;
  volatile uint32_t core0_serial_poll_last_us = 0;
  volatile uint32_t core0_serial_poll_max_us = 0;
  volatile uint32_t core0_control_poll_last_us = 0;
  volatile uint32_t core0_control_poll_max_us = 0;
  volatile uint32_t queue_dropped_midi = 0;
  volatile uint32_t queue_dropped_web = 0;
  volatile uint32_t queue_dropped_hardware = 0;
  volatile uint8_t last_init_error = static_cast<uint8_t>(InitErrorCode::None);
  volatile uint8_t runtime_status = static_cast<uint8_t>(RuntimeStatus::Booting);
};

}  // namespace pico2w_synth
