#include "app/application.h"

#include <cstdio>

#include "hardware/gpio.h"
#include "hardware/watchdog.h"
#include "pico/stdlib.h"
#include "pico/time.h"
#include "pico2w_synth/config.h"

namespace pico2w_synth {

bool Application::Init() {
  stdio_init_all();
  std::printf("[boot] pico2w-synth init start\n");
  InitStatusLed();

  diagnostics_.runtime_status = static_cast<uint8_t>(RuntimeStatus::Booting);
  diagnostics_.last_init_error = static_cast<uint8_t>(InitErrorCode::None);

  event_queue_.Init(&diagnostics_);
  registry_.ResetToDefaults(time_us_32());

  if (!midi_input_.Init(&event_queue_, &registry_)) {
    std::printf("[boot] midi init failed\n");
    diagnostics_.last_init_error = static_cast<uint8_t>(InitErrorCode::MidiInit);
    return false;
  }
  if (!control_input_.Init(&event_queue_, &registry_)) {
    std::printf("[boot] control init failed\n");
    diagnostics_.last_init_error = static_cast<uint8_t>(InitErrorCode::ControlInit);
    return false;
  }
  if (!serial_control_.Init(&registry_, &event_queue_, &diagnostics_)) {
    std::printf("[boot] serial control init failed\n");
    diagnostics_.last_init_error = static_cast<uint8_t>(InitErrorCode::SerialInit);
    return false;
  }
  std::printf("[boot] external-control mode (wifi/web/display removed)\n");

  if (!audio_worker_.Start(&event_queue_)) {
    std::printf("[boot] audio worker init failed\n");
    diagnostics_.last_init_error = static_cast<uint8_t>(InitErrorCode::AudioWorkerInit);
    return false;
  }

  QueueDefaultParams();

  diagnostics_.runtime_status = static_cast<uint8_t>(RuntimeStatus::Running);
  last_core1_heartbeat_ = audio_worker_.Heartbeat();
  last_core1_change_ms_ = to_ms_since_boot(get_absolute_time());

  // Feed only when both cores keep progressing.
  if (kEnableWatchdog) {
    watchdog_enable(2000, true);
  }
  std::printf("[boot] init complete\n");

  return true;
}

void Application::Run() {
  auto update_max = [](volatile uint32_t* max_slot, uint32_t value) {
    if (max_slot != nullptr && value > *max_slot) {
      *max_slot = value;
    }
  };

  while (true) {
    uint32_t loop_start_us = time_us_32();
    ++diagnostics_.core0_heartbeat;

    uint32_t phase_start_us = loop_start_us;
    midi_input_.Poll();
    uint32_t midi_us = time_us_32() - phase_start_us;
    diagnostics_.core0_midi_poll_last_us = midi_us;
    update_max(&diagnostics_.core0_midi_poll_max_us, midi_us);

    phase_start_us = time_us_32();
    serial_control_.Poll();
    uint32_t serial_us = time_us_32() - phase_start_us;
    diagnostics_.core0_serial_poll_last_us = serial_us;
    update_max(&diagnostics_.core0_serial_poll_max_us, serial_us);

    phase_start_us = time_us_32();
    control_input_.Poll();
    uint32_t control_us = time_us_32() - phase_start_us;
    diagnostics_.core0_control_poll_last_us = control_us;
    update_max(&diagnostics_.core0_control_poll_max_us, control_us);

    diagnostics_.core1_heartbeat = audio_worker_.Heartbeat();
    diagnostics_.audio_overruns = audio_worker_.Overruns();
    diagnostics_.audio_frames = audio_worker_.Frames();
    diagnostics_.audio_callback_last_us = audio_worker_.CallbackLastUs();
    diagnostics_.audio_callback_max_us = audio_worker_.CallbackMaxUs();
    diagnostics_.audio_callback_over_budget = audio_worker_.CallbackOverBudget();

    RefreshRuntimeState();
    UpdateStatusLed();

    if (kEnableWatchdog) {
      uint32_t core0 = diagnostics_.core0_heartbeat;
      uint32_t core1 = diagnostics_.core1_heartbeat;
      if (core0 != last_watchdog_core0_heartbeat_ && core1 != last_watchdog_core1_heartbeat_) {
        watchdog_update();
        last_watchdog_core0_heartbeat_ = core0;
        last_watchdog_core1_heartbeat_ = core1;
      }
    }

    uint32_t loop_us = time_us_32() - loop_start_us;
    diagnostics_.core0_loop_last_us = loop_us;
    update_max(&diagnostics_.core0_loop_max_us, loop_us);

    sleep_us(kMainLoopSleepUs);
  }
}

void Application::InitStatusLed() {
#ifdef PICO_DEFAULT_LED_PIN
  gpio_init(PICO_DEFAULT_LED_PIN);
  gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
  status_led_use_gpio_ = true;
  SetStatusLed(false);
#endif
}

void Application::SetStatusLed(bool on) {
  if (status_led_on_ == on) {
    return;
  }

  if (status_led_use_gpio_) {
#ifdef PICO_DEFAULT_LED_PIN
    gpio_put(PICO_DEFAULT_LED_PIN, on ? 1 : 0);
    status_led_on_ = on;
#endif
  }
}

void Application::UpdateStatusLed() {
  uint32_t now_ms = to_ms_since_boot(get_absolute_time());
  RuntimeStatus status = static_cast<RuntimeStatus>(diagnostics_.runtime_status);

  bool on = false;
  switch (status) {
    case RuntimeStatus::Booting:
      on = ((now_ms % 500u) < 80u);
      break;

    case RuntimeStatus::Running:
      on = ((now_ms % 1000u) < 500u);
      break;

    case RuntimeStatus::QueuePressure: {
      uint32_t phase = now_ms % 1000u;
      on = (phase < 120u) || (phase >= 240u && phase < 360u);
      break;
    }

    case RuntimeStatus::Core1Stall:
      on = true;
      break;
  }

  SetStatusLed(on);
}

void Application::RefreshRuntimeState() {
  uint32_t now_ms = to_ms_since_boot(get_absolute_time());

  uint32_t core1_hb = diagnostics_.core1_heartbeat;
  if (core1_hb != last_core1_heartbeat_) {
    last_core1_heartbeat_ = core1_hb;
    last_core1_change_ms_ = now_ms;
  }

  if ((now_ms - last_core1_change_ms_) > 400u) {
    diagnostics_.runtime_status = static_cast<uint8_t>(RuntimeStatus::Core1Stall);
    return;
  }

  size_t queue_depth = event_queue_.Size();
  if (queue_depth > (kEventQueueCapacity * 3u / 4u)) {
    diagnostics_.runtime_status = static_cast<uint8_t>(RuntimeStatus::QueuePressure);
    return;
  }

  diagnostics_.runtime_status = static_cast<uint8_t>(RuntimeStatus::Running);
}

void Application::QueueDefaultParams() {
  uint32_t t = time_us_32();
  for (size_t i = 0; i < static_cast<size_t>(ParamId::ParamCount); ++i) {
    ParamId id = static_cast<ParamId>(i);
    uint8_t value = registry_.Get(id);

    Event ev{};
    ev.type = EventType::ParamSet;
    ev.channel = 0;
    ev.data0 = 0;
    ev.data1 = value;
    ev.param_id = static_cast<uint16_t>(id);
    ev.timestamp_us = t;
    ev.source = ParamSource::System;

    if (!event_queue_.TryPush(ev)) {
      event_queue_.RecordDrop(ParamSource::System);
    }
  }
}

}  // namespace pico2w_synth
