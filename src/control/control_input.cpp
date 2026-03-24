#include "control/control_input.h"

#include "control/hardware_controls_stub.h"
#include "pico/time.h"

namespace pico2w_synth {

bool ControlInput::Init(SynthEventQueue* queue, ParamRegistry* registry) {
  queue_ = queue;
  registry_ = registry;
  HardwareControlsInit();
  return queue_ != nullptr && registry_ != nullptr;
}

void ControlInput::Poll() {
  if (queue_ == nullptr) {
    return;
  }

  HardwareControlChange change{};
  while (HardwareControlsPoll(&change)) {
    if (!change.valid) {
      continue;
    }

    Event ev{};
    ev.type = EventType::ParamSet;
    ev.channel = 0;
    ev.data0 = 0;
    ev.data1 = change.value;
    ev.param_id = static_cast<uint16_t>(change.param);
    ev.timestamp_us = time_us_32();
    ev.source = ParamSource::Hardware;
    if (!queue_->TryPush(ev)) {
      queue_->RecordDrop(ParamSource::Hardware);
      continue;
    }
    if (registry_ != nullptr) {
      registry_->Set(change.param, change.value, ev.timestamp_us, ParamSource::Hardware);
    }
  }
}

}  // namespace pico2w_synth
