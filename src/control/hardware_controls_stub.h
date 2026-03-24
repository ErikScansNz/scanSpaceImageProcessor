#pragma once

#include <cstdint>

#include "params/param_schema.h"

namespace pico2w_synth {

struct HardwareControlChange {
  bool valid;
  ParamId param;
  uint8_t value;
};

void HardwareControlsInit();
bool HardwareControlsPoll(HardwareControlChange* out_change);

}  // namespace pico2w_synth
