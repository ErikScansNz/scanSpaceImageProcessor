#pragma once

#include "util/event_queue.h"

namespace pico2w_synth {

class ControlInput {
 public:
  bool Init(SynthEventQueue* queue, ParamRegistry* registry);
  void Poll();

 private:
  SynthEventQueue* queue_ = nullptr;
  ParamRegistry* registry_ = nullptr;
};

}  // namespace pico2w_synth
