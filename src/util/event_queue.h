#pragma once

#include <cstddef>
#include <cstdint>

#include "pico/util/queue.h"

#include "app/runtime_diagnostics.h"
#include "pico2w_synth/config.h"
#include "params/param_registry.h"
#include "params/param_schema.h"

namespace pico2w_synth {

enum class EventType : uint8_t {
  NoteOn = 0,
  NoteOff,
  PitchBend,
  ChannelPressure,
  PolyAftertouch,
  ParamSet,
};

struct Event {
  EventType type;
  uint8_t channel;
  uint8_t data0;
  uint8_t data1;
  uint16_t param_id;
  uint32_t timestamp_us;
  ParamSource source;
};

class SynthEventQueue {
 public:
  void Init(RuntimeDiagnostics* diagnostics);

  bool TryPush(const Event& e);
  bool TryPop(Event* out);

  size_t Size() const;
  void RecordDrop(ParamSource source);

 private:
  RuntimeDiagnostics* diagnostics_ = nullptr;
  queue_t queue_{};
  bool initialized_ = false;
};

}  // namespace pico2w_synth
