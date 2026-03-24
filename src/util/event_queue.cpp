#include "util/event_queue.h"

namespace pico2w_synth {

void SynthEventQueue::Init(RuntimeDiagnostics* diagnostics) {
  diagnostics_ = diagnostics;
  queue_init(&queue_, sizeof(Event), kEventQueueCapacity);
  initialized_ = true;
}

bool SynthEventQueue::TryPush(const Event& e) {
  if (!initialized_) {
    return false;
  }
  return queue_try_add(&queue_, &e);
}

bool SynthEventQueue::TryPop(Event* out) {
  if (!initialized_ || out == nullptr) {
    return false;
  }
  return queue_try_remove(&queue_, out);
}

size_t SynthEventQueue::Size() const {
  if (!initialized_) {
    return 0;
  }
  return static_cast<size_t>(queue_get_level(const_cast<queue_t*>(&queue_)));
}

void SynthEventQueue::RecordDrop(ParamSource source) {
  if (diagnostics_ == nullptr) {
    return;
  }

  switch (source) {
    case ParamSource::Midi:
      ++diagnostics_->queue_dropped_midi;
      break;
    case ParamSource::Web:
      ++diagnostics_->queue_dropped_web;
      break;
    case ParamSource::Hardware:
      ++diagnostics_->queue_dropped_hardware;
      break;
    case ParamSource::System:
      break;
  }
}

}  // namespace pico2w_synth
