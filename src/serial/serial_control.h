#pragma once

#include <cstddef>
#include <cstdint>

#include "app/runtime_diagnostics.h"
#include "params/param_registry.h"
#include "util/event_queue.h"

namespace pico2w_synth {

class SerialControl {
 public:
  bool Init(ParamRegistry* registry, SynthEventQueue* queue, RuntimeDiagnostics* diagnostics = nullptr);
  void Poll();

 private:
  void HandleLine(char* line);
  void FlushTx();

  bool QueueRaw(const char* data, size_t len);
  bool QueueText(const char* text);
  bool QueueError(const char* reason);
  bool QueueState();
  bool QueueMeta();
  bool QueueStats();
  bool QueueAckSet(uint16_t id, uint8_t value);
  bool QueueAckBatch(size_t applied);

  bool HandleSet(unsigned id, unsigned value);
  bool HandleNoteCommand(bool note_on, unsigned note, unsigned velocity);
  bool HandleBatch(char* args, size_t* applied_out);

  ParamRegistry* registry_ = nullptr;
  SynthEventQueue* queue_ = nullptr;
  RuntimeDiagnostics* diagnostics_ = nullptr;
  bool initialized_ = false;

  static constexpr size_t kRxLineMax = 512;
  static constexpr size_t kTxMax = 12288;
  char rx_line_[kRxLineMax] = {};
  size_t rx_used_ = 0;
  char tx_[kTxMax] = {};
  size_t tx_used_ = 0;
  size_t tx_sent_ = 0;

  uint32_t rx_lines_ = 0;
  uint32_t rx_overflow_ = 0;
  uint32_t rx_parse_errors_ = 0;
  uint32_t tx_lines_ = 0;
  uint32_t tx_dropped_ = 0;

  uint32_t last_stats_audio_frames_ = 0;
  uint32_t last_stats_over_budget_ = 0;
  bool stats_baseline_valid_ = false;
};

}  // namespace pico2w_synth
