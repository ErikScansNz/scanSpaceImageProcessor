#pragma once

#include <cstdint>

#include "midi/midi_uart.h"
#include "midi/midi_usb.h"
#include "util/event_queue.h"

namespace pico2w_synth {

class MidiInput {
 public:
  bool Init(SynthEventQueue* queue, ParamRegistry* registry);
  void Poll();

 private:
  void ProcessByte(uint8_t b, ParamSource source, uint32_t timestamp_us);
  void DispatchMessage(uint8_t status, uint8_t data0, uint8_t data1, uint32_t timestamp_us);
  bool PushEvent(const Event& event);

  SynthEventQueue* queue_ = nullptr;
  ParamRegistry* registry_ = nullptr;
  MidiUart uart_;
  MidiUsb usb_;

  uint8_t running_status_ = 0;
  uint8_t data_bytes_[2] = {0, 0};
  uint8_t data_count_ = 0;
  bool in_sysex_ = false;
};

}  // namespace pico2w_synth
