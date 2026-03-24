#pragma once

#include <cstdint>

namespace pico2w_synth {

class MidiUsb {
 public:
  bool Init();
  void ServiceUsbTask();
  bool ReadByte(uint8_t* out);

 private:
  uint8_t buffer_[64] = {};
  uint32_t buffer_used_ = 0;
  uint32_t buffer_index_ = 0;
};

}  // namespace pico2w_synth
