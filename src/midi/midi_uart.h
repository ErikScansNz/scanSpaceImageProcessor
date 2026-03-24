#pragma once

#include <cstdint>

namespace pico2w_synth {

class MidiUart {
 public:
  bool Init();
  bool ReadByte(uint8_t* out);
};

}  // namespace pico2w_synth
