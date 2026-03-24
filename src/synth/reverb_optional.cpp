#include <cstdint>

namespace pico2w_synth {

// Placeholder for optional reverb module inspired by pico_synthesizer.
// Disabled by default via compile definition PICO2W_SYNTH_ENABLE_REVERB=0.
int32_t ApplyOptionalReverb(int32_t left, int32_t right, int32_t* right_out) {
  if (right_out) {
    *right_out = right;
  }
  return left;
}

}  // namespace pico2w_synth
