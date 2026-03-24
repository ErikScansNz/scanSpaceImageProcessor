#pragma once

#include <cstdint>

#include "params/param_schema.h"

namespace pico2w_synth {

void SynthRuntimeHandleNoteOn(uint8_t channel, uint8_t note, uint8_t velocity);
void SynthRuntimeHandleNoteOff(uint8_t channel, uint8_t note, uint8_t velocity);
void SynthRuntimeHandlePitchBend(uint8_t channel, uint16_t bend_14bit);
void SynthRuntimeHandleChannelPressure(uint8_t channel, uint8_t pressure);
void SynthRuntimeApplyParam(ParamId id, uint8_t value);
void SynthRuntimeProcessFrame(int16_t* left_out, int16_t* right_out);

}  // namespace pico2w_synth
