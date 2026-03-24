#pragma once

#include <cstdint>

#include "params/param_schema.h"
#include "synth/pra32/pra32-u2-amp.h"
#include "synth/pra32/pra32-u2-chorus-fx.h"
#include "synth/pra32/pra32-u2-delay-fx.h"
#include "synth/pra32/pra32-u2-eg.h"
#include "synth/pra32/pra32-u2-filter.h"
#include "synth/pra32/pra32-u2-lfo.h"
#include "synth/pra32/pra32-u2-noise-gen.h"
#include "synth/pra32/pra32-u2-osc.h"
#include "synth/pra32/pra32-u2-panner.h"

namespace pico2w_synth {

class SynthEngine {
 public:
  SynthEngine();

  void HandleNoteOn(uint8_t channel, uint8_t note, uint8_t velocity);
  void HandleNoteOff(uint8_t channel, uint8_t note, uint8_t velocity);
  void HandlePitchBend(uint8_t channel, uint16_t bend_14bit);
  void HandleChannelPressure(uint8_t channel, uint8_t pressure);

  void ApplyParam(ParamId id, uint8_t value);
  void ProcessFrame(int16_t* left_out, int16_t* right_out);

 private:
  template <uint8_t VoiceIndex>
  void NoteOnVoice(uint8_t note, uint8_t velocity);

  template <uint8_t VoiceIndex>
  void NoteOffVoice();

  uint8_t FindVoiceForNoteOn() const;
  void ApplySupersawMode(bool enabled);
  void StartPitchEnvelope();
  void UpdatePitchEnvelopeOnControlTick();
  void ApplyCombinedPitchBend();
  static int16_t MapPitchEnvBendTo14Bit(uint8_t value);
  static uint32_t MapPitchEnvTimeToControlSteps(uint8_t value);
  void UpdateAuxOscState();
  void UpdateAuxOscForVoice(uint8_t voice, uint8_t note);
  int32_t RenderAuxOscSample(uint8_t voice, uint32_t& phase, uint32_t phase_inc, uint8_t wave_sel, uint8_t level_q7);

  PRA32_U2_NoiseGen noise_gen_;
  PRA32_U2_Osc osc_;
  PRA32_U2_Filter filter_[4];
  PRA32_U2_EG eg_[8];
  PRA32_U2_LFO lfo_;
  PRA32_U2_Amp amp_[4];
  PRA32_U2_Panner panner_;
  PRA32_U2_ChorusFx chorus_fx_;
  PRA32_U2_DelayFx delay_fx_;

  uint8_t voice_note_[4] = {NOTE_NUMBER_INVALID, NOTE_NUMBER_INVALID, NOTE_NUMBER_INVALID, NOTE_NUMBER_INVALID};
  bool voice_pressed_[4] = {false, false, false, false};
  uint8_t round_robin_voice_ = 0;
  bool monophonic_ = false;
  bool bypass_fx_ = false;
  bool supersaw_mode_ = false;
  uint32_t sample_count_ = 0;
  uint8_t osc1_level_ = 127;
  uint8_t osc2_level_ = 127;
  uint8_t osc3_level_ = 0;
  uint8_t osc4_level_ = 0;
  uint8_t osc3_wave_ = 64;
  uint8_t osc4_wave_ = 84;
  uint8_t osc_pair_level_q7_ = 127;
  uint8_t osc_pair_mix_ = 64;
  uint8_t osc_detune_ = 64;
  uint8_t tune_offset_ = 64;
  uint8_t osc3_pitch_ = 64;
  uint8_t osc4_pitch_ = 64;
  int16_t midi_pitch_bend_ = 0;
  uint8_t pitch_env_bend_ = 64;
  uint8_t pitch_env_time_ = 0;
  int32_t pitch_env_current_q16_ = 0;
  uint32_t pitch_env_step_q16_ = 0;
  bool pitch_env_active_ = false;
  bool infinite_drop_enabled_ = false;
  int32_t infinite_drop_q16_ = 0;
  uint32_t osc3_phase_[4] = {0u, 0u, 0u, 0u};
  uint32_t osc4_phase_[4] = {0u, 0u, 0u, 0u};
  uint32_t osc3_phase_inc_[4] = {0u, 0u, 0u, 0u};
  uint32_t osc4_phase_inc_[4] = {0u, 0u, 0u, 0u};
};

}  // namespace pico2w_synth
