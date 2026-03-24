#include "synth/synth_engine.h"
#include "synth/synth_runtime.h"

#include <cmath>
#include <climits>

#include "pico2w_synth/config.h"

namespace pico2w_synth {

namespace {

int16_t Clamp24ToInt16(int32_t x) {
  int32_t clamped = clamp(x, (-(INT16_MAX << 8)), (+(INT16_MAX << 8)));
  return static_cast<int16_t>(clamped >> 8);
}

int32_t ApplyMasterBoost10dB(int32_t x) {
  // +10 dB ~= 3.162x linear gain.
  constexpr int32_t kBoostQ8 = 810;
  int64_t scaled = (static_cast<int64_t>(x) * static_cast<int64_t>(kBoostQ8)) >> 8;
  if (scaled > static_cast<int64_t>(INT32_MAX)) {
    return INT32_MAX;
  }
  if (scaled < static_cast<int64_t>(INT32_MIN)) {
    return INT32_MIN;
  }
  return static_cast<int32_t>(scaled);
}

int16_t ScaleLfoOutput3x(int16_t value) {
  int32_t scaled = static_cast<int32_t>(value) * 3;
  if (scaled > INT16_MAX) {
    return INT16_MAX;
  }
  if (scaled < INT16_MIN) {
    return INT16_MIN;
  }
  return static_cast<int16_t>(scaled);
}

uint8_t RemapLfoRateFromUi(uint8_t value) {
  // Re-anchor the LFO rate range so UI "0" starts around previous value 65.
  constexpr uint8_t kUiMinMappedTo = 65u;
  constexpr uint8_t kUiMaxMappedTo = 127u;
  uint16_t span = static_cast<uint16_t>(kUiMaxMappedTo - kUiMinMappedTo);
  uint16_t mapped = static_cast<uint16_t>(kUiMinMappedTo) +
                    static_cast<uint16_t>((static_cast<uint32_t>(value) * span + 63u) / 127u);
  return static_cast<uint8_t>(mapped);
}

uint8_t RemapSubNoiseForPresence(uint8_t value) {
  // Center (64) is neutral.
  // Positive side should reach audible sub sooner; negative side should
  // introduce noise more gradually to avoid overwhelming hiss.
  int16_t delta = static_cast<int16_t>(value) - 64;
  if (delta == 0) {
    return 64u;
  }
  float mag = static_cast<float>((delta < 0) ? -delta : delta) / 64.0f;
  float shaped = (delta > 0) ? std::pow(mag, 0.60f) : std::pow(mag, 1.35f);
  int16_t mapped_delta = static_cast<int16_t>(std::lround(shaped * 64.0f));
  int16_t mapped = (delta > 0) ? (64 + mapped_delta) : (64 - mapped_delta);
  if (mapped < 0) {
    mapped = 0;
  } else if (mapped > 127) {
    mapped = 127;
  }
  return static_cast<uint8_t>(mapped);
}

uint8_t MapOscShapeSelectorToWaveControl(uint8_t value) {
  // Ordered selector: triangle, square, saw, sine.
  if (value < 32u) {
    return 51u;  // triangle bucket in PRA32 waveform selector
  }
  if (value < 64u) {
    return 26u;  // square bucket
  }
  if (value < 96u) {
    return 0u;   // saw bucket
  }
  return 76u;    // sine bucket
}

float MidiNoteToHz(uint8_t note, int16_t tune_offset) {
  float semitone = static_cast<float>(note) - 69.0f;
  float tune = (static_cast<float>(tune_offset) - 64.0f) / 64.0f;
  return 440.0f * std::pow(2.0f, (semitone + tune) / 12.0f);
}

uint32_t FrequencyToPhaseIncrement(float hz) {
  if (!(hz > 0.0f)) {
    return 0u;
  }
  constexpr double kPhaseScale = 4294967296.0 / static_cast<double>(kSampleRate);
  double scaled = static_cast<double>(hz) * kPhaseScale;
  if (scaled < 0.0) {
    return 0u;
  }
  if (scaled > static_cast<double>(UINT32_MAX)) {
    return UINT32_MAX;
  }
  return static_cast<uint32_t>(scaled);
}

uint8_t QuantizeWaveToShape(uint8_t value) {
  if (value < 21u) {
    return 2u;   // saw
  }
  if (value < 53u) {
    return 1u;   // square
  }
  if (value < 74u) {
    return 0u;   // triangle
  }
  return 3u;     // sine
}

int32_t RenderBasicWave24(uint32_t phase, uint8_t waveform) {
  switch (waveform) {
    case 0u: {  // triangle
      int32_t saw = static_cast<int32_t>(phase >> 8) - (1 << 23);
      int32_t tri = (saw < 0) ? (saw + (1 << 23)) : ((1 << 23) - 1 - saw);
      tri = (tri << 1) - (1 << 23);
      return tri;
    }
    case 1u: {  // square
      return (phase < 0x80000000u) ? ((1 << 23) - 1) : -(1 << 23);
    }
    case 3u: {  // sine
      uint16_t phase16 = static_cast<uint16_t>(phase >> 16);
      uint16_t curr_index = phase16 >> (16 - OSC_WAVE_TABLE_SAMPLES_BITS);
      uint16_t next_weight = phase16 & ((1u << (16 - OSC_WAVE_TABLE_SAMPLES_BITS)) - 1u);
      int16_t curr_data = g_osc_sine_wave_table_h1[curr_index + 0u];
      int16_t next_data = g_osc_sine_wave_table_h1[curr_index + 1u];
      int32_t sine = static_cast<int32_t>(
          curr_data + (((next_data - curr_data) * static_cast<int32_t>(next_weight)) >> (16 - OSC_WAVE_TABLE_SAMPLES_BITS)));
      return sine << 8;
    }
    case 2u:
    default: {  // saw
      return static_cast<int32_t>(phase >> 8) - (1 << 23);
    }
  }
}

#if PICO2W_SYNTH_RESTRICT_OSC
constexpr bool kRestrictOsc = true;
#else
constexpr bool kRestrictOsc = false;
#endif

SynthEngine g_runtime_synth;
constexpr uint8_t kSupersawForcedWave = 0;
constexpr uint8_t kSupersawMorphFull = 127;
constexpr uint8_t kSupersawLinearSawMode = 0;
constexpr int16_t kPitchBend14Min = -8192;
constexpr int16_t kPitchBend14Max = 8191;
constexpr int32_t kInfiniteDropBend14PerSecond = 2048;
constexpr int32_t kInfiniteDropStepQ16 =
    (kInfiniteDropBend14PerSecond * 65536 + static_cast<int32_t>(kSampleRate / (2u * kControlRateDiv))) /
    static_cast<int32_t>(kSampleRate / kControlRateDiv);

}  // namespace

int32_t ApplyOptionalReverb(int32_t left, int32_t right, int32_t* right_out);

SynthEngine::SynthEngine() {
  // Initialize default gain and expression
  for (int i = 0; i < 4; ++i) {
    amp_[i].set_gain(100);
    amp_[i].set_expression(127);
  }

  // Filter defaults
  for (int i = 0; i < 4; ++i) {
    filter_[i].set_cutoff(96);
    filter_[i].set_resonance(24);
    filter_[i].set_cutoff_lfo_amt(0, 64);
    filter_[i].set_cutoff_eg_amt(0, 64);
  }

  osc_.set_pitch_bend_range(8);
  osc_.set_portamento<0>(0);
  osc_.set_portamento<1>(0);
  osc_.set_portamento<2>(0);
  osc_.set_portamento<3>(0);
  UpdateAuxOscState();
  ApplyCombinedPitchBend();
}

void SynthEngine::ApplySupersawMode(bool enabled) {
  supersaw_mode_ = enabled;
  if (!supersaw_mode_) {
    return;
  }

  // Supersaw mode is implemented by forcing both osc banks to saw and
  // driving osc1 morph to the 7-voice detuned saw path.
  osc_.set_osc_waveform<0>(kSupersawForcedWave);
  osc_.set_osc_waveform<1>(kSupersawForcedWave);
  osc_.set_saw_wave_mode(kSupersawLinearSawMode);
  osc_.set_osc1_morph_control(kSupersawMorphFull);
}

void SynthEngine::UpdateAuxOscState() {
  uint16_t mix_sum = static_cast<uint16_t>(osc1_level_) + static_cast<uint16_t>(osc2_level_);
  if (mix_sum > 0u) {
    uint16_t mix = static_cast<uint16_t>((static_cast<uint32_t>(osc2_level_) * 127u + (mix_sum / 2u)) / mix_sum);
    osc_pair_mix_ = static_cast<uint8_t>(mix);
    osc_.set_mixer_osc_mix(osc_pair_mix_);
  }
  osc_pair_level_q7_ = static_cast<uint8_t>((mix_sum > 254u) ? 127u : (mix_sum >> 1));
}

void SynthEngine::UpdateAuxOscForVoice(uint8_t voice, uint8_t note) {
  if (voice >= 4u) {
    return;
  }
  float base_hz = MidiNoteToHz(note, static_cast<int16_t>(tune_offset_));
  float detune_norm = (static_cast<int16_t>(osc_detune_) - 64) / 64.0f;
  float spread_cents = 7.0f + (detune_norm * 14.0f);
  float osc3_semitones = static_cast<float>(static_cast<int16_t>(osc3_pitch_) - 64);
  float osc4_semitones = static_cast<float>(static_cast<int16_t>(osc4_pitch_) - 64);
  float osc3_hz = base_hz * std::pow(2.0f, osc3_semitones / 12.0f) * std::pow(2.0f, (-spread_cents / 1200.0f));
  float osc4_hz = base_hz * std::pow(2.0f, osc4_semitones / 12.0f) * std::pow(2.0f, (+spread_cents / 1200.0f));
  osc3_phase_inc_[voice] = FrequencyToPhaseIncrement(osc3_hz);
  osc4_phase_inc_[voice] = FrequencyToPhaseIncrement(osc4_hz);
}

int32_t SynthEngine::RenderAuxOscSample(uint8_t voice,
                                        uint32_t& phase,
                                        uint32_t phase_inc,
                                        uint8_t wave_sel,
                                        uint8_t level_q7) {
  if (voice >= 4u || phase_inc == 0u || level_q7 == 0u) {
    return 0;
  }
  phase += phase_inc;
  int32_t sample = RenderBasicWave24(phase, QuantizeWaveToShape(wave_sel));
  return (sample * static_cast<int32_t>(level_q7)) >> 7;
}

int16_t SynthEngine::MapPitchEnvBendTo14Bit(uint8_t value) {
  int16_t centered = static_cast<int16_t>(static_cast<int16_t>(value) - 64);
  if (centered == 0) {
    return 0;
  }
  if (centered > 0) {
    int32_t scaled = (static_cast<int32_t>(centered) * static_cast<int32_t>(kPitchBend14Max) + 31) / 63;
    return static_cast<int16_t>(scaled);
  }
  int32_t scaled = (static_cast<int32_t>(-centered) * static_cast<int32_t>(-kPitchBend14Min) + 32) / 64;
  return static_cast<int16_t>(-scaled);
}

uint32_t SynthEngine::MapPitchEnvTimeToControlSteps(uint8_t value) {
  if (value == 0u) {
    return 0u;
  }
  constexpr uint32_t kMaxSweepMs = 2500u;
  constexpr uint32_t kSquareMax = 127u * 127u;
  uint64_t max_steps = (static_cast<uint64_t>(kSampleRate) * static_cast<uint64_t>(kMaxSweepMs)) /
                       (1000ull * static_cast<uint64_t>(kControlRateDiv));
  uint32_t squared = static_cast<uint32_t>(value) * static_cast<uint32_t>(value);
  uint64_t scaled = static_cast<uint64_t>(squared) * max_steps;
  uint32_t steps = static_cast<uint32_t>((scaled + (static_cast<uint64_t>(kSquareMax) / 2ull)) /
                                         static_cast<uint64_t>(kSquareMax));
  if (steps == 0u) {
    steps = 1u;
  }
  return steps;
}

void SynthEngine::ApplyCombinedPitchBend() {
  int32_t env14 = pitch_env_current_q16_ >> 16;
  if (pitch_env_current_q16_ > 0) {
    env14 = (pitch_env_current_q16_ + 0x00008000) >> 16;
  } else if (pitch_env_current_q16_ < 0) {
    env14 = (pitch_env_current_q16_ - 0x00008000) >> 16;
  }
  int32_t drop14 = infinite_drop_q16_ >> 16;
  if (infinite_drop_q16_ > 0) {
    drop14 = (infinite_drop_q16_ + 0x00008000) >> 16;
  } else if (infinite_drop_q16_ < 0) {
    drop14 = (infinite_drop_q16_ - 0x00008000) >> 16;
  }
  int32_t combined = static_cast<int32_t>(midi_pitch_bend_) + env14 + drop14;
  if (combined < static_cast<int32_t>(kPitchBend14Min)) {
    combined = static_cast<int32_t>(kPitchBend14Min);
  } else if (combined > static_cast<int32_t>(kPitchBend14Max)) {
    combined = static_cast<int32_t>(kPitchBend14Max);
  }
  osc_.set_pitch_bend(static_cast<int16_t>(combined));
}

void SynthEngine::StartPitchEnvelope() {
  int16_t start_bend = MapPitchEnvBendTo14Bit(pitch_env_bend_);
  uint32_t control_steps = MapPitchEnvTimeToControlSteps(pitch_env_time_);
  if (start_bend == 0 || control_steps == 0u) {
    pitch_env_current_q16_ = 0;
    pitch_env_step_q16_ = 0u;
    pitch_env_active_ = false;
    ApplyCombinedPitchBend();
    return;
  }

  pitch_env_current_q16_ = static_cast<int32_t>(start_bend) << 16;
  int32_t magnitude_q16_signed = (pitch_env_current_q16_ >= 0) ? pitch_env_current_q16_ : -pitch_env_current_q16_;
  uint32_t magnitude_q16 = static_cast<uint32_t>(magnitude_q16_signed);
  pitch_env_step_q16_ = (magnitude_q16 + control_steps - 1u) / control_steps;
  if (pitch_env_step_q16_ == 0u) {
    pitch_env_step_q16_ = 1u;
  }
  pitch_env_active_ = true;
  ApplyCombinedPitchBend();
}

void SynthEngine::UpdatePitchEnvelopeOnControlTick() {
  bool changed = false;
  if (pitch_env_active_) {
    int32_t step_q16 = static_cast<int32_t>(pitch_env_step_q16_);
    if (pitch_env_current_q16_ > 0) {
      pitch_env_current_q16_ -= step_q16;
      if (pitch_env_current_q16_ <= 0) {
        pitch_env_current_q16_ = 0;
        pitch_env_active_ = false;
        pitch_env_step_q16_ = 0u;
      }
    } else if (pitch_env_current_q16_ < 0) {
      pitch_env_current_q16_ += step_q16;
      if (pitch_env_current_q16_ >= 0) {
        pitch_env_current_q16_ = 0;
        pitch_env_active_ = false;
        pitch_env_step_q16_ = 0u;
      }
    } else {
      pitch_env_active_ = false;
      pitch_env_step_q16_ = 0u;
    }
    changed = true;
  }

  if (infinite_drop_enabled_) {
    int32_t next = infinite_drop_q16_ - kInfiniteDropStepQ16;
    int32_t min_q16 = static_cast<int32_t>(kPitchBend14Min) << 16;
    if (next < min_q16) {
      next = min_q16;
    }
    if (next != infinite_drop_q16_) {
      infinite_drop_q16_ = next;
      changed = true;
    }
  } else if (infinite_drop_q16_ != 0) {
    infinite_drop_q16_ = 0;
    changed = true;
  }

  if (changed) {
    ApplyCombinedPitchBend();
  }
}

void SynthEngine::HandleNoteOn(uint8_t channel, uint8_t note, uint8_t velocity) {
  static_cast<void>(channel);
  if (infinite_drop_enabled_) {
    infinite_drop_q16_ = 0;
  }

  uint8_t voice = monophonic_ ? 0 : FindVoiceForNoteOn();
  if (!monophonic_) {
    round_robin_voice_ = static_cast<uint8_t>((voice + 1u) & 0x03u);
  }

  switch (voice) {
    case 0:
      NoteOnVoice<0>(note, velocity);
      break;
    case 1:
      NoteOnVoice<1>(note, velocity);
      break;
    case 2:
      NoteOnVoice<2>(note, velocity);
      break;
    default:
      NoteOnVoice<3>(note, velocity);
      break;
  }

  StartPitchEnvelope();
  lfo_.trigger_lfo();
}

void SynthEngine::HandleNoteOff(uint8_t channel, uint8_t note, uint8_t velocity) {
  static_cast<void>(channel);
  static_cast<void>(velocity);

  // Release exactly one matching voice for this note-off event.
  // Releasing all matching voices causes audible dropouts when the same note
  // is retriggered quickly and multiple instances overlap.
  for (int v = 3; v >= 0; --v) {
    if (voice_note_[v] == note) {
      switch (v) {
        case 0:
          NoteOffVoice<0>();
          break;
        case 1:
          NoteOffVoice<1>();
          break;
        case 2:
          NoteOffVoice<2>();
          break;
        default:
          NoteOffVoice<3>();
          break;
      }
      break;
    }
  }
}

void SynthEngine::HandlePitchBend(uint8_t channel, uint16_t bend_14bit) {
  static_cast<void>(channel);
  int32_t signed_bend = static_cast<int32_t>(bend_14bit) - 8192;
  if (signed_bend < static_cast<int32_t>(kPitchBend14Min)) {
    signed_bend = static_cast<int32_t>(kPitchBend14Min);
  } else if (signed_bend > static_cast<int32_t>(kPitchBend14Max)) {
    signed_bend = static_cast<int32_t>(kPitchBend14Max);
  }
  midi_pitch_bend_ = static_cast<int16_t>(signed_bend);
  ApplyCombinedPitchBend();
}

void SynthEngine::HandleChannelPressure(uint8_t channel, uint8_t pressure) {
  static_cast<void>(channel);
  lfo_.set_pressure<0>(pressure);
  lfo_.set_pressure<1>(pressure);
  lfo_.set_pressure<2>(pressure);
  lfo_.set_pressure<3>(pressure);
}

void SynthEngine::ApplyParam(ParamId id, uint8_t value) {
  auto refresh_aux_for_active_voices = [this]() {
    for (uint8_t v = 0; v < 4u; ++v) {
      if (voice_pressed_[v]) {
        UpdateAuxOscForVoice(v, voice_note_[v]);
      }
    }
  };
  switch (id) {
    case ParamId::Osc1Wave:
      if (!supersaw_mode_) {
        osc_.set_osc_waveform<0>(value);
      }
      break;
    case ParamId::Osc2Wave:
      if (!supersaw_mode_) {
        osc_.set_osc_waveform<1>(value);
      }
      break;
    case ParamId::OscMix:
      osc_pair_mix_ = value;
      osc_.set_mixer_osc_mix(value);
      break;
    case ParamId::Osc1Shape:
      if (!supersaw_mode_) {
        osc_.set_osc_waveform<0>(MapOscShapeSelectorToWaveControl(value));
      }
      break;
    case ParamId::Osc1Morph:
      osc_.set_osc1_morph_control(value);
      break;
    case ParamId::SubNoiseMix:
      osc_.set_mixer_sub_osc_control(RemapSubNoiseForPresence(value));
      break;
    case ParamId::Osc2Coarse:
      osc_.set_osc2_pitch(value);
      break;
    case ParamId::Osc2Detune:
      osc_detune_ = value;
      osc_.set_osc2_detune(value);
      refresh_aux_for_active_voices();
      break;
    case ParamId::OscDrift:
      osc_.set_drift(0);
      osc_.set_tune_offset(value);
      tune_offset_ = value;
      refresh_aux_for_active_voices();
      break;
    case ParamId::Osc1Freq:
      osc_.set_tune_offset(value);
      tune_offset_ = value;
      refresh_aux_for_active_voices();
      break;
    case ParamId::Osc3Freq:
      osc3_pitch_ = value;
      refresh_aux_for_active_voices();
      break;
    case ParamId::Osc4Freq:
      osc4_pitch_ = value;
      refresh_aux_for_active_voices();
      break;
    case ParamId::Portamento:
      osc_.set_portamento<0>(value);
      osc_.set_portamento<1>(value);
      osc_.set_portamento<2>(value);
      osc_.set_portamento<3>(value);
      break;

    case ParamId::FilterCutoff:
      filter_[0].set_cutoff(value);
      filter_[1].set_cutoff(value);
      filter_[2].set_cutoff(value);
      filter_[3].set_cutoff(value);
      break;
    case ParamId::FilterReso:
      filter_[0].set_resonance(value);
      filter_[1].set_resonance(value);
      filter_[2].set_resonance(value);
      filter_[3].set_resonance(value);
      break;
    case ParamId::FilterMode:
      filter_[0].set_filter_mode(value);
      filter_[1].set_filter_mode(value);
      filter_[2].set_filter_mode(value);
      filter_[3].set_filter_mode(value);
      // Preserve filter state across LP/HP mode changes.
      // Resetting on mode switch can produce a one-shot emphasis transient
      // ("resonance on first note only") instead of stable behavior.
      break;
    case ParamId::FilterEgAmt:
      filter_[0].set_cutoff_eg_amt(0, value);
      filter_[1].set_cutoff_eg_amt(0, value);
      filter_[2].set_cutoff_eg_amt(0, value);
      filter_[3].set_cutoff_eg_amt(0, value);
      break;
    case ParamId::FilterLfoAmt:
      filter_[0].set_cutoff_lfo_amt(0, value);
      filter_[1].set_cutoff_lfo_amt(0, value);
      filter_[2].set_cutoff_lfo_amt(0, value);
      filter_[3].set_cutoff_lfo_amt(0, value);
      break;
    case ParamId::FilterKeyTrack:
      filter_[0].set_cutoff_pitch_amt(value);
      filter_[1].set_cutoff_pitch_amt(value);
      filter_[2].set_cutoff_pitch_amt(value);
      filter_[3].set_cutoff_pitch_amt(value);
      break;

    case ParamId::EgAttack:
      eg_[0].set_attack(value);
      eg_[2].set_attack(value);
      eg_[4].set_attack(value);
      eg_[6].set_attack(value);
      break;
    case ParamId::EgDecay:
      eg_[0].set_decay(value);
      eg_[2].set_decay(value);
      eg_[4].set_decay(value);
      eg_[6].set_decay(value);
      break;
    case ParamId::EgSustain:
      eg_[0].set_sustain(value);
      eg_[2].set_sustain(value);
      eg_[4].set_sustain(value);
      eg_[6].set_sustain(value);
      break;
    case ParamId::EgRelease:
      eg_[0].set_release(value);
      eg_[2].set_release(value);
      eg_[4].set_release(value);
      eg_[6].set_release(value);
      break;

    case ParamId::AmpAttack:
      eg_[1].set_attack(value);
      eg_[3].set_attack(value);
      eg_[5].set_attack(value);
      eg_[7].set_attack(value);
      break;
    case ParamId::AmpDecay:
      eg_[1].set_decay(value);
      eg_[3].set_decay(value);
      eg_[5].set_decay(value);
      eg_[7].set_decay(value);
      break;
    case ParamId::AmpSustain:
      eg_[1].set_sustain(value);
      eg_[3].set_sustain(value);
      eg_[5].set_sustain(value);
      eg_[7].set_sustain(value);
      break;
    case ParamId::AmpRelease:
      eg_[1].set_release(value);
      eg_[3].set_release(value);
      eg_[5].set_release(value);
      eg_[7].set_release(value);
      break;
    case ParamId::PitchEnvBend:
      pitch_env_bend_ = value;
      break;
    case ParamId::PitchEnvTime:
      pitch_env_time_ = value;
      break;
    case ParamId::InfiniteDrop:
      infinite_drop_enabled_ = (value >= 64u);
      if (!infinite_drop_enabled_) {
        infinite_drop_q16_ = 0;
      } else {
        infinite_drop_q16_ = 0;
      }
      ApplyCombinedPitchBend();
      break;

    case ParamId::LfoWave:
      lfo_.set_lfo_waveform(value);
      break;
    case ParamId::LfoRate:
      lfo_.set_lfo_rate_extended(RemapLfoRateFromUi(value), value);
      break;
    case ParamId::LfoDepth:
      lfo_.set_lfo_depth<0>(value);
      break;
    case ParamId::LfoFade:
      lfo_.set_lfo_fade_time(value);
      break;
    case ParamId::LfoOscAmt:
      osc_.set_pitch_lfo_amt<0>(value);
      osc_.set_pitch_lfo_amt<1>(value);
      break;
    case ParamId::LfoOscDst:
#if PICO2W_SYNTH_ENABLE_EXPENSIVE_MOD_DEST
      if (value >= 96) {
        osc_.set_shape_lfo_amt(127);
      } else {
        osc_.set_shape_lfo_amt(64);
      }
#else
      osc_.set_shape_lfo_amt(64);
#endif
      break;
    case ParamId::EgOscAmt:
      osc_.set_pitch_eg_amt<0>(value);
      osc_.set_pitch_eg_amt<1>(value);
      break;
    case ParamId::EgOscDst:
#if PICO2W_SYNTH_ENABLE_EXPENSIVE_MOD_DEST
      if (value >= 96) {
        osc_.set_shape_eg_amt(127);
      } else {
        osc_.set_shape_eg_amt(64);
      }
#else
      osc_.set_shape_eg_amt(64);
#endif
      break;

    case ParamId::ChorusMix:
#if PICO2W_SYNTH_ENABLE_CHORUS
      chorus_fx_.set_chorus_level(value);
#else
      static_cast<void>(value);
#endif
      break;
    case ParamId::ChorusRate:
#if PICO2W_SYNTH_ENABLE_CHORUS
      chorus_fx_.set_chorus_rate(value);
#else
      static_cast<void>(value);
#endif
      break;
    case ParamId::ChorusDepth:
#if PICO2W_SYNTH_ENABLE_CHORUS
      chorus_fx_.set_chorus_depth(value);
#else
      static_cast<void>(value);
#endif
      break;
    case ParamId::ChorusDelay:
#if PICO2W_SYNTH_ENABLE_CHORUS
      chorus_fx_.set_chorus_delay_time(value);
#else
      static_cast<void>(value);
#endif
      break;
    case ParamId::DelayMix:
#if PICO2W_SYNTH_ENABLE_DELAY
      delay_fx_.set_delay_level(value);
#else
      static_cast<void>(value);
#endif
      break;
    case ParamId::DelayTime:
#if PICO2W_SYNTH_ENABLE_DELAY
      delay_fx_.set_delay_time(value);
#else
      static_cast<void>(value);
#endif
      break;
    case ParamId::DelayFeedback:
#if PICO2W_SYNTH_ENABLE_DELAY
      delay_fx_.set_delay_feedback(value);
#else
      static_cast<void>(value);
#endif
      break;
    case ParamId::DelayMode:
#if PICO2W_SYNTH_ENABLE_DELAY
      delay_fx_.set_delay_mode(value);
#else
      static_cast<void>(value);
#endif
      break;

    case ParamId::AmpGain:
      amp_[0].set_gain(value);
      amp_[1].set_gain(value);
      amp_[2].set_gain(value);
      amp_[3].set_gain(value);
      break;
    case ParamId::Pan:
      panner_.set_pan(value);
      break;

    case ParamId::VoiceMode:
      monophonic_ = (value >= 64);
      break;
    case ParamId::PitchBendRange:
      {
        uint32_t scaled = static_cast<uint32_t>(value) * 4u;
        if (scaled > 127u) {
          scaled = 127u;
        }
        osc_.set_pitch_bend_range(static_cast<uint8_t>(scaled));
      }
      break;
    case ParamId::SystemBypassFx:
      bypass_fx_ = (value >= 64);
      break;
    case ParamId::OscSupersaw:
      ApplySupersawMode(value >= 64);
      break;
    case ParamId::Osc3Wave:
      osc3_wave_ = value;
      break;
    case ParamId::Osc4Wave:
      osc4_wave_ = value;
      break;
    case ParamId::Osc1Level:
      osc1_level_ = value;
      UpdateAuxOscState();
      break;
    case ParamId::Osc2Level:
      osc2_level_ = value;
      UpdateAuxOscState();
      break;
    case ParamId::Osc3Level:
      osc3_level_ = value;
      break;
    case ParamId::Osc4Level:
      osc4_level_ = value;
      break;
    case ParamId::ReverbMix:
    case ParamId::ParamCount:
      break;
  }
}

void SynthEngine::ProcessFrame(int16_t* left_out, int16_t* right_out) {
  ++sample_count_;
  int16_t noise_int15 = noise_gen_.process();

  if ((sample_count_ & (kControlRateDiv - 1u)) == 0) {
    uint32_t c = sample_count_ / kControlRateDiv;
    UpdatePitchEnvelopeOnControlTick();
    uint8_t ph = static_cast<uint8_t>(c & 0x03u);

    lfo_.process_at_low_rate(static_cast<uint8_t>(c), noise_int15);

    if (ph == 0) {
      eg_[0].process_at_low_rate();
      eg_[1].process_at_low_rate();
      int16_t l = ScaleLfoOutput3x(lfo_.get_output<0>());
      osc_.process_at_low_rate<0>(c, l, eg_[0].get_output(), noise_int15);
      filter_[0].process_at_low_rate(c, eg_[0].get_output(), l, osc_.get_osc_pitch(0));
      amp_[0].process_at_low_rate(eg_[1].get_output());
    } else if (ph == 1) {
      eg_[2].process_at_low_rate();
      eg_[3].process_at_low_rate();
      int16_t l = ScaleLfoOutput3x(lfo_.get_output<1>());
      osc_.process_at_low_rate<1>(c, l, eg_[2].get_output(), noise_int15);
      filter_[1].process_at_low_rate(c, eg_[2].get_output(), l, osc_.get_osc_pitch(1));
      amp_[1].process_at_low_rate(eg_[3].get_output());
      panner_.process_at_low_rate();
      osc_.process_at_low_rate_global();
    } else if (ph == 2) {
      eg_[4].process_at_low_rate();
      eg_[5].process_at_low_rate();
      int16_t l = ScaleLfoOutput3x(lfo_.get_output<2>());
      osc_.process_at_low_rate<2>(c, l, eg_[4].get_output(), noise_int15);
      filter_[2].process_at_low_rate(c, eg_[4].get_output(), l, osc_.get_osc_pitch(2));
      amp_[2].process_at_low_rate(eg_[5].get_output());
      if (!bypass_fx_) {
#if PICO2W_SYNTH_ENABLE_CHORUS
        chorus_fx_.process_at_low_rate(static_cast<uint8_t>(c));
#endif
      }
    } else {
      eg_[6].process_at_low_rate();
      eg_[7].process_at_low_rate();
      int16_t l = ScaleLfoOutput3x(lfo_.get_output<3>());
      osc_.process_at_low_rate<3>(c, l, eg_[6].get_output(), noise_int15);
      filter_[3].process_at_low_rate(c, eg_[6].get_output(), l, osc_.get_osc_pitch(3));
      amp_[3].process_at_low_rate(eg_[7].get_output());
      if (!bypass_fx_) {
#if PICO2W_SYNTH_ENABLE_DELAY
        delay_fx_.process_at_low_rate(static_cast<uint8_t>(c));
#endif
      }
    }
  }

  int32_t osc_out0 = osc_.process<0, kRestrictOsc>(noise_int15);
  int32_t osc_out1 = osc_.process<1, kRestrictOsc>(noise_int15);
  int32_t osc_out2 = osc_.process<2, kRestrictOsc>(noise_int15);
  int32_t osc_out3 = osc_.process<3, kRestrictOsc>(noise_int15);

  osc_out0 = (osc_out0 * static_cast<int32_t>(osc_pair_level_q7_)) >> 7;
  osc_out1 = (osc_out1 * static_cast<int32_t>(osc_pair_level_q7_)) >> 7;
  osc_out2 = (osc_out2 * static_cast<int32_t>(osc_pair_level_q7_)) >> 7;
  osc_out3 = (osc_out3 * static_cast<int32_t>(osc_pair_level_q7_)) >> 7;

  osc_out0 += RenderAuxOscSample(0u, osc3_phase_[0], osc3_phase_inc_[0], osc3_wave_, osc3_level_);
  osc_out0 += RenderAuxOscSample(0u, osc4_phase_[0], osc4_phase_inc_[0], osc4_wave_, osc4_level_);
  osc_out1 += RenderAuxOscSample(1u, osc3_phase_[1], osc3_phase_inc_[1], osc3_wave_, osc3_level_);
  osc_out1 += RenderAuxOscSample(1u, osc4_phase_[1], osc4_phase_inc_[1], osc4_wave_, osc4_level_);
  osc_out2 += RenderAuxOscSample(2u, osc3_phase_[2], osc3_phase_inc_[2], osc3_wave_, osc3_level_);
  osc_out2 += RenderAuxOscSample(2u, osc4_phase_[2], osc4_phase_inc_[2], osc4_wave_, osc4_level_);
  osc_out3 += RenderAuxOscSample(3u, osc3_phase_[3], osc3_phase_inc_[3], osc3_wave_, osc3_level_);
  osc_out3 += RenderAuxOscSample(3u, osc4_phase_[3], osc4_phase_inc_[3], osc4_wave_, osc4_level_);

  int32_t sig0 = amp_[0].process(filter_[0].process(osc_out0));
  int32_t sig1 = amp_[1].process(filter_[1].process(osc_out1));
  int32_t sig2 = amp_[2].process(filter_[2].process(osc_out2));
  int32_t sig3 = amp_[3].process(filter_[3].process(osc_out3));

  int32_t voice_mix = sig0 + sig1 + sig2 + sig3;

  int32_t pan_r = 0;
  int32_t pan_l = panner_.process(voice_mix, pan_r);

  int32_t fx_l = pan_l;
  int32_t fx_r = pan_r;

  if (!bypass_fx_) {
#if PICO2W_SYNTH_ENABLE_CHORUS
    int32_t chorus_r = 0;
    int32_t chorus_l = chorus_fx_.process(fx_l, fx_r, chorus_r);
    fx_l = chorus_l;
    fx_r = chorus_r;
#endif
#if PICO2W_SYNTH_ENABLE_DELAY
    int32_t delay_r = 0;
    fx_l = delay_fx_.process(fx_l, fx_r, delay_r);
    fx_r = delay_r;
#endif
#if PICO2W_SYNTH_ENABLE_REVERB
    int32_t rv_r = 0;
    fx_l = ApplyOptionalReverb(fx_l, fx_r, &rv_r);
    fx_r = rv_r;
#endif
  }

  *left_out = Clamp24ToInt16(ApplyMasterBoost10dB(fx_l));
  *right_out = Clamp24ToInt16(ApplyMasterBoost10dB(fx_r));
}

template <uint8_t VoiceIndex>
void SynthEngine::NoteOnVoice(uint8_t note, uint8_t velocity) {
  osc_.note_on<VoiceIndex>(note);
  eg_[VoiceIndex * 2].note_on(velocity);
  eg_[VoiceIndex * 2 + 1].note_on(velocity);
  voice_note_[VoiceIndex] = note;
  voice_pressed_[VoiceIndex] = true;
  osc3_phase_[VoiceIndex] = 0u;
  osc4_phase_[VoiceIndex] = 0u;
  UpdateAuxOscForVoice(VoiceIndex, note);
}

template <uint8_t VoiceIndex>
void SynthEngine::NoteOffVoice() {
  osc_.note_off<VoiceIndex>();
  eg_[VoiceIndex * 2].note_off();
  eg_[VoiceIndex * 2 + 1].note_off();
  voice_note_[VoiceIndex] = NOTE_NUMBER_INVALID;
  voice_pressed_[VoiceIndex] = false;
}

uint8_t SynthEngine::FindVoiceForNoteOn() const {
  for (uint8_t i = 0; i < 4; ++i) {
    if (!voice_pressed_[i]) {
      return i;
    }
  }

  // Voice stealing fallback: deterministic round-robin.
  return round_robin_voice_;
}

void SynthRuntimeHandleNoteOn(uint8_t channel, uint8_t note, uint8_t velocity) {
  g_runtime_synth.HandleNoteOn(channel, note, velocity);
}

void SynthRuntimeHandleNoteOff(uint8_t channel, uint8_t note, uint8_t velocity) {
  g_runtime_synth.HandleNoteOff(channel, note, velocity);
}

void SynthRuntimeHandlePitchBend(uint8_t channel, uint16_t bend_14bit) {
  g_runtime_synth.HandlePitchBend(channel, bend_14bit);
}

void SynthRuntimeHandleChannelPressure(uint8_t channel, uint8_t pressure) {
  g_runtime_synth.HandleChannelPressure(channel, pressure);
}

void SynthRuntimeApplyParam(ParamId id, uint8_t value) {
  g_runtime_synth.ApplyParam(id, value);
}

void SynthRuntimeProcessFrame(int16_t* left_out, int16_t* right_out) {
  g_runtime_synth.ProcessFrame(left_out, right_out);
}

}  // namespace pico2w_synth
