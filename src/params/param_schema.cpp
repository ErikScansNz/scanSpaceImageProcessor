#include "params/param_schema.h"

#include "pico2w_synth/config.h"

namespace pico2w_synth {

namespace {

constexpr uint8_t kDefaultBypassFxValue = (PICO2W_SYNTH_DEFAULT_BYPASS_FX != 0) ? 127u : 0u;
constexpr uint8_t kOscTonalWaveMaxValue = 88u;
constexpr uint8_t kOsc2CoarseMinValue = 20u;
constexpr uint8_t kOsc2CoarseMaxValue = 110u;
constexpr uint8_t kPitchBendRangeMaxValue = 31u;
constexpr uint8_t kDelayTimeMaxValue = 112u;
constexpr uint8_t kChorusDepthMaxValue = 126u;

}  // namespace

static constexpr ParamMeta kParamSchema[] = {
    {ParamId::Osc1Wave, "osc1_wave", ParamPage::Osc, 0, 0, kOscTonalWaveMaxValue, SmoothingPolicy::None},
    {ParamId::Osc2Wave, "osc2_wave", ParamPage::Osc, 0, 0, kOscTonalWaveMaxValue, SmoothingPolicy::None},
    {ParamId::OscMix, "osc_mix", ParamPage::Osc, 64, 0, 127, SmoothingPolicy::Slow},
    {ParamId::Osc1Shape, "osc1_shape", ParamPage::Osc, 84, 0, 127, SmoothingPolicy::Fast},
    {ParamId::Osc1Morph, "osc1_morph", ParamPage::Osc, 0, 0, 127, SmoothingPolicy::Fast},
    {ParamId::SubNoiseMix, "sub_noise_mix", ParamPage::Osc, 64, 0, 127, SmoothingPolicy::Fast},
    {ParamId::Osc2Coarse, "osc2_coarse", ParamPage::Osc, 64, kOsc2CoarseMinValue, kOsc2CoarseMaxValue, SmoothingPolicy::None},
    {ParamId::Osc2Detune, "osc2_detune", ParamPage::Osc, 64, 0, 127, SmoothingPolicy::Fast},
    {ParamId::OscDrift, "osc_drift", ParamPage::Osc, 64, 0, 127, SmoothingPolicy::Slow},
    {ParamId::Portamento, "portamento", ParamPage::Osc, 0, 0, 127, SmoothingPolicy::Slow},

    {ParamId::FilterCutoff, "filter_cutoff", ParamPage::Filter, 96, 0, 127, SmoothingPolicy::Fast},
    {ParamId::FilterReso, "filter_reso", ParamPage::Filter, 24, 0, 127, SmoothingPolicy::Fast},
    {ParamId::FilterMode, "filter_mode", ParamPage::Filter, 0, 0, 127, SmoothingPolicy::None},
    {ParamId::FilterEgAmt, "filter_eg_amt", ParamPage::Filter, 64, 0, 127, SmoothingPolicy::Fast},
    {ParamId::FilterLfoAmt, "filter_lfo_amt", ParamPage::Filter, 64, 0, 127, SmoothingPolicy::Fast},
    {ParamId::FilterKeyTrack, "filter_key_track", ParamPage::Filter, 64, 0, 127, SmoothingPolicy::Fast},

    {ParamId::EgAttack, "eg_attack", ParamPage::Env, 16, 0, 127, SmoothingPolicy::Fast},
    {ParamId::EgDecay, "eg_decay", ParamPage::Env, 48, 0, 127, SmoothingPolicy::Fast},
    {ParamId::EgSustain, "eg_sustain", ParamPage::Env, 96, 0, 127, SmoothingPolicy::Fast},
    {ParamId::EgRelease, "eg_release", ParamPage::Env, 48, 0, 127, SmoothingPolicy::Fast},
    {ParamId::AmpAttack, "amp_attack", ParamPage::Env, 0, 0, 127, SmoothingPolicy::Fast},
    {ParamId::AmpDecay, "amp_decay", ParamPage::Env, 48, 0, 127, SmoothingPolicy::Fast},
    {ParamId::AmpSustain, "amp_sustain", ParamPage::Env, 127, 0, 127, SmoothingPolicy::Fast},
    {ParamId::AmpRelease, "amp_release", ParamPage::Env, 48, 0, 127, SmoothingPolicy::Fast},
    {ParamId::PitchEnvBend, "pitch_env_bend", ParamPage::Env, 64, 0, 127, SmoothingPolicy::None},
    {ParamId::PitchEnvTime, "pitch_env_time", ParamPage::Env, 0, 0, 127, SmoothingPolicy::Fast},

    {ParamId::LfoWave, "lfo_wave", ParamPage::Lfo, 0, 0, 127, SmoothingPolicy::None},
    {ParamId::LfoRate, "lfo_rate", ParamPage::Lfo, 40, 0, 127, SmoothingPolicy::Fast},
    {ParamId::LfoDepth, "lfo_depth", ParamPage::Lfo, 0, 0, 127, SmoothingPolicy::Fast},
    {ParamId::LfoFade, "lfo_fade", ParamPage::Lfo, 0, 0, 127, SmoothingPolicy::Fast},
    {ParamId::LfoOscAmt, "lfo_osc_amt", ParamPage::Lfo, 64, 0, 127, SmoothingPolicy::Fast},
    {ParamId::LfoOscDst, "lfo_osc_dst", ParamPage::Lfo, 0, 0, 127, SmoothingPolicy::None},
    {ParamId::EgOscAmt, "eg_osc_amt", ParamPage::Lfo, 64, 0, 127, SmoothingPolicy::Fast},
    {ParamId::EgOscDst, "eg_osc_dst", ParamPage::Lfo, 0, 0, 127, SmoothingPolicy::None},

    {ParamId::ChorusMix, "chorus_mix", ParamPage::Fx, 0, 0, 127, SmoothingPolicy::Fast},
    {ParamId::ChorusRate, "chorus_rate", ParamPage::Fx, 64, 0, 127, SmoothingPolicy::Fast},
    {ParamId::ChorusDepth, "chorus_depth", ParamPage::Fx, 64, 0, kChorusDepthMaxValue, SmoothingPolicy::Fast},
    {ParamId::ChorusDelay, "chorus_delay", ParamPage::Fx, 64, 0, 127, SmoothingPolicy::Fast},
    {ParamId::DelayMix, "delay_mix", ParamPage::Fx, 0, 0, 127, SmoothingPolicy::Fast},
    {ParamId::DelayTime, "delay_time", ParamPage::Fx, 93, 0, kDelayTimeMaxValue, SmoothingPolicy::Fast},
    {ParamId::DelayFeedback, "delay_feedback", ParamPage::Fx, 64, 0, 127, SmoothingPolicy::Fast},
    {ParamId::DelayMode, "delay_mode", ParamPage::Fx, 0, 0, 127, SmoothingPolicy::None},
    {ParamId::ReverbMix, "reverb_mix", ParamPage::Fx, 0, 0, 127, SmoothingPolicy::Fast},

    {ParamId::AmpGain, "amp_gain", ParamPage::Amp, 100, 0, 127, SmoothingPolicy::Fast},
    {ParamId::Pan, "pan", ParamPage::Amp, 64, 0, 127, SmoothingPolicy::Fast},

    {ParamId::VoiceMode, "voice_mode", ParamPage::System, 0, 0, 127, SmoothingPolicy::None},
    {ParamId::PitchBendRange, "pb_range", ParamPage::System, 2, 0, kPitchBendRangeMaxValue, SmoothingPolicy::None},
    {ParamId::SystemBypassFx, "bypass_fx", ParamPage::System, kDefaultBypassFxValue, 0, 127, SmoothingPolicy::None},
    {ParamId::InfiniteDrop, "infinite_drop", ParamPage::System, 0, 0, 127, SmoothingPolicy::None},
    {ParamId::OscSupersaw, "osc_supersaw", ParamPage::System, 0, 0, 127, SmoothingPolicy::None},
    {ParamId::Osc3Wave, "osc3_wave", ParamPage::Osc, 64, 0, kOscTonalWaveMaxValue, SmoothingPolicy::None},
    {ParamId::Osc4Wave, "osc4_wave", ParamPage::Osc, 84, 0, kOscTonalWaveMaxValue, SmoothingPolicy::None},
    {ParamId::Osc1Level, "osc1_level", ParamPage::Osc, 127, 0, 127, SmoothingPolicy::Fast},
    {ParamId::Osc2Level, "osc2_level", ParamPage::Osc, 127, 0, 127, SmoothingPolicy::Fast},
    {ParamId::Osc3Level, "osc3_level", ParamPage::Osc, 0, 0, 127, SmoothingPolicy::Fast},
    {ParamId::Osc4Level, "osc4_level", ParamPage::Osc, 0, 0, 127, SmoothingPolicy::Fast},
    {ParamId::Osc1Freq, "osc1_freq", ParamPage::Osc, 64, kOsc2CoarseMinValue, kOsc2CoarseMaxValue, SmoothingPolicy::None},
    {ParamId::Osc3Freq, "osc3_freq", ParamPage::Osc, 64, kOsc2CoarseMinValue, kOsc2CoarseMaxValue, SmoothingPolicy::None},
    {ParamId::Osc4Freq, "osc4_freq", ParamPage::Osc, 64, kOsc2CoarseMinValue, kOsc2CoarseMaxValue, SmoothingPolicy::None},
};

const ParamMeta* GetParamSchema() { return kParamSchema; }

size_t GetParamSchemaCount() { return sizeof(kParamSchema) / sizeof(kParamSchema[0]); }

const ParamMeta* FindParamMeta(ParamId id) {
  for (size_t i = 0; i < GetParamSchemaCount(); ++i) {
    if (kParamSchema[i].id == id) {
      return &kParamSchema[i];
    }
  }
  return nullptr;
}

uint8_t ClampParamValue(ParamId id, uint8_t value) {
  const ParamMeta* meta = FindParamMeta(id);
  uint8_t min_value = meta ? meta->min_value : kParamMinValue;
  uint8_t max_value = meta ? meta->max_value : kParamMaxValue;
  if (min_value > max_value) {
    min_value = kParamMinValue;
    max_value = kParamMaxValue;
  }

  if (value < min_value) {
    value = min_value;
  } else if (value > max_value) {
    value = max_value;
  }

  switch (id) {
    case ParamId::Osc1Shape:
      if (value < 32u) {
        value = 0u;
      } else if (value < 64u) {
        value = 42u;
      } else if (value < 96u) {
        value = 84u;
      } else {
        value = 127u;
      }
      break;

    case ParamId::VoiceMode:
    case ParamId::SystemBypassFx:
    case ParamId::OscSupersaw:
    case ParamId::InfiniteDrop:
      value = (value >= 64u) ? 127u : 0u;
      break;

    default:
      break;
  }

  return value;
}

}  // namespace pico2w_synth
