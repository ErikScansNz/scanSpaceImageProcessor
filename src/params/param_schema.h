#pragma once

#include <cstdint>
#include <cstddef>

namespace pico2w_synth {

enum class ParamPage : uint8_t {
  Osc = 0,
  Filter,
  Env,
  Lfo,
  Fx,
  Amp,
  System,
  Count
};

enum class SmoothingPolicy : uint8_t {
  None = 0,
  Slow,
  Fast,
};

enum class ParamId : uint16_t {
  Osc1Wave = 0,
  Osc2Wave,
  OscMix,
  Osc1Shape,
  Osc1Morph,
  SubNoiseMix,
  Osc2Coarse,
  Osc2Detune,
  OscDrift,
  Portamento,

  FilterCutoff,
  FilterReso,
  FilterMode,
  FilterEgAmt,
  FilterLfoAmt,
  FilterKeyTrack,

  EgAttack,
  EgDecay,
  EgSustain,
  EgRelease,
  AmpAttack,
  AmpDecay,
  AmpSustain,
  AmpRelease,

  LfoWave,
  LfoRate,
  LfoDepth,
  LfoFade,
  LfoOscAmt,
  LfoOscDst,
  EgOscAmt,
  EgOscDst,

  ChorusMix,
  ChorusRate,
  ChorusDepth,
  ChorusDelay,
  DelayMix,
  DelayTime,
  DelayFeedback,
  DelayMode,
  ReverbMix,

  AmpGain,
  Pan,

  VoiceMode,
  PitchBendRange,
  SystemBypassFx,
  OscSupersaw,
  Osc3Wave,
  Osc4Wave,
  Osc1Level,
  Osc2Level,
  Osc3Level,
  Osc4Level,
  Osc1Freq,
  Osc3Freq,
  Osc4Freq,
  PitchEnvBend,
  PitchEnvTime,
  InfiniteDrop,
  ParamCount
};

struct ParamMeta {
  ParamId id;
  const char* name;
  ParamPage page;
  uint8_t default_value;
  uint8_t min_value;
  uint8_t max_value;
  SmoothingPolicy smoothing;
};

constexpr uint8_t kParamMinValue = 0;
constexpr uint8_t kParamMaxValue = 127;

const ParamMeta* GetParamSchema();
size_t GetParamSchemaCount();
const ParamMeta* FindParamMeta(ParamId id);
uint8_t ClampParamValue(ParamId id, uint8_t value);

}  // namespace pico2w_synth
