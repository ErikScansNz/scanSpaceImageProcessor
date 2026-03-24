#include "midi/midi_mapper.h"

namespace pico2w_synth {

namespace {

constexpr CcMapEntry kCcMap[] = {
    {102, ParamId::Osc1Wave},       {104, ParamId::Osc2Wave},
    {21, ParamId::OscMix},          {19, ParamId::Osc1Shape},
    {20, ParamId::Osc1Morph},       {23, ParamId::SubNoiseMix},
    {85, ParamId::Osc2Coarse},      {76, ParamId::Osc2Detune},
    {82, ParamId::OscDrift},        {5, ParamId::Portamento},

    {74, ParamId::FilterCutoff},    {71, ParamId::FilterReso},
    {39, ParamId::FilterMode},      {24, ParamId::FilterEgAmt},
    {25, ParamId::FilterLfoAmt},    {9, ParamId::FilterKeyTrack},

    {73, ParamId::EgAttack},        {75, ParamId::EgDecay},
    {30, ParamId::EgSustain},       {72, ParamId::EgRelease},
    {52, ParamId::AmpAttack},       {53, ParamId::AmpDecay},
    {54, ParamId::AmpSustain},      {55, ParamId::AmpRelease},

    {33, ParamId::LfoWave},         {3, ParamId::LfoRate},
    {17, ParamId::LfoDepth},        {56, ParamId::LfoFade},
    {13, ParamId::LfoOscAmt},       {103, ParamId::LfoOscDst},
    {89, ParamId::EgOscAmt},        {8, ParamId::EgOscDst},

    {93, ParamId::ChorusMix},       {58, ParamId::ChorusRate},
    {59, ParamId::ChorusDepth},     {22, ParamId::ChorusDelay},
    {94, ParamId::DelayMix},        {90, ParamId::DelayTime},
    {92, ParamId::DelayFeedback},   {35, ParamId::DelayMode},
    {91, ParamId::ReverbMix},

    {15, ParamId::AmpGain},         {10, ParamId::Pan},
    {18, ParamId::VoiceMode},       {57, ParamId::PitchBendRange},
    {95, ParamId::SystemBypassFx},
};

}  // namespace

bool TryMapCcToParam(uint8_t cc, ParamId* out_param) {
  for (size_t i = 0; i < sizeof(kCcMap) / sizeof(kCcMap[0]); ++i) {
    if (kCcMap[i].cc == cc) {
      if (out_param != nullptr) {
        *out_param = kCcMap[i].param;
      }
      return true;
    }
  }
  return false;
}

const CcMapEntry* GetCcMap(size_t* out_count) {
  if (out_count != nullptr) {
    *out_count = sizeof(kCcMap) / sizeof(kCcMap[0]);
  }
  return kCcMap;
}

}  // namespace pico2w_synth
