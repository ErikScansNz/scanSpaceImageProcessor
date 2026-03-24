#include "params/param_registry.h"

#include <cstdio>

namespace pico2w_synth {

namespace {
constexpr size_t ToIndex(ParamId id) {
  return static_cast<size_t>(id);
}
}  // namespace

ParamRegistry::ParamRegistry() {
  ResetToDefaults(0);
}

void ParamRegistry::ResetToDefaults(uint32_t timestamp_us) {
  for (size_t i = 0; i < states_.size(); ++i) {
    const ParamMeta* meta = FindParamMeta(static_cast<ParamId>(i));
    states_[i].value = meta ? meta->default_value : 0;
    states_[i].timestamp_us = timestamp_us;
    states_[i].source = ParamSource::System;
    states_[i].dirty = true;
  }
}

bool ParamRegistry::Set(ParamId id, uint8_t value, uint32_t timestamp_us, ParamSource source) {
  value = ClampParamValue(id, value);

  ParamState& st = states_[ToIndex(id)];
  if (timestamp_us < st.timestamp_us) {
    return false;
  }
  bool changed = (st.value != value) || (st.timestamp_us != timestamp_us) || (st.source != source);
  st.value = value;
  st.timestamp_us = timestamp_us;
  st.source = source;
  st.dirty = true;
  return changed;
}

uint8_t ParamRegistry::Get(ParamId id) const {
  return states_[ToIndex(id)].value;
}

const ParamState& ParamRegistry::GetState(ParamId id) const {
  return states_[ToIndex(id)];
}

bool ParamRegistry::PopDirty(ParamId* id_out, ParamState* state_out) {
  for (size_t i = 0; i < states_.size(); ++i) {
    if (states_[i].dirty) {
      states_[i].dirty = false;
      if (id_out) {
        *id_out = static_cast<ParamId>(i);
      }
      if (state_out) {
        *state_out = states_[i];
      }
      return true;
    }
  }
  return false;
}

size_t ParamRegistry::WriteStateJson(char* out, size_t out_len) const {
  if (out_len == 0) {
    return 0;
  }

  size_t used = 0;
  int w = std::snprintf(out + used, out_len - used, "{");
  if (w <= 0) {
    return 0;
  }
  used += static_cast<size_t>(w);

  for (size_t i = 0; i < states_.size(); ++i) {
    const char* sep = (i == 0) ? "" : ",";
    w = std::snprintf(out + used, out_len - used, "%s\"%u\":%u", sep,
                      static_cast<unsigned>(i),
                      static_cast<unsigned>(states_[i].value));
    if (w <= 0 || static_cast<size_t>(w) >= (out_len - used)) {
      break;
    }
    used += static_cast<size_t>(w);
  }

  if (used + 2 < out_len) {
    out[used++] = '}';
    out[used] = '\0';
  }

  return used;
}

}  // namespace pico2w_synth
