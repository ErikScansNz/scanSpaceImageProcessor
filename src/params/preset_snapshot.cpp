#include "params/preset_snapshot.h"

#include <cctype>
#include <cstdlib>

namespace pico2w_synth {

namespace {

bool ParseNextUnsigned(const char** p, unsigned* out) {
  const char* s = *p;
  while (*s != '\0' && !std::isdigit(static_cast<unsigned char>(*s))) {
    ++s;
  }
  if (*s == '\0') {
    return false;
  }

  char* end = nullptr;
  unsigned value = static_cast<unsigned>(std::strtoul(s, &end, 10));
  if (end == s) {
    return false;
  }

  *out = value;
  *p = end;
  return true;
}

}  // namespace

size_t ExportPresetSnapshotJson(const ParamRegistry& registry, char* out, size_t out_len) {
  return registry.WriteStateJson(out, out_len);
}

PresetSnapshotResult ImportPresetSnapshotJson(ParamRegistry* registry,
                                              const char* json,
                                              uint32_t timestamp_us,
                                              ParamSource source) {
  if (registry == nullptr || json == nullptr) {
    return {false, 0};
  }

  size_t applied = 0;
  const char* p = json;
  while (*p != '\0') {
    unsigned id = 0;
    if (!ParseNextUnsigned(&p, &id)) {
      break;
    }

    unsigned value = 0;
    if (!ParseNextUnsigned(&p, &value)) {
      break;
    }

    if (id < static_cast<unsigned>(ParamId::ParamCount)) {
      registry->Set(static_cast<ParamId>(id), static_cast<uint8_t>(value), timestamp_us, source);
      ++applied;
    }
  }

  return {true, applied};
}

}  // namespace pico2w_synth
