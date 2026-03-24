#pragma once

#include <cstddef>
#include <cstdint>

#include "params/param_registry.h"

namespace pico2w_synth {

struct PresetSnapshotResult {
  bool ok;
  size_t applied;
};

// Export current parameter registry as compact JSON.
size_t ExportPresetSnapshotJson(const ParamRegistry& registry, char* out, size_t out_len);

// Import snapshot JSON and apply values into registry.
PresetSnapshotResult ImportPresetSnapshotJson(ParamRegistry* registry,
                                              const char* json,
                                              uint32_t timestamp_us,
                                              ParamSource source);

}  // namespace pico2w_synth
