#pragma once

#include <array>
#include <cstdint>
#include <cstddef>

#include "params/param_schema.h"

namespace pico2w_synth {

enum class ParamSource : uint8_t {
  Midi = 0,
  Web,
  Hardware,
  System,
};

struct ParamState {
  uint8_t value;
  uint32_t timestamp_us;
  ParamSource source;
  bool dirty;
};

class ParamRegistry {
 public:
  ParamRegistry();

  void ResetToDefaults(uint32_t timestamp_us);
  bool Set(ParamId id, uint8_t value, uint32_t timestamp_us, ParamSource source);
  uint8_t Get(ParamId id) const;
  const ParamState& GetState(ParamId id) const;

  // Pop one dirty parameter for low-rate application.
  bool PopDirty(ParamId* id_out, ParamState* state_out);

  // Write all state values as a compact JSON object body: "id":value,...
  size_t WriteStateJson(char* out, size_t out_len) const;

 private:
  std::array<ParamState, static_cast<size_t>(ParamId::ParamCount)> states_;
};

}  // namespace pico2w_synth
