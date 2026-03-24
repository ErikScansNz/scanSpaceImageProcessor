#pragma once

#include <cstddef>
#include <cstdint>

#include "params/param_schema.h"

namespace pico2w_synth {

struct CcMapEntry {
  uint8_t cc;
  ParamId param;
};

bool TryMapCcToParam(uint8_t cc, ParamId* out_param);
const CcMapEntry* GetCcMap(size_t* out_count);

}  // namespace pico2w_synth
