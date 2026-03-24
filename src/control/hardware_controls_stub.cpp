#include "control/hardware_controls_stub.h"

#include <cstddef>

#include "hardware/adc.h"
#include "hardware/gpio.h"

#include "pico2w_synth/config.h"

namespace pico2w_synth {

namespace {

void InitAdcPinIfValid(uint8_t gpio) {
  if (gpio <= 29u) {
    adc_gpio_init(gpio);
  }
}

}  // namespace

void HardwareControlsInit() {
#if PICO2W_SYNTH_ENABLE_HW_CONTROLS
  adc_init();
  InitAdcPinIfValid(kPinPot1);
  InitAdcPinIfValid(kPinPot2);
  InitAdcPinIfValid(kPinPot3);

  const uint8_t pins[] = {
      kPinEnc1A, kPinEnc1B, kPinEnc1Sw,
      kPinEnc2A, kPinEnc2B, kPinEnc2Sw,
      kPinEnc3A, kPinEnc3B, kPinEnc3Sw,
      kPinEnc4A, kPinEnc4B, kPinEnc4Sw,
  };

  for (size_t i = 0; i < (sizeof(pins) / sizeof(pins[0])); ++i) {
    gpio_init(pins[i]);
    gpio_set_dir(pins[i], GPIO_IN);
    gpio_pull_up(pins[i]);
  }
#endif
}

bool HardwareControlsPoll(HardwareControlChange* out_change) {
  if (out_change != nullptr) {
    out_change->valid = false;
    out_change->param = ParamId::OscMix;
    out_change->value = 0;
  }

#if PICO2W_SYNTH_ENABLE_HW_CONTROLS
  // Hardware controls are intentionally deferred for phase-1 web-first bring-up.
  // This hook is left for a follow-up implementation.
#endif

  return false;
}

}  // namespace pico2w_synth
