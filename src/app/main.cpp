#include "app/application.h"

#include "hardware/gpio.h"
#include "pico/stdlib.h"
#include "tusb.h"

namespace {

void SetFailLed(bool on) {
#ifdef PICO_DEFAULT_LED_PIN
  gpio_put(PICO_DEFAULT_LED_PIN, on ? 1 : 0);
#else
  static_cast<void>(on);
#endif
}

void SetBoardLed(bool on) {
  SetFailLed(on);
}

void BlinkDelayMs(uint32_t delay_ms) {
  const uint32_t step_ms = 10u;
  uint32_t elapsed = 0u;
  while (elapsed < delay_ms) {
    if (tusb_inited()) {
      tud_task();
    }
    sleep_ms(step_ms);
    elapsed += step_ms;
  }
}

void InitFailLed() {
#ifdef PICO_DEFAULT_LED_PIN
  gpio_init(PICO_DEFAULT_LED_PIN);
  gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
#endif
}

void BootSignatureBlink() {
  InitFailLed();
  // One long, one short blink to indicate normal firmware boot path.
  SetBoardLed(true);
  BlinkDelayMs(300);
  SetBoardLed(false);
  BlinkDelayMs(180);
  SetBoardLed(true);
  BlinkDelayMs(100);
  SetBoardLed(false);
  BlinkDelayMs(220);
}

[[noreturn]] void FailBlinkLoop() {
  InitFailLed();
  while (true) {
    SetBoardLed(true);
    BlinkDelayMs(120);
    SetBoardLed(false);
    BlinkDelayMs(120);
  }
}

#if PICO2W_SYNTH_USB_ENUM_ONLY
[[noreturn]] void UsbEnumOnlyLoop() {
  stdio_init_all();
  bool usb_ok = tusb_init();

  while (true) {
    if (usb_ok && tusb_inited()) {
      tud_task();
    }
    sleep_ms(1);
  }
}
#endif

#if PICO2W_SYNTH_DIAGNOSTIC_LED_ONLY
[[noreturn]] void DiagnosticLedLoop() {
  InitFailLed();

  // Distinctive pattern: two short flashes, then a pause.
  while (true) {
    SetBoardLed(true);
    BlinkDelayMs(120);
    SetBoardLed(false);
    BlinkDelayMs(120);
    SetBoardLed(true);
    BlinkDelayMs(120);
    SetBoardLed(false);
    BlinkDelayMs(640);
  }
}
#endif

}  // namespace

int main() {
#if PICO2W_SYNTH_USB_ENUM_ONLY
  UsbEnumOnlyLoop();
#elif PICO2W_SYNTH_DIAGNOSTIC_LED_ONLY
  DiagnosticLedLoop();
#else
  BootSignatureBlink();

  pico2w_synth::Application app;
  if (!app.Init()) {
    FailBlinkLoop();
  }

  app.Run();
  return 0;
#endif
}
