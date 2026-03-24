#include "midi/midi_uart.h"

#include "hardware/gpio.h"
#include "hardware/uart.h"

#include "pico2w_synth/config.h"

namespace pico2w_synth {

bool MidiUart::Init() {
  uart_init(uart0, kMidiUartBaud);
  gpio_set_function(kPinMidiUartRx, GPIO_FUNC_UART);
  gpio_set_function(kPinMidiUartTx, GPIO_FUNC_UART);
  uart_set_hw_flow(uart0, false, false);
  uart_set_format(uart0, 8, 1, UART_PARITY_NONE);
  uart_set_fifo_enabled(uart0, true);
  return true;
}

bool MidiUart::ReadByte(uint8_t* out) {
  if (out == nullptr || !uart_is_readable(uart0)) {
    return false;
  }
  *out = static_cast<uint8_t>(uart_getc(uart0));
  return true;
}

}  // namespace pico2w_synth
