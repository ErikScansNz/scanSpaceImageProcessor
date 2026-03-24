#include "midi/midi_usb.h"

#include "pico/time.h"
#include "pico/stdlib.h"
#include "tusb.h"

namespace pico2w_synth {

bool MidiUsb::Init() {
  if (!tusb_init()) {
    return false;
  }

  // Give hosts a stable window for initial descriptor exchange before the
  // rest of bring-up. This reduces Windows descriptor failures if later init
  // work briefly delays USB servicing.
  absolute_time_t settle_deadline = make_timeout_time_ms(1500);
  while (!time_reached(settle_deadline)) {
    tud_task();
    if (tud_mounted()) {
      break;
    }
    sleep_ms(1);
  }

  return true;
}

void MidiUsb::ServiceUsbTask() {
  tud_task();
}

bool MidiUsb::ReadByte(uint8_t* out) {
  if (out == nullptr) {
    return false;
  }

#if !CFG_TUD_MIDI
  // USB enumeration-only profile may disable MIDI class.
  (void)out;
  return false;
#else
  if (buffer_index_ < buffer_used_) {
    *out = buffer_[buffer_index_++];
    return true;
  }

  buffer_used_ = 0;
  buffer_index_ = 0;

  uint32_t available = static_cast<uint32_t>(tud_midi_n_available(0, 0));
  if (available == 0) {
    return false;
  }

  uint32_t read_len = tud_midi_n_stream_read(0, 0, buffer_, sizeof(buffer_));
  if (read_len == 0) {
    return false;
  }

  buffer_used_ = read_len;
  *out = buffer_[buffer_index_++];
  return true;
#endif
}

}  // namespace pico2w_synth
