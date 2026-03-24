#include "midi/midi_input.h"

#include "midi/midi_mapper.h"
#include "pico/time.h"

namespace pico2w_synth {

namespace {

constexpr uint32_t kMaxUsbBytesPerPoll = 128u;
constexpr uint32_t kMaxUartBytesPerPoll = 128u;

uint8_t ExpectedDataBytes(uint8_t status) {
  uint8_t command = status & 0xF0u;
  if (command == 0xC0u || command == 0xD0u) {
    return 1;
  }
  if (command >= 0x80u && command <= 0xE0u) {
    return 2;
  }
  return 0;
}

}  // namespace

bool MidiInput::Init(SynthEventQueue* queue, ParamRegistry* registry) {
  queue_ = queue;
  registry_ = registry;
  running_status_ = 0;
  data_count_ = 0;
  in_sysex_ = false;
  bool ok_uart = uart_.Init();
  bool ok_usb = usb_.Init();
  return ok_uart && ok_usb;
}

void MidiInput::Poll() {
  usb_.ServiceUsbTask();

  uint8_t b = 0;
  uint32_t t = time_us_32();
  uint32_t usb_budget = kMaxUsbBytesPerPoll;
  while (usb_budget > 0u && usb_.ReadByte(&b)) {
    ProcessByte(b, ParamSource::Midi, t);
    t = time_us_32();
    --usb_budget;
  }

  t = time_us_32();
  uint32_t uart_budget = kMaxUartBytesPerPoll;
  while (uart_budget > 0u && uart_.ReadByte(&b)) {
    ProcessByte(b, ParamSource::Midi, t);
    t = time_us_32();
    --uart_budget;
  }
}

void MidiInput::ProcessByte(uint8_t b, ParamSource source, uint32_t timestamp_us) {
  static_cast<void>(source);

  if (b >= 0xF8u) {
    return;
  }

  if ((b & 0x80u) != 0u) {
    if (b == 0xF0u) {
      in_sysex_ = true;
      running_status_ = 0;
      data_count_ = 0;
      return;
    }
    if (b == 0xF7u) {
      in_sysex_ = false;
      return;
    }
    if (b >= 0xF0u) {
      running_status_ = 0;
      data_count_ = 0;
      return;
    }

    in_sysex_ = false;
    running_status_ = b;
    data_count_ = 0;
    return;
  }

  if (in_sysex_ || running_status_ == 0) {
    return;
  }

  uint8_t needed = ExpectedDataBytes(running_status_);
  if (needed == 0) {
    return;
  }

  data_bytes_[data_count_++] = b;
  if (data_count_ < needed) {
    return;
  }

  uint8_t data0 = data_bytes_[0];
  uint8_t data1 = (needed > 1) ? data_bytes_[1] : 0;
  DispatchMessage(running_status_, data0, data1, timestamp_us);
  data_count_ = 0;
}

void MidiInput::DispatchMessage(uint8_t status, uint8_t data0, uint8_t data1, uint32_t timestamp_us) {
  Event ev{};
  ev.channel = static_cast<uint8_t>(status & 0x0Fu);
  ev.timestamp_us = timestamp_us;
  ev.source = ParamSource::Midi;

  switch (status & 0xF0u) {
    case 0x80u:
      ev.type = EventType::NoteOff;
      ev.data0 = data0;
      ev.data1 = data1;
      ev.param_id = 0;
      PushEvent(ev);
      return;

    case 0x90u:
      ev.type = (data1 == 0) ? EventType::NoteOff : EventType::NoteOn;
      ev.data0 = data0;
      ev.data1 = data1;
      ev.param_id = 0;
      PushEvent(ev);
      return;

    case 0xA0u:
#if PICO2W_SYNTH_POLY_AFTERTOUCH
      ev.type = EventType::PolyAftertouch;
      ev.data0 = data0;
      ev.data1 = data1;
      ev.param_id = 0;
      PushEvent(ev);
#endif
      return;

    case 0xB0u: {
      ParamId param{};
      if (TryMapCcToParam(data0, &param)) {
        ev.type = EventType::ParamSet;
        ev.data0 = data0;
        ev.data1 = data1;
        ev.param_id = static_cast<uint16_t>(param);
        PushEvent(ev);
      }
      return;
    }

    case 0xD0u:
      ev.type = EventType::ChannelPressure;
      ev.data0 = data0;
      ev.data1 = 0;
      ev.param_id = 0;
      PushEvent(ev);
      return;

    case 0xE0u:
      ev.type = EventType::PitchBend;
      ev.data0 = data0;
      ev.data1 = data1;
      ev.param_id = 0;
      PushEvent(ev);
      return;

    default:
      return;
  }
}

bool MidiInput::PushEvent(const Event& event) {
  if (queue_ == nullptr) {
    return false;
  }
  if (!queue_->TryPush(event)) {
    queue_->RecordDrop(ParamSource::Midi);
    return false;
  }
  if (registry_ != nullptr && event.type == EventType::ParamSet &&
      event.param_id < static_cast<uint16_t>(ParamId::ParamCount)) {
    registry_->Set(static_cast<ParamId>(event.param_id), event.data1, event.timestamp_us, event.source);
  }
  return true;
}

}  // namespace pico2w_synth
