#include "serial/serial_control.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "pico2w_synth/config.h"
#include "params/param_schema.h"
#include "pico/time.h"
#include "tusb.h"

#ifndef PICO2W_SYNTH_ENABLE_USB_SERIAL_DEBUG
#define PICO2W_SYNTH_ENABLE_USB_SERIAL_DEBUG 0
#endif

namespace pico2w_synth {

#if !PICO2W_SYNTH_ENABLE_USB_SERIAL_DEBUG

bool SerialControl::Init(ParamRegistry* registry, SynthEventQueue* queue, RuntimeDiagnostics* diagnostics) {
  registry_ = registry;
  queue_ = queue;
  diagnostics_ = diagnostics;
  initialized_ = (registry != nullptr && queue != nullptr);
  return initialized_;
}

void SerialControl::Poll() {}

#else

namespace {

constexpr char kPageNames[][8] = {"osc", "filter", "env", "lfo", "fx", "amp", "system"};
constexpr char kSmoothingNames[][8] = {"none", "slow", "fast"};
constexpr uint32_t kMaxRxCharsPerPoll = 256u;

const char* PageName(ParamPage page) {
  uint8_t idx = static_cast<uint8_t>(page);
  if (idx >= static_cast<uint8_t>(ParamPage::Count)) {
    return "unknown";
  }
  return kPageNames[idx];
}

const char* SmoothingName(SmoothingPolicy policy) {
  uint8_t idx = static_cast<uint8_t>(policy);
  if (idx > static_cast<uint8_t>(SmoothingPolicy::Fast)) {
    return "none";
  }
  return kSmoothingNames[idx];
}

void TrimInPlace(char* text) {
  if (text == nullptr) {
    return;
  }

  size_t len = std::strlen(text);
  size_t start = 0;
  while (start < len && std::isspace(static_cast<unsigned char>(text[start])) != 0) {
    ++start;
  }

  size_t end = len;
  while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }

  if (start > 0) {
    std::memmove(text, text + start, end - start);
  }
  text[end - start] = '\0';
}

void SkipWs(char** p) {
  if (p == nullptr || *p == nullptr) {
    return;
  }
  while (**p != '\0' && std::isspace(static_cast<unsigned char>(**p)) != 0) {
    ++(*p);
  }
}

bool ParseUnsignedToken(char** p, unsigned* out) {
  if (p == nullptr || *p == nullptr || out == nullptr) {
    return false;
  }
  SkipWs(p);
  if (**p == '\0') {
    return false;
  }

  char* end = nullptr;
  unsigned long parsed = std::strtoul(*p, &end, 10);
  if (end == *p) {
    return false;
  }
  if (parsed > static_cast<unsigned long>(std::numeric_limits<unsigned>::max())) {
    return false;
  }

  *p = end;
  *out = static_cast<unsigned>(parsed);
  return true;
}

bool ParseSetArgs(char* args, unsigned* out_id, unsigned* out_value) {
  if (args == nullptr || out_id == nullptr || out_value == nullptr) {
    return false;
  }
  char* p = args;
  if (!ParseUnsignedToken(&p, out_id)) {
    return false;
  }
  if (!ParseUnsignedToken(&p, out_value)) {
    return false;
  }
  SkipWs(&p);
  return *p == '\0';
}

}  // namespace

bool SerialControl::Init(ParamRegistry* registry, SynthEventQueue* queue, RuntimeDiagnostics* diagnostics) {
  if (registry == nullptr || queue == nullptr) {
    return false;
  }
  registry_ = registry;
  queue_ = queue;
  diagnostics_ = diagnostics;
  initialized_ = true;
  rx_used_ = 0;
  tx_used_ = 0;
  tx_sent_ = 0;
  rx_lines_ = 0;
  rx_overflow_ = 0;
  rx_parse_errors_ = 0;
  tx_lines_ = 0;
  tx_dropped_ = 0;
  last_stats_audio_frames_ = 0;
  last_stats_over_budget_ = 0;
  stats_baseline_valid_ = false;
  return true;
}

void SerialControl::Poll() {
  if (!initialized_) {
    return;
  }

  FlushTx();

  uint32_t rx_budget = kMaxRxCharsPerPoll;
  while (rx_budget > 0u && tud_cdc_available() > 0) {
    char c = static_cast<char>(tud_cdc_read_char());
    --rx_budget;
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      rx_line_[rx_used_] = '\0';
      if (rx_used_ > 0) {
        HandleLine(rx_line_);
      }
      rx_used_ = 0;
      continue;
    }

    if (rx_used_ + 1 >= kRxLineMax) {
      rx_used_ = 0;
      ++rx_overflow_;
      static_cast<void>(QueueError("line_too_long"));
      continue;
    }

    rx_line_[rx_used_++] = c;
  }
}

void SerialControl::HandleLine(char* line) {
  if (line == nullptr) {
    return;
  }

  TrimInPlace(line);
  if (line[0] == '\0') {
    return;
  }
  ++rx_lines_;

  if (std::strcmp(line, "PING") == 0) {
    static_cast<void>(QueueText("{\"type\":\"pong\"}"));
    return;
  }

  if (std::strcmp(line, "META") == 0) {
    static_cast<void>(QueueMeta());
    return;
  }

  if (std::strcmp(line, "STATE") == 0) {
    static_cast<void>(QueueState());
    return;
  }

  if (std::strcmp(line, "HELP") == 0) {
    static_cast<void>(QueueText(
        "{\"type\":\"help\",\"commands\":[\"PING\",\"META\",\"STATE\",\"STATS\",\"NOTEON <note> <vel>\",\"NOTEOFF <note> <vel>\",\"SET <id> <value>\",\"SETQ <id> <value>\",\"BATCH <id>=<value>,...\",\"BATCHQ <id>=<value>,...\"]}"));
    return;
  }

  if (std::strcmp(line, "STATS") == 0) {
    static_cast<void>(QueueStats());
    return;
  }

  if (std::strncmp(line, "NOTEON ", 7) == 0) {
    unsigned note = 0;
    unsigned velocity = 0;
    if (!ParseSetArgs(line + 7, &note, &velocity)) {
      ++rx_parse_errors_;
      static_cast<void>(QueueError("bad_noteon"));
      return;
    }
    if (!HandleNoteCommand(true, note, velocity)) {
      ++rx_parse_errors_;
      static_cast<void>(QueueError("noteon_failed"));
    }
    return;
  }

  if (std::strncmp(line, "NOTEOFF ", 8) == 0) {
    unsigned note = 0;
    unsigned velocity = 0;
    if (!ParseSetArgs(line + 8, &note, &velocity)) {
      ++rx_parse_errors_;
      static_cast<void>(QueueError("bad_noteoff"));
      return;
    }
    if (!HandleNoteCommand(false, note, velocity)) {
      ++rx_parse_errors_;
      static_cast<void>(QueueError("noteoff_failed"));
    }
    return;
  }

  if (std::strncmp(line, "SET ", 4) == 0) {
    unsigned id = 0;
    unsigned value = 0;
    if (!ParseSetArgs(line + 4, &id, &value)) {
      ++rx_parse_errors_;
      static_cast<void>(QueueError("bad_set"));
      return;
    }
    if (!HandleSet(id, value)) {
      ++rx_parse_errors_;
      static_cast<void>(QueueError("set_failed"));
      return;
    }
    uint8_t applied_value = registry_->Get(static_cast<ParamId>(id));
    static_cast<void>(QueueAckSet(static_cast<uint16_t>(id), applied_value));
    return;
  }

  if (std::strncmp(line, "SETQ ", 5) == 0) {
    unsigned id = 0;
    unsigned value = 0;
    if (!ParseSetArgs(line + 5, &id, &value)) {
      ++rx_parse_errors_;
      static_cast<void>(QueueError("bad_set"));
      return;
    }
    if (!HandleSet(id, value)) {
      ++rx_parse_errors_;
      static_cast<void>(QueueError("set_failed"));
    }
    return;
  }

  if (std::strncmp(line, "BATCH ", 6) == 0) {
    size_t applied = 0;
    if (!HandleBatch(line + 6, &applied)) {
      ++rx_parse_errors_;
      static_cast<void>(QueueError("batch_failed"));
      return;
    }
    static_cast<void>(QueueAckBatch(applied));
    return;
  }

  if (std::strncmp(line, "BATCHQ ", 7) == 0) {
    size_t applied = 0;
    if (!HandleBatch(line + 7, &applied)) {
      ++rx_parse_errors_;
      static_cast<void>(QueueError("batch_failed"));
    }
    return;
  }

  ++rx_parse_errors_;
  static_cast<void>(QueueError("unknown_command"));
}

void SerialControl::FlushTx() {
  if (tx_sent_ >= tx_used_) {
    return;
  }

  while (tx_sent_ < tx_used_) {
    uint32_t avail = tud_cdc_write_available();
    if (avail == 0) {
      break;
    }
    size_t remaining = tx_used_ - tx_sent_;
    size_t chunk = std::min(remaining, static_cast<size_t>(avail));
    uint32_t written = tud_cdc_write(tx_ + tx_sent_, static_cast<uint32_t>(chunk));
    if (written == 0) {
      break;
    }
    tx_sent_ += static_cast<size_t>(written);
  }

  tud_cdc_write_flush();

  if (tx_sent_ >= tx_used_) {
    tx_used_ = 0;
    tx_sent_ = 0;
    return;
  }

  if (tx_sent_ > 0) {
    size_t pending = tx_used_ - tx_sent_;
    std::memmove(tx_, tx_ + tx_sent_, pending);
    tx_used_ = pending;
    tx_sent_ = 0;
  }
}

bool SerialControl::QueueRaw(const char* data, size_t len) {
  if (data == nullptr) {
    return false;
  }
  if (len + 1 > kTxMax) {
    ++tx_dropped_;
    return false;
  }

  size_t pending = tx_used_ - tx_sent_;
  if (pending > 0 && tx_sent_ > 0) {
    std::memmove(tx_, tx_ + tx_sent_, pending);
    tx_used_ = pending;
    tx_sent_ = 0;
  } else if (pending == 0) {
    tx_used_ = 0;
    tx_sent_ = 0;
  }

  if (pending + len + 1 > kTxMax) {
    ++tx_dropped_;
    return false;
  }

  std::memcpy(tx_ + tx_used_, data, len);
  tx_[tx_used_ + len] = '\n';
  tx_used_ += len + 1;
  ++tx_lines_;
  return true;
}

bool SerialControl::QueueText(const char* text) {
  if (text == nullptr) {
    return false;
  }
  return QueueRaw(text, std::strlen(text));
}

bool SerialControl::QueueError(const char* reason) {
  char response[128] = {};
  std::snprintf(response,
                sizeof(response),
                "{\"type\":\"error\",\"message\":\"%s\"}",
                (reason != nullptr) ? reason : "error");
  return QueueText(response);
}

bool SerialControl::QueueState() {
  if (registry_ == nullptr) {
    return QueueError("registry_unavailable");
  }

  char values[2048] = {};
  registry_->WriteStateJson(values, sizeof(values));

  char response[2400] = {};
  std::snprintf(response, sizeof(response), "{\"type\":\"state\",\"values\":%s}", values);
  return QueueText(response);
}

bool SerialControl::QueueMeta() {
  char body[8192] = {};
  size_t used = 0;

  int w = std::snprintf(body, sizeof(body), "{\"type\":\"meta\",\"pages\":[");
  if (w <= 0 || static_cast<size_t>(w) >= sizeof(body)) {
    return QueueError("meta_overflow");
  }
  used = static_cast<size_t>(w);

  for (uint8_t i = 0; i < static_cast<uint8_t>(ParamPage::Count); ++i) {
    const char* sep = (i == 0) ? "" : ",";
    w = std::snprintf(body + used, sizeof(body) - used, "%s\"%s\"", sep, kPageNames[i]);
    if (w <= 0 || static_cast<size_t>(w) >= (sizeof(body) - used)) {
      return QueueError("meta_overflow");
    }
    used += static_cast<size_t>(w);
  }

  w = std::snprintf(body + used, sizeof(body) - used, "],\"params\":[");
  if (w <= 0 || static_cast<size_t>(w) >= (sizeof(body) - used)) {
    return QueueError("meta_overflow");
  }
  used += static_cast<size_t>(w);

  const ParamMeta* schema = GetParamSchema();
  size_t schema_count = GetParamSchemaCount();
  for (size_t i = 0; i < schema_count; ++i) {
    const char* sep = (i == 0) ? "" : ",";
    w = std::snprintf(body + used,
                      sizeof(body) - used,
                      "%s{\"id\":%u,\"name\":\"%s\",\"page\":\"%s\",\"default\":%u,\"min\":%u,\"max\":%u,\"smoothing\":\"%s\"}",
                      sep,
                      static_cast<unsigned>(schema[i].id),
                      schema[i].name,
                      PageName(schema[i].page),
                      static_cast<unsigned>(schema[i].default_value),
                      static_cast<unsigned>(schema[i].min_value),
                      static_cast<unsigned>(schema[i].max_value),
                      SmoothingName(schema[i].smoothing));
    if (w <= 0 || static_cast<size_t>(w) >= (sizeof(body) - used)) {
      return QueueError("meta_overflow");
    }
    used += static_cast<size_t>(w);
  }

  w = std::snprintf(body + used, sizeof(body) - used, "]}");
  if (w <= 0 || static_cast<size_t>(w) >= (sizeof(body) - used)) {
    return QueueError("meta_overflow");
  }
  used += static_cast<size_t>(w);

  return QueueRaw(body, used);
}

bool SerialControl::QueueStats() {
  size_t queue_depth = (queue_ != nullptr) ? queue_->Size() : 0u;
  uint32_t dropped_midi = 0;
  uint32_t dropped_web = 0;
  uint32_t dropped_hardware = 0;
  uint32_t audio_overruns = 0;
  uint32_t audio_frames = 0;
  uint32_t audio_cb_last_us = 0;
  uint32_t audio_cb_max_us = 0;
  uint32_t audio_cb_over_budget = 0;
  uint32_t audio_cb_over_budget_delta = 0;
  uint32_t audio_cb_over_budget_pct_x100 = 0;
  uint32_t core0_loop_last_us = 0;
  uint32_t core0_loop_max_us = 0;
  uint32_t core0_midi_poll_last_us = 0;
  uint32_t core0_midi_poll_max_us = 0;
  uint32_t core0_serial_poll_last_us = 0;
  uint32_t core0_serial_poll_max_us = 0;
  uint32_t core0_control_poll_last_us = 0;
  uint32_t core0_control_poll_max_us = 0;
  uint32_t runtime_status = 0;

  if (diagnostics_ != nullptr) {
    dropped_midi = diagnostics_->queue_dropped_midi;
    dropped_web = diagnostics_->queue_dropped_web;
    dropped_hardware = diagnostics_->queue_dropped_hardware;
    audio_overruns = diagnostics_->audio_overruns;
    audio_frames = diagnostics_->audio_frames;
    audio_cb_last_us = diagnostics_->audio_callback_last_us;
    audio_cb_max_us = diagnostics_->audio_callback_max_us;
    audio_cb_over_budget = diagnostics_->audio_callback_over_budget;
    core0_loop_last_us = diagnostics_->core0_loop_last_us;
    core0_loop_max_us = diagnostics_->core0_loop_max_us;
    core0_midi_poll_last_us = diagnostics_->core0_midi_poll_last_us;
    core0_midi_poll_max_us = diagnostics_->core0_midi_poll_max_us;
    core0_serial_poll_last_us = diagnostics_->core0_serial_poll_last_us;
    core0_serial_poll_max_us = diagnostics_->core0_serial_poll_max_us;
    core0_control_poll_last_us = diagnostics_->core0_control_poll_last_us;
    core0_control_poll_max_us = diagnostics_->core0_control_poll_max_us;
    runtime_status = diagnostics_->runtime_status;
  }

  if (stats_baseline_valid_) {
    uint32_t frames_delta = audio_frames - last_stats_audio_frames_;
    audio_cb_over_budget_delta = audio_cb_over_budget - last_stats_over_budget_;
    if (frames_delta > 0u) {
      // Two-decimal fixed-point percentage in [0, 10000] == [0.00%, 100.00%].
      uint64_t scaled = static_cast<uint64_t>(audio_cb_over_budget_delta) * 10000ull;
      audio_cb_over_budget_pct_x100 = static_cast<uint32_t>(scaled / static_cast<uint64_t>(frames_delta));
      if (audio_cb_over_budget_pct_x100 > 10000u) {
        audio_cb_over_budget_pct_x100 = 10000u;
      }
    }
  } else {
    stats_baseline_valid_ = true;
  }
  last_stats_audio_frames_ = audio_frames;
  last_stats_over_budget_ = audio_cb_over_budget;

  char response[1024] = {};
  std::snprintf(response,
                sizeof(response),
                "{\"type\":\"stats\",\"serial\":{\"rx_lines\":%u,\"rx_overflow\":%u,\"rx_parse_errors\":%u,"
                "\"tx_lines\":%u,\"tx_dropped\":%u},\"runtime\":{\"queue_depth\":%u,\"dropped_midi\":%u,"
                "\"dropped_web\":%u,\"dropped_hardware\":%u,\"audio_overruns\":%u,\"audio_cb_last_us\":%u,"
                "\"audio_cb_max_us\":%u,\"audio_frames\":%u,\"audio_cb_over_budget\":%u,"
                "\"audio_cb_over_budget_delta\":%u,\"audio_cb_over_budget_pct_x100\":%u,\"core0_loop_last_us\":%u,"
                "\"core0_loop_max_us\":%u,\"core0_midi_poll_last_us\":%u,\"core0_midi_poll_max_us\":%u,"
                "\"core0_serial_poll_last_us\":%u,\"core0_serial_poll_max_us\":%u,"
                "\"core0_control_poll_last_us\":%u,\"core0_control_poll_max_us\":%u,\"sample_rate_hz\":%u,"
                "\"audio_frame_interval_us\":%u,\"audio_render_div\":%u,\"audio_callback_interval_us\":%u,"
                "\"main_loop_sleep_us\":%u,\"chorus_enabled\":%u,\"delay_enabled\":%u,"
                "\"exp_mod_dest_enabled\":%u,\"restrict_osc_enabled\":%u,\"runtime_status\":%u}}",
                static_cast<unsigned>(rx_lines_),
                static_cast<unsigned>(rx_overflow_),
                static_cast<unsigned>(rx_parse_errors_),
                static_cast<unsigned>(tx_lines_),
                static_cast<unsigned>(tx_dropped_),
                static_cast<unsigned>(queue_depth),
                static_cast<unsigned>(dropped_midi),
                static_cast<unsigned>(dropped_web),
                static_cast<unsigned>(dropped_hardware),
                static_cast<unsigned>(audio_overruns),
                static_cast<unsigned>(audio_cb_last_us),
                static_cast<unsigned>(audio_cb_max_us),
                static_cast<unsigned>(audio_frames),
                static_cast<unsigned>(audio_cb_over_budget),
                static_cast<unsigned>(audio_cb_over_budget_delta),
                static_cast<unsigned>(audio_cb_over_budget_pct_x100),
                static_cast<unsigned>(core0_loop_last_us),
                static_cast<unsigned>(core0_loop_max_us),
                static_cast<unsigned>(core0_midi_poll_last_us),
                static_cast<unsigned>(core0_midi_poll_max_us),
                static_cast<unsigned>(core0_serial_poll_last_us),
                static_cast<unsigned>(core0_serial_poll_max_us),
                static_cast<unsigned>(core0_control_poll_last_us),
                static_cast<unsigned>(core0_control_poll_max_us),
                static_cast<unsigned>(kSampleRate),
                static_cast<unsigned>(kAudioFrameIntervalUs),
                static_cast<unsigned>(kAudioRenderDiv),
                static_cast<unsigned>(kAudioCallbackIntervalUs),
                static_cast<unsigned>(kMainLoopSleepUs),
                static_cast<unsigned>(PICO2W_SYNTH_ENABLE_CHORUS != 0),
                static_cast<unsigned>(PICO2W_SYNTH_ENABLE_DELAY != 0),
                static_cast<unsigned>(PICO2W_SYNTH_ENABLE_EXPENSIVE_MOD_DEST != 0),
                static_cast<unsigned>(PICO2W_SYNTH_RESTRICT_OSC != 0),
                static_cast<unsigned>(runtime_status));
  return QueueText(response);
}

bool SerialControl::QueueAckSet(uint16_t id, uint8_t value) {
  char response[128] = {};
  std::snprintf(response,
                sizeof(response),
                "{\"type\":\"ack\",\"cmd\":\"set\",\"id\":%u,\"value\":%u}",
                static_cast<unsigned>(id),
                static_cast<unsigned>(value));
  return QueueText(response);
}

bool SerialControl::QueueAckBatch(size_t applied) {
  char response[128] = {};
  std::snprintf(response,
                sizeof(response),
                "{\"type\":\"ack\",\"cmd\":\"batch\",\"applied\":%u}",
                static_cast<unsigned>(applied));
  return QueueText(response);
}

bool SerialControl::HandleSet(unsigned id, unsigned value) {
  if (queue_ == nullptr || registry_ == nullptr) {
    return false;
  }
  if (id >= static_cast<unsigned>(ParamId::ParamCount) || value > static_cast<unsigned>(kParamMaxValue)) {
    return false;
  }

  ParamId param_id = static_cast<ParamId>(id);
  uint8_t clamped_value = ClampParamValue(param_id, static_cast<uint8_t>(value));

  Event ev{};
  ev.type = EventType::ParamSet;
  ev.channel = 0;
  ev.data0 = 0;
  ev.data1 = clamped_value;
  ev.param_id = static_cast<uint16_t>(id);
  ev.timestamp_us = time_us_32();
  ev.source = ParamSource::Web;

  if (!queue_->TryPush(ev)) {
    queue_->RecordDrop(ParamSource::Web);
    return false;
  }

  registry_->Set(param_id, clamped_value, ev.timestamp_us, ev.source);
  return true;
}

bool SerialControl::HandleNoteCommand(bool note_on, unsigned note, unsigned velocity) {
  if (queue_ == nullptr) {
    return false;
  }
  if (note > 127u || velocity > 127u) {
    return false;
  }

  Event ev{};
  ev.type = note_on ? EventType::NoteOn : EventType::NoteOff;
  ev.channel = 0;
  ev.data0 = static_cast<uint8_t>(note);
  ev.data1 = static_cast<uint8_t>(velocity);
  ev.param_id = 0;
  ev.timestamp_us = time_us_32();
  ev.source = ParamSource::Web;

  if (!queue_->TryPush(ev)) {
    queue_->RecordDrop(ParamSource::Web);
    return false;
  }

  return true;
}

bool SerialControl::HandleBatch(char* args, size_t* applied_out) {
  if (applied_out == nullptr) {
    return false;
  }
  *applied_out = 0;
  if (args == nullptr) {
    return false;
  }

  char* p = args;
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t' || *p == ',') {
      ++p;
    }
    if (*p == '\0') {
      break;
    }

    char* end = nullptr;
    unsigned long id = std::strtoul(p, &end, 10);
    if (end == p || *end != '=') {
      return false;
    }
    if (id > static_cast<unsigned long>(std::numeric_limits<unsigned>::max())) {
      return false;
    }
    p = end + 1;

    unsigned long value = std::strtoul(p, &end, 10);
    if (end == p) {
      return false;
    }
    if (value > static_cast<unsigned long>(std::numeric_limits<unsigned>::max())) {
      return false;
    }

    if (!HandleSet(static_cast<unsigned>(id), static_cast<unsigned>(value))) {
      return false;
    }
    ++(*applied_out);

    p = end;
    while (*p == ' ' || *p == '\t') {
      ++p;
    }
    if (*p == ',') {
      ++p;
    } else if (*p != '\0') {
      return false;
    }
  }

  return (*applied_out) > 0;
}

#endif  // PICO2W_SYNTH_ENABLE_USB_SERIAL_DEBUG

}  // namespace pico2w_synth
