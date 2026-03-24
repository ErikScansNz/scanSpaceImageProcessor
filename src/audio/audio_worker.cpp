#include "audio/audio_worker.h"

#include <algorithm>

#include "pico/multicore.h"
#include "pico/stdlib.h"
#include "pico/time.h"
#include "pico2w_synth/config.h"
#include "synth/synth_runtime.h"
#include "tusb.h"

#ifndef PICO2W_SYNTH_DISABLE_AUDIO_ENGINE
#define PICO2W_SYNTH_DISABLE_AUDIO_ENGINE 0
#endif

namespace pico2w_synth {

namespace {

int16_t Clamp16(int32_t x) {
  if (x > 32767) {
    return 32767;
  }
  if (x < -32768) {
    return -32768;
  }
  return static_cast<int16_t>(x);
}

constexpr uint32_t kEventDrainMaxPerSlice = 24u;

}  // namespace

AudioWorker* AudioWorker::g_instance_ = nullptr;

bool AudioWorker::Start(SynthEventQueue* queue) {
  if (queue == nullptr || running_) {
    return false;
  }

  queue_ = queue;
  core1_ready_ = false;
  stop_requested_ = false;
  running_ = false;
  heartbeat_ = 0;
  overrun_count_ = 0;
  frame_count_ = 0;
  sample_count_ = 0;
  active_param_count_ = 0;

  for (size_t i = 0; i < target_values_.size(); ++i) {
    target_values_[i] = 0;
    smoothed_values_[i] = 0;
    active_param_ids_[i] = 0;
    active_param_present_[i] = 0;
  }

  g_instance_ = this;
  multicore_reset_core1();
  multicore_launch_core1(&AudioWorker::Core1Entry);

  absolute_time_t deadline = make_timeout_time_ms(1000);
  while (!core1_ready_ && !time_reached(deadline)) {
    if (tusb_inited()) {
      tud_task();
    }
    sleep_ms(1);
  }

  return core1_ready_;
}

void AudioWorker::Stop() {
  stop_requested_ = true;
}

uint32_t AudioWorker::Heartbeat() const {
  return heartbeat_;
}

uint32_t AudioWorker::Overruns() const {
  return overrun_count_;
}

uint32_t AudioWorker::Frames() const {
  return frame_count_;
}

uint32_t AudioWorker::CallbackLastUs() const {
  return callback_last_us_;
}

uint32_t AudioWorker::CallbackMaxUs() const {
  return callback_max_us_;
}

uint32_t AudioWorker::CallbackOverBudget() const {
  return callback_over_budget_;
}

void AudioWorker::Core1Entry() {
  if (g_instance_ != nullptr) {
    g_instance_->RunCore1();
  }
  while (true) {
    sleep_ms(1000);
  }
}

void AudioWorker::RunCore1() {
  for (size_t i = 0; i < static_cast<size_t>(ParamId::ParamCount); ++i) {
    const ParamMeta* meta = FindParamMeta(static_cast<ParamId>(i));
    uint8_t def = meta ? meta->default_value : 0;
    target_values_[i] = def;
    smoothed_values_[i] = def;
    SynthRuntimeApplyParam(static_cast<ParamId>(i), def);
  }

  if (!audio_backend_.Start(kSampleRate)) {
    core1_ready_ = false;
    running_ = false;
    return;
  }

  if (!add_repeating_timer_us(-static_cast<int64_t>(kAudioCallbackIntervalUs),
                              &AudioWorker::AudioTimerThunk,
                              this,
                              &audio_timer_)) {
    core1_ready_ = false;
    running_ = false;
    audio_backend_.Stop();
    return;
  }

  running_ = true;
  core1_ready_ = true;

  while (!stop_requested_) {
    ++heartbeat_;
    sleep_ms(10);
  }

  cancel_repeating_timer(&audio_timer_);
  audio_backend_.Stop();
  running_ = false;
}

bool AudioWorker::AudioTimerThunk(struct repeating_timer* timer) {
  if (timer == nullptr || timer->user_data == nullptr) {
    return true;
  }
  AudioWorker* worker = static_cast<AudioWorker*>(timer->user_data);
  return worker->OnAudioTimer();
}

bool AudioWorker::OnAudioTimer() {
  uint32_t t_start_us = time_us_32();

  if (audio_busy_) {
    ++overrun_count_;
    audio_backend_.WriteStereo(0, 0);
    uint32_t elapsed_busy = time_us_32() - t_start_us;
    callback_last_us_ = elapsed_busy;
    if (elapsed_busy > callback_max_us_) {
      callback_max_us_ = elapsed_busy;
    }
    if (elapsed_busy > kAudioCallbackIntervalUs) {
      ++callback_over_budget_;
    }
    return true;
  }
  audio_busy_ = true;

  ++sample_count_;
  if ((sample_count_ & (kControlRateDiv - 1u)) == 0u) {
    // Only touch the cross-core event queue at control-rate cadence.
    // This materially reduces lock traffic between cores while keeping
    // sub-millisecond control latency.
    Event ev{};
    uint32_t drained = 0;
    while (drained < kEventDrainMaxPerSlice && queue_ != nullptr && queue_->TryPop(&ev)) {
      HandleEvent(ev);
      ++drained;
    }
    UpdateSmoothedParams();
  }

#if PICO2W_SYNTH_DISABLE_AUDIO_ENGINE
  audio_backend_.WriteStereo(0, 0);
#else
  int16_t left = 0;
  int16_t right = 0;
  SynthRuntimeProcessFrame(&left, &right);
  ApplyOutputSafety(&left, &right);
  audio_backend_.WriteStereo(left, right);
#endif

  ++frame_count_;
  ++heartbeat_;
  uint32_t elapsed_us = time_us_32() - t_start_us;
  callback_last_us_ = elapsed_us;
  if (elapsed_us > callback_max_us_) {
    callback_max_us_ = elapsed_us;
  }
  if (elapsed_us > kAudioCallbackIntervalUs) {
    ++callback_over_budget_;
  }
  audio_busy_ = false;
  return true;
}

void AudioWorker::HandleEvent(const Event& ev) {
  switch (ev.type) {
    case EventType::NoteOn:
      SynthRuntimeHandleNoteOn(ev.channel, ev.data0, ev.data1);
      return;

    case EventType::NoteOff:
      SynthRuntimeHandleNoteOff(ev.channel, ev.data0, ev.data1);
      return;

    case EventType::PitchBend: {
      uint16_t value14 = static_cast<uint16_t>(ev.data0) | (static_cast<uint16_t>(ev.data1) << 7u);
      SynthRuntimeHandlePitchBend(ev.channel, value14);
      return;
    }

    case EventType::ChannelPressure:
      SynthRuntimeHandleChannelPressure(ev.channel, ev.data0);
      return;

    case EventType::PolyAftertouch:
      SynthRuntimeHandleChannelPressure(ev.channel, ev.data1);
      return;

    case EventType::ParamSet:
      if (ev.param_id < static_cast<uint16_t>(ParamId::ParamCount)) {
        uint16_t id = ev.param_id;
        uint8_t clamped = ClampParamValue(static_cast<ParamId>(id), ev.data1);
        target_values_[id] = clamped;
        if (smoothed_values_[id] != clamped && active_param_present_[id] == 0u) {
          if (active_param_count_ < static_cast<uint8_t>(active_param_ids_.size())) {
            active_param_ids_[active_param_count_++] = static_cast<uint8_t>(id);
            active_param_present_[id] = 1u;
          }
        }
      }
      return;
  }
}

void AudioWorker::UpdateSmoothedParams() {
  uint8_t write_index = 0u;
  for (uint8_t read_index = 0u; read_index < active_param_count_; ++read_index) {
    uint8_t id_u8 = active_param_ids_[read_index];
    size_t i = static_cast<size_t>(id_u8);
    ParamId id = static_cast<ParamId>(i);
    uint8_t target = target_values_[i];
    uint8_t current = smoothed_values_[i];

    const ParamMeta* meta = FindParamMeta(id);
    SmoothingPolicy policy = meta ? meta->smoothing : SmoothingPolicy::None;

    uint8_t next = current;
    if (current != target) {
      switch (policy) {
        case SmoothingPolicy::None:
          next = target;
          break;

        case SmoothingPolicy::Slow:
          if (current < target) {
            next = static_cast<uint8_t>(std::min<int>(current + 1, target));
          } else {
            next = static_cast<uint8_t>(std::max<int>(current - 1, target));
          }
          break;

        case SmoothingPolicy::Fast:
          if (current < target) {
            next = static_cast<uint8_t>(std::min<int>(current + 4, target));
          } else {
            int candidate = static_cast<int>(current) - 4;
            next = static_cast<uint8_t>(candidate > static_cast<int>(target) ? candidate : target);
          }
          break;
      }
    }

    if (next != current) {
      smoothed_values_[i] = next;
      SynthRuntimeApplyParam(id, next);
    }

    if (next != target) {
      active_param_ids_[write_index++] = id_u8;
    } else {
      active_param_present_[i] = 0u;
    }
  }
  active_param_count_ = write_index;
}

void AudioWorker::ApplyOutputSafety(int16_t* left, int16_t* right) {
  int32_t l = static_cast<int32_t>(*left);
  int32_t r = static_cast<int32_t>(*right);

  l = (l * 31) / 32;
  r = (r * 31) / 32;

  if (dither_enabled_) {
    dither_lfsr_ = (dither_lfsr_ >> 1u) ^ (-(static_cast<int32_t>(dither_lfsr_ & 1u)) & 0xD0000001u);
    int32_t d = static_cast<int32_t>(dither_lfsr_ & 0x3u) - 1;
    l += d;
    r -= d;
  }

  if (noise_shaping_enabled_) {
    int32_t shaped_l = l + ns_error_l_;
    int32_t shaped_r = r + ns_error_r_;
    int16_t ql = Clamp16(shaped_l);
    int16_t qr = Clamp16(shaped_r);
    ns_error_l_ = shaped_l - ql;
    ns_error_r_ = shaped_r - qr;
    *left = ql;
    *right = qr;
    return;
  }

  *left = Clamp16(l);
  *right = Clamp16(r);
}

}  // namespace pico2w_synth
