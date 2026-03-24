#pragma once

#include <cstdint>

#ifndef PICO2W_SYNTH_ENABLE_WATCHDOG
#define PICO2W_SYNTH_ENABLE_WATCHDOG 0
#endif

#ifndef PICO2W_SYNTH_ENABLE_CHORUS
#define PICO2W_SYNTH_ENABLE_CHORUS 0
#endif

#ifndef PICO2W_SYNTH_ENABLE_DELAY
#define PICO2W_SYNTH_ENABLE_DELAY 0
#endif

#ifndef PICO2W_SYNTH_ENABLE_EXPENSIVE_MOD_DEST
#define PICO2W_SYNTH_ENABLE_EXPENSIVE_MOD_DEST 0
#endif

#ifndef PICO2W_SYNTH_RESTRICT_OSC
#define PICO2W_SYNTH_RESTRICT_OSC 0
#endif

#ifndef PICO2W_SYNTH_DEFAULT_SAMPLE_RATE
#define PICO2W_SYNTH_DEFAULT_SAMPLE_RATE 48000
#endif

#ifndef PICO2W_SYNTH_CONTROL_RATE_DIV
#define PICO2W_SYNTH_CONTROL_RATE_DIV 4
#endif

#ifndef PICO2W_SYNTH_MAIN_LOOP_SLEEP_US
#define PICO2W_SYNTH_MAIN_LOOP_SLEEP_US 500
#endif

#ifndef PICO2W_SYNTH_DEFAULT_BYPASS_FX
#define PICO2W_SYNTH_DEFAULT_BYPASS_FX 1
#endif

#ifndef PICO2W_SYNTH_AUDIO_RENDER_DIV
#define PICO2W_SYNTH_AUDIO_RENDER_DIV 1
#endif

#ifndef PICO2W_SYNTH_ENABLE_PWM_MIRROR_18_19
#define PICO2W_SYNTH_ENABLE_PWM_MIRROR_18_19 1
#endif

namespace pico2w_synth {

constexpr uint32_t kSampleRate = static_cast<uint32_t>(PICO2W_SYNTH_DEFAULT_SAMPLE_RATE);
constexpr uint32_t kControlRateDiv = static_cast<uint32_t>(PICO2W_SYNTH_CONTROL_RATE_DIV);
constexpr uint32_t kAudioFrameIntervalUs = (1000000u + (kSampleRate / 2u)) / kSampleRate;
constexpr uint32_t kAudioRenderDiv = static_cast<uint32_t>(PICO2W_SYNTH_AUDIO_RENDER_DIV);
constexpr uint32_t kAudioCallbackIntervalUs = kAudioFrameIntervalUs * kAudioRenderDiv;
constexpr uint32_t kMainLoopSleepUs = static_cast<uint32_t>(PICO2W_SYNTH_MAIN_LOOP_SLEEP_US);
constexpr uint32_t kEventQueueCapacity = 512;

static_assert(kSampleRate >= 8000u && kSampleRate <= 96000u, "kSampleRate must be in [8000, 96000]");
static_assert(kControlRateDiv != 0u && ((kControlRateDiv & (kControlRateDiv - 1u)) == 0u),
              "kControlRateDiv must be a non-zero power of two");
static_assert(kAudioRenderDiv != 0u && kAudioRenderDiv <= 8u,
              "kAudioRenderDiv must be in [1, 8]");
static_assert(kAudioCallbackIntervalUs > 0u, "kAudioCallbackIntervalUs must be non-zero");
static_assert(kMainLoopSleepUs <= 5000u, "kMainLoopSleepUs is unexpectedly high");

// Prototype pin map (frozen v1)
constexpr uint8_t kPinMidiUartTx = 0;
constexpr uint8_t kPinMidiUartRx = 1;

constexpr uint8_t kPinI2cSda = 2;
constexpr uint8_t kPinI2cScl = 3;

constexpr uint8_t kPinEnc1A = 4;
constexpr uint8_t kPinEnc1B = 5;
constexpr uint8_t kPinEnc1Sw = 6;

constexpr uint8_t kPinEnc2A = 7;
constexpr uint8_t kPinEnc2B = 8;
constexpr uint8_t kPinEnc2Sw = 9;

constexpr uint8_t kPinEnc3A = 10;
constexpr uint8_t kPinEnc3B = 11;
constexpr uint8_t kPinEnc3Sw = 12;

constexpr uint8_t kPinEnc4A = 13;
constexpr uint8_t kPinEnc4B = 14;
constexpr uint8_t kPinEnc4Sw = 15;

constexpr uint8_t kPinPwmLeft = 27;
constexpr uint8_t kPinPwmRight = 28;
constexpr uint8_t kPinPwmLegacyLeft = 18;
constexpr uint8_t kPinPwmLegacyRight = 19;

constexpr uint8_t kPinPot1 = 26;
constexpr uint8_t kPinUnused = 0xFFu;
constexpr uint8_t kPinPot2 = kPinUnused;
constexpr uint8_t kPinPot3 = kPinUnused;

constexpr uint32_t kMidiUartBaud = 31250;

constexpr bool kEnableWatchdog = (PICO2W_SYNTH_ENABLE_WATCHDOG != 0);
constexpr bool kEnablePwmMirrorLegacyPins = (PICO2W_SYNTH_ENABLE_PWM_MIRROR_18_19 != 0);

}  // namespace pico2w_synth
