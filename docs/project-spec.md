# Project Specification: External-Control Runtime

## Overview
Build a Pico 2 W based digital synth firmware rooted in PRA32-U2, optimized for:
- low-latency audio
- external control only (USB serial + MIDI)
- no onboard Wi-Fi/web server or display runtime

## Requirements Summary
| Category | Requirement |
|---|---|
| MCU | RP2350 (Pico 2 W) |
| Audio Out | PWM stereo on GP27/GP28 + RC filter |
| MIDI In | UART DIN/TRS MIDI @ 31250 bps, optional USB MIDI |
| Control UI | Externally hosted Web Serial page over USB CDC |
| Local Controls | 4 encoders + 1 pot mapped on GP26 (`POT2`/`POT3` reserved) + buttons |

## Detailed Requirements
### Synthesis Engine
- PRA32-U2 oscillator/filter/EG/LFO/amp/chorus/delay architecture.
- Deterministic fixed-point signal path where practical.
- 4-voice processing and monophonic mode option.

### Parameter Architecture
- Shared registry with `ParamId` and `0..127` values.
- Metadata: name, page, default, smoothing policy.
- Conflict resolution: timestamped last-write-wins.

### Input Subsystems
- USB serial command channel (TinyUSB CDC).
- USB MIDI parser (optional, build flag).
- UART MIDI parser for DIN/TRS receiver.
- Unified non-blocking event queue for all control sources.

### Audio Backends
- Active: PWM stereo backend.
- Stubs: I2S and Bluetooth audio backends.

## Constraints
- Legacy repos are read-only references.
- Audio callback avoids blocking I/O and dynamic allocation.
- Onboard networking/display paths are removed from active runtime.
- Pin map frozen for prototype validation.
