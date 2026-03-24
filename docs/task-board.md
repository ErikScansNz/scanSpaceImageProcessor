# Task Board

Status legend: `[ ]` not started, `[-]` in progress, `[x]` done.

## Phase 1 - Bootstrap + Architecture
- [x] `[Architect]` Create `pico2w-synth` project scaffold and module layout.
- [x] `[Architect]` Set board target `pico2_w` and compile definitions.
- [x] `[Architect]` Add deterministic fixed-point oriented compile/runtime defaults.
- [x] `[QA]` Add CI workflow skeleton for Pico SDK build.

## Phase 2 - DSP Port (PRA32-first)
- [x] `[DSP]` Port oscillator/filter/EG/LFO/amp/panner/chorus/delay modules.
- [x] `[DSP]` Integrate DSP into `SynthEngine` with 4-voice processing.
- [x] `[DSP]` Add optional reverb hook (disabled by default).
- [-] `[QA]` Validate callback sample timing and CPU headroom on target hardware.

## Phase 3 - Event + Parameter Bus
- [x] `[MIDI/IO]` Add ring-queue event bus for note/mod/param events.
- [x] `[Architect]` Implement unified `ParamId` schema and metadata.
- [x] `[Architect]` Implement timestamped registry (last-write-wins).
- [x] `[Architect]` Add preset snapshot JSON export/import helpers.
- [-] `[QA]` Stress test queue under simultaneous MIDI + web updates.

## Phase 4 - MIDI Integration
- [x] `[MIDI/IO]` Add UART MIDI byte reader (31250 bps).
- [x] `[MIDI/IO]` Add USB MIDI polling/parser via TinyUSB.
- [x] `[MIDI/IO]` Merge USB+UART into one event stream.
- [x] `[MIDI/IO]` Map mandatory CC set for osc/filter/env/mod/fx.
- [-] `[QA]` Validate deterministic merge order with simultaneous sources.

## Phase 5 - Web UI (AP Mode)
- [x] `[Web UI]` Bring up Pico W AP mode.
- [x] `[Web UI]` Implement `GET /`, `GET /api/meta`, `GET /api/state`, `POST /api/set`, `POST /api/batch`.
- [x] `[Web UI]` Build mobile-first single-page controller UI.
- [x] `[Web UI]` Implement delta-only throttled updates on client.
- [-] `[QA]` Verify cold-boot connect flow and API behavior on-device.

## Phase 6 - Audio Output + Safety
- [x] `[DSP]` Implement PWM stereo output backend.
- [x] `[DSP]` Add callback-side clamp/limiter stage.
- [x] `[DSP]` Add optional dither/noise-shaping toggles in runtime path.
- [x] `[DSP]` Add watchdog-based recovery guard for runtime stalls.
- [-] `[QA]` Measure audible noise floor and overrun thresholds.

## Phase 7 - Display + Hardware Control Hooks
- [x] `[Hardware]` Implement SSD1306 status/page renderer.
- [x] `[Hardware]` Add 4-encoder/3-pot abstraction stubs behind feature flag.
- [x] `[Architect]` Ensure MIDI/web/hardware all route through shared registry API.
- [-] `[Hardware]` Implement full encoder detent decoding and pot filtering.

## Phase 8 - Wiring + Validation Artifacts
- [x] `[Hardware]` Publish prototype pin map and interface wiring docs.
- [x] `[Hardware]` Publish MIDI DIN/TRS input front-end guidance.
- [x] `[Hardware]` Publish PWM RC filter and line-out guidance.
- [x] `[Hardware]` Publish SSD1306 I2C wiring constraints.
- [-] `[Hardware]` Build and test prototype harness against docs.

## Phase 9 - Documentation + Governance
- [x] `[Architect]` Create workspace-level `AGENTS.md`.
- [x] `[Architect]` Create project README and architecture notes.
- [x] `[QA]` Create acceptance criteria checklist.
- [-] `[QA]` Execute checklist on physical hardware and capture evidence.

## Phase 10 - USB Serial Control Bring-up
- [x] `[MIDI/IO]` Add USB enumeration-only diagnostic firmware profile (`-UsbEnumOnly`).
- [x] `[MIDI/IO]` Make USB MIDI reader compile-safe when MIDI class is disabled.
- [x] `[MIDI/IO]` Harden USB CDC serial control parser and add `STATS` diagnostics command.
- [x] `[Web UI]` Rework local Web Serial page with queued TX writer and runtime stats panel.
- [x] `[Architect]` Make Wi-Fi HTTP server compile to stub when Wi-Fi build flag is disabled.
- [x] `[MIDI/IO]` Add build toggle to disable USB MIDI and run UART/GPIO MIDI-only with USB CDC control.
- [x] `[Hardware]` Add SSD1306 timeout hardening during prior USB bring-up investigation (historical).
- [x] `[Web UI]` Move controller to client-authoritative local schema and quiet write-only serial updates.
- [-] `[QA]` Validate long-run USB CDC stability under sustained parameter sweeps.

## Phase 11 - External-Control-Only Runtime
- [x] `[Architect]` Remove onboard Wi-Fi/web server and SSD1306 display from active runtime path.
- [x] `[Architect]` Remove Wi-Fi/display build options and dependency links from default build chain.
- [x] `[Docs]` Update README to external-control-only architecture and build flow.
