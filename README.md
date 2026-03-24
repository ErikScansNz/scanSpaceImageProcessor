# pico2w-synth

Pico 2 W digital synth firmware with an external-control-only runtime:
- no onboard Wi-Fi/web server
- no onboard display runtime
- control comes from USB CDC serial and MIDI inputs

## Runtime Model
- Core0:
  - USB CDC serial control parser (`SETQ`/`BATCHQ` style updates)
  - USB MIDI + UART MIDI ingest
  - hardware controls ingest
  - watchdog gating (optional)
- Core1:
  - synth runtime + PWM audio only

All control sources route through one shared non-blocking event/parameter bus.

## Phase 1 Scope
- PRA32-derived synth engine:
  - dual oscillator, filter, envelopes, LFO, modulation routing
  - chorus + delay FX chain (reverb disabled by default)
- Unified `ParamId` registry (`0..127`)
- MIDI input:
  - UART MIDI (`GP1` RX, 31250 bps)
  - USB MIDI (optional)
- USB serial control:
  - TinyUSB CDC command channel
  - protocol supports `SET/SETQ`, `BATCH/BATCHQ`, plus diagnostics commands
- Audio output:
  - PWM stereo backend (active)
  - I2S/Bluetooth backends as stubs

## Project Layout
- `src/app` runtime orchestration + diagnostics
- `src/audio` PWM backend + core1 worker
- `src/synth` synth engine and PRA32 DSP modules
- `src/midi` USB/UART MIDI parsing + mapping
- `src/serial` USB CDC serial protocol
- `src/params` schema/registry/snapshots
- `src/control` hardware control abstraction/stub
- `web/serial_controller.html` local, externally hosted Web Serial UI

## Build
Prerequisites:
- Raspberry Pi Pico SDK 2.x
- ARM GCC toolchain (`arm-none-eabi-gcc`)
- CMake + Ninja

Known-good Windows build:
```powershell
cd C:\softwareDevelopment\picoSillynth\pico2w-synth
$tc = "C:/Program Files (x86)/Arm/GNU Toolchain mingw-w64-i686-arm-none-eabi/bin"

cmake -S . -B build-ninja -G Ninja `
  -DCMAKE_MAKE_PROGRAM=C:/ninja/ninja.exe `
  -DPICO_BOARD=pico2_w `
  -DPICO_SDK_PATH=C:/softwareDevelopment/picoSillynth/pico-sdk `
  -Dpioasm_DIR=C:/softwareDevelopment/picoSillynth/pioasm `
  "-DPICO_TOOLCHAIN_PATH=$tc" `
  -DPICO_NO_PICOTOOL=1 `
  -DPICO_NO_UF2=1

cmake --build build-ninja --parallel 8
```

Automated build/flash:
```powershell
.\scripts\build_flash.ps1
```

Common options:
```powershell
# Build only
.\scripts\build_flash.ps1 -NoFlash

# Clean rebuild
.\scripts\build_flash.ps1 -Clean

# USB enum-only diagnostic firmware
.\scripts\build_flash.ps1 -Clean -UsbEnumOnly

# Disable USB CDC serial control
.\scripts\build_flash.ps1 -Clean -NoUsbSerialDebug

# Enable USB MIDI (composite USB MIDI + CDC)
.\scripts\build_flash.ps1 -Clean -EnableUsbMidi

# Enable chorus/delay DSP paths (disabled by default for stability)
.\scripts\build_flash.ps1 -Clean -EnableChorus -EnableDelay

# Enable expensive modulation destinations (shape/morph routes)
.\scripts\build_flash.ps1 -Clean -EnableExpensiveModDest

# Enable reduced oscillator complexity mode (fallback stability throttle)
.\scripts\build_flash.ps1 -Clean -RestrictOsc

# Build low-CPU profile (24kHz + lighter control cadence)
.\scripts\build_flash.ps1 -Clean -LowCpuProfile

# Keep FX chain enabled at boot (default is bypassed for stability)
.\scripts\build_flash.ps1 -Clean -DisableFxBypassAtBoot

# Disable synth DSP in audio worker (silent output, serial/control debug)
.\scripts\build_flash.ps1 -Clean -DisableAudioEngine

# Enable SDK stdio over USB
.\scripts\build_flash.ps1 -Clean -EnableUsbStdio

# Enable watchdog gating
.\scripts\build_flash.ps1 -Clean -EnableWatchdog
```

## Runtime Defaults
- External control path: USB serial + MIDI
- USB CDC serial debug: enabled
- USB MIDI: disabled
- USB stdio: disabled
- Watchdog: disabled
- Chorus DSP: disabled
- Delay DSP: disabled
- Expensive modulation destinations: disabled
- Oscillator complexity: full mode (`-RestrictOsc` enables reduced-complexity mode)
- FX chain: bypassed at boot (can be toggled with `ParamId::SystemBypassFx`)
- Audio sample target: 48 kHz (`-LowCpuProfile` uses 24 kHz)
- Full-rate profile uses divided synth render cadence for USB/control stability
- Audio engine DSP: enabled (`-DisableAudioEngine` for silent debug build)
- Master output stage: +10 dB gain trim to increase max loudness at full `AMP GAIN`
- Pico 2 W onboard LED may remain off in this external-control build (firmware no longer uses CYW43 LED path)

## Diagnostics (STATS)
`STATS` includes:
- queue depth + dropped event counters
- audio overrun counters
- audio callback timing (`audio_cb_last_us`, `audio_cb_max_us`, `audio_cb_over_budget`)
- audio callback load counters (`audio_frames`, `audio_cb_over_budget_delta`, `audio_cb_over_budget_pct_x100`)
- runtime timing constants (`sample_rate_hz`, `audio_frame_interval_us`, `audio_render_div`, `audio_callback_interval_us`, `main_loop_sleep_us`)
- DSP feature flags (`chorus_enabled`, `delay_enabled`, `exp_mod_dest_enabled`, `restrict_osc_enabled`)
- core0 timing breakdown (`core0_loop_*`, `core0_midi_poll_*`, `core0_serial_poll_*`, `core0_control_poll_*`)

## USB Serial Controller (Local Page)
- Local page: `web/serial_controller.html`
- Host it from your PC:
  - `python -m http.server 8080`
  - open `http://localhost:8080/web/serial_controller.html`
- Two controller profiles are available:
  - **Hardware (Pico over USB CDC):**
    - `http://localhost:8080/web/serial_controller.html?mode=usb`
    - shortcut: `http://localhost:8080/web/serial_controller_usb.html`
  - **Local Emulator (no hardware required):**
    - `http://localhost:8080/web/serial_controller.html?mode=emu`
    - shortcut: `http://localhost:8080/web/serial_controller_emu.html`
  - You can also switch modes at runtime with the `Mode: USB` / `Mode: Emulator` button in the UI.
- Controller is client-authoritative:
  - browser owns UI schema/defaults
  - sends parameter writes to device (`SETQ`/`BATCHQ`)
  - includes a local playable test keyboard (mouse/touch + computer-key mapping) using `NOTEON`/`NOTEOFF`
  - enforces the same min/max parameter clamps as firmware (`META` range contract)
  - tonal safety ranges are enabled by default for oscillator wave selection (`osc1_wave`/`osc2_wave` limited to saw/square/triangle/sine bucket range)
  - oscillator architecture now exposes 4 mixable oscillators in the UI:
    - per-osc waveform selection for osc2/osc3/osc4
    - dedicated level sliders for `osc1_level`, `osc2_level`, `osc3_level`, `osc4_level`
  - heavy controls (wave/shape/fx/mod-destination) commit on release to reduce USB/control load
  - control taxonomy is explicit in UI (`knob`, `slider`, `click-knob`, `toggle`, `switch`) with per-control guide text
  - includes `Simple Mode`/`Advanced Mode` control-density toggle:
    - `Simple Mode` shows a reduced parameter set per page
    - `Advanced Mode` exposes the full parameter surface
  - osc page includes `Supersaw` switch:
    - when enabled, firmware forces both oscillator banks to saw and engages the internal 7-voice detuned saw path
  - filter page uses:
    - `Filter Mode` as LP/HP toggle (not numeric)
    - dedicated `Lowpass Cutoff` and `Highpass Cutoff` knobs; moving either knob auto-selects its mode
  - discrete label conventions are aligned to PRA32-style mappings (wave/filter/LFO mode text) and checked against local reference repos (`digital-synth-pra32-u2`, `pico_synthesizer`, `Picophonica`, `synthiota`)
  - no onboard web/self-hosted UI required
  - local emulator mode implements the same line protocol (`PING`, `META`, `STATE`, `STATS`, `SETQ`, `BATCHQ`, `NOTEON`, `NOTEOFF`) so control-flow can be tested before flashing hardware
  - local emulator mode now includes a browser DSP engine (Web Audio) so keyboard notes and parameter edits produce audible synth output without hardware (approximate behavior; not bit-identical to RP2350 DSP path)
  - LFO speed range has a widened high-end curve so top-end values reach substantially faster modulation rates

## Pin Map
- MIDI UART RX `GP1`, TX `GP0` (TX optional)
- PWM L `GP27`, PWM R `GP28`
- Compatibility mirror (same PWM signal): `GP18/GP19` (enabled by default for bring-up)
- Pots `GP26` active (`POT2`/`POT3` unassigned in current firmware pin map)
- Enc1 `GP4/GP5`, SW `GP6`
- Enc2 `GP7/GP8`, SW `GP9`
- Enc3 `GP10/GP11`, SW `GP12`
- Enc4 `GP13/GP14`, SW `GP15`

## Governance
- Workspace rules: `../AGENTS.md`
- Task board: `docs/task-board.md`
- Acceptance checklist: `docs/acceptance-criteria.md`
- Serial protocol: `docs/serial-protocol.md`
- Wiring: `docs/wiring.md`
