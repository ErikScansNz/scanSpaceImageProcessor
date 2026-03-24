# USB Communication Fix Notes (2026-03-23)

## Symptom
- Windows Device Manager showed `Unknown USB Device (Device Descriptor Request Failed)`.
- Firmware LEDs blinked, but no stable CDC COM port appeared.
- In early builds, Wi-Fi and runtime init made USB diagnosis ambiguous.

## What Changed
- Added `-UsbEnumOnly` firmware profile:
  - Initializes TinyUSB and runs `tud_task()` only.
  - Disables MIDI class in this profile for CDC-only validation.
- Hardened TinyUSB MIDI path:
  - `midi_usb.cpp` now compiles and runs correctly when MIDI class is disabled.
- Hardened CDC serial command path:
  - strict command parsing for `SET`.
  - buffered TX responses (prevents drops during bursty host traffic).
  - new `STATS` command for parser/runtime telemetry.
- Updated local browser controller:
  - queued TX writes to avoid overlapping async `writer.write()` calls.
  - switched to client-authoritative local schema/defaults and quiet write-only controls (`SETQ`/`BATCHQ`) for minimal device load.
- USB interface profile decoupling:
  - default build now uses CDC-only (`PICO2W_SYNTH_ENABLE_USB_MIDI=0`) for stable serial control bring-up.
  - USB MIDI can be re-enabled explicitly when needed.
- Runtime simplification (current):
  - onboard Wi-Fi/web and SSD1306 display paths removed from active firmware build/runtime.
- Watchdog bring-up mode:
  - watchdog is build-toggled and defaults OFF during USB bring-up to avoid reset loops masking serial-connect issues.

## Validation Flow
1. Flash enum-only profile:
   - `.\scripts\build_flash.ps1 -Clean -UsbEnumOnly`
2. Confirm COM port appears in Device Manager.
3. Flash default USB serial profile:
   - `.\scripts\build_flash.ps1 -Clean`
4. Open `web/serial_controller.html`, connect, and verify:
   - connection succeeds without browser freeze.
   - parameter edits send quiet commands (`SETQ`/`BATCHQ`).
   - firmware reports errors only when parsing/queue failures occur.

## Remaining QA
- Run sustained sweep/load tests over USB CDC + USB MIDI simultaneously.
- Confirm no dropped control updates under 5+ minute continuous UI activity.
