# Changelog

All notable changes to `pico2w-synth` are recorded here.

Format:
- date: `YYYY-MM-DD`
- scope: affected subsystem(s)
- summary: user-visible behavior changes
- files: primary files edited
- validation: quick verification status

## 2026-03-25

### Added
- Web Serial keyboard now supports fixed 4-octave rendering with Auto Chord controls:
  - chord enable toggle
  - chord root selector
  - chord octave selector
  - chord type selector (major/minor/dom7/maj7/min7/sus2/sus4/dim/aug/power)

### Changed
- Keyboard panel is now sticky in the synth shell, so it remains accessible while scrolling.
- UI app container now expands to full browser width (removed fixed max width cap).
- Auto Chord mapping now follows the key pressed:
  - chord shape is transposed from keyboard position instead of always sounding only from the fixed root note.
- Panel section layout is now modular with fixed-size clusters:
  - sections no longer stretch to fill a 12-column fluid grid
  - clusters auto-pack by available width (`auto-fill` fixed tile tracks)
  - oscillator cluster spans 2 tiles, while filter/env/lfo/fx/amp/system each span 1 tile
  - enables layouts like filter beside oscillator when room exists, with remaining clusters flowing below.

### Files
- `web/serial_controller.html`
- `CHANGELOG.md`

### Validation
- Static sanity check completed (element IDs and handler wiring verified).
- Runtime browser validation pending user refresh/retest.
