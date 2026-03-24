# Coding Conventions

## Language and Tooling
- C++17 for firmware modules, C for TinyUSB descriptors.
- Pico SDK target: `pico2_w`.

## Compile Defaults
- `PICO2W_SYNTH_FIXED_POINT=1`
- `PICO2W_SYNTH_DEFAULT_SAMPLE_RATE=48000`
- `PICO2W_SYNTH_ENABLE_REVERB=0` (phase 1 default)
- `PICO2W_SYNTH_ENABLE_HW_CONTROLS=0` (phase 1 default)

## Warnings / Flags
- `-Wall -Wextra -Wshadow -Wconversion -Wdouble-promotion`
- `-fno-exceptions -fno-rtti`

## Runtime Rules
- Audio callback must not block on I/O.
- Audio callback must not allocate memory.
- All external control sources must route through event bus + parameter registry.
- Parameter conflict policy: timestamped last-write-wins.

## Module Boundaries
- `synth`: DSP and synthesis voice logic only.
- `audio`: transport/backends only.
- `midi`: byte parsing + event translation.
- `serial`: USB CDC transport + command parsing.
- `control`: physical input abstraction.
- `app`: orchestration and lifecycle.
