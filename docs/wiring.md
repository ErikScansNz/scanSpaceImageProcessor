# Wiring Plan (Prototype v1)

This document is the phase-1 wiring baseline for Pico 2 W synth bring-up.

## 1. Frozen Pin Map
- MIDI UART RX: `GP1`
- MIDI UART TX (optional debug/thru): `GP0`
- SSD1306 I2C SDA: `GP2`
- SSD1306 I2C SCL: `GP3`
- Encoder 1: `A=GP4`, `B=GP5`, `SW=GP6`
- Encoder 2: `A=GP7`, `B=GP8`, `SW=GP9`
- Encoder 3: `A=GP10`, `B=GP11`, `SW=GP12`
- Encoder 4: `A=GP13`, `B=GP14`, `SW=GP15`
- PWM audio left: `GP27`
- PWM audio right: `GP28`
- Optional compatibility mirror output: `GP18`/`GP19` carries the same PWM during bring-up
- Potentiometer 1: `GP26` (ADC0)
- Potentiometer 2: `UNASSIGNED` (reserved)
- Potentiometer 3: `UNASSIGNED` (reserved)

## 2. Power and Ground
- Pico 2 W on USB 5V input.
- Logic domain is `3V3`.
- All controls/display/audio filter grounds return to Pico ground.
- Use a star-ish return for audio filter and output jack ground where possible.

## 3. MIDI Input Front-End (DIN-5 or TRS Type-A)

### Required
- MIDI input must be opto-isolated.
- UART pin `GP1` is 3.3V logic only.

### Recommended Receiver Topology
- Use a standard MIDI IN optocoupler stage (6N138, 6N137, or H11L1 class).
- DIN/TRS current loop side:
  - 220R series resistor network per MIDI receiver reference.
  - Reverse-protection diode across opto LED.
- Logic output side:
  - Opto output pulled up to `3V3` with `4.7k` to `10k`.
  - Output to `GP1`.
  - Optional RC edge cleanup: `100R` series + `100pF` to ground at MCU side.

### Connector Notes
- DIN-5: follow MIDI 1.0 receiver wiring for pins 4/5 loop.
- TRS: support Type-A mapping by default (document if adapter used).

## 4. Encoders + Push Buttons
- Each encoder A/B channel:
  - channel pin -> GPIO
  - common -> GND
  - use internal pull-ups (or external 10k pull-up if cable/noisy)
- Push switch:
  - SW pin -> GPIO
  - other side -> GND
  - internal pull-up enabled
- Optional hardware debounce per line: `1k` series + `100nF` to ground.

## 5. Potentiometers (10k)
- Pot end 1 -> `3V3`
- Pot end 2 -> `GND`
- Wiper -> ADC pin (`GP26`)
- Recommended anti-noise on each wiper:
  - `1k` series to ADC pin
  - `100nF` from ADC pin to GND

## 6. PWM Stereo Test Output

### Phase-1 Goal
- Produce listenable test output before I2S DAC is available.

### Per Channel Filter (minimum)
- `GPIO PWM` -> `1k` series -> output node
- output node -> `10nF` to GND
- cutoff approx `15.9kHz` (single-pole)

### Better Two-Pole Passive (recommended)
- Stage 1: `1k` + `10nF`
- Stage 2: another `1k` + `10nF`
- AC-couple to line output with `1uF` capacitor
- Add `100k` resistor to GND at jack side as reference bleed

### Protection/Usability
- Optional `220R` series on each final L/R line into jack.
- Keep PWM filter physically close to Pico pins.

## 7. SSD1306 I2C 0.96" Display
- `VCC` -> `3V3` (or module-rated supply)
- `GND` -> GND
- `SDA` -> `GP2`
- `SCL` -> `GP3`
- If module has no pull-ups, add `4.7k` pull-up on SDA/SCL to 3V3.
- Keep I2C wires short (<20cm) for reliable 400kHz operation.

## 8. Prototype Bring-Up Order
1. Power + USB flash + serial sanity check.
2. PWM audio RC output and basic tone check.
3. SSD1306 connectivity check.
4. USB MIDI input validation.
5. UART MIDI input front-end validation.
6. Add one encoder + one pot, then scale to full panel.

## 9. Net Naming (for schematic)
- `VCC_3V3`, `VBUS_5V`, `GND`
- `MIDI_RX_UART0`, `MIDI_TX_UART0`
- `I2C1_SDA`, `I2C1_SCL`
- `PWM_AUDIO_L`, `PWM_AUDIO_R`
- `POT1_ADC0` (`POT2_ADC1` / `POT3_ADC2` reserved for future remap)
- `ENC1_A/B/SW` ... `ENC4_A/B/SW`
