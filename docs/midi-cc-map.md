# MIDI CC Map (Phase 1)

| CC | Param |
|---:|---|
| 102 | Osc1Wave |
| 104 | Osc2Wave |
| 21 | OscMix |
| 19 | Osc1Shape |
| 20 | Osc1Morph |
| 23 | SubNoiseMix |
| 85 | Osc2Coarse |
| 76 | Osc2Detune |
| 82 | OscDrift |
| 5 | Portamento |
| 74 | FilterCutoff |
| 71 | FilterReso |
| 39 | FilterMode |
| 24 | FilterEgAmt |
| 25 | FilterLfoAmt |
| 9 | FilterKeyTrack |
| 73 | EgAttack |
| 75 | EgDecay |
| 30 | EgSustain |
| 72 | EgRelease |
| 52 | AmpAttack |
| 53 | AmpDecay |
| 54 | AmpSustain |
| 55 | AmpRelease |
| 33 | LfoWave |
| 3 | LfoRate |
| 17 | LfoDepth |
| 56 | LfoFade |
| 13 | LfoOscAmt |
| 103 | LfoOscDst |
| 89 | EgOscAmt |
| 8 | EgOscDst |
| 93 | ChorusMix |
| 58 | ChorusRate |
| 59 | ChorusDepth |
| 22 | ChorusDelay |
| 94 | DelayMix |
| 90 | DelayTime |
| 92 | DelayFeedback |
| 35 | DelayMode |
| 91 | ReverbMix |
| 15 | AmpGain |
| 10 | Pan |
| 18 | VoiceMode |
| 57 | PitchBendRange |
| 95 | SystemBypassFx |

Notes:
- `Osc1Shape` is quantized to 4 selector states: triangle, square, saw, sine.
- `OscDrift` is bipolar in current firmware: `64` = neutral, below = tune down, above = tune up.
