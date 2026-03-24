# Acceptance Criteria

## Build and Artifact
- [ ] Clean configure/build for `pico2_w` completes.
- [ ] `.elf`, `.bin`, `.hex` artifacts are generated.
- [ ] Size report captured and tracked.

## Audio and DSP
- [ ] PWM backend produces stable stereo output.
- [ ] 4-voice note allocation works without stuck voices.
- [ ] ADSR response is correct for filter and amp envelopes.
- [ ] LFO/EG routing audibly affects pitch/shape/cutoff.
- [ ] FX chain order is verified: `mix -> pan -> chorus -> delay`.
- [ ] No clipping artifacts under nominal gain staging.

## Real-Time Stability
- [ ] No audible dropouts under sustained MIDI note stream.
- [ ] No audible dropouts under rapid USB serial parameter updates.
- [ ] Overrun counter remains within target budget.
- [ ] Watchdog behavior (if enabled) is defined and validated.

## MIDI and Serial Input
- [ ] USB CDC serial connect/control path is stable.
- [ ] UART MIDI Note On/Off handling passes.
- [ ] USB MIDI handling passes when enabled.
- [ ] Simultaneous MIDI + serial updates remain deterministic.
- [ ] CC mapping updates target parameters correctly.
- [ ] Unsupported/ignored messages do not destabilize runtime.

## Control Equivalence
- [ ] Same parameter update from MIDI and serial produces same synthesis result.
- [ ] Hardware-control path (when enabled) uses shared registry semantics.
- [ ] Timestamp conflict behavior is last-write-wins.

## Documentation and Governance
- [ ] Workspace `AGENTS.md` present and current.
- [ ] Project README reflects external-control architecture.
- [ ] Wiring docs match active firmware defaults.
- [ ] Task board reflects current implementation status.
