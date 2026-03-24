# USB Serial Control Protocol

Transport:
- USB CDC (TinyUSB), UTF-8 text lines terminated by `\n`
- Command parser is case-sensitive
- Web UI also includes a local emulator transport that mirrors this command/response contract for no-hardware testing.

Commands:
- `PING`
- `META`
- `STATE`
- `STATS`
- `NOTEON <note> <velocity>`
- `NOTEOFF <note> <velocity>`
- `SET <id> <value>`
- `SETQ <id> <value>` (quiet, no ack on success)
- `BATCH <id>=<value>,<id>=<value>,...`
- `BATCHQ <id>=<value>,<id>=<value>,...` (quiet, no ack on success)
- `HELP`

Responses are JSON lines:
- `{"type":"pong"}`
- `{"type":"meta","pages":[...],"params":[...]}`
- `{"type":"state","values":{"0":64,...}}`
- `{"type":"stats","serial":{"rx_lines":12,...},"runtime":{"queue_depth":1,...}}`
- `{"type":"ack","cmd":"set","id":12,"value":100}`
- `{"type":"ack","cmd":"batch","applied":6}`
- `{"type":"error","message":"..."}`

Notes:
- Parameter IDs are `0..127`.
- Incoming values are normalized per-parameter before apply:
  - global clamp: `0..127`
  - `osc1_wave`/`osc2_wave`: `0..88` (tonal wave buckets)
  - `osc2_coarse`: `4..123`
  - `chorus_depth`: `0..126`
  - `delay_time`: `0..112`
  - `pb_range`: `0..64`
  - `osc3_wave`/`osc4_wave`: `0..88` (tonal wave buckets)
  - `voice_mode`/`bypass_fx`: quantized to `0` or `127`
- Updates are routed to the shared synth event bus and registry on Core0.
- Note commands (`NOTEON`/`NOTEOFF`) inject note events directly into the shared synth event bus on channel 0.
- Parser rejects malformed `SET` payloads and trailing garbage.
- TX responses are buffered in-order in firmware so back-to-back `STATE`/`META` requests are not dropped.
- Quiet commands (`SETQ`/`BATCHQ`) avoid success responses for low-overhead control streams.
- `META` exposes `min`/`max` per parameter for host/UI clamping.
- Queue-full and parse failures return `type=error`.
- `STATS.runtime` also includes timing fields for profiling:
  - `audio_cb_last_us`, `audio_cb_max_us`, `audio_frames`
  - `audio_cb_over_budget` (lifetime cumulative counter)
  - `audio_cb_over_budget_delta` (per-STATS-poll increment)
  - `audio_cb_over_budget_pct_x100` (per-poll ratio in fixed-point percent)
  - `core0_loop_last_us`, `core0_loop_max_us`
  - `core0_midi_poll_last_us`, `core0_midi_poll_max_us`
  - `core0_serial_poll_last_us`, `core0_serial_poll_max_us`
  - `core0_control_poll_last_us`, `core0_control_poll_max_us`
  - `sample_rate_hz`, `audio_frame_interval_us`, `audio_render_div`, `audio_callback_interval_us`, `main_loop_sleep_us`
  - feature flags: `chorus_enabled`, `delay_enabled`, `exp_mod_dest_enabled`, `restrict_osc_enabled`
