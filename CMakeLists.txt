cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
  set(USERHOME $ENV{USERPROFILE})
else()
  set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.1.1)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.1.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
  include(${picoVscode})
endif()

set(PICO_BOARD pico2_w CACHE STRING "Board type")

include(pico_sdk_import.cmake)

project(pico2w_synth C CXX ASM)

pico_sdk_init()

option(PICO2W_SYNTH_DIAGNOSTIC_LED_ONLY "Build LED-only diagnostic firmware mode" OFF)
if(PICO2W_SYNTH_DIAGNOSTIC_LED_ONLY)
  set(PICO2W_SYNTH_DIAGNOSTIC_LED_ONLY_VALUE 1)
else()
  set(PICO2W_SYNTH_DIAGNOSTIC_LED_ONLY_VALUE 0)
endif()

option(PICO2W_SYNTH_USB_ENUM_ONLY "Build minimal USB-enumeration-only firmware profile" OFF)
if(PICO2W_SYNTH_USB_ENUM_ONLY)
  set(PICO2W_SYNTH_USB_ENUM_ONLY_VALUE 1)
else()
  set(PICO2W_SYNTH_USB_ENUM_ONLY_VALUE 0)
endif()

option(PICO2W_SYNTH_ENABLE_USB_SERIAL_DEBUG "Expose USB CDC serial debug console alongside USB MIDI" ON)
if(PICO2W_SYNTH_ENABLE_USB_SERIAL_DEBUG)
  set(PICO2W_SYNTH_ENABLE_USB_SERIAL_DEBUG_VALUE 1)
else()
  set(PICO2W_SYNTH_ENABLE_USB_SERIAL_DEBUG_VALUE 0)
endif()

option(PICO2W_SYNTH_ENABLE_USB_MIDI "Expose USB MIDI interface in composite USB device" OFF)
if(PICO2W_SYNTH_ENABLE_USB_MIDI)
  set(PICO2W_SYNTH_ENABLE_USB_MIDI_VALUE 1)
else()
  set(PICO2W_SYNTH_ENABLE_USB_MIDI_VALUE 0)
endif()

option(PICO2W_SYNTH_ENABLE_WATCHDOG "Enable runtime watchdog feeding/gating" OFF)
if(PICO2W_SYNTH_ENABLE_WATCHDOG)
  set(PICO2W_SYNTH_ENABLE_WATCHDOG_VALUE 1)
else()
  set(PICO2W_SYNTH_ENABLE_WATCHDOG_VALUE 0)
endif()

option(PICO2W_SYNTH_ENABLE_CHORUS "Enable chorus DSP path" OFF)
if(PICO2W_SYNTH_ENABLE_CHORUS)
  set(PICO2W_SYNTH_ENABLE_CHORUS_VALUE 1)
else()
  set(PICO2W_SYNTH_ENABLE_CHORUS_VALUE 0)
endif()

option(PICO2W_SYNTH_ENABLE_DELAY "Enable delay DSP path" OFF)
if(PICO2W_SYNTH_ENABLE_DELAY)
  set(PICO2W_SYNTH_ENABLE_DELAY_VALUE 1)
else()
  set(PICO2W_SYNTH_ENABLE_DELAY_VALUE 0)
endif()

option(PICO2W_SYNTH_ENABLE_EXPENSIVE_MOD_DEST "Enable expensive modulation destinations (shape/morph routes)" OFF)
if(PICO2W_SYNTH_ENABLE_EXPENSIVE_MOD_DEST)
  set(PICO2W_SYNTH_ENABLE_EXPENSIVE_MOD_DEST_VALUE 1)
else()
  set(PICO2W_SYNTH_ENABLE_EXPENSIVE_MOD_DEST_VALUE 0)
endif()

option(PICO2W_SYNTH_RESTRICT_OSC "Enable reduced oscillator complexity mode" OFF)
if(PICO2W_SYNTH_RESTRICT_OSC)
  set(PICO2W_SYNTH_RESTRICT_OSC_VALUE 1)
else()
  set(PICO2W_SYNTH_RESTRICT_OSC_VALUE 0)
endif()

option(PICO2W_SYNTH_LOW_CPU_PROFILE "Reduce DSP/runtime load to prioritize USB/IO stability" OFF)
if(PICO2W_SYNTH_LOW_CPU_PROFILE)
  set(PICO2W_SYNTH_DEFAULT_SAMPLE_RATE_VALUE 24000)
  set(PICO2W_SYNTH_CONTROL_RATE_DIV_VALUE 8)
  set(PICO2W_SYNTH_AUDIO_RENDER_DIV_VALUE 1)
else()
  set(PICO2W_SYNTH_DEFAULT_SAMPLE_RATE_VALUE 48000)
  set(PICO2W_SYNTH_CONTROL_RATE_DIV_VALUE 8)
  set(PICO2W_SYNTH_AUDIO_RENDER_DIV_VALUE 2)
endif()

option(PICO2W_SYNTH_DEFAULT_BYPASS_FX "Boot with FX chain bypass enabled" ON)
if(PICO2W_SYNTH_DEFAULT_BYPASS_FX)
  set(PICO2W_SYNTH_DEFAULT_BYPASS_FX_VALUE 1)
else()
  set(PICO2W_SYNTH_DEFAULT_BYPASS_FX_VALUE 0)
endif()

option(PICO2W_SYNTH_DISABLE_AUDIO_ENGINE "Run audio worker without synth frame DSP (USB/control debug mode)" OFF)
if(PICO2W_SYNTH_DISABLE_AUDIO_ENGINE)
  set(PICO2W_SYNTH_DISABLE_AUDIO_ENGINE_VALUE 1)
else()
  set(PICO2W_SYNTH_DISABLE_AUDIO_ENGINE_VALUE 0)
endif()

option(PICO2W_SYNTH_ENABLE_USB_STDIO "Enable SDK stdio over USB (separate from CDC serial control channel)" OFF)
if(PICO2W_SYNTH_ENABLE_USB_STDIO)
  set(PICO2W_SYNTH_ENABLE_USB_STDIO_VALUE 1)
else()
  set(PICO2W_SYNTH_ENABLE_USB_STDIO_VALUE 0)
endif()

add_executable(pico2w_synth
  src/app/main.cpp
  src/app/application.cpp
  src/audio/audio_backend.cpp
  src/audio/audio_worker.cpp
  src/audio/pwm_stereo_backend.cpp
  src/audio/i2s_backend_stub.cpp
  src/audio/bluetooth_audio_backend_stub.cpp
  src/params/param_registry.cpp
  src/params/param_schema.cpp
  src/params/preset_snapshot.cpp
  src/util/event_queue.cpp
  src/synth/synth_engine.cpp
  src/synth/reverb_optional.cpp
  src/midi/midi_input.cpp
  src/midi/midi_uart.cpp
  src/midi/midi_usb.cpp
  src/midi/midi_mapper.cpp
  src/serial/serial_control.cpp
  src/control/control_input.cpp
  src/control/hardware_controls_stub.cpp
  src/midi/usb_descriptors.c
)

target_include_directories(pico2w_synth PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${CMAKE_CURRENT_LIST_DIR}/src/midi
  ${CMAKE_CURRENT_LIST_DIR}/src/synth/pra32
)

target_compile_definitions(pico2w_synth PRIVATE
  PICO2W_SYNTH_ENABLE_REVERB=0
  PICO2W_SYNTH_ENABLE_HW_CONTROLS=0
  PICO2W_SYNTH_FIXED_POINT=1
  PICO2W_SYNTH_POLY_AFTERTOUCH=0
  PICO2W_SYNTH_DEFAULT_SAMPLE_RATE=${PICO2W_SYNTH_DEFAULT_SAMPLE_RATE_VALUE}
  PICO2W_SYNTH_CONTROL_RATE_DIV=${PICO2W_SYNTH_CONTROL_RATE_DIV_VALUE}
  PICO2W_SYNTH_AUDIO_RENDER_DIV=${PICO2W_SYNTH_AUDIO_RENDER_DIV_VALUE}
  PICO2W_SYNTH_DEFAULT_BYPASS_FX=${PICO2W_SYNTH_DEFAULT_BYPASS_FX_VALUE}
  PICO2W_SYNTH_DISABLE_AUDIO_ENGINE=${PICO2W_SYNTH_DISABLE_AUDIO_ENGINE_VALUE}
  PICO2W_SYNTH_DIAGNOSTIC_LED_ONLY=${PICO2W_SYNTH_DIAGNOSTIC_LED_ONLY_VALUE}
  PICO2W_SYNTH_USB_ENUM_ONLY=${PICO2W_SYNTH_USB_ENUM_ONLY_VALUE}
  PICO2W_SYNTH_ENABLE_USB_SERIAL_DEBUG=${PICO2W_SYNTH_ENABLE_USB_SERIAL_DEBUG_VALUE}
  PICO2W_SYNTH_ENABLE_USB_MIDI=${PICO2W_SYNTH_ENABLE_USB_MIDI_VALUE}
  PICO2W_SYNTH_ENABLE_WATCHDOG=${PICO2W_SYNTH_ENABLE_WATCHDOG_VALUE}
  PICO2W_SYNTH_ENABLE_CHORUS=${PICO2W_SYNTH_ENABLE_CHORUS_VALUE}
  PICO2W_SYNTH_ENABLE_DELAY=${PICO2W_SYNTH_ENABLE_DELAY_VALUE}
  PICO2W_SYNTH_ENABLE_EXPENSIVE_MOD_DEST=${PICO2W_SYNTH_ENABLE_EXPENSIVE_MOD_DEST_VALUE}
  PICO2W_SYNTH_RESTRICT_OSC=${PICO2W_SYNTH_RESTRICT_OSC_VALUE}
  CFG_TUSB_CONFIG_FILE=\"midi/tusb_config.h\"
)

target_compile_options(pico2w_synth PRIVATE
  -Wall
  -Wextra
  -Wshadow
  -Wconversion
  -Wdouble-promotion
  -Wno-unused-parameter
  $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
  $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
)

# Imported PRA32 headers are conversion-heavy; keep global warnings strict
# while suppressing conversion noise in the single translation unit that wraps them.
set_source_files_properties(src/synth/synth_engine.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-conversion"
)

target_link_libraries(pico2w_synth
  pico_stdlib
  hardware_pwm
  hardware_uart
  hardware_adc
  hardware_watchdog
  pico_multicore
  tinyusb_device
  tinyusb_board
)

pico_enable_stdio_usb(pico2w_synth ${PICO2W_SYNTH_ENABLE_USB_STDIO_VALUE})
pico_enable_stdio_uart(pico2w_synth 0)

pico_add_extra_outputs(pico2w_synth)
