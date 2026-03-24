[CmdletBinding()]
param(
  [string]$BuildDir = "build-ninja",
  [string]$Board = "pico2_w",
  [string]$PicoSdkPath = "C:/softwareDevelopment/picoSillynth/pico-sdk",
  [string]$ToolchainPath = "C:/Program Files (x86)/Arm/GNU Toolchain mingw-w64-i686-arm-none-eabi/bin",
  [string]$PioasmDir = "C:/softwareDevelopment/picoSillynth/pioasm",
  [string]$NinjaPath = "C:/ninja/ninja.exe",
  [int]$Jobs = 8,
  [switch]$Clean,
  [switch]$NoConfigure,
  [switch]$NoBuild,
  [switch]$NoFlash,
  [switch]$DiagLedOnly,
  [switch]$UsbEnumOnly,
  [switch]$NoUsbSerialDebug,
  [switch]$EnableUsbMidi,
  [switch]$EnableUsbStdio,
  [switch]$EnableChorus,
  [switch]$EnableDelay,
  [switch]$EnableExpensiveModDest,
  [switch]$RestrictOsc,
  [switch]$LowCpuProfile,
  [switch]$DisableFxBypassAtBoot,
  [switch]$DisableAudioEngine,
  [switch]$EnableWatchdog,
  [string]$PicotoolPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-CommandPath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name
  )

  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if ($null -eq $cmd) {
    return $null
  }
  return $cmd.Source
}

function Resolve-PicotoolPath {
  param(
    [string]$ExplicitPath,
    [string]$ProjectRootPath
  )

  if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
    if (Test-Path $ExplicitPath) {
      return $ExplicitPath
    }
    throw "Picotool path was provided but does not exist: $ExplicitPath"
  }

  $fromPath = Resolve-CommandPath "picotool"
  if (-not [string]::IsNullOrWhiteSpace($fromPath)) {
    return $fromPath
  }

  $candidates = @(
    "C:/softwareDevelopment/picoSillynth/picotool/picotool.exe",
    "C:/softwareDevelopment/picoSillynth/picotool/build/picotool.exe",
    "C:/softwareDevelopment/picoSillynth/pico-sdk/build/picotool/picotool.exe",
    "C:/softwareDevelopment/picoSillynth/pico-sdk/tools/picotool/picotool.exe",
    "C:/tools/picotool.exe",
    "C:/Program Files/Raspberry Pi/Picotool/picotool.exe",
    "C:/Program Files (x86)/Raspberry Pi/Picotool/picotool.exe",
    "$ProjectRootPath/build-ninja/_deps/picotool/picotool.exe",
    "$ProjectRootPath/build-ninja/_deps/picotool/picotool",
    "$ProjectRootPath/build-ninja/_deps/picotool-build/picotool.exe",
    "$ProjectRootPath/build-ninja/_deps/picotool-build/picotool"
  )

  foreach ($path in $candidates) {
    if (Test-Path $path) {
      return $path
    }
  }

  # Try wildcard SDK/tool installs.
  $globCandidates = @(
    "C:/Program Files/Raspberry Pi/*/picotool.exe",
    "C:/Program Files (x86)/Raspberry Pi/*/picotool.exe"
  )
  foreach ($pattern in $globCandidates) {
    $hit = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $hit) {
      return $hit.FullName
    }
  }

  $searched = ($candidates + $globCandidates) -join "`n  - "
  throw "picotool was not found on PATH or known locations.`nSearched:`n  - $searched`nInstall picotool or pass -PicotoolPath."
}

function Require-Path {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PathValue,
    [Parameter(Mandatory = $true)]
    [string]$Description
  )
  if (-not (Test-Path $PathValue)) {
    throw "$Description not found: $PathValue"
  }
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildPath = Join-Path $ProjectRoot $BuildDir
$ElfPath = Join-Path $BuildPath "pico2w_synth.elf"

$CmakePath = Resolve-CommandPath "cmake"
if ([string]::IsNullOrWhiteSpace($CmakePath)) {
  throw "cmake was not found on PATH."
}

if (-not $NoFlash) {
  $PicotoolPath = Resolve-PicotoolPath -ExplicitPath $PicotoolPath -ProjectRootPath $ProjectRoot
  Write-Host "Using picotool: $PicotoolPath"
}

Require-Path -PathValue $PicoSdkPath -Description "PICO SDK path"
Require-Path -PathValue $ToolchainPath -Description "ARM toolchain path"
Require-Path -PathValue $PioasmDir -Description "pioasm directory"
Require-Path -PathValue $NinjaPath -Description "Ninja executable"

if ($Clean -and (Test-Path $BuildPath)) {
  Write-Host "Cleaning build directory: $BuildPath"
  Remove-Item -Recurse -Force $BuildPath
}

Push-Location $ProjectRoot
try {
  $diagLedOnlyValue = if ($DiagLedOnly) { 1 } else { 0 }
  $usbEnumOnlyValue = if ($UsbEnumOnly) { 1 } else { 0 }
  $usbSerialDebugValue = if ($NoUsbSerialDebug) { 0 } else { 1 }
  $usbMidiValue = if ($EnableUsbMidi) { 1 } else { 0 }
  $usbStdioValue = if ($EnableUsbStdio) { 1 } else { 0 }
  $enableChorusValue = if ($EnableChorus) { 1 } else { 0 }
  $enableDelayValue = if ($EnableDelay) { 1 } else { 0 }
  $enableExpensiveModDestValue = if ($EnableExpensiveModDest) { 1 } else { 0 }
  $restrictOscValue = if ($RestrictOsc) { 1 } else { 0 }
  $lowCpuProfileValue = if ($LowCpuProfile) { 1 } else { 0 }
  $defaultBypassFxValue = if ($DisableFxBypassAtBoot) { 0 } else { 1 }
  $disableAudioEngineValue = if ($DisableAudioEngine) { 1 } else { 0 }
  $watchdogValue = if ($EnableWatchdog) { 1 } else { 0 }

  if (-not $NoConfigure) {
    Write-Host "Configuring ($Board) ..."
    if ($DiagLedOnly) {
      Write-Host "Diagnostic firmware mode: LED-only"
    }
    if ($UsbEnumOnly) {
      Write-Host "Firmware profile: USB enumeration only"
    }
    if ($NoUsbSerialDebug) {
      Write-Host "USB serial debug: disabled"
    } else {
      Write-Host "USB serial debug: enabled"
    }
    if ($EnableUsbMidi) {
      Write-Host "USB MIDI: enabled"
    } else {
      Write-Host "USB MIDI: disabled (UART/GPIO MIDI only)"
    }
    if ($EnableChorus) {
      Write-Host "Chorus DSP: enabled"
    } else {
      Write-Host "Chorus DSP: disabled"
    }
    if ($EnableDelay) {
      Write-Host "Delay DSP: enabled"
    } else {
      Write-Host "Delay DSP: disabled"
    }
    if ($EnableExpensiveModDest) {
      Write-Host "Expensive modulation destinations: enabled"
    } else {
      Write-Host "Expensive modulation destinations: disabled"
    }
    if ($RestrictOsc) {
      Write-Host "Oscillator complexity: restricted mode enabled"
    } else {
      Write-Host "Oscillator complexity: full mode"
    }
    if ($LowCpuProfile) {
      Write-Host "Performance profile: low CPU (24kHz audio, reduced control-rate pressure)"
    } else {
      Write-Host "Performance profile: full-rate output (48kHz PWM, synth render divided for stability)"
    }
    if ($DisableFxBypassAtBoot) {
      Write-Host "FX chain default: enabled at boot"
    } else {
      Write-Host "FX chain default: bypassed at boot (for stability/profiling)"
    }
    if ($DisableAudioEngine) {
      Write-Host "Audio engine: DSP disabled (serial/control debug mode, silent output)"
    } else {
      Write-Host "Audio engine: enabled"
    }
    Write-Host "Web/Wi-Fi/display runtime: removed (external-control firmware)"
    if ($EnableWatchdog) {
      Write-Host "Watchdog: enabled"
    } else {
      Write-Host "Watchdog: disabled"
    }
    if ($EnableUsbStdio) {
      Write-Host "USB stdio: enabled"
    } else {
      Write-Host "USB stdio: disabled"
    }
    & $CmakePath -S . -B $BuildDir -G Ninja `
      "-DCMAKE_MAKE_PROGRAM=$NinjaPath" `
      "-DPICO_BOARD=$Board" `
      "-DPICO_SDK_PATH=$PicoSdkPath" `
      "-Dpioasm_DIR=$PioasmDir" `
      "-DPICO_TOOLCHAIN_PATH=$ToolchainPath" `
      "-DPICO2W_SYNTH_DIAGNOSTIC_LED_ONLY=$diagLedOnlyValue" `
      "-DPICO2W_SYNTH_USB_ENUM_ONLY=$usbEnumOnlyValue" `
      "-DPICO2W_SYNTH_ENABLE_USB_SERIAL_DEBUG=$usbSerialDebugValue" `
      "-DPICO2W_SYNTH_ENABLE_USB_MIDI=$usbMidiValue" `
      "-DPICO2W_SYNTH_ENABLE_USB_STDIO=$usbStdioValue" `
      "-DPICO2W_SYNTH_ENABLE_CHORUS=$enableChorusValue" `
      "-DPICO2W_SYNTH_ENABLE_DELAY=$enableDelayValue" `
      "-DPICO2W_SYNTH_ENABLE_EXPENSIVE_MOD_DEST=$enableExpensiveModDestValue" `
      "-DPICO2W_SYNTH_RESTRICT_OSC=$restrictOscValue" `
      "-DPICO2W_SYNTH_LOW_CPU_PROFILE=$lowCpuProfileValue" `
      "-DPICO2W_SYNTH_DEFAULT_BYPASS_FX=$defaultBypassFxValue" `
      "-DPICO2W_SYNTH_DISABLE_AUDIO_ENGINE=$disableAudioEngineValue" `
      "-DPICO2W_SYNTH_ENABLE_WATCHDOG=$watchdogValue" `
      -DPICO_NO_PICOTOOL=1 `
      -DPICO_NO_UF2=1

    if ($LASTEXITCODE -ne 0) {
      throw "CMake configure failed."
    }
  }

  if (-not $NoBuild) {
    Write-Host "Building ..."
    & $CmakePath --build $BuildDir --parallel $Jobs -- -k 1
    if ($LASTEXITCODE -ne 0) {
      throw "Build failed."
    }
  }

  if (-not (Test-Path $ElfPath)) {
    throw "ELF artifact missing: $ElfPath"
  }

  Write-Host "ELF ready: $ElfPath"

  if (-not $NoFlash) {
    Write-Host "Flashing with picotool ..."
    & $PicotoolPath load -x $ElfPath
    if ($LASTEXITCODE -ne 0) {
      throw "Flash failed. Try BOOTSEL mode and rerun."
    }
    Write-Host "Flash complete."
  } else {
    Write-Host "Skipping flash (-NoFlash)."
  }
}
finally {
  Pop-Location
}
