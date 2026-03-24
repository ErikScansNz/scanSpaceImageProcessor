$ErrorActionPreference = 'Stop'
$cmake = 'C:/Program Files/CMake/bin/cmake.exe'
$tc = 'C:/Program Files (x86)/Arm/GNU Toolchain mingw-w64-i686-arm-none-eabi/bin'

Set-Location 'C:/softwareDevelopment/picoSillynth/pico2w-synth'

Remove-Item -Recurse -Force build-ninja -ErrorAction SilentlyContinue

& $cmake -S . -B build-ninja -G Ninja `
  -DCMAKE_MAKE_PROGRAM=C:/ninja/ninja.exe `
  -DPICO_BOARD=pico2_w `
  -DPICO_SDK_PATH=C:/softwareDevelopment/picoSillynth/pico-sdk `
  -Dpioasm_DIR=C:/softwareDevelopment/picoSillynth/pioasm `
  -DPICO_NO_PICOTOOL=1 `
  -DPICO_NO_UF2=1 `
  -DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON `
  "-DPICO_TOOLCHAIN_PATH=$tc"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $cmake --build build-ninja --parallel 8 -- -k 1
exit $LASTEXITCODE
