$cmake = 'C:/Program Files/CMake/bin/cmake.exe'
$out = 'C:/softwareDevelopment/picoSillynth/pico2w-synth/cmake_version.log'
Remove-Item $out -ErrorAction SilentlyContinue
& $cmake --version *> $out
Write-Output "exit=$LASTEXITCODE"
