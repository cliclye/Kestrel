# Build windhover-server.exe + stage Tauri externalBin for Windows.
# Usage (from repo root, in PowerShell or cmd with MinGW make already done):
#   powershell -File packaging/build_sidecar.ps1 -Triple x86_64-pc-windows-msvc
param(
  [string]$Triple = "x86_64-pc-windows-msvc"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path "app/dist/index.html")) {
  Push-Location app
  npm ci
  npm run build
  Pop-Location
}

python -m pip install -q pyinstaller "huggingface_hub>=0.23"
# Fail fast if hub is missing — Library downloads need it inside the frozen sidecar.
python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"
python -m PyInstaller packaging/windhover-server.spec --noconfirm --distpath packaging/dist --workpath packaging/build

$BinDir = Join-Path $Root "desktop/src-tauri/binaries"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

$Server = Join-Path $Root "packaging/dist/windhover-server.exe"
if (-not (Test-Path $Server)) { throw "missing $Server" }
Copy-Item -Force $Server (Join-Path $BinDir "windhover-server-$Triple.exe")
Write-Host "Staged windhover-server-$Triple.exe"

$Eng = Join-Path $Root "engine/windhover-engine.exe"
if (Test-Path $Eng) {
  Copy-Item -Force $Eng (Join-Path $BinDir "windhover-engine-$Triple.exe")
  Write-Host "Staged windhover-engine-$Triple.exe"
}
