#!/usr/bin/env bash
# Build windhover-server sidecar and stage Tauri externalBin names.
# Usage (from repo root):
#   ./packaging/build_sidecar.sh [target-triple]
# Example:
#   ./packaging/build_sidecar.sh x86_64-pc-windows-msvc
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRIPLE="${1:-}"
cd "$ROOT"

if [[ ! -d app/dist ]]; then
  (cd app && npm ci && npm run build)
fi

python3 -m pip install -q pyinstaller "huggingface_hub>=0.23"
python3 -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"
python3 -m PyInstaller packaging/windhover-server.spec --noconfirm --distpath packaging/dist --workpath packaging/build

BIN_DIR="$ROOT/desktop/src-tauri/binaries"
mkdir -p "$BIN_DIR"

SERVER_SRC="$ROOT/packaging/dist/windhover-server"
if [[ -f "${SERVER_SRC}.exe" ]]; then
  SERVER_SRC="${SERVER_SRC}.exe"
  EXT=".exe"
else
  EXT=""
fi

if [[ -z "$TRIPLE" ]]; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) TRIPLE="aarch64-apple-darwin" ;;
    Darwin-x86_64) TRIPLE="x86_64-apple-darwin" ;;
    Linux-x86_64) TRIPLE="x86_64-unknown-linux-gnu" ;;
    Linux-aarch64) TRIPLE="aarch64-unknown-linux-gnu" ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT*)
      if [[ "${PROCESSOR_ARCHITECTURE:-}" == "ARM64" ]]; then
        TRIPLE="aarch64-pc-windows-msvc"
      else
        TRIPLE="x86_64-pc-windows-msvc"
      fi
      ;;
    *) TRIPLE="unknown" ;;
  esac
fi

cp -f "$SERVER_SRC" "$BIN_DIR/windhover-server-${TRIPLE}${EXT}"
echo "Staged $BIN_DIR/windhover-server-${TRIPLE}${EXT}"

# Stage engine next to it when present
ENG="$ROOT/engine/windhover-engine"
if [[ -f "${ENG}.exe" ]]; then ENG="${ENG}.exe"; EXT=".exe"; elif [[ -f "$ENG" ]]; then EXT=""; else ENG=""; fi
if [[ -n "$ENG" ]]; then
  cp -f "$ENG" "$BIN_DIR/windhover-engine-${TRIPLE}${EXT}"
  echo "Staged $BIN_DIR/windhover-engine-${TRIPLE}${EXT}"
fi
