# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: builds windhover-server[.exe] for the Tauri sidecar.
# Run from repo root:
#   pyinstaller packaging/windhover-server.spec --noconfirm

import os
import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

# SPECPATH is the directory that contains this .spec (packaging/), not the file path.
ROOT = Path(SPECPATH).resolve()  # packaging/
REPO = ROOT.parent

# Extensionless `windhover` is invisible to PyInstaller's import graph (and to
# importlib.util.spec_from_file_location on Python 3.14+). Copy to a .py module
# so Analysis traces stdlib deps and the frozen entry can `import` it.
_bundled = ROOT / "bundled_windhover.py"
shutil.copyfile(REPO / "windhover", _bundled)

block_cipher = None

datas = [
    (str(REPO / "windhover"), "."),
    (str(_bundled), "."),
    (str(REPO / "app" / "dist"), "app/dist"),
    (str(REPO / "app" / "public" / "catalog.json"), "app/public"),
    (str(REPO / "app" / "public" / "windhover-icon.png"), "app/public"),
    (str(REPO / "tools" / "agent_workspace.py"), "tools"),
    # In-process KPK convert (numpy, no torch) for Phi/Gemma after Library install.
    (str(REPO / "tools" / "kestrel_pack.py"), "tools"),
]

binaries = []
hiddenimports = [
    "bundled_windhover",
    "agent_workspace",
    "kestrel_pack",
    "http.server",
    "http.client",
    "urllib.parse",
    "urllib.request",
    "uuid",
    "argparse",
    "json",
    "socketserver",
    "email",
    "mimetypes",
    "html",
    "html.parser",
    "hashlib",
    "base64",
    "ssl",
    "resource",
    "secrets",
    "gzip",
    "concurrent.futures",
]

# Library downloads need huggingface_hub (+ httpx stack) inside the frozen sidecar.
# KPK convert needs numpy + safetensors (torch stays excluded — too large).
# Lazy imports in windhover are invisible to Analysis — collect packages explicitly.
for _pkg in (
    "huggingface_hub",
    "httpx",
    "httpcore",
    "anyio",
    "h11",
    "idna",
    "certifi",
    "filelock",
    "fsspec",
    "yaml",
    "tqdm",
    "packaging",
    "click",
    "numpy",
    "safetensors",
    "ml_dtypes",
):
    try:
        _d, _b, _h = collect_all(_pkg)
    except Exception as e:
        # yaml may be imported as PyYAML; hf-xet / ml_dtypes optional.
        print(f"[windhover-server.spec] skip collect_all({_pkg!r}): {e}")
        continue
    datas += _d
    binaries += _b
    hiddenimports += list(_h)
    hiddenimports.append(_pkg)

try:
    _d, _b, _h = collect_all("hf_xet")
    datas += _d
    binaries += _b
    hiddenimports += list(_h) + ["hf_xet"]
except Exception as e:
    print(f"[windhover-server.spec] hf_xet not collected (ok on some platforms): {e}")

# Optional icon for Windows
icon = None
ico = REPO / "desktop" / "src-tauri" / "icons" / "icon.ico"
if ico.is_file() and sys.platform == "win32":
    icon = str(ico)

a = Analysis(
    [str(ROOT / "server_entry.py")],
    pathex=[str(REPO), str(ROOT), str(REPO / "tools")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "pyi_rth_windhover_utf8.py")],
    # torch/transformers stay out (gigabytes). Chat uses windhover-engine;
    # KPK convert uses numpy + safetensors only.
    excludes=["torch", "transformers", "tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="windhover-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    # Windowed on Windows installs; set WINDHOVER_SERVER_CONSOLE=1 in CI for logs.
    console=(
        True
        if sys.platform != "win32"
        or os.environ.get("WINDHOVER_SERVER_CONSOLE", "").lower() in ("1", "true", "yes")
        else False
    ),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)
