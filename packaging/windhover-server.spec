# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: builds windhover-server[.exe] for the Tauri sidecar.
# Run from repo root:
#   pyinstaller packaging/windhover-server.spec --noconfirm

import os
import shutil
import sys
from pathlib import Path

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
]

# Optional icon for Windows
icon = None
ico = REPO / "desktop" / "src-tauri" / "icons" / "icon.ico"
if ico.is_file() and sys.platform == "win32":
    icon = str(ico)

a = Analysis(
    [str(ROOT / "server_entry.py")],
    pathex=[str(REPO), str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "bundled_windhover",
        "agent_workspace",
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
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "pyi_rth_windhover_utf8.py")],
    excludes=["torch", "transformers", "numpy", "tkinter"],
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
