# Download Windhover

## macOS (Apple Silicon)

Stable URL (always the latest GitHub Release asset):

**https://github.com/cliclye/Windhover/releases/latest/download/Windhover-macOS-arm64.dmg**

Browse releases: https://github.com/cliclye/Windhover/releases/latest

### Install

1. Download `Windhover-macOS-arm64.dmg`
2. Open it and drag **Windhover** into Applications
3. Launch Windhover from Applications / Launchpad

### First launch — “damaged” / Gatekeeper (unsigned builds)

CI ships an **ad-hoc signed but not Apple-notarized** DMG (no Developer ID yet). After download, macOS may say:

- **“Windhover is damaged and can’t be opened”**, or
- “cannot be opened because it is from an unidentified developer”

That is Gatekeeper quarantine on community builds — **not** a corrupt download.

**Fix (recommended):**

```bash
xattr -cr /Applications/Windhover.app
```

Then open Windhover again (or right-click → **Open** → **Open**).

Alternatives:

- Right-click **Windhover** → **Open** → **Open**
- **System Settings → Privacy & Security** → **Open Anyway**

### Requirements

- macOS 12+
- Apple Silicon (M1 / M2 / M3 / M4) — primary supported target

Intel Macs: build from source for now (see README).

---

## Windows 11 (x64 + ARM64)

Stable URLs:

- **https://github.com/cliclye/Windhover/releases/latest/download/Windhover-Windows-x64.exe** — Intel/AMD 64-bit
- **https://github.com/cliclye/Windhover/releases/latest/download/Windhover-Windows-arm64.exe** — Snapdragon / ARM64 PCs

### Install

1. Download the installer for your architecture
2. Run the NSIS setup (current-user install by default)
3. Launch **Windhover** from the Start menu

The installer embeds `windhover-server` + `windhover-engine` — no repo checkout, Python, or MinGW required for end users. Models live under `%USERPROFILE%\.windhover\models`.

### First launch — SmartScreen (unsigned builds)

CI ships **unsigned** installers until Authenticode signing secrets are configured. Windows SmartScreen may say “Windows protected your PC.”

Allow it once:

- Click **More info** → **Run anyway**

Same honesty as the macOS Gatekeeper path: community builds are not notarized/signed yet.

### Requirements

- Windows 11 (x64 or ARM64)
- WebView2 runtime (usually preinstalled on Windows 11)

---

## Cut a release (maintainers)

1. Ensure `desktop/src-tauri/tauri.conf.json` `version` matches the tag you want (e.g. `0.1.0`).
2. Push a tag from `main` (or merge to `main` first):

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. GitHub Actions workflows build and attach:
   - **Release macOS** → `Windhover-macOS-arm64.dmg` (ad-hoc codesigned)
   - **Release Windows** → `Windhover-Windows-x64.exe` + `Windhover-Windows-arm64.exe`
4. Confirm the stable latest URLs work:

   ```bash
   curl -sI https://github.com/cliclye/Windhover/releases/latest/download/Windhover-macOS-arm64.dmg | head
   curl -sI https://github.com/cliclye/Windhover/releases/latest/download/Windhover-Windows-x64.exe | head
   curl -sI https://github.com/cliclye/Windhover/releases/latest/download/Windhover-Windows-arm64.exe | head
   ```

Manual dry-run: **Actions → Release macOS / Release Windows → Run workflow**.

### Optional: signing

- **macOS:** Apple Developer secrets — see [Tauri macOS signing](https://v2.tauri.app/distribute/sign-macos/).
- **Windows:** Authenticode — see [Tauri Windows signing](https://v2.tauri.app/distribute/sign-windows/).

Until then, keep the Gatekeeper / SmartScreen notes above honest and visible.

---

## Build from source

### macOS

```bash
./windhover build
cd app && npm ci && npm run build && cd ..
cd desktop && cargo tauri build --bundles app,dmg
open src-tauri/target/release/bundle/macos/Windhover.app
```

### Windows (MinGW/Clang engine + NSIS)

```powershell
cd engine
make

cd ..
powershell -File packaging/build_sidecar.ps1 -Triple x86_64-pc-windows-msvc

cd desktop
cargo tauri build --bundles nsis --target x86_64-pc-windows-msvc
```

See [`desktop/README.md`](../desktop/README.md).
