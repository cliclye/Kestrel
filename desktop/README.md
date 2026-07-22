# Windhover (macOS)

Native **Mac app** (Tauri v2) for the Windhover Library + Chat UI in [`../app`](../app).

On launch the desktop app starts `../windhover app` (local engine API + UI on `http://127.0.0.1:8000`) and opens it in a branded native window (**Windhover**, bundle ID `ai.vexilo.windhover`).

## Prerequisites

```sh
./windhover build
cd app && npm ci && npm run build && cd ..
cargo install tauri-cli --version "^2.0.0" --locked
```

## Develop

```sh
cd desktop
cargo tauri dev
```

## Release bundle

```sh
cd desktop
cargo tauri build --bundles app,dmg
open src-tauri/target/release/bundle/macos/Windhover.app
```

Prefer the project venv (`../c/.venv`) so Chat previews have `torch` / `transformers`.
