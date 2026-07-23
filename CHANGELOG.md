# Changelog

## [0.3.4] — 2026-07-23

### Bugs fixed
- **Windows Library download `ModuleNotFoundError: No module named 'huggingface_hub'`:** package `huggingface_hub` (and deps) into the `windhover-server` PyInstaller sidecar so Install from Library works without a system Python.

## [0.3.3] — 2026-07-23

### Bugs fixed
- **Windows Library install crash:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'` during model download — UTF-8-safe stdio, HF/tqdm progress bars disabled on Windows, and ASCII-safe progress messages.
- **Windows setup “Error opening file for writing: …\windhover-server.exe”:** NSIS preinstall hooks stop `Windhover` / `windhover-server` / `windhover-engine` and delete locked sidecars before copy.
- **Windows app freezes ~1 minute on first launch:** backend sidecar starts on a background thread so the UI window is responsive immediately; packaged startup defers impostor cleanup until after `/health` is up.
- **Windows release CI:** catalog/health smoke uses `curl.exe` (not flaky `Invoke-WebRequest` / `ResponseEnded`); do not redirect stdout+stderr to the same file in `Start-Process`; JSON catalog/meta reads always use UTF-8; WinARM engine links `winpthread` for static OpenMP builds.

## [0.3.1] — 2026-07-22

### App / API
- **Engine inactive errors:** missing binary, launch failure, or non-zero engine exit now return HTTP 503 with `code: engine_inactive` so Chat/Agent clearly show the engine is not active (no silent pretend-success).
- Chat UI banner + status pill when engine binary is missing or the last reply fell back / failed.
- Soft transformers fallback (unusable decode) still works when HF weights exist, but is marked `engine_active: false`.

### Catalog
- **Qwen3 8B · engine Q4** (`Qwen/Qwen3-8B`) — dense `qwen3`, Windhover int4/KPK path for ~9B-class local engine chat.
- **Qwen3.5 9B** listed as `soon` / blocked: official `qwen3_5` hybrid multimodal is not supported by windhover-engine yet.

## [0.3.0] — 2026-07-22

### Highlights
- **Phi-4 Mini engine fidelity:** windhover-engine now matches transformers quality on math/reasoning while staying faster and lighter.
- **Partial RoPE + longrope attention scale** for `phi3` packs (`partial_rotary_factor`, attn scale ≈ 1.19).
- **Higher-precision KPK quant for Phi-class models** (D ≤ 4096): int8 `o`/`gate`/`up`/`down`, AWQ kept for near-tied tokens.
- Chat prefers accurate engine path for Phi again; transformers remains fallback for garbage/unsupported RoPE types.

### Engine
- Parse `partial_rotary_factor`, longrope `attention_factor`, and related config in `model_desc.h`.
- Apply RoPE only on `rope_dim` with correct inv-freq base and attn scale (`windhover.c`).
- Support int8 transposed `down_proj` (was int4-only).
- Default `WH_SPARSE=0` for quality; Chat sets it unless overridden.
- Multi-stop / denser stop-token handling for instruct packs.

### Pack / convert
- `tools/kestrel_pack.py`: int8 FFN path for medium dense models; `./windhover convert` uses the ML venv Python.
- Phi-4 Mini KPK rebuild: ~4.5 GB, AWQ + int8 FFN.

### App / API
- Cursor-like shell (Agent/Chat/Library), live engine status strip.
- Catalog refresh (Gemma 4, Phi-4, Qwen3.6, …) and incomplete-pack / download UX fixes.
- Engine→transformers fallback when replies look degenerate.

### Benchmarks (Phi-4 Mini Instruct)
- Engine accuracy suite: **100%** on math/reasoning prompts; mean **~20 tok/s**, peak **~3.8 GB** RSS.
- Transformers (MPS): mean **~2.5 tok/s**, peak **~5.3 GB** — engine ~1.5 GB lighter.
- Details: `docs/phi4_engine_acc.json`, `docs/phi4_detail_bench.json`.

### Downloads
- macOS: `Windhover-macOS-arm64.dmg`
- Windows: `Windhover-Windows-x64.exe`, `Windhover-Windows-arm64.exe`

## [0.2.0] — prior
- Ollama bridge, macOS DMG Gatekeeper fixes, Windows ARM64 engine, PyInstaller sidecar.
