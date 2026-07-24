

### Fixed — Windows tok/s + RAM (Phi / Qwen / dense KPK)
- **RAM stuck at 0 on Windows:** Process RSS used Unix-only `resource.getrusage`, which always failed on native Windows CPython. Now uses `GetProcessMemoryInfo` (WorkingSet). Chat stats also prefer the engine’s `rss_gb` / `footprint_gb` when present so the UI reflects real model memory without inflating it via fake counters.
- **Extremely slow decode (~300s) on Windows x64:** KPK / dense kernels (`windhover.c`, `dense.c`) were ARM-NEON-only and fell back to **scalar** matmul on x86. Added AVX2 (+ AVX-VNNI when available) IDOT helpers shared via `engine/runtime/idot_avx.h` so Phi-4 Mini, Qwen, and other dense packs run at normal CPU tok/s.
- **OpenMP defaults missing on desktop Windows:** Engine spawn now applies the same Windows OMP/I/O env as `coli` (`OMP_NUM_THREADS` = physical cores, `OMP_WAIT_POLICY=active`, `DIRECT`/`PIPE`/`PILOT_REAL`) **before** launching `windhover-engine.exe`, so libgomp actually sees them.

## [0.3.12] — 2026-07-23

### Fixed
- **Windows WinError 267** when chatting: packaged app launched `windhover-engine` with
  cwd=`bundle/engine` (often missing). Use the engine binary’s directory (or another
  existing path) so CreateProcess succeeds.

## [0.3.11] — 2026-07-23

### Engine — Universal Windhover Model IR (WMIR)
- Replaced architecture allowlists with a layer-typed IR (`windhover.wmir` in `kestrel.json`).
- HF configs lower via `tools/wmir/` (Gemma 4, Qwen3.5/3.6, Llama 4, Kimi, DeepSeek V4, MiniMax, Mistral Large 3, plus classic dense/GLM).
- Runtime executes ops by kind: GQA (+ KV share, chunked/CSA/MSA windows), linear GDN, double-wide MLP, MoE-stream markers.
- Catalog installable entries now require a WMIR lowerer + registered kernels (`tools/catalog_engine_audit.py`).

## [0.3.10] — 2026-07-23

### Catalog
- **Engine-truth audit:** only models Windhover can actually run stay installable. Gemma 4 / Qwen3.5–3.6 hybrid / Llama 4 / Kimi K2.x / DeepSeek V4 / MiniMax M3 / broken HF ids are marked `soon` + blocked.
- Ready list is Qwen2.5/3 dense + SmolLM2 + DeepSeek R1 Distill (chat immediately after download) and Phi-4 Mini (install finishes with KPK convert). Same paths on Mac and Windows.
- Added `tools/catalog_engine_audit.py` to verify HF `model_type` vs engine support before release.

### Bugs fixed
- **Windows Phi-4 / Gemma engine prepare:** first chat no longer hard-fails while converting. Convert runs in the background with progress; install fails clearly if Phi/Gemma KPK convert fails; Chat returns `engine_preparing` and the UI polls until ready.

## [0.3.8] — 2026-07-23

### Bugs fixed
- **UI squashed into top quarter of the window:** shell grid reserved an empty `1fr` row when the update banner was hidden. Titlebar + banners are one chrome block again so the app fills the window.

## [0.3.7] — 2026-07-23

### Bugs fixed
- **Windows Phi-4 “KPK convert failed”:** torch-free safetensors BF16 reads were returning flat arrays, which broke Phi’s fused `qkv` / `gate_up` split during convert. Tensor shapes are restored from the safetensors header so Phi-4 Mini converts and chats via windhover-engine without torch.

## [0.3.6] — 2026-07-23

### App
- **In-app updates:** when a newer GitHub Release exists, Windhover shows an **Update** button. Windows downloads the NSIS installer and upgrades in place (no uninstall). macOS opens the new DMG. Also under Advanced → Check for updates.

## [0.3.5] — 2026-07-23

### Bugs fixed
- **Windows Chat `No module named 'torch'` after Library download:** chat no longer routes downloaded models through transformers preview when KPK is missing. Packaged apps use windhover-engine (dense load-time quant for Qwen/Llama/Mistral; torch-free numpy KPK convert for Phi/Gemma).
- **Packaging:** sidecar bundles `numpy` / `safetensors` / `kestrel_pack` for convert; selfcheck covers chat routing helpers; torch/transformers stay excluded on purpose.

## [0.3.4] — 2026-07-23

### Bugs fixed
- **Windows Library download `ModuleNotFoundError: No module named 'huggingface_hub'`:** package `huggingface_hub` (and deps) into the `windhover-server` PyInstaller sidecar so Install from Library works without a system Python.
- **Packaging hardening:** also collect `httpx` / hub runtime stack; `windhover-server --sidecar-selfcheck` fails the sidecar build if download imports are missing.

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
