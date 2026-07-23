# Packaging checklist (Windows / macOS sidecars)

Use this before cutting a release so Library Install / Chat / NSIS / smoke do not regress.

## Frozen `windhover-server` must include download + convert deps

Library Install calls `huggingface_hub.snapshot_download` inside the PyInstaller
sidecar. Chat uses **windhover-engine** (not torch). Phi/Gemma need an in-process
KPK convert via `numpy` + `safetensors`. Lazy imports are **invisible** to Analysis.

- [ ] `packaging/build_sidecar.ps1` / `.sh` install `huggingface_hub` **and**
      runtime deps (`httpx`, `filelock`, `fsspec`, `PyYAML`, `tqdm`, `packaging`,
      `click`, `hf-xet`, `numpy`, `safetensors`, `ml_dtypes`).
- [ ] `packaging/windhover-server.spec` uses `collect_all(...)` for those packages
      and ships `tools/kestrel_pack.py`.
- [ ] Spec **excludes** `torch` / `transformers` (do not bundle them).
- [ ] Build runs `windhover-server --sidecar-selfcheck` and exits **0** (hub +
      numpy + safetensors + kestrel_pack + `_dense_loadtime_ok` fixture).
- [ ] CI Release Windows smoke still hits `/health` and `/v1/catalog` with **curl.exe**
      (not `Invoke-WebRequest`).

## Chat routing (must stay engine-first)

- [ ] Downloaded HF weights with `chat: engine` must **not** force transformers
      preview when KPK is missing (`_chat_mode` → `engine` for dense / convertible).
- [ ] Qwen2/3 / Llama / Mistral: engine dense load-time quant on raw HF is enough.
- [ ] Phi / Gemma: `_ensure_engine_pack` runs torch-free KPK convert before chat.
- [ ] Engine→transformers fallback only runs when `_transformers_available()`.

## Do not

- Redirect `Start-Process` stdout **and** stderr to the **same** log file on Windows
  (fails immediately).
- Tell packaged users to `pip install torch` for Library models.
- Ship a Windows build without re-running sidecar selfcheck after changing
  `windhover` download/chat code.

## Known intentional limits

- Packaged Windows app does **not** bundle `torch`/`transformers` — that is
  correct. Product chat path is windhover-engine (+ optional numpy KPK convert).
- Chat-template formatting falls back to a plain prompt if `transformers` is missing.
- Yarn/dynamic/linear RoPE packs may still need a dev transformers install.
