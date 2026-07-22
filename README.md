<p align="center">
  <img src="docs/screenshots/icon.png" alt="Windhover" width="96" height="96" />
</p>

<h1 align="center">Windhover</h1>

<p align="center">
  <strong>Local LLM runtime for macOS</strong><br />
  Run open models on Apple Silicon with a hard RAM ceiling — without thrashing your Mac.
</p>

<p align="center">
  <a href="https://github.com/cliclye/Kestrel/releases/latest/download/Windhover-macOS-arm64.dmg"><strong>Download for macOS</strong></a>
  ·
  <a href="https://github.com/cliclye/Kestrel/releases/latest">Releases</a>
  ·
  <a href="docs/DOWNLOAD.md">Install notes</a>
  ·
  <a href="#benchmarks">Benchmarks</a>
  ·
  <a href="#how-it-works">How it works</a>
  ·
  <a href="#license">License</a>
</p>

<p align="center">
  <a href="https://github.com/cliclye/Kestrel/releases/latest/download/Windhover-macOS-arm64.dmg">
    <img src="https://img.shields.io/badge/Download-macOS%20DMG%20(Apple%20Silicon)-d4a574?style=for-the-badge&logo=apple&logoColor=white" alt="Download for macOS" />
  </a>
</p>

---

## Why Windhover exists

On a 16 GB Mac, a stock 7B chat model in fp16 is rough:

- Weights alone are ~14 GB
- Decode is **memory-bandwidth bound**
- When RSS climbs toward the machine’s comfort zone, macOS starts swapping — and tokens/sec collapses

**Windhover** keeps a sparse working set in RAM (mmap’d KPK packs + activation-unit budgeting) so real models stay usable under a hard ceiling.

The ship binary is **`windhover-engine`**. The Mac app wraps Library · Chat · Agent · Advanced.

> All headline numbers below were measured **locally on a MacBook Air M4 · 16 GB RAM** (2026-07-22), with a **≤ 9 GB** process budget and swap-abort enabled. Not projected. Not cloud.

---

## Benchmarks

### Qwen2.5-7B Instruct — with vs without Windhover

Same model: [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)  
Same machine: **MacBook Air M4 · 16 GB** · budget **≤ 9 GB RSS**

| Path | Backend | Decode | Peak / mean RSS | Fits ≤ 9 GB? | Status |
|---|---|---:|---:|---|---|
| **Without Windhover** | stock `transformers` · CPU · fp16 | **0.013 tok/s** | **8.82 GB**\* | borderline | swap-bound |
| **With Windhover** | **`windhover-engine` · KPK** | **9.80 tok/s** | **4.19 GB** | **yes** | **ok** |

\*Historical thrash finish. A live without-engine attempt was **aborted at 6.44 GB RSS** after swap jumped **+1.5 GB**, to avoid cooking the machine. The 0.013 tok/s figure is from an earlier run that was allowed to finish under thrash.

**Verdict:** under a 9 GB ceiling on this M4 Air, Windhover is the usable path — about **9.8 decode tok/s at ~4.2 GB RSS**. Without it, the same 7B is effectively dead (~0.01 tok/s).

```text
Decode tok/s          Without │ 0.013  ▏
                      Windhover │ 9.80   ████████████████████████████  (~780×)

RSS vs 9 GB budget    Without │ 8.82 GB ████████████████████████░░  swap-bound*
                      Windhover │ 4.19 GB ███████████░░░░░░░░░░░░░░  fits

* Live without-engine attempt aborted at 6.44 GB RSS after +1.5 GB swap.
```

![Qwen2.5-7B with vs without Windhover](docs/screenshots/bench-qwen7b-m4.svg)

### What the gap means

| | Without | With Windhover | Δ |
|---|---:|---:|---:|
| Decode | 0.013 tok/s | **9.80 tok/s** | **~780×** |
| RSS | 8.82 GB (thrash) | **4.19 GB** | **−53%** |
| Usable chat on 16 GB Mac | no | **yes** | — |

### Host pressure (same day, same Mac)

After the without-engine swap stress, Windhover was re-run on the same machine:

| Condition | Decode | RSS |
|---|---:|---:|
| Quiet (earlier same day) | **9.80 tok/s** | 4.19 GB |
| After without-engine swap stress | 5.62 tok/s | 4.19 GB |

RSS stays flat. Throughput drops under thermal / memory pressure — honest, not hidden.

![Host pressure](docs/screenshots/bench-qwen7b-pressure.svg)

<details>
<summary>Bench notes</summary>

- Decode-only where available · greedy
- Windhover: `NGEN=32`
- Stock path: CPU fp16 via `transformers`
- Budget: ≤ 9 GB with swap abort
- Full dumps live under [`docs/`](docs/) (`qwen7b_bench.json`, related)

</details>

---

## Theory

### Decode is a bandwidth problem

LLM decode mostly streams weight bytes from memory into the CPU every token. On Apple Silicon laptops the ceiling is finite (tens of GB/s for a handful of performance cores). If every step touches a dense 7B fp16 footprint, you:

1. Blow past a healthy RAM budget
2. Enter swap
3. Watch tok/s fall off a cliff

That is exactly what “without Windhover” looks like on the M4 Air for Qwen2.5-7B.

### Sparse working set

Windhover treats a model as a set of **activation units (AUs)** under one **byte ledger**:

- **Dense models** (Qwen, Llama, …): FFN neuron bundles gated by magnitude (CATS-style sparsity)
- **MoE models**: routed experts as AUs

Hot AUs stay resident / mlocked. Cold AUs stay **mmap’d** on SSD. Kernels only touch the predicted bytes for the next token.

```text
┌──────────────┐     gate / router      ┌─────────────────┐
│  KPK pack    │ ─────────────────────▶ │  AU tiers       │
│  group-int4  │                        │  hot · warm ·   │
│  mmap weights│                        │  cold (SSD)     │
└──────────────┘                        └────────┬────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │ RAM ledger      │
                                        │ hard ceiling    │
                                        │ (RAM_GB / cap)  │
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                        bandwidth-first
                                        CPU kernels
```

### KPK packs

Models are converted into a **KPK** on-disk format:

- Grouped int4 weights (mmap-friendly)
- Arch descriptor + tokenizer
- Hotness / gate metadata for AU placement

You download once, convert once, then run under the engine’s RAM budget instead of loading a full fp16 heap.

### Why this beats “just quantize”

Quantization helps, but a naive load still wants a large contiguous working set. Windhover’s point is **residency control**: only the active slice competes for RAM, so a 7B stays near **~4 GB RSS** on this machine instead of thrashing toward **9 GB+**.

---

## How it works

```text
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Mac app /  │────▶│  ./windhover │────▶│  windhover-engine │
│  Library UI │     │  app :8000   │     │  SNAP = model dir │
└─────────────┘     └──────────────┘     └───────────────────┘
                           │
                           ├─ Library  /v1/catalog · pull · uninstall
                           ├─ Chat     /v1/chat/...
                           ├─ Agent    /api/workspace · /api/agent
                           └─ Advanced /api/stats
```

1. **Install a pack** from Library (Mac 16 GB models download real HF weights; frontier MoEs are honest about size).
2. **Convert** to KPK when needed (`./windhover convert …`).
3. **Chat / Agent** talk to `windhover-engine` with a hard RAM ceiling.
4. **Advanced** shows live RSS, tok/s, footprint, sparsity, AU hit — no silent model swap.

Numerics lineage (Apache-2.0): [UPSTREAM.md](UPSTREAM.md).

---

## Features

| Feature | What you get |
|---|---|
| **Hard RAM ceiling** | Process stays inside a budget (`RAM_GB` / hard-cap ledger) |
| **KPK + mmap** | Group-int4 packs; cold weights stay on SSD |
| **Sparse AUs** | Dense FFN bundles + MoE experts under one working-set policy |
| **Library** | Browse / install / uninstall local packs |
| **Chat** | Markdown streaming UI; only chat-capable installs appear |
| **Agent** | Folder-scoped list / read / write on device |
| **Advanced** | Live telemetry: RSS, latency, tok/s, sparsity, AU hit |
| **Honest catalog** | No fake stubs for frontier MoEs |

---

## Screenshots

### Library

Browse Mac-16 GB packs and larger models. Install what you actually have disk for.

![Windhover Library](docs/screenshots/library.png)

### Chat

Local replies with speed / RSS chips per message.

![Windhover Chat](docs/screenshots/chat.png)

### Agent

Pick a folder on your Mac. The local model lists / reads / writes under that root only — lightweight on-device coding agent.

![Windhover Agent](docs/screenshots/agent.png)

### Advanced

Live engine telemetry — RSS, tok/s, Windhover decode stats.

![Windhover Advanced](docs/screenshots/advanced.png)

---

## Install

### Easiest — DMG

[Download Windhover for macOS (Apple Silicon)](https://github.com/cliclye/Kestrel/releases/latest/download/Windhover-macOS-arm64.dmg)

See [docs/DOWNLOAD.md](docs/DOWNLOAD.md) for Gatekeeper notes on unsigned builds.

> GitHub repo may still be named `Kestrel`; the product is **Windhover**.

### From source

```bash
git clone <repo> && cd Windhover   # folder may still be named Kestrel
./windhover build
cd app && npm ci && npm run build && cd ..
cd desktop && cargo tauri build --bundles app,dmg
open desktop/src-tauri/target/release/bundle/macos/Windhover.app
```

Dev: `cd desktop && cargo tauri dev` (starts or reuses `./windhover app` on `:8000`).

---

## Quick start (CLI)

```bash
./windhover build
./windhover doctor
./windhover pull Qwen/Qwen2.5-7B-Instruct --weights
./windhover convert ~/.windhover/models/Qwen__Qwen2.5-7B-Instruct
./windhover app                 # http://127.0.0.1:8000
```

```bash
./windhover chat --model ~/.windhover/models/Qwen__Qwen2.5-7B-Instruct/kpk \
  --prompt "Hello" --ngen 64
```

```bash
./windhover bench --windhover
./windhover uninstall Qwen/Qwen2.5-7B-Instruct
```

Model home: `~/.windhover/models` (falls back to `~/.kestrel/models` if present).  
(`./kestrel` remains a thin shim to `./windhover`.)

---

## Layout

| Path | Role |
|------|------|
| [`engine/`](engine/) | **`windhover-engine`** (MoE + dense KPK) |
| [`engine/runtime/windhover.c`](engine/runtime/windhover.c) | Dense Windhover runtime |
| [`tools/kestrel_pack.py`](tools/kestrel_pack.py) | HF → KPK converter |
| [`windhover`](windhover) | CLI + Library / Chat API |
| [`app/`](app/) | Vite / React UI |
| [`desktop/`](desktop/) | Tauri macOS app |
| [`docs/`](docs/) | Benches and notes |

---

## Requirements

- **macOS 12+** · **Apple Silicon recommended**
- Headline benches: **MacBook Air M4 · 16 GB**
- Xcode CLT · Rust (Tauri) · Node 18+ (from source)
- Python 3.10+ with `torch` + `transformers` for preview / convert (`c/.venv`)
- Optional: Hugging Face CLI for `--weights` pulls

---

## License

Apache-2.0 — see [LICENSE](LICENSE). Upstream attribution in [UPSTREAM.md](UPSTREAM.md).
