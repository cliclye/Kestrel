# Kestrel engine

Clean-slate CPU runtime. Primary binary: `kestrel-engine`.

- **GLM MoE** (`glm_moe_dsa`) — MLA + streamed experts (oracle: `fixtures/glm_tiny`)
- **Dense** (Qwen2 / Llama / Mistral) — GQA + SwiGLU, int8 weights + NEON IDOT; auto-detected from `config.json`

## Layout

| Path | Role |
|------|------|
| `io/` | safetensors, tokenizer, json headers |
| `memory/` | hard RAM budget (`budget.c`) |
| `runtime/engine.c` | MoE forward / generate / TF oracle / CLI |
| `runtime/dense.c` | dense Qwen2/Llama/Mistral path |
| `attn/` `moe/` `model/` `tensor/` | module boundaries for further splits |
| `fixtures/` | `glm_tiny` + `ref_glm.json` |

## Build & oracle

```bash
make ARCH=native
make test-oracle   # expect 32/32

# Dense example (after ./kestrel pull … --weights):
SNAP=~/.kestrel/models/Qwen__Qwen2.5-Coder-1.5B-Instruct \
  PROMPT='Say hi' NGEN=32 ./kestrel-engine 64 4 4
```

Attribution: numerics lineage documented in repo-root `UPSTREAM.md` (Apache-2.0).
