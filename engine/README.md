# Windhover engine

Clean-slate CPU runtime. Primary binary: `windhover-engine` (`kestrel-engine` symlink kept for older scripts).

- **GLM MoE** (`glm_moe_dsa`) — MLA + streamed experts (oracle: `fixtures/glm_tiny`)
- **Dense / Windhover KPK** (Qwen2 / Llama / Mistral / Gemma / Phi) — sparse working-set + int8 KV

## Layout

| Path | Role |
|------|------|
| `io/` | safetensors, tokenizer, json headers |
| `memory/` | hard RAM budget (`budget.c`) |
| `runtime/engine.c` | MoE forward / generate / TF oracle / CLI |
| `runtime/dense.c` | dense path (`WH=0`) |
| `runtime/windhover.c` | KPK Windhover runtime |
| `attn/` `moe/` `model/` `tensor/` | module boundaries for further splits |
| `fixtures/` | `glm_tiny` + `ref_glm.json` |

## Build & oracle

```bash
make ARCH=native
make test-oracle   # expect 32/32

# Dense / Windhover example (after ./windhover pull … --weights && convert):
SNAP=~/.windhover/models/Qwen__Qwen2.5-Coder-1.5B-Instruct/kpk \
  PROMPT='Say hi' NGEN=32 ./windhover-engine 64 4 4
```

Attribution: numerics lineage documented in repo-root `UPSTREAM.md` (Apache-2.0).
