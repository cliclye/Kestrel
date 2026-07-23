# Windhover engine

Clean-slate CPU runtime. Primary binary: `windhover-engine` (`kestrel-engine` symlink kept for older scripts).

- **WMIR** (`kestrel.json` → `windhover.wmir`) — architecture-agnostic layer graph; packers lower HF configs via `tools/wmir/`
- **Dense / Windhover KPK** — GQA (+ KV share, linear GDN, chunked/CSA/MSA windows), SwiGLU/GELU/double-wide MLP
- **GLM MoE** (`glm_moe_dsa`) — MLA + streamed experts (oracle: `fixtures/glm_tiny`); other MoE families emit WMIR + stream markers

## Layout

| Path | Role |
|------|------|
| `io/` | safetensors, tokenizer, json headers |
| `memory/` | hard RAM budget (`budget.c`) |
| `runtime/wmir.h` | WMIR schema + loader |
| `runtime/engine.c` | MoE forward / generate / TF oracle / CLI |
| `runtime/dense.c` | dense path (`WH=0`) |
| `runtime/windhover.c` | KPK Windhover runtime (WMIR interpreter) |
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

Attribution: Apache-2.0 — see repo-root [LICENSE](../LICENSE).
