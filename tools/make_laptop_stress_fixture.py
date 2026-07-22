#!/usr/bin/env python3
"""Build a laptop-limit MoE stress fixture for Apple Silicon (≤16GB class).

This is still a *synthetic* GLM-MoE-DSA dataflow fixture (random weights) — not
GLM-5.2 or Kimi. It is sized to stress a MacBook Air M4 16GB: ~1GB on disk,
multi-expert MoE layers, enough to push CPU + unified memory harder than glm_tiny.

  python3 tools/make_laptop_stress_fixture.py
  # → engine/fixtures/glm_stress/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "c" / "tools"))

import torch
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM


def build_config() -> GlmMoeDsaConfig:
    # ~0.5–0.6B params → ~1.0–1.2GB fp16 — fits ~3GB free while stressing MoE.
    return GlmMoeDsaConfig(
        vocab_size=8192,
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size=512,
        num_hidden_layers=12,
        first_k_dense_replace=3,
        num_attention_heads=16,
        num_key_value_heads=16,
        n_routed_experts=40,
        num_experts_per_tok=8,
        n_shared_experts=1,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        index_topk=4096,
        index_head_dim=32,
        index_n_heads=4,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=False,
        rms_norm_eps=1e-5,
        attention_bias=False,
        max_position_embeddings=4096,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default=str(Path.home() / ".windhover" / "models" / "kestrel__glm-stress"),
        help="output SNAP directory (default: ~/.kestrel/models/kestrel__glm-stress)",
    )
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    cfg = build_config()
    cfg._attn_implementation = "eager"
    print(f"building GlmMoeDsa stress fixture → {out}")
    print(
        f"  layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
        f"experts={cfg.n_routed_experts} topk={cfg.num_experts_per_tok}"
    )
    model = GlmMoeDsaForCausalLM(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 2:
                param.normal_(0, 0.02)
        for layer in model.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.weight.normal_(0, 0.02)

    # Save as bf16 safetensors (engine-friendly, half the fp32 disk)
    model.to(dtype=torch.bfloat16)
    model.save_pretrained(out, safe_serialization=True)
    tok = {
        "model_type": "glm_moe_dsa",
        "note": "synthetic tokenizer stub — generate path may use ids only",
    }
    # Minimal tokenizer.json so SNAP looks complete; engine may need real tok —
    # copy from glm_tiny if present for structure.
    tiny_tok = ROOT / "engine" / "fixtures" / "glm_tiny" / "tokenizer.json"
    if tiny_tok.is_file():
        (out / "tokenizer.json").write_bytes(tiny_tok.read_bytes())
    else:
        (out / "tokenizer.json").write_text(json.dumps(tok) + "\n")

    meta = {
        "id": "kestrel/glm-stress",
        "name": "GLM Stress (laptop-limit synthetic MoE)",
        "synthetic": True,
        "not_a_real_model": True,
        "purpose": "Push MacBook Air M4 16GB without/with windhover-engine harder than glm_tiny",
        "config": {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "n_routed_experts": cfg.n_routed_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
        },
    }
    (out / "kestrel.json").write_text(json.dumps(meta, indent=2) + "\n")

    bytes_ = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"done: {out} ({bytes_ / 1e9:.2f} GB on disk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
