#!/usr/bin/env python3
"""Audit catalog.json against WMIR lowerers + Hugging Face configs.

Installable models (status ready|download, chat!=blocked) must have a WMIR
lowerer whose required_ops are in the kernel registry. engine_path must match:

  - dense: immediate qwen2/3/llama/mistral
  - kpk:  convert-required dense/hybrid (phi/gemma/gemma4/qwen3_5/…)
  - moe:  streamed MoE (glm / llama4 / kimi / deepseek_v4 / …)
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from wmir import can_lower, lower_config, missing_ops  # noqa: E402

CATALOG = ROOT / "app" / "public" / "catalog.json"

DENSE_IMMEDIATE = {"qwen2", "qwen3", "llama", "mistral"}
MOE_FAMILIES = {
    "glm_moe_dsa", "llama4", "llama4_text", "kimi_k25", "kimi_k2",
    "deepseek_v4", "minimax_m3", "minimax_m3_vl", "mistral_large3",
    "qwen3_5_moe",
}


def fetch_cfg(repo: str) -> dict:
    if repo in STUB_CFGS:
        try:
            # Prefer live config when the mirror is public; else stub.
            url = f"https://huggingface.co/{repo}/resolve/main/config.json"
            req = urllib.request.Request(url, headers={"User-Agent": "WindhoverCatalogAudit/2.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return dict(STUB_CFGS[repo])
    for name in ("config.json", "params.json"):
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        req = urllib.request.Request(url, headers={"User-Agent": "WindhoverCatalogAudit/2.0"})
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                cfg = json.loads(resp.read().decode())
            if name == "params.json":
                cfg.setdefault("model_type", "mistral_large3")
            return cfg
        except Exception:
            continue
    if repo in STUB_CFGS:
        return dict(STUB_CFGS[repo])
    raise RuntimeError(f"no config.json/params.json for {repo}")


def expected_path(cfg: dict, wmir: dict) -> str:
    family = str(wmir.get("family") or "").lower()
    mt = str(cfg.get("model_type") or "").lower()
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    mt_text = str(tc.get("model_type") or mt).lower()
    # Hybrid MoE that still packs through kestrel_pack (linear + routed).
    if family == "qwen3_5_moe":
        return "kpk"
    if family in MOE_FAMILIES or mt in MOE_FAMILIES:
        return "moe"
    if family in DENSE_IMMEDIATE or mt_text in DENSE_IMMEDIATE:
        if family in DENSE_IMMEDIATE and not cfg.get("vision_config"):
            return "dense"
    return "kpk"


# Gated / mirror-unfriendly repos: enough fields for WMIR lowerer checks.
STUB_CFGS: dict[str, dict] = {
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "model_type": "llama4",
        "text_config": {
            "model_type": "llama4_text",
            "hidden_size": 5120,
            "num_hidden_layers": 48,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 202048,
            "num_local_experts": 16,
            "num_experts_per_tok": 1,
            "attention_chunk_size": 8192,
            "rms_norm_eps": 1e-5,
        },
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": {
        "model_type": "llama4",
        "text_config": {
            "model_type": "llama4_text",
            "hidden_size": 5120,
            "num_hidden_layers": 48,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 202048,
            "num_local_experts": 128,
            "num_experts_per_tok": 1,
            "attention_chunk_size": 8192,
            "rms_norm_eps": 1e-5,
        },
    },
}


def main() -> int:
    models = json.loads(CATALOG.read_text(encoding="utf-8")).get("models") or []
    installable = [
        m
        for m in models
        if m.get("status") in ("ready", "download") and m.get("chat") not in ("blocked",)
    ]
    errors: list[str] = []
    print(f"Auditing {len(installable)} installable catalog models (WMIR)…\n")
    for m in installable:
        mid = m["id"]
        repo = m.get("hf_repo") or mid
        try:
            cfg = fetch_cfg(repo)
            if not can_lower(cfg):
                raise RuntimeError(f"no WMIR lowerer for model_type={cfg.get('model_type')}")
            wmir = lower_config(cfg)
            miss = missing_ops(wmir.get("required_ops") or [])
            if miss:
                raise RuntimeError(f"missing kernels: {', '.join(miss)}")
            verdict = f"WMIR:{wmir.get('family')}"
            expect = expected_path(cfg, wmir)
        except Exception as e:
            errors.append(f"{mid}: {type(e).__name__}: {e}")
            print(f"FAIL  {mid}: {e}")
            continue
        path = m.get("engine_path")
        ok = True
        if path and path != expect:
            # Allow kpk for models that could also be dense (Phi always kpk).
            if not (expect == "dense" and path == "kpk"):
                ok = False
                errors.append(f"{mid}: engine_path={path!r} expected {expect!r}")
        if expect == "moe" and m.get("convert") is None and "glm" in (wmir.get("family") or ""):
            if m.get("convert") != "fp8_to_int4":
                # GLM keeps fp8 convert; other MoE use WMIR-only stream.
                pass
        if expect == "moe" and (wmir.get("family") or "") == "glm_moe_dsa":
            if m.get("convert") != "fp8_to_int4":
                ok = False
                errors.append(f"{mid}: GLM MoE should set convert=fp8_to_int4")
        mark = "OK  " if ok else "FAIL"
        print(f"{mark}  {mid}: {verdict} path={path or expect} status={m.get('status')}")
        if not ok and mid not in " ".join(errors):
            pass
    print()
    if errors:
        print(f"{len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("All installable catalog models have a WMIR lowerer + kernel ops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
