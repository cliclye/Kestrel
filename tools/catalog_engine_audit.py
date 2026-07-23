#!/usr/bin/env python3
"""Audit catalog.json against Hugging Face config.json + Windhover engine paths.

Verifies every installable (status ready|download, chat!=blocked) entry is one of:
  - IMMEDIATE_DENSE: qwen2/qwen3/llama/mistral → chat after download (Mac+Windows)
  - KPK_AFTER_CONVERT: phi3/gemma2/gemma3 → install must convert (Mac+Windows)
  - MOE_GLM: glm_moe_dsa → FP8 convert then MoE engine

Exits non-zero if a ready/download engine model is unsupported or mislabeled.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "app" / "public" / "catalog.json"

DENSE_IMMEDIATE = {"qwen2", "qwen3", "llama", "mistral"}
KPK_NEEDED = {"phi3", "gemma2", "gemma3"}


def fetch_cfg(repo: str) -> dict:
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "WindhoverCatalogAudit/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode())


def classify(cfg: dict) -> tuple[str, str]:
    mt = str(cfg.get("model_type") or "").lower()
    arches = " ".join(str(a) for a in (cfg.get("architectures") or []))
    moe = bool(
        cfg.get("n_routed_experts")
        or cfg.get("num_experts")
        or cfg.get("num_local_experts")
        or "Moe" in arches
        or "Mixtral" in arches
    )
    hybrid = mt in {"qwen3_5", "qwen3.5", "qwen3_5_moe", "gemma4", "gemma4_unified", "kimi_k25", "minimax_m3_vl"} or bool(
        cfg.get("vision_config")
    )
    if mt == "glm_moe_dsa":
        return "MOE_GLM", mt
    if moe:
        return "UNSUPPORTED_MOE", mt
    if hybrid and mt not in DENSE_IMMEDIATE and mt not in KPK_NEEDED:
        return "UNSUPPORTED_HYBRID", mt
    if mt in DENSE_IMMEDIATE:
        return "IMMEDIATE_DENSE", mt
    if mt in KPK_NEEDED:
        return "KPK_AFTER_CONVERT", mt
    return "UNSUPPORTED", mt


def main() -> int:
    models = json.loads(CATALOG.read_text(encoding="utf-8")).get("models") or []
    installable = [
        m
        for m in models
        if m.get("status") in ("ready", "download") and m.get("chat") not in ("blocked",)
    ]
    errors: list[str] = []
    print(f"Auditing {len(installable)} installable catalog models…\n")
    for m in installable:
        mid = m["id"]
        repo = m.get("hf_repo") or mid
        try:
            cfg = fetch_cfg(repo)
            verdict, mt = classify(cfg)
        except Exception as e:
            errors.append(f"{mid}: fetch failed: {type(e).__name__}: {e}")
            print(f"FAIL  {mid}: fetch {e}")
            continue
        path = m.get("engine_path")
        ok = True
        if verdict == "IMMEDIATE_DENSE":
            expect = "dense"
            if path and path != expect:
                ok = False
                errors.append(f"{mid}: engine_path={path!r} expected {expect!r}")
        elif verdict == "KPK_AFTER_CONVERT":
            expect = "kpk"
            if path and path != expect:
                ok = False
                errors.append(f"{mid}: engine_path={path!r} expected {expect!r}")
        elif verdict == "MOE_GLM":
            expect = "moe"
            if path and path != expect:
                ok = False
                errors.append(f"{mid}: engine_path={path!r} expected {expect!r}")
            if m.get("convert") != "fp8_to_int4":
                ok = False
                errors.append(f"{mid}: GLM MoE should set convert=fp8_to_int4")
        else:
            ok = False
            errors.append(
                f"{mid}: listed installable but engine does not support model_type={mt} ({verdict})"
            )
        mark = "OK  " if ok else "FAIL"
        print(f"{mark}  {mid}: {verdict} model_type={mt} status={m.get('status')}")
    print()
    if errors:
        print(f"{len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("All installable catalog models match a supported Mac/Windows engine path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
