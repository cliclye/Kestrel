#!/usr/bin/env python3
"""Windhover A/B bench: KPK packs on kestrel-engine (Windhover vs legacy dense).

Measures decode-only tok/s, prefill tok/s, RSS, footprint, sparsity from
engine stderr / @@WH_STATS@@. Never invents numbers.

  ./kestrel bench --windhover
  WH_BENCH_MODELS=1.5b,7b NGEN=32 python3 tools/windhover_bench.py

Writes docs/windhover_bench.json.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_host import host_info

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "kestrel-engine"
OUT = ROOT / "docs" / "windhover_bench.json"
MODELS_DIR = Path.home() / ".kestrel" / "models"
NGEN = int(os.environ.get("BENCH_NGEN", "32"))
TRIALS = int(os.environ.get("BENCH_TRIALS", "2"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "1"))
PROMPT = os.environ.get(
    "BENCH_PROMPT",
    "Write four short bullet points about local MoE inference on a laptop.",
)

# id -> preferred SNAP roots (first existing wins)
CATALOG = {
    "1.5b": [
        MODELS_DIR / "Qwen__Qwen2.5-Coder-1.5B-Instruct" / "kpk",
        MODELS_DIR / "Qwen__Qwen2.5-Coder-1.5B-Instruct",
    ],
    "0.6b": [
        MODELS_DIR / "Qwen__Qwen3-0.6B" / "kpk",
        MODELS_DIR / "Qwen__Qwen3-0.6B",
    ],
    "7b": [
        MODELS_DIR / "Qwen__Qwen2.5-7B-Instruct" / "kpk",
        MODELS_DIR / "Qwen__Qwen2.5-7B-Instruct",
    ],
}


def _host() -> dict:
    return host_info()


def _is_kpk(snap: Path) -> bool:
    kj = snap / "kestrel.json"
    if not kj.is_file():
        return False
    try:
        return "windhover" in json.loads(kj.read_text(encoding="utf-8"))
    except Exception:
        return False


def _resolve(keys: list[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for k in keys:
        for p in CATALOG.get(k, []):
            if p.is_dir() and (_is_kpk(p) or (p / "config.json").is_file()):
                found[k] = p
                break
    return found


def _chat_prompt(snap: Path, user: str) -> str:
    # Prefer HF tokenizer at parent if snap is …/kpk
    root = snap.parent if snap.name == "kpk" else snap
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True)
        return tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def _parse_wh(stderr: str, stdout: str) -> dict:
    r: dict = {}
    m = re.search(
        r"\[wh\] decode ([\d.]+) tok/s \((\d+) tok, (\d+) fwd\) \| "
        r"prefill ([\d.]+) tok/s \| RSS ([\d.]+) GB \| footprint ([\d.]+) GB",
        stderr,
    )
    if m:
        r.update(
            decode_tok_s=float(m.group(1)),
            tokens=int(m.group(2)),
            forwards=int(m.group(3)),
            prefill_tok_s=float(m.group(4)),
            rss_gb=float(m.group(5)),
            footprint_gb=float(m.group(6)),
        )
    sp = re.search(r"\[wh\] sparsity ([\d.]+)%", stderr)
    if sp:
        r["sparsity_pct"] = float(sp.group(1))
    pf = re.search(r"\[wh\] prefill (\d+) tok in [\d.]+s \(([\d.]+) tok/s\)", stderr)
    if pf:
        r["prefill_tokens"] = int(pf.group(1))
        r.setdefault("prefill_tok_s", float(pf.group(2)))
    if "@@WH_STATS@@" in stdout:
        _, _, tail = stdout.partition("@@WH_STATS@@")
        try:
            js = json.loads(tail.strip().splitlines()[0])
            r.update({k: v for k, v in js.items() if v is not None})
        except Exception:
            pass
    # legacy dense path lines
    dm = re.search(r"decode\s+([\d.]+)\s+tok/s.*?for\s+(\d+)\s+toks", stderr)
    if dm and "decode_tok_s" not in r:
        r["decode_tok_s"] = float(dm.group(1))
        r["tokens"] = int(dm.group(2))
    return r


def _hf_root(snap: Path) -> Path:
    return snap.parent if snap.name == "kpk" else snap


def _run_without_transformers(snap: Path, prompt: str) -> dict:
    """Decode-only tok/s via stock transformers (without Kestrel)."""
    root = _hf_root(snap)
    code = r"""
import json, sys, time, resource
snap, prompt, ngen = sys.argv[1:4]
ngen = int(ngen)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    snap, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True
)
model.eval()
inputs = tok(prompt, return_tensors="pt")
eos = tok.eos_token_id
with torch.inference_mode():
    out = model(**inputs, use_cache=True)
    past = out.past_key_values
    next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    ids = [int(next_id.item())]
    t_dec = time.perf_counter()
    for _ in range(ngen - 1):
        if eos is not None and ids[-1] == eos:
            break
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        ids.append(int(next_id.item()))
    decode_s = time.perf_counter() - t_dec
ntok = len(ids)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_gb = rss / (1024**3) if sys.platform == "darwin" else rss / (1024**2)
print(json.dumps({
    "ok": True,
    "path": "transformers cpu fp16",
    "decode_tok_s": round(ntok / decode_s, 3) if decode_s > 0 else 0,
    "tokens": ntok,
    "rss_gb": round(rss_gb, 3),
    "metric": "decode_only",
}))
"""
    p = subprocess.run(
        [sys.executable, "-c", code, str(root), prompt, str(NGEN)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if p.returncode != 0:
        return {"ok": False, "error": (p.stderr or p.stdout)[-800:], "rc": p.returncode}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"ok": False, "error": f"parse: {e}", "stdout": (p.stdout or "")[-500:]}


def _run(snap: Path, prompt: str, *, windhover: bool) -> dict:
    if not ENGINE.is_file():
        return {"ok": False, "error": "missing kestrel-engine — run ./kestrel build"}
    env = os.environ.copy()
    env.update(
        {
            "SNAP": str(snap),
            "PROMPT": prompt,
            "COLI_PROMPT": prompt,
            "NGEN": str(NGEN),
            "TEMP": "0",
            "QUIET": "0",
            "DRAFT": "0",
            "WH_JSON_STATS": "1",
            "WH_STATS": "1",
        }
    )
    if windhover:
        env.pop("WH", None)
    else:
        env["WH"] = "0"
    t0 = time.perf_counter()
    p = subprocess.run(
        [str(ENGINE), "64", "4", "4"],
        cwd=str(ROOT / "engine"),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    wall = time.perf_counter() - t0
    stats = _parse_wh(p.stderr or "", p.stdout or "")
    ok = p.returncode == 0 and stats.get("decode_tok_s") is not None
    return {
        "ok": ok,
        "rc": p.returncode,
        "wall_s": round(wall, 3),
        "path": "windhover" if windhover else "legacy-dense (WH=0)",
        "text": (p.stdout or "").split("@@WH_STATS@@")[0].strip()[:240],
        "stderr_tail": (p.stderr or "")[-600:],
        **stats,
    }


def _mean(runs: list[dict], key: str) -> float | None:
    xs = [float(r[key]) for r in runs if r.get("ok") and r.get(key) is not None]
    return round(statistics.mean(xs), 3) if xs else None


def main() -> int:
    want = [
        x.strip()
        for x in os.environ.get("WH_BENCH_MODELS", "1.5b,7b").split(",")
        if x.strip()
    ]
    snaps = _resolve(want)
    if not snaps:
        print(
            "no KPK/SNAP packs found under ~/.kestrel/models — "
            "run ./kestrel pull … && ./kestrel convert …",
            file=sys.stderr,
        )
        return 2
    if not ENGINE.is_file():
        print("missing engine — run ./kestrel build", file=sys.stderr)
        return 2

    doc: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": _host(),
        "protocol": {
            "prompt": PROMPT,
            "max_new_tokens": NGEN,
            "trials": TRIALS,
            "warmup": WARMUP,
            "metric": "decode_only from [wh] / @@WH_STATS@@",
            "without": "transformers · CPU · float16 · greedy decode loop",
            "windhover": "default kestrel-engine (KPK mmap + CATS + int8 KV)",
            "legacy": "WH=0 dense.c path when pack still loads",
        },
        "models": {},
    }

    want_without = os.environ.get("WH_BENCH_WITHOUT", "1") == "1"
    want_legacy = os.environ.get("WH_BENCH_LEGACY", "0") == "1"

    for key, snap in snaps.items():
        print(f"\n======== {key}  snap={snap} ========", flush=True)
        prompt = _chat_prompt(snap, PROMPT)
        wh_runs: list[dict] = []
        leg_runs: list[dict] = []
        wo_runs: list[dict] = []
        for i in range(WARMUP + TRIALS):
            tag = "warmup" if i < WARMUP else f"trial-{i - WARMUP + 1}"
            if want_without and (_hf_root(snap) / "config.json").is_file():
                print(f"--- without transformers ({tag}) ---", flush=True)
                r_wo = _run_without_transformers(snap, prompt)
                print(
                    json.dumps(
                        {
                            k: r_wo.get(k)
                            for k in ("ok", "decode_tok_s", "rss_gb", "tokens", "error")
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if i >= WARMUP:
                    wo_runs.append(r_wo)
            print(f"--- windhover ({tag}) ---", flush=True)
            r = _run(snap, prompt, windhover=True)
            print(
                json.dumps(
                    {
                        k: r.get(k)
                        for k in (
                            "ok",
                            "decode_tok_s",
                            "prefill_tok_s",
                            "rss_gb",
                            "footprint_gb",
                            "sparsity_pct",
                            "tokens",
                            "error",
                        )
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if i >= WARMUP:
                wh_runs.append(r)
            if want_legacy:
                print(f"--- legacy WH=0 ({tag}) ---", flush=True)
                r0 = _run(snap, prompt, windhover=False)
                print(
                    json.dumps(
                        {
                            k: r0.get(k)
                            for k in ("ok", "decode_tok_s", "rss_gb", "error")
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if i >= WARMUP:
                    leg_runs.append(r0)

        entry = {
            "snap": str(snap),
            "kpk": _is_kpk(snap),
            "windhover": {
                "runs": wh_runs,
                "mean_decode_tok_s": _mean(wh_runs, "decode_tok_s"),
                "mean_prefill_tok_s": _mean(wh_runs, "prefill_tok_s"),
                "mean_rss_gb": _mean(wh_runs, "rss_gb"),
                "mean_footprint_gb": _mean(wh_runs, "footprint_gb"),
                "mean_sparsity_pct": _mean(wh_runs, "sparsity_pct"),
            },
        }
        if wo_runs:
            entry["without"] = {
                "runs": wo_runs,
                "mean_decode_tok_s": _mean(wo_runs, "decode_tok_s"),
                "mean_rss_gb": _mean(wo_runs, "rss_gb"),
            }
            wo = entry["without"]["mean_decode_tok_s"]
            wi = entry["windhover"]["mean_decode_tok_s"]
            wo_rss = entry["without"]["mean_rss_gb"]
            wi_rss = entry["windhover"]["mean_rss_gb"]
            if wo and wi and wo > 0:
                entry["delta_decode_pct_vs_without"] = round(100.0 * (wi - wo) / wo, 1)
            if wo_rss and wi_rss and wo_rss > 0:
                entry["delta_rss_pct_vs_without"] = round(
                    100.0 * (wi_rss - wo_rss) / wo_rss, 1
                )
        if leg_runs:
            entry["legacy"] = {
                "runs": leg_runs,
                "mean_decode_tok_s": _mean(leg_runs, "decode_tok_s"),
                "mean_rss_gb": _mean(leg_runs, "rss_gb"),
            }
            wo = entry["legacy"]["mean_decode_tok_s"]
            wi = entry["windhover"]["mean_decode_tok_s"]
            if wo and wi and wo > 0:
                entry["delta_decode_pct"] = round(100.0 * (wi - wo) / wo, 1)
        doc["models"][key] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    for key, entry in doc["models"].items():
        wh = entry["windhover"]
        line = (
            f"  {key}: windhover decode={wh['mean_decode_tok_s']} tok/s  "
            f"rss={wh['mean_rss_gb']} GB  sparsity={wh['mean_sparsity_pct']}%"
        )
        if entry.get("without"):
            wo = entry["without"]
            line += (
                f"  | without={wo['mean_decode_tok_s']} tok/s "
                f"rss={wo['mean_rss_gb']} GB  "
                f"Δdecode={entry.get('delta_decode_pct_vs_without')}% "
                f"Δrss={entry.get('delta_rss_pct_vs_without')}%"
            )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
