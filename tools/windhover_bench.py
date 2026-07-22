#!/usr/bin/env python3
"""Windhover A/B bench: KPK packs on windhover-engine.

Measures decode-only tok/s, prefill tok/s, RSS, footprint, sparsity from
engine stderr / @@WH_STATS@@. Never invents numbers.

  ./windhover bench --windhover
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

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "windhover-engine"
OUT = ROOT / "docs" / "windhover_bench.json"
def _models_dir():
    wh = Path.home() / ".windhover" / "models"
    ke = Path.home() / ".kestrel" / "models"
    return wh if wh.is_dir() or not ke.is_dir() else ke


MODELS_DIR = _models_dir()
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
    out: dict = {"platform": sys.platform}
    for key, flag in (
        ("logical_cpu", "hw.logicalcpu"),
        ("physical_cpu", "hw.physicalcpu"),
        ("mem_bytes", "hw.memsize"),
        ("cpu_brand", "machdep.cpu.brand_string"),
    ):
        try:
            out[key] = subprocess.check_output(
                ["sysctl", "-n", flag], text=True, timeout=2
            ).strip()
        except Exception:
            pass
    if "mem_bytes" in out:
        try:
            out["mem_gb"] = round(int(out["mem_bytes"]) / (1024**3), 1)
        except Exception:
            pass
    return out


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


def _run(snap: Path, prompt: str, *, windhover: bool) -> dict:
    if not ENGINE.is_file():
        return {"ok": False, "error": "missing windhover-engine — run ./windhover build"}
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
        print("missing engine — run ./windhover build", file=sys.stderr)
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
            "windhover": "default windhover-engine (KPK mmap + CATS + int8 KV)",
            "legacy": "WH=0 dense.c path when pack still loads",
        },
        "models": {},
    }

    for key, snap in snaps.items():
        print(f"\n======== {key}  snap={snap} ========", flush=True)
        prompt = _chat_prompt(snap, PROMPT)
        wh_runs: list[dict] = []
        leg_runs: list[dict] = []
        for i in range(WARMUP + TRIALS):
            tag = "warmup" if i < WARMUP else f"trial-{i - WARMUP + 1}"
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
            # legacy only when not a pure-KPK-only install (optional)
            if os.environ.get("WH_BENCH_LEGACY", "0") == "1":
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
        print(
            f"  {key}: decode={wh['mean_decode_tok_s']} tok/s  "
            f"prefill={wh['mean_prefill_tok_s']}  "
            f"rss={wh['mean_rss_gb']} GB  "
            f"footprint={wh['mean_footprint_gb']} GB  "
            f"sparsity={wh['mean_sparsity_pct']}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
