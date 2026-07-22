#!/usr/bin/env python3
"""Laptop-limit bench: same Mac without vs with windhover-engine on glm_stress.

Builds the synthetic stress SNAP if missing, then interleaved generate trials.
Honest labeling: not GLM-5.2 / Kimi — max feasible MoE stress on ~16GB Apple Silicon.
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
KESTREL_BIN = ROOT / "engine" / "windhover-engine"
BASELINE_BIN = Path(os.environ.get("BASELINE_BIN", "/tmp/colibri-clone/c/glm"))
SNAP = Path(
    os.environ.get(
        "KESTREL_SNAP",
        str(Path.home() / ".windhover" / "models" / "kestrel__glm-stress"),
    )
)
OUT = ROOT / "docs" / "laptop_limit_bench.json"
TOK_RE = re.compile(r"([\d.]+)\s*tok/s", re.I)


def _rss_mb() -> float:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0
    except Exception:
        return 0.0


def _ensure_fixture() -> None:
    if (SNAP / "config.json").is_file() and any(SNAP.glob("*.safetensors")):
        return
    print(f"building stress fixture → {SNAP}")
    rc = subprocess.call(
        [sys.executable, str(ROOT / "tools" / "make_laptop_stress_fixture.py"), "--output", str(SNAP)],
        cwd=str(ROOT),
    )
    if rc != 0:
        raise SystemExit(rc)


def _run(binary: Path, prompt: str, ngen: int) -> dict:
    # Intentionally minimal env — a polluted PROMPT/QUIET from the parent shell
    # previously made windhover-engine skip real generation.
    env = {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "SNAP": str(SNAP),
        "COLI_PROMPT": prompt,
        "NGEN": str(ngen),
        "TEMP": "0",
        "DRAFT": "0",
        "RAM_GB": os.environ.get("RAM_GB", "14"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
    }
    cwd = str(ROOT / "engine") if binary.resolve() == KESTREL_BIN.resolve() else str(binary.parent)
    t0 = time.perf_counter()
    p = subprocess.run(
        [str(binary), "256", "8", "8"],
        cwd=cwd,
        env=env,
        capture_output=True,
    )
    wall = time.perf_counter() - t0
    out = (p.stdout or b"").decode("utf-8", errors="replace") + "\n" + (
        p.stderr or b""
    ).decode("utf-8", errors="replace")
    # Prefer the last progress line tok/s (steady-state)
    rates = [float(x) for x in TOK_RE.findall(out)]
    rss = None
    rm = re.search(r"RSS\s+([\d.]+)\s*GB", out)
    if rm:
        rss = float(rm.group(1)) * 1024.0
    return {
        "ok": p.returncode == 0 and ("[t=" in out or "tok/s" in out.lower() or "prefill" in out.lower()),
        "wall_s": wall,
        "tok_s": rates[-1] if rates else None,
        "tok_s_mean_lines": statistics.fmean(rates) if rates else None,
        "rss_mb": rss,
        "rc": p.returncode,
        "tail": out[-1500:],
    }


def main() -> int:
    _ensure_fixture()
    if not KESTREL_BIN.is_file():
        print("missing windhover-engine — ./windhover build", file=sys.stderr)
        return 1
    if not BASELINE_BIN.is_file():
        print(f"missing baseline {BASELINE_BIN}", file=sys.stderr)
        return 1

    size = sum(p.stat().st_size for p in SNAP.rglob("*") if p.is_file())
    cfg = json.loads((SNAP / "config.json").read_text())
    prompt = os.environ.get(
        "BENCH_PROMPT",
        "You are running on a MacBook Air with 16GB unified memory. "
        "Explain mixture-of-experts routing, expert capacity, and KV cache pressure "
        "in five dense technical paragraphs for systems engineers.",
    )
    ngen = int(os.environ.get("BENCH_NGEN", "128"))
    trials = int(os.environ.get("BENCH_TRIALS", "8"))
    warmup = int(os.environ.get("BENCH_WARMUP", "1"))

    print("=== Laptop-limit bench (synthetic MoE stress · not GLM-5.2/Kimi) ===")
    print(
        json.dumps(
            {
                "snap": str(SNAP),
                "size_gb": round(size / 1e9, 3),
                "hidden": cfg.get("hidden_size"),
                "layers": cfg.get("num_hidden_layers"),
                "experts": cfg.get("n_routed_experts"),
                "topk": cfg.get("num_experts_per_tok"),
                "trials": trials,
                "ngen": ngen,
            },
            indent=2,
        )
    )

    for i in range(warmup):
        for label, binary in (("without", BASELINE_BIN), ("with", KESTREL_BIN)):
            r = _run(binary, prompt, ngen)
            print(f"  warmup {label}: ok={r['ok']} wall={r['wall_s']:.3f}s tok/s={r['tok_s']}")
            if not r["ok"]:
                print(r["tail"], file=sys.stderr)
                return 1

    sides = {"without": [], "with": []}
    for i in range(trials):
        for label, binary in (("without", BASELINE_BIN), ("with", KESTREL_BIN)):
            r = _run(binary, prompt, ngen)
            sides[label].append(r)
            print(
                f"  [{i+1:02d}/{trials}] {label:7s}: ok={r['ok']}  "
                f"wall={r['wall_s']:.3f}s  tok/s={r['tok_s']}"
            )
            if not r["ok"]:
                print(r["tail"], file=sys.stderr)
                return 1

    def summarize(xs: list[dict]) -> dict:
        walls = [x["wall_s"] for x in xs]
        rates = [x["tok_s"] for x in xs if x["tok_s"] is not None]
        rss = [x["rss_mb"] for x in xs if x.get("rss_mb") is not None]
        out = {
            "n": len(xs),
            "wall_s_mean": statistics.fmean(walls),
            "wall_s_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
            "all_ok": all(x["ok"] for x in xs),
        }
        if rates:
            out["tok_s_mean"] = statistics.fmean(rates)
            out["tok_s_stdev"] = statistics.stdev(rates) if len(rates) > 1 else 0.0
        if rss:
            out["rss_mb_mean"] = statistics.fmean(rss)
            out["rss_mb_max"] = max(rss)
        return out

    without = summarize(sides["without"])
    with_ = summarize(sides["with"])
    wall_delta = 100.0 * (with_["wall_s_mean"] - without["wall_s_mean"]) / without["wall_s_mean"]
    report = {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "framing": "same MacBook — without Kestrel vs with Kestrel",
        "fixture": "glm_stress synthetic MoE (laptop-limit)",
        "not_a_real_hf_model": True,
        "not_glm52_or_kimi": True,
        "snap": str(SNAP),
        "size_gb": round(size / 1e9, 3),
        "config": {
            "hidden_size": cfg.get("hidden_size"),
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "n_routed_experts": cfg.get("n_routed_experts"),
            "num_experts_per_tok": cfg.get("num_experts_per_tok"),
            "vocab_size": cfg.get("vocab_size"),
        },
        "prompt": prompt,
        "ngen": ngen,
        "without_kestrel": without,
        "with_kestrel": with_,
        "wall_delta_pct": wall_delta,
        "host_rss_mb_reporter": round(_rss_mb(), 1),
    }
    if without.get("tok_s_mean") and with_.get("tok_s_mean"):
        report["tok_s_delta_pct"] = (
            100.0 * (with_["tok_s_mean"] - without["tok_s_mean"]) / without["tok_s_mean"]
        )
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("\n=== RESULTS ===")
    print(
        f"Without wall {without['wall_s_mean']:.3f}s  "
        f"With wall {with_['wall_s_mean']:.3f}s  Δwall {wall_delta:+.1f}%"
    )
    if "tok_s_mean" in without and "tok_s_mean" in with_:
        print(
            f"Without tok/s {without['tok_s_mean']:.2f}  "
            f"With tok/s {with_['tok_s_mean']:.2f}  "
            f"Δ {report.get('tok_s_delta_pct', float('nan')):+.1f}%"
        )
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
