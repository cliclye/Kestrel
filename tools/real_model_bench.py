#!/usr/bin/env python3
"""Real-model fair bench: same laptop without vs with windhover-engine.

This is NOT for glm_tiny. That fixture is a synthetic ~2MB oracle (vocab=256,
hidden=128) used only for numerics / scheduling — it is not GLM-5.2, Kimi, or
any published model.

To bench a frontier MoE you need a converted local SNAP, e.g.:

  # GLM-5.2 (~756 GB FP8 download, then convert)
  ./kestrel pull zai-org/GLM-5.2-FP8 --weights
  python3 c/tools/convert_fp8_to_int4.py --indir <fp8> --outdir <int4>

  # Kimi K2.6 / K2.7 Code (~600 GB) — same idea once HF weights are local

Then:

  KESTREL_SNAP=~/.kestrel/models/<converted> \\
  BASELINE_BIN=/path/to/baseline/glm \\
    python3 tools/real_model_bench.py

On this machine today (~2 GB free), downloading GLM-5.2 / Kimi is not possible.
The script exits with a clear status instead of inventing numbers.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KESTREL_BIN = ROOT / "engine" / "windhover-engine"
BASELINE_BIN = Path(os.environ.get("BASELINE_BIN", "/tmp/colibri-clone/c/glm"))
OUT = ROOT / "docs" / "real_model_bench.json"

TOK_RE = re.compile(r"([\d.]+)\s*tok/s", re.I)
POS_RE = re.compile(r"([\d.]+)\s*pos/s", re.I)

# Catalog download sizes (approx GB) — for honest refusal messages
FRONTIER = {
    "zai-org/GLM-5.2-FP8": 756,
    "zai-org/GLM-5.1-FP8": 700,
    "moonshotai/Kimi-K2.6": 600,
    "moonshotai/Kimi-K2.7-Code": 600,
    "moonshotai/Kimi-K2-Thinking": 600,
}


def _dir_bytes(path: Path) -> int:
    total = 0
    if not path.is_dir():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _looks_like_glm_tiny(snap: Path) -> bool:
    cfg = snap / "config.json"
    if not cfg.is_file():
        return False
    try:
        c = json.loads(cfg.read_text())
    except json.JSONDecodeError:
        return False
    return int(c.get("vocab_size") or 0) <= 512 and int(c.get("hidden_size") or 0) <= 256


def _free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def _run_once(binary: Path, snap: Path, prompt: str, ngen: int) -> dict:
    env = os.environ.copy()
    for k in list(env):
        if k.startswith(("BENCH_", "COLI_", "TF", "SNAP", "PROMPT", "NGEN", "TEMP", "QUIET")):
            env.pop(k, None)
    env["SNAP"] = str(snap)
    env["PROMPT"] = prompt
    env["NGEN"] = str(ngen)
    env["TEMP"] = "0"
    env["QUIET"] = "1"
    env["DRAFT"] = "0"
    t0 = time.perf_counter()
    p = subprocess.run(
        [str(binary), "64", "4", "4"],
        cwd=str(binary.parent if binary.name != "windhover-engine" else ROOT / "engine"),
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - t0
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    tok = None
    m = TOK_RE.search(out) or POS_RE.search(out)
    if m:
        tok = float(m.group(1))
    return {
        "ok": p.returncode == 0,
        "wall_s": wall,
        "reported_rate": tok,
        "rc": p.returncode,
        "tail": out[-800:],
    }


def main() -> int:
    snap_env = os.environ.get("WINDHOVER_SNAP", os.environ.get("KESTREL_SNAP", "").strip()
    free = _free_gb(Path.home())
    print("=== Real-model bench (without vs with Windhover) ===")
    print(
        json.dumps(
            {
                "glm_tiny_is_real_model": False,
                "glm_tiny_note": (
                    "Synthetic ~2MB TF oracle (vocab=256, hidden=128). "
                    "Not GLM-5.2, not Kimi, not any HF checkpoint."
                ),
                "disk_free_gb": round(free, 2),
                "frontier_download_gb": FRONTIER,
                "snap": snap_env or None,
            },
            indent=2,
        )
    )

    if not snap_env:
        need = min(FRONTIER.values())
        print(
            f"\nNo KESTREL_SNAP set.\n"
            f"  Need a converted frontier MoE dir (Kimi / GLM-5.2), typically ≥{need} GB download.\n"
            f"  Free disk now: {free:.1f} GB — not enough to pull GLM-5.2 (~756 GB) or Kimi (~600 GB).\n"
            f"  Refusing to invent numbers. Micro-fixture results stay in docs/full_bench.json "
            f"(clearly labeled synthetic).",
            file=sys.stderr,
        )
        report = {
            "status": "blocked_no_snap",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "glm_tiny_is_real_model": False,
            "disk_free_gb": free,
            "required_examples_gb": FRONTIER,
            "message": "Set KESTREL_SNAP to a converted real MoE checkpoint to run this bench.",
        }
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {OUT}")
        return 2

    snap = Path(snap_env).expanduser().resolve()
    if not snap.is_dir():
        print(f"SNAP not a directory: {snap}", file=sys.stderr)
        return 1
    if _looks_like_glm_tiny(snap):
        print(
            "Refusing: SNAP looks like glm_tiny (synthetic micro-fixture).\n"
            "  Point KESTREL_SNAP at a real converted MoE (GLM-5.2 / Kimi / …).",
            file=sys.stderr,
        )
        return 2

    size_gb = _dir_bytes(snap) / (1024**3)
    if size_gb < 1.0:
        print(
            f"Refusing: SNAP is only {size_gb:.3f} GB — too small for a frontier MoE bench.",
            file=sys.stderr,
        )
        return 2

    if not KESTREL_BIN.is_file():
        print(f"missing {KESTREL_BIN} — run ./windhover build", file=sys.stderr)
        return 1
    if not BASELINE_BIN.is_file():
        print(
            f"missing baseline binary {BASELINE_BIN}\n"
            f"  Set BASELINE_BIN to the non-Windhover engine on this laptop.",
            file=sys.stderr,
        )
        return 1

    prompt = os.environ.get(
        "BENCH_PROMPT",
        "Write a short Python function that returns the Fibonacci sequence up to n.",
    )
    ngen = int(os.environ.get("BENCH_NGEN", "64"))
    trials = int(os.environ.get("BENCH_TRIALS", "5"))
    warmup = int(os.environ.get("BENCH_WARMUP", "1"))

    print(f"SNAP={snap} ({size_gb:.1f} GB)  trials={trials} ngen={ngen}")

    results = {"without": [], "with": []}
    for i in range(warmup):
        for label, binary in (("without", BASELINE_BIN), ("with", KESTREL_BIN)):
            r = _run_once(binary, snap, prompt, ngen)
            print(f"  warmup {label} {i+1}/{warmup}: ok={r['ok']} wall={r['wall_s']:.3f}s rate={r['reported_rate']}")
            if not r["ok"]:
                print(r["tail"][-500:], file=sys.stderr)
                return 1

    for i in range(trials):
        for label, binary, bucket in (
            ("without", BASELINE_BIN, results["without"]),
            ("with", KESTREL_BIN, results["with"]),
        ):
            r = _run_once(binary, snap, prompt, ngen)
            bucket.append(r)
            print(
                f"  [{i+1:02d}/{trials}] {label:7s}: ok={r['ok']}  "
                f"wall={r['wall_s']:.3f}s  rate={r['reported_rate']}"
            )
            if not r["ok"]:
                print(r["tail"][-500:], file=sys.stderr)
                return 1

    def summarize(side: list[dict]) -> dict:
        walls = [x["wall_s"] for x in side]
        rates = [x["reported_rate"] for x in side if x["reported_rate"] is not None]
        out = {
            "n": len(side),
            "wall_s_mean": statistics.fmean(walls),
            "wall_s_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
        }
        if rates:
            out["rate_mean"] = statistics.fmean(rates)
            out["rate_stdev"] = statistics.stdev(rates) if len(rates) > 1 else 0.0
        return out

    without = summarize(results["without"])
    with_ = summarize(results["with"])
    wall_delta = 100.0 * (with_["wall_s_mean"] - without["wall_s_mean"]) / without["wall_s_mean"]
    report = {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "framing": "same laptop — without Windhover vs with Windhover",
        "snap": str(snap),
        "snap_size_gb": size_gb,
        "prompt": prompt,
        "ngen": ngen,
        "trials": trials,
        "without_kestrel": without,
        "with_kestrel": with_,
        "wall_delta_pct": wall_delta,
        "note": "Real converted MoE SNAP — not glm_tiny.",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("\n=== RESULTS ===")
    print(f"Without wall {without['wall_s_mean']:.3f}s  With wall {with_['wall_s_mean']:.3f}s  Δ {wall_delta:+.1f}%")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
