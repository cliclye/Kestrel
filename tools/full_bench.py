#!/usr/bin/env python3
"""Full fair benchmark: same laptop without vs with windhover-engine.

Accuracy rules:
  - Baseline = reference local MoE engine binary; Kestrel = engine/windhover-engine.
  - Both ARCH=native; identical fixture; no in-process loop cheats on the baseline.
  - One trial = BATCH consecutive TF process runs (default 40), timed as one wall.
  - Warmup batches discarded; report mean/median/stdev/95% CI + Welch test.
  - Require 32/32 oracle every process.
  - Identical stripped env for both engines.

Limitation: glm_tiny only — not full GLM-5.2 tok/s or RAM.
Public framing: "Without Kestrel" vs "With Kestrel" (JSON keys remain stock/kestrel).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KESTREL_GLM = ROOT / "engine" / "windhover-engine"
STOCK_GLM = Path("/tmp/colibri-clone/c/glm")
KESTREL_C = ROOT / "engine"
STOCK_C = Path("/tmp/colibri-clone/c")
# Kestrel fixtures live under engine/fixtures; SNAP still ./glm_tiny via symlink in engine/
FIXTURE_SNAP = "./fixtures/glm_tiny"

POS_RE = re.compile(
    r"PREFILL \(teacher-forcing\) C vs oracle: (\d+)/(\d+) positions \| ([\d.]+) pos/s"
)
# macOS `/usr/bin/time -l` and Linux `/usr/bin/time -v`
RSS_RE_DARWIN = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.M)
RSS_RE_LINUX = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)", re.M)
TIME_LINE_RE = re.compile(
    r"^\s*([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys\s*$", re.M
)
TIME_USER_LINUX = re.compile(r"User time \(seconds\):\s*([\d.]+)")
TIME_SYS_LINUX = re.compile(r"System time \(seconds\):\s*([\d.]+)")


def _time_cmd(glm: Path) -> list[str]:
    """External time wrapper that works on both macOS and Linux."""
    if sys.platform == "darwin":
        return ["/usr/bin/time", "-l", str(glm), "64", "16", "16"]
    return ["/usr/bin/time", "-v", str(glm), "64", "16", "16"]


def _parse_time_rss(out: str) -> tuple[int | None, float | None, float | None]:
    rss = user = sys_ = None
    m = RSS_RE_DARWIN.search(out)
    if m:
        rss = int(m.group(1))  # bytes on Darwin
    else:
        m = RSS_RE_LINUX.search(out)
        if m:
            rss = int(m.group(1)) * 1024  # kB → bytes
    tm = TIME_LINE_RE.search(out)
    if tm:
        user = float(tm.group(2))
        sys_ = float(tm.group(3))
    else:
        um = TIME_USER_LINUX.search(out)
        sm = TIME_SYS_LINUX.search(out)
        if um:
            user = float(um.group(1))
        if sm:
            sys_ = float(sm.group(1))
    return rss, user, sys_


@dataclass
class ProcResult:
    ok: int
    n: int
    pos_s: float
    max_rss_bytes: int | None = None
    user_s: float | None = None
    sys_s: float | None = None


@dataclass
class BatchTrial:
    wall_s: float  # total wall for batch
    wall_per_proc: float
    pos_mean: float
    pos_median: float
    all_ok: bool
    n_procs: int
    procs: list[ProcResult]


def clean_env(snap: str = "./glm_tiny") -> dict[str, str]:
    env = os.environ.copy()
    drop_prefixes = (
        "BENCH_",
        "COLI_",
        "EMBED_",
        "EXPERT_",
        "TF",
        "SNAP",
        "PROMPT",
        "NGEN",
        "TEMP",
        "DRAFT",
        "REPLAY",
        "REF_",
        "DEBUG_",
        "PIN",
        "LOOKA",
        "PIPE",
        "OMP_",
        "KESTREL_",
        "WINDHOVER_",
    )
    for k in list(env):
        if k.startswith(drop_prefixes) or k in ("TF", "SNAP"):
            env.pop(k, None)
    env["TF"] = "1"
    env["SNAP"] = snap
    return env


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_proc(glm: Path, cwd: Path, with_time: bool, snap: str = "./glm_tiny") -> ProcResult:
    env = clean_env(snap)
    if with_time:
        cmd = _time_cmd(glm)
    else:
        cmd = [str(glm), "64", "16", "16"]
    p = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"{glm} exit {p.returncode}:\n{out[-2000:]}")
    m = POS_RE.search(out)
    if not m:
        raise RuntimeError(f"no PREFILL line from {glm}:\n{out[-1500:]}")
    ok, n, pos = int(m.group(1)), int(m.group(2)), float(m.group(3))
    rss = user = sys_ = None
    if with_time:
        rss, user, sys_ = _parse_time_rss(out)
    return ProcResult(ok, n, pos, rss, user, sys_)


def run_batch(
    glm: Path, cwd: Path, batch: int, with_time_every: int, snap: str = "./glm_tiny"
) -> BatchTrial:
    procs: list[ProcResult] = []
    t0 = time.perf_counter()
    for i in range(batch):
        with_time = with_time_every > 0 and (i % with_time_every == 0)
        procs.append(run_proc(glm, cwd, with_time=with_time, snap=snap))
    wall = time.perf_counter() - t0
    poss = [p.pos_s for p in procs]
    all_ok = all(p.ok == p.n for p in procs)
    return BatchTrial(
        wall_s=wall,
        wall_per_proc=wall / batch,
        pos_mean=statistics.fmean(poss),
        pos_median=statistics.median(poss),
        all_ok=all_ok,
        n_procs=batch,
        procs=procs,
    )


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    n = len(xs)
    mean = statistics.fmean(xs)
    stdev = statistics.stdev(xs) if n > 1 else 0.0
    se = stdev / math.sqrt(n) if n else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": statistics.median(xs),
        "stdev": stdev,
        "min": min(xs),
        "max": max(xs),
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def welch_t(a: list[float], b: list[float]) -> dict:
    na, nb = len(a), len(b)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va = statistics.variance(a) if na > 1 else 0.0
    vb = statistics.variance(b) if nb > 1 else 0.0
    se2 = va / na + vb / nb
    if se2 <= 0:
        return {"t": 0.0, "df": float(na + nb - 2), "p_approx": 1.0}
    t = (ma - mb) / math.sqrt(se2)
    num = se2 * se2
    den = 0.0
    if na > 1 and va > 0:
        den += (va / na) ** 2 / (na - 1)
    if nb > 1 and vb > 0:
        den += (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else float(na + nb - 2)
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return {"t": t, "df": df, "p_approx": p}


def delta_pct(base: float, new: float) -> float:
    if base == 0:
        return float("nan")
    return 100.0 * (new - base) / base


def run_side(
    name: str,
    glm: Path,
    cwd: Path,
    batches: int,
    batch_size: int,
    warmup: int,
    with_time_every: int,
) -> dict:
    print(f"\n== {name}: warmup_batches={warmup} measured_batches={batches} "
          f"batch_size={batch_size} ==")
    for i in range(warmup):
        b = run_batch(glm, cwd, batch_size, with_time_every=0)
        print(
            f"  warmup {i+1}/{warmup}: ok={b.all_ok}  "
            f"pos_mean={b.pos_mean:.1f}  batch_wall={b.wall_s:.4f}s  "
            f"per_proc={b.wall_per_proc*1000:.2f}ms"
        )
        if not b.all_ok:
            raise RuntimeError(f"{name} warmup correctness failed")

    trials: list[BatchTrial] = []
    for i in range(batches):
        b = run_batch(glm, cwd, batch_size, with_time_every=with_time_every)
        trials.append(b)
        print(
            f"  batch {i+1:02d}/{batches}: ok={b.all_ok}  "
            f"pos_mean={b.pos_mean:.1f}  median={b.pos_median:.1f}  "
            f"batch_wall={b.wall_s:.4f}s  per_proc={b.wall_per_proc*1000:.2f}ms"
        )
        if not b.all_ok:
            bad = next(p for p in b.procs if p.ok != p.n)
            raise RuntimeError(f"{name} correctness failed: {bad.ok}/{bad.n}")

    # Flatten all process pos/s for overall throughput distribution
    all_pos = [p.pos_s for t in trials for p in t.procs]
    batch_walls = [t.wall_s for t in trials]
    per_proc_walls = [t.wall_per_proc for t in trials]
    batch_pos_means = [t.pos_mean for t in trials]

    rss = [p.max_rss_bytes / (1024 * 1024) for t in trials for p in t.procs if p.max_rss_bytes]
    user = [p.user_s for t in trials for p in t.procs if p.user_s is not None]
    sys_ = [p.sys_s for t in trials for p in t.procs if p.sys_s is not None]

    out = {
        "name": name,
        "binary": str(glm),
        "sha256": sha256(glm),
        "all_correct": all(t.all_ok for t in trials),
        "oracle": "32/32",
        "n_batches": batches,
        "batch_size": batch_size,
        "n_processes_total": batches * batch_size,
        # Primary: mean of per-batch mean pos/s (stable), plus all-proc summary
        "pos_per_s_batch_means": summarize(batch_pos_means),
        "pos_per_s_all_procs": summarize(all_pos),
        "batch_wall_s": summarize(batch_walls),
        "wall_per_proc_s": summarize(per_proc_walls),
    }
    if rss:
        out["max_rss_mb"] = summarize(rss)
    if user:
        out["user_s"] = summarize(user)
    if sys_:
        out["sys_s"] = summarize(sys_)
    if user and sys_ and len(user) == len(sys_):
        out["cpu_s"] = summarize([u + s for u, s in zip(user, sys_)])

    out["_batch_pos_means"] = batch_pos_means
    out["_batch_walls"] = batch_walls
    out["_per_proc_walls"] = per_proc_walls
    return out


def main() -> int:
    batches = int(os.environ.get("BENCH_BATCHES", "12"))
    batch_size = int(os.environ.get("BENCH_BATCH_SIZE", "40"))
    warmup = int(os.environ.get("BENCH_WARMUP", "2"))
    # Sample RSS every Nth process inside a batch (0 = never; 10 = every 10th)
    with_time_every = int(os.environ.get("BENCH_TIME_EVERY", "10"))

    if not KESTREL_GLM.is_file() or not STOCK_GLM.is_file():
        print(f"missing binary (kestrel={KESTREL_GLM} stock={STOCK_GLM})", file=sys.stderr)
        return 1

    # Sync fixtures: stock uses ./glm_tiny; kestrel uses ./fixtures/glm_tiny
    shutil.copytree(
        KESTREL_C / "fixtures" / "glm_tiny", STOCK_C / "glm_tiny", dirs_exist_ok=True
    )
    shutil.copy2(KESTREL_C / "fixtures" / "ref_glm.json", STOCK_C / "ref_glm.json")
    # engine cwd also needs ref_glm.json at root (symlink ok)
    ref_link = KESTREL_C / "ref_glm.json"
    if not ref_link.exists():
        shutil.copy2(KESTREL_C / "fixtures" / "ref_glm.json", ref_link)

    upstream = subprocess.check_output(
        ["git", "-C", str(STOCK_C.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = "72d3d37231e922a6fa9afca16e08fa45842d5eb4"

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "mode": "TF teacher-forcing (windhover-engine fixtures/glm_tiny vs stock glm_tiny)",
            "trial_definition": (
                f"one trial = {batch_size} consecutive processes; "
                f"{batches} trials after {warmup} warmup batches"
            ),
            "fairness": [
                "no BENCH_LOOPS (stock does not have it)",
                "identical stripped env",
                "both ARCH=native OpenMP",
                "same glm_tiny + ref_glm.json",
                "page-cache warm (warmup discarded) — steady-state, not cold-start",
                "primary binary: engine/windhover-engine",
            ],
            "primary_metrics": [
                "pos/s: engine-reported forward_all throughput (excludes load)",
                "batch_wall_s: external wall for batch_size processes",
                "wall_per_proc_s: batch_wall / batch_size",
            ],
            "limitations": [
                "glm_tiny only — do not extrapolate to GLM-5.2",
                "pos/s on tiny model is short; use batch means + CI",
            ],
        },
        "stock_upstream_sha": upstream,
        "expected_upstream_sha": expected,
        "sha_match": upstream == expected,
        "batches": batches,
        "batch_size": batch_size,
        "warmup_batches": warmup,
    }

    print("=== Fair benchmark: without Kestrel vs with Kestrel (same laptop) ===")
    print(json.dumps({
        "framing": "without_kestrel (baseline) vs with_kestrel (windhover-engine)",
        "baseline_sha_match": meta["sha_match"],
        "batches": batches,
        "batch_size": batch_size,
        "warmup": warmup,
        "total_procs_per_side": batches * batch_size,
        "kestrel_binary": str(KESTREL_GLM),
        "schedule": "interleaved (baseline,kestrel)xN — cancels thermal/scheduling bias",
    }, indent=2))

    # Shared warmup on both, then interleaved measured batches
    print(f"\n== shared warmup ({warmup} batches each) ==")
    for i in range(warmup):
        for label, glm, cwd, snap in (
            ("stock", STOCK_GLM, STOCK_C, "./glm_tiny"),
            ("kestrel", KESTREL_GLM, KESTREL_C, FIXTURE_SNAP),
        ):
            b = run_batch(glm, cwd, batch_size, with_time_every=0, snap=snap)
            print(
                f"  warmup {label} {i+1}/{warmup}: ok={b.all_ok}  "
                f"pos_mean={b.pos_mean:.1f}  batch_wall={b.wall_s:.4f}s"
            )
            if not b.all_ok:
                raise RuntimeError(f"{label} warmup correctness failed")

    stock_trials: list[BatchTrial] = []
    kestrel_trials: list[BatchTrial] = []
    print(f"\n== interleaved measured batches ({batches}) ==")
    for i in range(batches):
        for label, glm, cwd, bucket, snap in (
            ("stock", STOCK_GLM, STOCK_C, stock_trials, "./glm_tiny"),
            ("kestrel", KESTREL_GLM, KESTREL_C, kestrel_trials, FIXTURE_SNAP),
        ):
            b = run_batch(
                glm, cwd, batch_size, with_time_every=with_time_every, snap=snap
            )
            bucket.append(b)
            print(
                f"  [{i+1:02d}/{batches}] {label:7s}: ok={b.all_ok}  "
                f"pos_mean={b.pos_mean:.1f}  batch_wall={b.wall_s:.4f}s  "
                f"per_proc={b.wall_per_proc*1000:.2f}ms"
            )
            if not b.all_ok:
                raise RuntimeError(f"{label} correctness failed")

    def pack(name: str, glm: Path, trials: list[BatchTrial]) -> dict:
        all_pos = [p.pos_s for t in trials for p in t.procs]
        batch_walls = [t.wall_s for t in trials]
        per_proc_walls = [t.wall_per_proc for t in trials]
        batch_pos_means = [t.pos_mean for t in trials]
        rss = [
            p.max_rss_bytes / (1024 * 1024)
            for t in trials
            for p in t.procs
            if p.max_rss_bytes
        ]
        user = [p.user_s for t in trials for p in t.procs if p.user_s is not None]
        sys_ = [p.sys_s for t in trials for p in t.procs if p.sys_s is not None]
        out = {
            "name": name,
            "binary": str(glm),
            "sha256": sha256(glm),
            "all_correct": all(t.all_ok for t in trials),
            "oracle": "32/32",
            "n_batches": batches,
            "batch_size": batch_size,
            "n_processes_total": batches * batch_size,
            "pos_per_s_batch_means": summarize(batch_pos_means),
            "pos_per_s_all_procs": summarize(all_pos),
            "batch_wall_s": summarize(batch_walls),
            "wall_per_proc_s": summarize(per_proc_walls),
            "_batch_pos_means": batch_pos_means,
            "_batch_walls": batch_walls,
            "_per_proc_walls": per_proc_walls,
        }
        if rss:
            out["max_rss_mb"] = summarize(rss)
        if user:
            out["user_s"] = summarize(user)
        if sys_:
            out["sys_s"] = summarize(sys_)
        if user and sys_ and len(user) == len(sys_):
            out["cpu_s"] = summarize([u + s for u, s in zip(user, sys_)])
        return out

    stock = pack("stock", STOCK_GLM, stock_trials)
    kestrel = pack("kestrel", KESTREL_GLM, kestrel_trials)
    meta["methodology"]["fairness"].append(
        "interleaved stock/kestrel batches (not stock-then-kestrel)"
    )

    sp = stock["pos_per_s_batch_means"]["mean"]
    kp = kestrel["pos_per_s_batch_means"]["mean"]
    sw = stock["batch_wall_s"]["mean"]
    kw = kestrel["batch_wall_s"]["mean"]
    swp = stock["wall_per_proc_s"]["mean"]
    kwp = kestrel["wall_per_proc_s"]["mean"]

    comparison = {
        "throughput_pos_per_s": {
            "stock_mean": sp,
            "kestrel_mean": kp,
            "delta_pct": delta_pct(sp, kp),
            "welch_on_batch_means": welch_t(
                stock["_batch_pos_means"], kestrel["_batch_pos_means"]
            ),
            "ci95_stock": [
                stock["pos_per_s_batch_means"]["ci95_low"],
                stock["pos_per_s_batch_means"]["ci95_high"],
            ],
            "ci95_kestrel": [
                kestrel["pos_per_s_batch_means"]["ci95_low"],
                kestrel["pos_per_s_batch_means"]["ci95_high"],
            ],
        },
        "batch_wall_s": {
            "stock_mean": sw,
            "kestrel_mean": kw,
            "delta_pct": delta_pct(sw, kw),
            "welch": welch_t(stock["_batch_walls"], kestrel["_batch_walls"]),
        },
        "wall_per_proc_s": {
            "stock_mean": swp,
            "kestrel_mean": kwp,
            "delta_pct": delta_pct(swp, kwp),
        },
        "correctness": {
            "stock_all_ok": stock["all_correct"],
            "kestrel_all_ok": kestrel["all_correct"],
        },
        "goal_10pct_throughput": delta_pct(sp, kp) >= 10.0,
    }

    # Strip private keys for JSON
    def public(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    report = {
        "meta": meta,
        "stock": public(stock),
        "kestrel": public(kestrel),
        "comparison": comparison,
    }

    out_path = Path(os.environ.get("FULL_BENCH_OUT", str(ROOT / "docs" / "full_bench.json")))
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== RESULTS (primary = mean of per-batch mean pos/s) ===")
    print(
        f"Without  pos/s  {sp:.1f}  "
        f"95%CI=[{stock['pos_per_s_batch_means']['ci95_low']:.1f}, "
        f"{stock['pos_per_s_batch_means']['ci95_high']:.1f}]  "
        f"stdev={stock['pos_per_s_batch_means']['stdev']:.1f}"
    )
    print(
        f"With     pos/s  {kp:.1f}  "
        f"95%CI=[{kestrel['pos_per_s_batch_means']['ci95_low']:.1f}, "
        f"{kestrel['pos_per_s_batch_means']['ci95_high']:.1f}]  "
        f"stdev={kestrel['pos_per_s_batch_means']['stdev']:.1f}"
    )
    w = comparison["throughput_pos_per_s"]["welch_on_batch_means"]
    print(
        f"Δ throughput   {comparison['throughput_pos_per_s']['delta_pct']:+.2f}%  "
        f"(Welch t={w['t']:.2f}, p≈{w['p_approx']:.2e})"
    )
    print(
        f"Without  batch_wall  {sw:.4f}s  (per proc {swp*1000:.2f}ms)"
    )
    print(
        f"With     batch_wall  {kw:.4f}s  (per proc {kwp*1000:.2f}ms)"
    )
    print(f"Δ batch wall   {comparison['batch_wall_s']['delta_pct']:+.2f}%")
    if "max_rss_mb" in stock and "max_rss_mb" in kestrel:
        print(
            f"RSS MB         without={stock['max_rss_mb']['mean']:.2f}  "
            f"with={kestrel['max_rss_mb']['mean']:.2f}"
        )
    print(
        f"Correctness    without={stock['all_correct']}  "
        f"with={kestrel['all_correct']} (32/32)"
    )
    print(f"Goal ≥10%      {comparison['goal_10pct_throughput']}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
