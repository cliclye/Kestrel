#!/usr/bin/env python3
"""Push this MacBook toward its limit: concurrent MoE generate soak.

Uses the synthetic glm_stress SNAP (not GLM-5.2 / Kimi). Spawns N parallel
engine processes, samples vm_stat compressor pressure, and compares
without-Kestrel vs with-Kestrel aggregate tok/s under the same load.

  python3 tools/laptop_soak_bench.py
  → docs/laptop_soak_bench.json
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
OUT = ROOT / "docs" / "laptop_soak_bench.json"
TOK_RE = re.compile(r"([\d.]+)\s*tok/s", re.I)
RSS_RE = re.compile(r"RSS\s+([\d.]+)\s*GB", re.I)


def _vm_sample() -> dict:
    try:
        text = subprocess.check_output(["vm_stat"], text=True, timeout=5)
    except Exception as e:
        return {"error": str(e)}
    page = 16384.0
    d: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"([^:]+):\s+(\d+)", line.strip())
        if m:
            d[m.group(1).strip()] = int(m.group(2).rstrip("."))
    def gb(key: str) -> float:
        return round(d.get(key, 0) * page / 1e9, 3)

    return {
        "free_gb": round(
            (d.get("Pages free", 0) + d.get("Pages speculative", 0)) * page / 1e9, 3
        ),
        "wired_gb": gb("Pages wired down"),
        "active_gb": gb("Pages active"),
        "inactive_gb": gb("Pages inactive"),
        "compressor_gb": gb("Pages occupied by compressor"),
        "compressed_pages_gb": gb("Pages stored in compressor"),
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


def _ensure_fixture() -> None:
    if (SNAP / "config.json").is_file() and any(SNAP.glob("*.safetensors")):
        return
    rc = subprocess.call(
        [
            sys.executable,
            str(ROOT / "tools" / "make_laptop_stress_fixture.py"),
            "--output",
            str(SNAP),
        ],
        cwd=str(ROOT),
    )
    if rc != 0:
        raise SystemExit(rc)


def _one(binary: Path, prompt: str, ngen: int, omp: int) -> dict:
    env = {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "SNAP": str(SNAP),
        "COLI_PROMPT": prompt,
        "NGEN": str(ngen),
        "TEMP": "0",
        "DRAFT": "0",
        "RAM_GB": os.environ.get("RAM_GB", "14"),
        "OMP_NUM_THREADS": str(omp),
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
    rates = [float(x) for x in TOK_RE.findall(out)]
    rss = None
    rm = RSS_RE.search(out)
    if rm:
        rss = float(rm.group(1)) * 1024.0
    return {
        "ok": p.returncode == 0 and bool(rates),
        "wall_s": wall,
        "tok_s": rates[-1] if rates else None,
        "rss_mb": rss,
        "rc": p.returncode,
    }


def _soak(binary: Path, concurrency: int, ngen: int, omp: int, prompt: str) -> dict:
    samples: list[dict] = []
    stop = {"flag": False}

    def sampler() -> None:
        while not stop["flag"]:
            samples.append({"t": time.time(), **_vm_sample()})
            time.sleep(0.35)

    import threading

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    workers: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(_one, binary, f"{prompt} [worker {i+1}/{concurrency}]", ngen, omp)
            for i in range(concurrency)
        ]
        for fut in as_completed(futs):
            workers.append(fut.result())
    wall = time.perf_counter() - t0
    stop["flag"] = True
    th.join(timeout=1.0)

    rates = [w["tok_s"] for w in workers if w.get("tok_s") is not None]
    rss = [w["rss_mb"] for w in workers if w.get("rss_mb") is not None]
    peak_comp = max((s.get("compressor_gb") or 0.0) for s in samples) if samples else None
    min_free = min((s.get("free_gb") or 99.0) for s in samples) if samples else None
    return {
        "concurrency": concurrency,
        "wall_s": wall,
        "all_ok": all(w["ok"] for w in workers),
        "per_worker_tok_s_mean": statistics.fmean(rates) if rates else None,
        "aggregate_tok_s": (sum(rates) if rates else None),
        "worker_rss_mb_mean": statistics.fmean(rss) if rss else None,
        "peak_compressor_gb": peak_comp,
        "min_free_gb": min_free,
        "vm_samples": samples[-8:],  # keep tail only
        "workers_n": len(workers),
        "failed": sum(1 for w in workers if not w["ok"]),
    }


def main() -> int:
    _ensure_fixture()
    if not KESTREL_BIN.is_file() or not BASELINE_BIN.is_file():
        print("missing engine binaries", file=sys.stderr)
        return 1

    host = _host()
    ncpu = int(host.get("logical_cpu") or os.environ.get("BENCH_NCPU", "8"))
    levels = [
        int(x)
        for x in os.environ.get("BENCH_CONCURRENCY", f"1,4,{min(8, ncpu)}").split(",")
        if x.strip()
    ]
    ngen = int(os.environ.get("BENCH_NGEN", "256"))
    omp = int(os.environ.get("OMP_NUM_THREADS", "2"))
    prompt = os.environ.get(
        "BENCH_PROMPT",
        "Push unified memory: explain MoE expert capacity, KV cache growth, "
        "and macOS memory compression under concurrent decode.",
    )

    print("=== Laptop soak (push RAM/CPU · synthetic glm_stress) ===")
    print(json.dumps({"host": host, "levels": levels, "ngen": ngen, "omp": omp}, indent=2))

    # Warm once each side
    for label, binary in (("without", BASELINE_BIN), ("with", KESTREL_BIN)):
        r = _one(binary, prompt, min(64, ngen), omp)
        print(f"  warmup {label}: ok={r['ok']} tok/s={r['tok_s']}")
        if not r["ok"]:
            print(r, file=sys.stderr)
            return 1

    idle = _vm_sample()
    results: list[dict] = []
    for n in levels:
        row: dict = {"concurrency": n}
        for label, binary in (("without_kestrel", BASELINE_BIN), ("with_kestrel", KESTREL_BIN)):
            print(f"  soak n={n} {label} …", flush=True)
            s = _soak(binary, n, ngen, omp, prompt)
            row[label] = s
            print(
                f"    wall={s['wall_s']:.2f}s agg_tok/s={s['aggregate_tok_s']} "
                f"peak_comp={s['peak_compressor_gb']}G min_free={s['min_free_gb']}G "
                f"ok={s['all_ok']}"
            )
        results.append(row)

    report = {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "framing": "same MacBook — concurrent soak without vs with Kestrel",
        "fixture": "glm_stress synthetic MoE (laptop soak)",
        "not_a_real_hf_model": True,
        "not_glm52_or_kimi": True,
        "purpose": "Push M4 16GB toward memory-compressor / CPU saturation",
        "snap": str(SNAP),
        "host": host,
        "idle_vm": idle,
        "ngen": ngen,
        "omp_num_threads_per_worker": omp,
        "levels": results,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
