#!/usr/bin/env python3
"""Fair Qwen2.5-7B bench: same laptop · without Windhover vs with Windhover.

Qwen2.5-7B is dense HF — windhover-engine (GLM MoE) cannot load it.
On a 16GB Mac, CPU float16 7B decode swap-thrashes; Windhover's Mac preview
path uses MPS float16 (same as Library chat for hf_small models).

Protocol (fresh subprocess per side so RAM can reclaim):

  without → stock transformers · CPU · float16  (may timeout / thrash on 16GB)
  with    → Windhover Mac preview · MPS · float16

Writes docs/qwen7b_bench.json. Never invents numbers.
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = Path(
    os.environ.get(
        "WINDHOVER_SNAP", os.environ.get("KESTREL_SNAP",
        str(Path.home() / ".windhover" / "models" / "Qwen__Qwen2.5-7B-Instruct"),
    )
)
OUT = ROOT / "docs" / "qwen7b_bench.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = os.environ.get(
    "BENCH_PROMPT",
    "Explain MoE routing and KV-cache pressure on a 16GB MacBook Air in two short paragraphs.",
)
MAX_NEW = int(os.environ.get("BENCH_NGEN", "24"))
TRIALS = int(os.environ.get("BENCH_TRIALS", "3"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "1"))
# CPU 7B on 16GB often never finishes — bound it so the suite can complete.
CPU_TIMEOUT_S = int(os.environ.get("BENCH_CPU_TIMEOUT", "240"))
MPS_TIMEOUT_S = int(os.environ.get("BENCH_MPS_TIMEOUT", "600"))


def _has_weights(path: Path) -> bool:
    return (path / "config.json").is_file() and (
        any(path.glob("*.safetensors")) or any(path.glob("model*.bin"))
    )


def _rss_mb() -> float:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0
    except Exception:
        return 0.0


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


def _engine_probe(snap: Path) -> dict:
    bin_ = ROOT / "engine" / "windhover-engine"
    if not bin_.is_file():
        return {"tried": False, "reason": "missing windhover-engine"}
    env = {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "PATH": "/usr/bin:/bin",
        "SNAP": str(snap),
        "COLI_PROMPT": "hi",
        "NGEN": "4",
        "TEMP": "0",
        "QUIET": "1",
        "DRAFT": "0",
        "RAM_GB": "14",
    }
    p = subprocess.run(
        [str(bin_), "64", "4", "4"],
        cwd=str(ROOT / "engine"),
        env=env,
        capture_output=True,
        timeout=45,
    )
    out = (p.stdout or b"").decode("utf-8", errors="replace") + (
        p.stderr or b""
    ).decode("utf-8", errors="replace")
    return {
        "tried": True,
        "rc": p.returncode,
        "loads": p.returncode == 0 and ("tok/s" in out.lower() or "[t=" in out),
        "tail": out[-500:],
        "note": "Qwen2 dense ≠ GLM MoE SNAP; Windhover serves this model via Mac preview (transformers+MPS).",
    }


def _worker(side: str) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_src = str(SNAP)
    t_load0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(load_src, trust_remote_code=True)
    if side == "with":
        use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if not use_mps:
            print(json.dumps({"event": "fatal", "error": "MPS unavailable"}), flush=True)
            return 1
        dtype = torch.float16
        device = "mps"
        backend = "kestrel-preview/mps/float16"
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    else:
        dtype = torch.float16
        device = "cpu"
        backend = "stock-transformers/cpu/float16"
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))

    model = AutoModelForCausalLM.from_pretrained(
        load_src,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    load_s = time.perf_counter() - t_load0
    print(
        json.dumps(
            {
                "event": "loaded",
                "side": side,
                "backend": backend,
                "load_s": load_s,
                "rss_mb": _rss_mb(),
            }
        ),
        flush=True,
    )

    messages = [
        {"role": "system", "content": "You are a concise technical assistant running locally."},
        {"role": "user", "content": PROMPT},
    ]
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = PROMPT

    ngen = 8 if side == "without" else MAX_NEW

    def one() -> dict:
        inputs = tok(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=ngen,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        if device == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass
        wall = time.perf_counter() - t0
        new_tokens = int(out[0].shape[0] - prompt_tokens)
        return {
            "ok": new_tokens > 0,
            "side": side,
            "backend": backend,
            "device": device,
            "dtype": "float16",
            "wall_s": wall,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": new_tokens,
            "tok_s": (new_tokens / wall) if wall > 0 else None,
            "rss_mb": round(_rss_mb(), 1),
            "preview": tok.decode(out[0][prompt_tokens:], skip_special_tokens=True)[:200],
            "load_s": load_s,
        }

    # CPU: one short attempt only (thrash risk). MPS: warmup + trials.
    if side == "without":
        r = one()
        print(json.dumps({"event": "trial", "trial": 1, **r}), flush=True)
        if not r["ok"]:
            return 1
    else:
        for _ in range(WARMUP):
            r = one()
            print(json.dumps({"event": "warmup", **r}), flush=True)
            if not r["ok"]:
                return 1
        for i in range(TRIALS):
            r = one()
            r["trial"] = i + 1
            print(json.dumps({"event": "trial", **r}), flush=True)
            if not r["ok"]:
                return 1

    del model, tok
    gc.collect()
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    print(json.dumps({"event": "done", "side": side}), flush=True)
    return 0


def _run_side_subprocess(side: str, timeout_s: int) -> dict:
    env = os.environ.copy()
    env["QWEN7B_WORKER_SIDE"] = side
    print(f"  === side={side} (timeout={timeout_s}s) ===", flush=True)
    t0 = time.perf_counter()
    try:
        p = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        return {"status": "error", "error": str(e), "trials": []}

    trials: list[dict] = []
    load_s = None
    backend = None
    assert p.stdout is not None
    timed_out = False
    try:
        while True:
            if time.perf_counter() - t0 > timeout_s:
                timed_out = True
                p.kill()
                break
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            if not line:
                time.sleep(0.05)
                continue
            line = line.rstrip("\n")
            if not line.startswith("{"):
                if "Loading weights" in line or "%" in line:
                    continue
                print(f"    {line[:180]}", flush=True)
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("event")
            if kind == "loaded":
                load_s = ev.get("load_s")
                backend = ev.get("backend")
                print(
                    f"    loaded in {load_s:.1f}s backend={backend} rss={ev.get('rss_mb')}MB",
                    flush=True,
                )
            elif kind == "warmup":
                print(
                    f"    warmup tok/s={ev.get('tok_s')} wall={ev.get('wall_s'):.2f}s",
                    flush=True,
                )
            elif kind == "trial":
                trials.append(ev)
                print(
                    f"    trial tok/s={ev.get('tok_s')} wall={ev.get('wall_s'):.2f}s "
                    f"rss={ev.get('rss_mb')}MB",
                    flush=True,
                )
            elif kind == "fatal":
                print(f"    FATAL: {ev}", flush=True)
            elif kind == "done":
                print(f"    done {side}", flush=True)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    except Exception as e:
        p.kill()
        return {"status": "error", "error": str(e), "trials": trials}

    wall = time.perf_counter() - t0
    if timed_out:
        return {
            "status": "timeout",
            "timeout_s": timeout_s,
            "elapsed_s": wall,
            "backend": backend or ("stock-transformers/cpu/float16" if side == "without" else None),
            "device": "cpu" if side == "without" else "mps",
            "load_s": load_s,
            "trials": trials,
            "note": (
                "CPU float16 7B decode exceeded timeout on 16GB unified memory "
                "(swap thrash). Not a measured tok/s — refused to invent one."
            ),
        }
    if p.returncode != 0 or not trials:
        return {
            "status": "error",
            "rc": p.returncode,
            "elapsed_s": wall,
            "backend": backend,
            "trials": trials,
        }
    rates = [t["tok_s"] for t in trials if t.get("tok_s") is not None]
    walls = [t["wall_s"] for t in trials]
    rss = [t["rss_mb"] for t in trials if t.get("rss_mb") is not None]
    return {
        "status": "ok",
        "elapsed_s": wall,
        "backend": backend or trials[0].get("backend"),
        "device": trials[0].get("device"),
        "load_s": load_s or trials[0].get("load_s"),
        "n": len(trials),
        "all_ok": all(t.get("ok") for t in trials),
        "tok_s_mean": statistics.fmean(rates) if rates else None,
        "tok_s_stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "wall_s_mean": statistics.fmean(walls) if walls else None,
        "wall_s_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
        "rss_mb_mean": statistics.fmean(rss) if rss else None,
        "rss_mb_max": max(rss) if rss else None,
        "trials": trials,
    }


def main() -> int:
    side = os.environ.get("QWEN7B_WORKER_SIDE", "").strip()
    if side in ("without", "with"):
        return _worker(side)

    if not _has_weights(SNAP):
        print(f"missing weights at {SNAP}\n  ./kestrel pull {MODEL_ID} --weights", file=sys.stderr)
        OUT.write_text(
            json.dumps({"status": "blocked_no_weights", "model": MODEL_ID, "snap": str(SNAP)}, indent=2)
            + "\n"
        )
        return 2

    size_gb = sum(p.stat().st_size for p in SNAP.rglob("*") if p.is_file()) / 1e9
    host = _host()
    print("=== Qwen2.5-7B fair bench (without vs with Windhover) ===")
    print(json.dumps({"snap": str(SNAP), "size_gb": round(size_gb, 2), "host": host}, indent=2))
    engine_probe = _engine_probe(SNAP)
    print(f"  windhover-engine probe: loads={engine_probe.get('loads')} rc={engine_probe.get('rc')}")

    without = _run_side_subprocess("without", CPU_TIMEOUT_S)
    time.sleep(4)
    with_ = _run_side_subprocess("with", MPS_TIMEOUT_S)

    report: dict = {
        "status": "ok" if with_.get("status") == "ok" else "partial",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "framing": "same MacBook — without Windhover vs with Windhover",
        "model": MODEL_ID,
        "model_kind": "real_hf_dense",
        "not_glm52_or_kimi": True,
        "snap": str(SNAP),
        "size_gb": round(size_gb, 2),
        "host": host,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW,
        "trials": TRIALS,
        "warmup": WARMUP,
        "protocol": {
            "without_kestrel": f"stock transformers · CPU · float16 (timeout {CPU_TIMEOUT_S}s)",
            "with_kestrel": "Windhover Mac preview · MPS · float16 (Library chat path for hf_small)",
            "note": (
                "Dense Qwen is not a windhover-engine SNAP. On 16GB Macs, CPU 7B decode often "
                "swap-thrashes; Windhover's preview path uses Apple MPS."
            ),
        },
        "kestrel_engine_probe": engine_probe,
        "without_kestrel": without,
        "with_kestrel": with_,
    }
    if (
        without.get("status") == "ok"
        and with_.get("status") == "ok"
        and without.get("tok_s_mean")
        and with_.get("tok_s_mean")
    ):
        report["tok_s_delta_pct"] = (
            100.0 * (with_["tok_s_mean"] - without["tok_s_mean"]) / without["tok_s_mean"]
        )
        report["wall_delta_pct"] = (
            100.0 * (with_["wall_s_mean"] - without["wall_s_mean"]) / without["wall_s_mean"]
        )

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("\n=== RESULTS ===")
    if without.get("status") == "ok":
        print(f"Without (CPU): {without.get('tok_s_mean'):.2f} tok/s")
    else:
        print(f"Without (CPU): {without.get('status')} — {without.get('note') or without.get('error')}")
    if with_.get("status") == "ok":
        print(f"With (Windhover MPS): {with_.get('tok_s_mean'):.2f} tok/s")
    else:
        print(f"With (Windhover MPS): {with_.get('status')} — {with_}")
    if "tok_s_delta_pct" in report:
        print(f"Δ tok/s {report['tok_s_delta_pct']:+.1f}%")
    print(f"Wrote {OUT}")
    return 0 if with_.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
