#!/usr/bin/env python3
"""Fair dense-engine bench: Qwen2.5-Coder-1.5B without vs with windhover-engine.

Both sides report **decode-only** tok/s (prefill excluded), same chat-templated
prompt, greedy decoding, fixed max_new_tokens (no early-EOS short-circuiting the
timer unfairly — we still stop on EOS but require enough tokens for a stable rate).

  without → stock transformers · CPU (or BENCH_DEVICE=mps) · float16
  with    → windhover-engine dense · int8 + NEON IDOT

Writes docs/dense_qwen_bench.json. Never invents numbers.
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
def _default_snap() -> Path:
    for home in (Path.home() / ".windhover" / "models", Path.home() / ".kestrel" / "models"):
        cand = home / "Qwen__Qwen2.5-Coder-1.5B-Instruct"
        if cand.is_dir():
            return cand
    return Path.home() / ".windhover" / "models" / "Qwen__Qwen2.5-Coder-1.5B-Instruct"


SNAP = Path(os.environ.get("WINDHOVER_SNAP", os.environ.get("KESTREL_SNAP", str(_default_snap()))))
OUT = Path(os.environ.get("DENSE_BENCH_OUT", str(ROOT / "docs" / "dense_qwen_bench.json")))
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
PROMPT = os.environ.get(
    "BENCH_PROMPT",
    "Write four short bullet points about local MoE inference on a laptop.",
)
MAX_NEW = int(os.environ.get("BENCH_NGEN", "48"))
TRIALS = int(os.environ.get("BENCH_TRIALS", "3"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "1"))
DEVICE = os.environ.get("BENCH_DEVICE", "cpu")  # cpu | mps


def _has_weights(path: Path) -> bool:
    return (path / "config.json").is_file() and (
        any(path.glob("*.safetensors")) or any(path.glob("model*.bin"))
    )


def _host() -> dict:
    return host_info()


def _chat_prompt(snap: Path, user: str) -> str:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=True)
        return tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return (
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


def _run_without(snap: Path, prompt: str, ngen: int) -> dict:
    """Decode-only tok/s: prefill untimed, then timed greedy decode loop."""
    code = r"""
import json, sys, time, resource
snap, prompt, ngen, device = sys.argv[1:5]
ngen = int(ngen)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
dtype = torch.float16
model = AutoModelForCausalLM.from_pretrained(
    snap, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
)
use_mps = device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
if use_mps:
    model = model.to("mps")
    device = "mps"
else:
    device = "cpu"
model.eval()
load_s = time.perf_counter() - t0
inputs = tok(prompt, return_tensors="pt")
dev = next(model.parameters()).device
inputs = {k: v.to(dev) for k, v in inputs.items()}
eos = tok.eos_token_id
# Prefill (untimed for decode rate)
with torch.inference_mode():
    t_pre = time.perf_counter()
    out = model(**inputs, use_cache=True)
    prefill_s = time.perf_counter() - t_pre
    past = out.past_key_values
    next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    ids = [int(next_id.item())]
    # Decode-only window
    t_dec = time.perf_counter()
    for _ in range(ngen - 1):
        if eos is not None and ids[-1] == eos:
            break
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        ids.append(int(next_id.item()))
    if hasattr(torch, "mps") and device == "mps":
        torch.mps.synchronize()
    decode_s = time.perf_counter() - t_dec
ntok = len(ids)
text = tok.decode(ids, skip_special_tokens=True)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_gb = rss / (1024**3) if sys.platform == "darwin" else rss / (1024**2)
print(json.dumps({
    "ok": True,
    "device": device,
    "load_s": round(load_s, 3),
    "prefill_s": round(prefill_s, 3),
    "decode_s": round(decode_s, 3),
    "tokens": ntok,
    "tok_s": round(ntok / decode_s, 3) if decode_s > 0 else 0,
    "rss_gb": round(rss_gb, 3),
    "text": text[:240],
    "metric": "decode_only",
}))
"""
    p = subprocess.run(
        [sys.executable, "-c", code, str(snap), prompt, str(ngen), DEVICE],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if p.returncode != 0:
        return {"ok": False, "error": (p.stderr or p.stdout)[-800:], "rc": p.returncode}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"ok": False, "error": f"parse: {e}", "stdout": p.stdout[-500:]}


def _run_with_engine(snap: Path, prompt: str, ngen: int) -> dict:
    bin_ = ROOT / "engine" / "windhover-engine"
    if not bin_.is_file():
        return {"ok": False, "error": "missing windhover-engine — run ./windhover build"}
    env = os.environ.copy()
    env.update(
        {
            "SNAP": str(snap),
            "PROMPT": prompt,
            "COLI_PROMPT": prompt,
            "NGEN": str(ngen),
            "QUIET": "0",
            "TEMP": "0",
            "DRAFT": "0",
            # Force legacy dense.c path even when a sibling KPK pack exists.
            "WH": "0",
        }
    )
    t0 = time.perf_counter()
    p = subprocess.run(
        [str(bin_), "64", "4", "4"],
        cwd=str(ROOT / "engine"),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    wall = time.perf_counter() - t0
    err = p.stderr or ""
    out = p.stdout or ""
    m = re.search(r"decode\s+([\d.]+)\s+tok/s.*?for\s+(\d+)\s+toks", err)
    pre_m = re.search(r"prefill\s+([\d.]+)s\s+\(([\d.]+)\s+tok/s\)", err)
    load_m = re.search(r"loaded .* in ([\d.]+)s\s*\|\s*RSS\s*([\d.]+)\s*GB", err)
    rss_m = re.search(r"RSS\s+([\d.]+)\s+GB\s*\|\s*load", err)
    tok_s = float(m.group(1)) if m else None
    ntok = int(m.group(2)) if m else None
    return {
        "ok": p.returncode == 0 and tok_s is not None,
        "rc": p.returncode,
        "tok_s": tok_s,
        "tokens": ntok,
        "wall_s": round(wall, 3),
        "prefill_s": float(pre_m.group(1)) if pre_m else None,
        "prefill_tok_s": float(pre_m.group(2)) if pre_m else None,
        "load_s": float(load_m.group(1)) if load_m else None,
        "rss_gb": float(rss_m.group(1)) if rss_m else (float(load_m.group(2)) if load_m else None),
        "text": out.strip()[:240],
        "stderr_tail": err[-500:],
        "path": "windhover-engine dense (int8 + IDOT)",
        "metric": "decode_only",
    }


def main() -> int:
    if not _has_weights(SNAP):
        print(f"missing weights at {SNAP}", file=sys.stderr)
        print("run: ./windhover pull Qwen/Qwen2.5-Coder-1.5B-Instruct --weights", file=sys.stderr)
        return 2
    prompt = _chat_prompt(SNAP, PROMPT)
    print(f"snap={SNAP}")
    print(
        f"prompt_chars={len(prompt)} ngen={MAX_NEW} trials={TRIALS} "
        f"without_device={DEVICE} metric=decode_only"
    )

    without_runs: list[dict] = []
    with_runs: list[dict] = []
    for i in range(WARMUP + TRIALS):
        tag = "warmup" if i < WARMUP else f"trial-{i - WARMUP + 1}"
        print(f"\n=== without ({tag}) ===", flush=True)
        w0 = _run_without(SNAP, prompt, MAX_NEW)
        print(
            json.dumps(
                {k: w0.get(k) for k in ("ok", "tok_s", "tokens", "rss_gb", "text", "error")},
                ensure_ascii=False,
            )
        )
        if i >= WARMUP:
            without_runs.append(w0)
        print(f"\n=== with engine ({tag}) ===", flush=True)
        w1 = _run_with_engine(SNAP, prompt, MAX_NEW)
        print(
            json.dumps(
                {k: w1.get(k) for k in ("ok", "tok_s", "tokens", "rss_gb", "text", "error")},
                ensure_ascii=False,
            )
        )
        if i >= WARMUP:
            with_runs.append(w1)

    def _mean_tok(runs: list[dict]) -> float | None:
        xs = [float(r["tok_s"]) for r in runs if r.get("ok") and r.get("tok_s") is not None]
        return round(statistics.mean(xs), 3) if xs else None

    def _mean_rss(runs: list[dict]) -> float | None:
        xs = [float(r["rss_gb"]) for r in runs if r.get("ok") and r.get("rss_gb") is not None]
        return round(statistics.mean(xs), 3) if xs else None

    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "snap": str(SNAP),
        "host": _host(),
        "protocol": {
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW,
            "trials": TRIALS,
            "warmup": WARMUP,
            "metric": "decode_only (prefill excluded on both sides)",
            "without": f"transformers · {DEVICE} · float16 · greedy decode loop",
            "with": "windhover-engine dense · int8 weights · NEON IDOT when available",
            "note": (
                "Earlier short-prompt runs that timed transformers generate() end-to-end "
                "vs engine decode-only inflated the Windhover delta; this protocol matches metrics."
            ),
        },
        "without": {
            "runs": without_runs,
            "mean_tok_s": _mean_tok(without_runs),
            "mean_rss_gb": _mean_rss(without_runs),
        },
        "with": {
            "runs": with_runs,
            "mean_tok_s": _mean_tok(with_runs),
            "mean_rss_gb": _mean_rss(with_runs),
        },
    }
    wo, wi = doc["without"]["mean_tok_s"], doc["with"]["mean_tok_s"]
    wo_rss, wi_rss = doc["without"]["mean_rss_gb"], doc["with"]["mean_rss_gb"]
    if wo and wi and wo > 0:
        doc["delta_decode_pct"] = round(100.0 * (wi - wo) / wo, 1)
        doc["delta_pct"] = doc["delta_decode_pct"]  # back-compat
    if wo_rss and wi_rss and wo_rss > 0:
        doc["delta_rss_pct"] = round(100.0 * (wi_rss - wo_rss) / wo_rss, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    print(
        f"without={wo} tok/s (rss={wo_rss} GB)  "
        f"with={wi} tok/s (rss={wi_rss} GB)  "
        f"Δdecode={doc.get('delta_decode_pct')}%  "
        f"Δrss={doc.get('delta_rss_pct')}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
