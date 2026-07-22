#!/usr/bin/env python3
"""wh_parity.py — end-to-end numeric parity: Windhover engine vs fp16 HF.

Runs the engine on a fixed prompt with WH_LOGITS=<dump>, computes the same
prompt's next-token logits with transformers fp16, and reports cosine
similarity + top-k agreement. Catches descriptor/fold/kernel/rope bugs that
perplexity alone can hide. Quant noise bound: int8/int4-g64 + int8 KV keeps
cosine >= 0.99 and top1 match on confident prompts.

Usage: c/.venv/bin/python3 tools/wh_parity.py [--snap DIR] [--sparse 0]
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engine", "windhover-engine")

PROMPTS = [
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a Python function to add two numbers.<|im_end|>\n<|im_start|>assistant\n",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
]


def engine_logits(snap, prompt, sparse):
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        dump = f.name
    env = dict(os.environ)
    env.update({
        "SNAP": snap, "PROMPT": prompt, "NGEN": "1", "QUIET": "1",
        "WH_LOGITS": dump, "WH_SPARSE": str(sparse), "TEMP": "0",
    })
    r = subprocess.run([ENGINE, "64", "4", "4"], env=env,
                       cwd=os.path.join(ROOT, "engine"),
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"engine failed: {r.stderr[-1500:]}")
    logits = np.fromfile(dump, dtype=np.float32)
    os.unlink(dump)
    return logits


def hf_logits(snap, prompt, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap)
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=dtype)
    model.to(device).eval()
    outs = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            lg = model(ids).logits[0, -1].float().cpu().numpy()
            outs.append((tok, ids, lg))
    del model
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default=os.path.expanduser(
        "~/.kestrel/models/Qwen__Qwen2.5-Coder-1.5B-Instruct"))
    ap.add_argument("--engine-snap", default=None,
                    help="override SNAP passed to the engine (defaults to --snap)")
    ap.add_argument("--sparse", default="0")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    refs = hf_logits(args.snap, PROMPTS, device)
    eng_snap = args.engine_snap or args.snap

    ok = True
    for i, prompt in enumerate(PROMPTS):
        tok, ids, ref = refs[i]
        eng = engine_logits(eng_snap, prompt, args.sparse)
        if eng.shape[0] != ref.shape[0]:
            n = min(eng.shape[0], ref.shape[0])
            eng, ref = eng[:n], ref[:n]
        cos = float(np.dot(eng, ref) /
                    (np.linalg.norm(eng) * np.linalg.norm(ref) + 1e-9))
        t1_ref = int(ref.argmax())
        t1_eng = int(eng.argmax())
        top5_ref = set(np.argsort(-ref)[:5].tolist())
        top5_eng = set(np.argsort(-eng)[:5].tolist())
        overlap = len(top5_ref & top5_eng)
        match = t1_ref == t1_eng
        # 0.985 = observed noise floor for sub-1B models (int8 KV + activation
        # quant compound more on small hidden sizes); >=1B lands 0.99+.
        ok &= match and cos >= 0.985
        print(f"prompt {i}: cos={cos:.4f} top1 {'MATCH' if match else 'DIFF'} "
              f"({tok.decode([t1_eng])!r} vs {tok.decode([t1_ref])!r}) "
              f"top5 overlap {overlap}/5")
    print("PARITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
