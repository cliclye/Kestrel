#!/usr/bin/env python3
"""kestrel_pack.py — build a Windhover KPK pack from a dense HF snapshot.

One-time offline convert. Output keeps the engine's existing container
conventions (safetensors + sidecar scale tensors) so `engine/io/st.h` can read
it unchanged, and adds:

  * group-64 **asymmetric** int4 for o_proj / gate / up / down (and optionally
    lm_head): tensor `<name>` U8 nibble-packed (stored q+8 like the engine),
    `<name>.qs` F16 per-group scales, `<name>.qz` F16 per-group zeros
    (z = lo + 8*s, so w = s*(q-8) + z and SDOT stays signed).
  * per-row symmetric int8 for q/k/v (+ lm_head by default) — `<name>.qs`
    F32 row scales (existing engine format).
  * int8 embeddings (`<name>.qs` F32 row scales).
  * AWQ-style activation-aware scale folding (exact, zero runtime cost):
      - qkv + gate/up scales fold into the preceding RMSNorm weight
      - down_proj scales fold into up_proj output rows
      - o_proj scales fold into v_proj output rows
  * `kestrel.json` gains a "windhover" block: format tags per tensor class,
    an architecture descriptor, and CATS sparsity thresholds calibrated on a
    small prompt set (consumed by the engine's sparse FFN path).
  * down_proj is stored **transposed** ([inter, hidden], groups along hidden)
    so a sparse FFN can skip whole neuron rows of gate/up/down contiguously.
  * When calibration runs, FFN neurons are **permuted hottest-first** (gate
    and up rows plus down^T rows move together — semantics-invariant). The
    engine's AU cold tier then maps "hot prefix resident, cold tail on SSD"
    with zero indirection.

Weights are written 64-byte aligned by the safetensors layout (header first);
the engine mmaps the data range directly.

Usage:
  c/.venv/bin/python3 tools/kestrel_pack.py --snap ~/.kestrel/models/Qwen__X \
      [--out <dir>/kpk] [--no-awq] [--no-calib] [--lm-bits 8|4] [--attn-bits 8|4]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

GS = 64
CALIB_PROMPTS = [
    "Explain how a CPU cache works in two short paragraphs.",
    "Write a Python function that merges two sorted lists.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "The water cycle describes how water moves between the ocean, the air,",
    "In C, pointers and arrays are closely related. For example,",
    "Summarize the plot of a mystery novel where the detective is the culprit.",
    "SELECT name, COUNT(*) FROM orders GROUP BY name HAVING",
    "The difference between TCP and UDP is",
]


def log(msg):
    print(f"[kpk] {msg}", flush=True)


# ---------------------------------------------------------------- quantizers

def q_row_i8(w):
    """Per-row symmetric int8. Returns (int8 [O,I], f32 scales [O])."""
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / 127.0, 1e-8).astype(np.float32)
    q = np.clip(np.rint(w / s), -128, 127).astype(np.int8)
    return q, s[:, 0]


def q_g64_asym_i4(w, gs=GS):
    """Group-`gs` asymmetric int4.

    Returns (packed u8 [O, I/2], scales f16 [O, I/gs], zeros f16 [O, I/gs]).
    Storage matches the engine nibble convention: byte = (q0+8) | (q1+8)<<4
    with q in [-8, 7]; w ~= s*(q) + z  where z = lo + 8*s  (note: q here is
    the *stored* nibble minus 8, i.e. the signed value the SDOT kernel sees).
    """
    O, I = w.shape
    assert I % gs == 0
    ng = I // gs
    wg = w.reshape(O, ng, gs).astype(np.float32)
    lo = wg.min(axis=2, keepdims=True)
    hi = wg.max(axis=2, keepdims=True)
    s = np.maximum((hi - lo) / 15.0, 1e-8)
    qu = np.clip(np.rint((wg - lo) / s), 0, 15).astype(np.uint8)  # 0..15
    z = (lo + 8.0 * s)[:, :, 0].astype(np.float16)                # w = s*(qu-8)+z
    sc = s[:, :, 0].astype(np.float16)
    flat = qu.reshape(O, I)
    packed = (flat[:, 0::2] | (flat[:, 1::2] << 4)).astype(np.uint8)
    return packed, sc, z


def dequant_g64(packed, sc, z):
    O, half = packed.shape
    I = half * 2
    ng = sc.shape[1]
    q = np.empty((O, I), dtype=np.int8)
    q[:, 0::2] = (packed & 0xF).astype(np.int8) - 8
    q[:, 1::2] = (packed >> 4).astype(np.int8) - 8
    w = q.reshape(O, ng, GS).astype(np.float32) * sc[:, :, None].astype(np.float32) \
        + z[:, :, None].astype(np.float32)
    return w.reshape(O, I)


# ---------------------------------------------------------------- descriptor

ARCHES = {
    # model_type -> descriptor defaults
    "qwen2":   dict(act="silu", norm="rmsnorm", qkv_bias=True,  qk_norm=False,
                    rope="neox", softcap=0.0, sliding_window=0, post_norms=False),
    "qwen3":   dict(act="silu", norm="rmsnorm", qkv_bias=False, qk_norm=True,
                    rope="neox", softcap=0.0, sliding_window=0, post_norms=False),
    "llama":   dict(act="silu", norm="rmsnorm", qkv_bias=False, qk_norm=False,
                    rope="neox", softcap=0.0, sliding_window=0, post_norms=False),
    "mistral": dict(act="silu", norm="rmsnorm", qkv_bias=False, qk_norm=False,
                    rope="neox", softcap=0.0, sliding_window=4096, post_norms=False),
    "gemma2":  dict(act="gelu_tanh", norm="rmsnorm_gemma", qkv_bias=False, qk_norm=False,
                    rope="neox", softcap=50.0, sliding_window=4096, post_norms=True,
                    embed_scale="sqrt_hidden", attn_logit_softcap=50.0),
    "gemma3":  dict(act="gelu_tanh", norm="rmsnorm_gemma", qkv_bias=False, qk_norm=True,
                    rope="neox", softcap=0.0, sliding_window=1024, post_norms=True,
                    embed_scale="sqrt_hidden"),
    "phi3":    dict(act="silu", norm="rmsnorm", qkv_bias=False, qk_norm=False,
                    rope="neox", softcap=0.0, sliding_window=0, post_norms=False,
                    fused_qkv=True, fused_gate_up=True),
}

MOE_MARKERS = ("n_routed_experts", "num_experts", "num_local_experts",
               "glm_moe_dsa", "MixtralForCausalLM", "Qwen2MoeForCausalLM",
               "Qwen3MoeForCausalLM")


def build_descriptor(cfg):
    mt = cfg.get("model_type", "")
    raw = json.dumps(cfg)
    if any(m in raw for m in MOE_MARKERS):
        raise SystemExit("kpk: MoE packs use the engine's MoE convert path, not kestrel_pack")
    if mt not in ARCHES:
        raise SystemExit(f"kpk: unsupported model_type '{mt}' "
                         f"(supported: {', '.join(sorted(ARCHES))})")
    d = dict(ARCHES[mt])
    d["model_type"] = mt
    d["hidden"] = cfg["hidden_size"]
    d["layers"] = cfg["num_hidden_layers"]
    d["heads"] = cfg["num_attention_heads"]
    d["kv_heads"] = cfg.get("num_key_value_heads", d["heads"])
    d["head_dim"] = cfg.get("head_dim") or d["hidden"] // d["heads"]
    d["inter"] = cfg["intermediate_size"]
    d["vocab"] = cfg["vocab_size"]
    d["rope_theta"] = cfg.get("rope_theta", 10000.0)
    d["eps"] = cfg.get("rms_norm_eps", 1e-6)
    d["tie_embeddings"] = bool(cfg.get("tie_word_embeddings", False))
    d["eos"] = cfg.get("eos_token_id", -1)
    d["bos"] = cfg.get("bos_token_id", -1)
    if isinstance(d["eos"], list):
        d["eos"] = d["eos"][0] if d["eos"] else -1
    d["sliding_window"] = cfg.get("sliding_window") or d.get("sliding_window", 0) or 0
    if mt == "gemma2":
        d["softcap"] = cfg.get("final_logit_softcapping", 30.0) or 0.0
        d["attn_logit_softcap"] = cfg.get("attn_logit_softcapping", 50.0) or 0.0
    return d


# ---------------------------------------------------------------- calibration

def calibrate(snap, desc, awq):
    """Run a few prompts through the HF model to collect:
       - per-linear-input |x| channel means (AWQ folding)
       - per-layer CATS thresholds tau for sparsity levels 25/40%
       - per-layer FFN neuron hotness (|silu(g)| EMA) for AU pinning priors
    Returns (awq_scales: {name: np.f32[I]}, cats: [layers][levels], hot: [layers][inter])."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=dtype)
    model.to(device).eval()

    act_absmean = {}
    hooks = []
    if awq:
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                def mk(nm):
                    def pre(module, args):
                        x = args[0].detach()
                        a = x.abs().reshape(-1, x.shape[-1]).mean(0).float().cpu()
                        act_absmean[nm] = act_absmean.get(nm, 0) + a
                    return pre
                hooks.append(mod.register_forward_pre_hook(mk(name)))

    gate_samples = []   # list per layer of |silu(gate)| flat samples
    hot_sums = []
    mlps = [m for _, m in model.named_modules()
            if hasattr(m, "gate_proj") and hasattr(m, "act_fn")]
    for i, m in enumerate(mlps):
        gate_samples.append([])
        hot_sums.append(None)

        def mk(idx):
            def hook(module, inp, out):
                a = torch.nn.functional.silu(out.detach().float()) \
                    if desc["act"] == "silu" else \
                    torch.nn.functional.gelu(out.detach().float(), approximate="tanh")
                a = a.abs()
                flat = a.reshape(-1, a.shape[-1])
                gate_samples[idx].append(
                    flat.flatten()[:: max(1, flat.numel() // 8192)].cpu())
                s = flat.sum(0).cpu()
                hot_sums[idx] = s if hot_sums[idx] is None else hot_sums[idx] + s
            return hook
        hooks.append(m.gate_proj.register_forward_hook(mk(i)))

    with torch.no_grad():
        for p in CALIB_PROMPTS:
            x = tok(p, return_tensors="pt").input_ids.to(device)
            model(x)
    for h in hooks:
        h.remove()

    awq_scales = {}
    if awq:
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear) or name not in act_absmean:
                continue
            a = act_absmean[name].numpy().astype(np.float64)
            a = np.maximum(a / len(CALIB_PROMPTS), 1e-5)
            wmax = np.maximum(np.abs(mod.weight.data.float().cpu().numpy()).max(0), 1e-5)
            s = np.sqrt(a) / np.sqrt(wmax)
            s = np.clip(s / s.mean(), 0.2, 5.0)
            awq_scales[name] = s.astype(np.float32)

    cats = []
    hot = []
    for i in range(len(mlps)):
        sample = torch.cat(gate_samples[i]).numpy()
        cats.append({
            "p25": float(np.quantile(sample, 0.25)),
            "p40": float(np.quantile(sample, 0.40)),
            "p50": float(np.quantile(sample, 0.50)),
        })
        h = hot_sums[i].numpy()
        order = np.argsort(-h).astype(np.int32)
        hot.append(order)
    del model
    return awq_scales, cats, hot


# ---------------------------------------------------------------- converter

def load_shards(snap):
    from safetensors import safe_open
    files = sorted(f for f in os.listdir(snap) if f.endswith(".safetensors"))
    if not files:
        raise SystemExit(f"kpk: no safetensors in {snap}")
    tensors = {}
    for f in files:
        with safe_open(os.path.join(snap, f), framework="np") as sf:
            for k in sf.keys():
                tensors[k] = (f, None)
    return files, tensors


def _bf16_u16_to_f32(u16):
    """Convert little-endian bfloat16 bit patterns to float32 without torch."""
    bits = np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def read_tensor(snap, fname, key):
    """Load one tensor as float32.

    Prefers numpy / ml_dtypes so packaged Windows sidecars can convert without
    torch. Falls back to torch only when BF16 cannot be decoded otherwise.
    """
    from safetensors import safe_open

    path = os.path.join(snap, fname)
    # 1) numpy path (F32/F16; BF16 when ml_dtypes is installed)
    try:
        with safe_open(path, framework="np") as sf:
            t = sf.get_tensor(key)
        if t.dtype == np.float32:
            return np.ascontiguousarray(t)
        if t.dtype == np.float16:
            return np.ascontiguousarray(t.astype(np.float32))
        # ml_dtypes.bfloat16 or similar
        return np.ascontiguousarray(t.astype(np.float32))
    except Exception:
        pass

    # 2) torch (dev machines / calibration)
    try:
        import torch

        with safe_open(path, framework="pt") as sf:
            t = sf.get_tensor(key)
        return t.float().numpy()
    except Exception:
        pass

    # 3) Manual BF16/F16/F32 via header offsets (no torch / no ml_dtypes)
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n).decode("utf-8"))
        info = header[key]
        dtype = info["dtype"]
        begin, end = info["data_offsets"]
        f.seek(8 + n + begin)
        raw = f.read(end - begin)
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        return np.ascontiguousarray(_bf16_u16_to_f32(u16))
    if dtype == "F16":
        return np.ascontiguousarray(np.frombuffer(raw, dtype="<f2").astype(np.float32))
    if dtype == "F32":
        return np.ascontiguousarray(np.frombuffer(raw, dtype="<f4"))
    raise RuntimeError(f"kpk: unsupported dtype {dtype} for {key} (install torch or ml_dtypes)")


def convert(snap, outdir, awq=True, calib=True, attn_bits=8, lm_bits=8):
    with open(os.path.join(snap, "config.json")) as f:
        cfg = json.load(f)
    desc = build_descriptor(cfg)
    L, D, I = desc["layers"], desc["hidden"], desc["inter"]
    H, KV, hd = desc["heads"], desc["kv_heads"], desc["head_dim"]
    log(f"arch {desc['model_type']}: L={L} D={D} I={I} H={H}/{KV} hd={hd} "
        f"vocab={desc['vocab']} tie={desc['tie_embeddings']}")

    files, index = load_shards(snap)

    def get(key):
        if key not in index:
            return None
        return read_tensor(snap, index[key][0], key)

    awq_scales, cats, hot = ({}, None, None)
    if calib or awq:
        try:
            awq_scales, cats, hot = calibrate(snap, desc, awq)
            log(f"calibration done (awq tensors: {len(awq_scales)}, "
                f"cats layers: {len(cats) if cats else 0})")
        except Exception as e:  # torch missing etc. — pack still valid
            log(f"calibration skipped: {e}")
            awq_scales, cats, hot = {}, None, None

    out = {}          # name -> np array to write
    meta_fmt = {}     # tensor class formats for kestrel.json

    def put_i8(name, w):
        q, s = q_row_i8(w)
        out[name] = q.view(np.uint8) if q.dtype == np.int8 else q
        out[name + ".qs"] = s.astype(np.float32)

    def put_g64(name, w, outlier_cols=0):
        O, Iw = w.shape
        if Iw % GS:
            put_i8(name, w)  # ragged: keep int8 row
            return
        if outlier_cols > 0:
            # Residual-stream outlier channels: a few input dims carry values
            # 10-100x the group median and wreck asymmetric int4 (measured on
            # Qwen3-0.6B down^T: cos 0.855 -> 0.995 with 8 fp16 columns).
            colmag = np.abs(w).max(axis=0)
            idx = np.argsort(-colmag)[:outlier_cols].astype(np.int32)
            idx.sort()
            # NB: w[:, idx] fancy indexing can return an F-ordered buffer;
            # force C-order or safetensors serializes it transposed.
            oc = np.ascontiguousarray(w[:, idx], dtype=np.float16)
            w = w.copy()
            w[:, idx] = 0.0
            out[name + ".oc"] = oc
            out[name + ".oci"] = idx
        p, sc, z = q_g64_asym_i4(w)
        out[name] = p
        out[name + ".qs"] = sc
        out[name + ".qz"] = z

    def fold(name, w):
        """Multiply weight columns by AWQ scale for this linear (input dim)."""
        s = awq_scales.get(name)
        if s is None or s.shape[0] != w.shape[1]:
            return w, None
        return w * s[None, :], s

    # --- embeddings / lm_head ---
    emb = get("model.embed_tokens.weight")
    if emb is None:
        raise SystemExit("kpk: missing embed_tokens")
    put_i8("model.embed_tokens.weight", emb)
    lm = get("lm_head.weight")
    if lm is None:
        if not desc["tie_embeddings"]:
            raise SystemExit("kpk: missing lm_head and not tied")
        lm = emb
    # lm_head always written separately (untied on disk; G2: tied int4 corrupts embeds)
    if lm_bits == 4:
        put_g64("lm_head.weight", lm)
    else:
        put_i8("lm_head.weight", lm)
    meta_fmt["lm_head"] = f"int{lm_bits}" + ("_g64" if lm_bits == 4 else "_row")

    fn = get("model.norm.weight")
    if fn is None:
        raise SystemExit("kpk: missing final norm")
    out["model.norm.weight"] = fn

    prefix = "model.layers.{i}."
    for i in range(L):
        p = prefix.format(i=i)
        in_ln = get(p + "input_layernorm.weight")
        post_ln = get(p + "post_attention_layernorm.weight")
        wq = get(p + "self_attn.q_proj.weight")
        wk = get(p + "self_attn.k_proj.weight")
        wv = get(p + "self_attn.v_proj.weight")
        wo = get(p + "self_attn.o_proj.weight")
        wg = get(p + "mlp.gate_proj.weight")
        wu = get(p + "mlp.up_proj.weight")
        wd = get(p + "mlp.down_proj.weight")
        if desc.get("fused_qkv") and wq is None:
            qkv = get(p + "self_attn.qkv_proj.weight")
            wq = qkv[: H * hd]
            wk = qkv[H * hd: H * hd + KV * hd]
            wv = qkv[H * hd + KV * hd:]
        if desc.get("fused_gate_up") and wg is None:
            gu = get(p + "mlp.gate_up_proj.weight")
            wg, wu = gu[:I], gu[I:]
        if any(t is None for t in (in_ln, post_ln, wq, wk, wv, wo, wg, wu, wd)):
            raise SystemExit(f"kpk: layer {i} missing tensors")

        # --- exact AWQ folding ---
        # qkv: scale s folds out of input_layernorm weight
        sq = awq_scales.get(p + "self_attn.q_proj")  # same input for q/k/v
        if sq is not None and sq.shape[0] == D:
            wq, wk, wv = wq * sq[None, :], wk * sq[None, :], wv * sq[None, :]
            in_ln = in_ln / sq
        # FFN neuron permutation: hottest-first so the AU cold tier is a
        # simple "hot prefix resident, cold tail streamed" split. Output is
        # invariant (gate/up rows and down columns move together).
        if hot is not None and len(hot) > i and hot[i].shape[0] == I:
            perm = hot[i]
            wg, wu = wg[perm], wu[perm]
            wd = wd[:, perm]
            if awq_scales.get(p + "mlp.down_proj") is not None:
                awq_scales[p + "mlp.down_proj"] = \
                    awq_scales[p + "mlp.down_proj"][perm]
        # gate/up: folds out of post_attention_layernorm
        sg = awq_scales.get(p + "mlp.gate_proj")
        if sg is not None and sg.shape[0] == D:
            wg, wu = wg * sg[None, :], wu * sg[None, :]
            post_ln = post_ln / sg
        # down: folds into up rows
        sd = awq_scales.get(p + "mlp.down_proj")
        if sd is not None and sd.shape[0] == I:
            wd = wd * sd[None, :]
            wu = wu / sd[:, None]
        # o: folds into v_proj rows (+ v bias). Attention output channel
        # (h, d) carries v_{kv(h), d}, so exactness under GQA requires one
        # shared scale per (kv-head, d): average AWQ's per-q-head scales
        # within each group, broadcast back to all q heads of the group.
        so = awq_scales.get(p + "self_attn.o_proj")
        bv = get(p + "self_attn.v_proj.bias")
        if so is not None and so.shape[0] == H * hd:
            gqa = H // KV
            so_kv = so.reshape(KV, gqa, hd).mean(axis=1)          # [KV, hd]
            so_q = np.repeat(so_kv, gqa, axis=0).reshape(H * hd)  # broadcast
            wo = wo * so_q[None, :]
            wv = wv / so_kv.reshape(KV * hd)[:, None]
            if bv is not None:
                bv = bv / so_kv.reshape(KV * hd)

        out[p + "input_layernorm.weight"] = in_ln
        out[p + "post_attention_layernorm.weight"] = post_ln
        for nm in ("q_norm", "k_norm"):
            t = get(p + f"self_attn.{nm}.weight")
            if t is not None:
                out[p + f"self_attn.{nm}.weight"] = t
        for nm in ("q_proj", "k_proj"):
            b = get(p + f"self_attn.{nm}.bias")
            if b is not None:
                out[p + f"self_attn.{nm}.bias"] = b
        if bv is not None:
            out[p + "self_attn.v_proj.bias"] = bv
        if desc.get("post_norms"):
            for nm in ("pre_feedforward_layernorm", "post_feedforward_layernorm"):
                t = get(p + nm + ".weight")
                if t is not None:
                    out[p + nm + ".weight"] = t

        if attn_bits == 4:
            put_g64(p + "self_attn.q_proj.weight", wq)
            put_g64(p + "self_attn.k_proj.weight", wk)
            put_g64(p + "self_attn.v_proj.weight", wv)
        else:
            put_i8(p + "self_attn.q_proj.weight", wq)
            put_i8(p + "self_attn.k_proj.weight", wk)
            put_i8(p + "self_attn.v_proj.weight", wv)
        # o_proj int8 for Phi-class widths — int4 here also skews near-tied tokens.
        if D <= 4096:
            put_i8(p + "self_attn.o_proj.weight", wo)
        else:
            put_g64(p + "self_attn.o_proj.weight", wo)
        # Small/medium dense models are quant-sensitive on FFN up-projections.
        # Keep gate/up at int8 through Phi-4 Mini (D=3072); int4-g64 alone can
        # flip near-tied arithmetic tokens (323 vs 357 on 17×19).
        if D <= 4096:
            put_i8(p + "mlp.gate_proj.weight", wg)
            put_i8(p + "mlp.up_proj.weight", wu)
        else:
            put_g64(p + "mlp.gate_proj.weight", wg)
            put_g64(p + "mlp.up_proj.weight", wu)
        # down stored TRANSPOSED [inter, hidden]. Phi-class widths: int8 rows
        # (accuracy); larger models keep int4-g64 + outlier columns.
        if D <= 4096:
            put_i8(p + "mlp.down_proj.weight.t", np.ascontiguousarray(wd.T))
        else:
            put_g64(p + "mlp.down_proj.weight.t", np.ascontiguousarray(wd.T),
                    outlier_cols=16)
        if (i + 1) % 8 == 0 or i == L - 1:
            log(f"layer {i + 1}/{L} quantized")

    meta_fmt["attn_qkv"] = f"int{attn_bits}" + ("_g64" if attn_bits == 4 else "_row")
    if D <= 4096:
        meta_fmt["attn_o"] = "int8_row"
        meta_fmt["mlp"] = "int8_row"  # gate/up/down^T int8
    else:
        meta_fmt["attn_o"] = meta_fmt["mlp"] = "int4_g64_asym"
    meta_fmt["embed"] = "int8_row"

    # --- write shards (~1.8GB max per file) ---
    os.makedirs(outdir, exist_ok=True)
    from safetensors.numpy import save_file
    shard, size, si = {}, 0, 0
    written = []
    for k, v in out.items():
        shard[k] = v
        size += v.nbytes
        if size > 1_800_000_000:
            fn_ = f"kpk-{si:05d}.safetensors"
            save_file(shard, os.path.join(outdir, fn_))
            written.append(fn_)
            shard, size, si = {}, 0, si + 1
    if shard:
        fn_ = f"kpk-{si:05d}.safetensors"
        save_file(shard, os.path.join(outdir, fn_))
        written.append(fn_)

    # --- copy config/tokenizer ---
    import shutil
    for f in os.listdir(snap):
        if f in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json"):
            shutil.copy2(os.path.join(snap, f), os.path.join(outdir, f))

    # --- kestrel.json windhover block ---
    meta_path = os.path.join(snap, "kestrel.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta["windhover"] = {
        "version": 1,
        "group_size": GS,
        "formats": meta_fmt,
        "awq": bool(awq_scales),
        "down_transposed": True,
        "neurons_permuted_hot_first": hot is not None,
        "descriptor": desc,
        "cats_tau": cats,
        "shards": written,
    }
    with open(os.path.join(outdir, "kestrel.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total = sum(v.nbytes for v in out.values())
    log(f"wrote {len(written)} shard(s), {total / 1e9:.2f} GB -> {outdir}")
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-awq", action="store_true")
    ap.add_argument("--no-calib", action="store_true")
    ap.add_argument("--attn-bits", type=int, default=8, choices=(4, 8))
    ap.add_argument("--lm-bits", type=int, default=8, choices=(4, 8))
    args = ap.parse_args()
    snap = os.path.expanduser(args.snap)
    outdir = args.out or os.path.join(snap, "kpk")
    convert(snap, outdir, awq=not args.no_awq, calib=not args.no_calib,
            attn_bits=args.attn_bits, lm_bits=args.lm_bits)


if __name__ == "__main__":
    main()
