"""HF config → WMIR lowerers for every catalog family (text-only packs)."""

from __future__ import annotations

import math
from typing import Any, Callable

from .emit import build_wmir_block, synthesize_dense_layers
from .ops import missing_ops

LowerFn = Callable[[dict[str, Any]], dict[str, Any]]


def _text_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Unwrap nested VL configs (text_config / language_config)."""
    for key in ("text_config", "language_config", "llm_config"):
        tc = cfg.get(key)
        if isinstance(tc, dict) and (
            tc.get("hidden_size") or tc.get("num_hidden_layers") or tc.get("model_type")
        ):
            return tc
    return cfg


def _mt(cfg: dict[str, Any]) -> str:
    tc = _text_cfg(cfg)
    return str(tc.get("model_type") or cfg.get("model_type") or "").lower()


def _base_model(tc: dict[str, Any], *, family: str, **extra: Any) -> dict[str, Any]:
    hidden = int(tc.get("hidden_size") or 0)
    heads = int(tc.get("num_attention_heads") or 0)
    kv = int(tc.get("num_key_value_heads") or heads or 0)
    hd = int(tc.get("head_dim") or (hidden // heads if heads else 0))
    inter = int(tc.get("intermediate_size") or 0)
    layers = int(tc.get("num_hidden_layers") or 0)
    vocab = int(tc.get("vocab_size") or 0)
    eos = tc.get("eos_token_id", -1)
    if isinstance(eos, list):
        eos = eos[0] if eos else -1
    bos = tc.get("bos_token_id", -1)
    if isinstance(bos, list):
        bos = bos[0] if bos else -1
    m: dict[str, Any] = {
        "model_type": family,
        "hidden": hidden,
        "layers": layers,
        "heads": heads,
        "kv_heads": kv,
        "head_dim": hd,
        "inter": inter,
        "vocab": vocab,
        "rope_theta": float(tc.get("rope_theta") or 10000.0),
        "eps": float(tc.get("rms_norm_eps") or tc.get("layer_norm_eps") or 1e-6),
        "eos": int(eos) if eos is not None else -1,
        "bos": int(bos) if bos is not None else -1,
        "tie_embeddings": bool(tc.get("tie_word_embeddings", False)),
        "max_position": int(tc.get("max_position_embeddings") or 0),
        "act": "silu",
        "norm": "rmsnorm",
        "qkv_bias": False,
        "qk_norm": False,
        "post_norms": False,
        "sliding_window": int(tc.get("sliding_window") or 0) or 0,
        "sw_pattern": 0,
        "partial_rotary": float(tc.get("partial_rotary_factor") or 1.0),
        "embed_scale": 0.0,
        "attn_softcap": 0.0,
        "final_softcap": 0.0,
        "query_scale": 0.0,
    }
    m.update(extra)
    return m


def _finish(family: str, model: dict, layers: list, required: list[str],
            *, weight_prefix: str = "", text_only: bool = True,
            notes: str | None = None) -> dict[str, Any]:
    miss = missing_ops(required)
    if miss:
        raise RuntimeError(
            f"wmir: family {family} needs ops not in kernel registry: {', '.join(miss)}"
        )
    return build_wmir_block(
        family, model, layers, required,
        text_only=text_only, weight_prefix=weight_prefix, notes=notes,
    )


# ---- classic dense ---------------------------------------------------------

def _lower_classic(cfg: dict[str, Any], family: str) -> dict[str, Any]:
    tc = _text_cfg(cfg)
    model = _base_model(tc, family=family)
    if family == "qwen2":
        model["qkv_bias"] = True
        model["act"] = "silu"
    elif family == "qwen3":
        model["qk_norm"] = True
        model["act"] = "silu"
    elif family == "mistral":
        model["sliding_window"] = int(tc.get("sliding_window") or 4096)
        model["act"] = "silu"
    elif family == "gemma2":
        model["act"] = "gelu_tanh"
        model["norm"] = "rmsnorm_gemma"
        model["post_norms"] = True
        model["sliding_window"] = int(tc.get("sliding_window") or 4096)
        model["sw_pattern"] = 2
        model["attn_softcap"] = float(tc.get("attn_logit_softcapping") or 50.0)
        model["final_softcap"] = float(tc.get("final_logit_softcapping") or 30.0)
        model["embed_scale"] = math.sqrt(float(model["hidden"])) if model["hidden"] else 0.0
        qs = float(tc.get("query_pre_attn_scalar") or 0.0)
        if qs > 0:
            model["query_scale"] = 1.0 / math.sqrt(qs)
    elif family == "gemma3":
        model["act"] = "gelu_tanh"
        model["norm"] = "rmsnorm_gemma"
        model["post_norms"] = True
        model["qk_norm"] = True
        model["sliding_window"] = int(tc.get("sliding_window") or 1024)
        model["sw_pattern"] = int(tc.get("sliding_window_pattern") or 6)
        model["embed_scale"] = math.sqrt(float(model["hidden"])) if model["hidden"] else 0.0
    elif family == "phi3":
        model["act"] = "silu"
        model["fused_qkv"] = True
        model["fused_gate_up"] = True
    elif family == "llama":
        model["act"] = "silu"
    layers = synthesize_dense_layers(model)
    req = ["attn_gqa", "mlp_gelu" if model["act"] in ("gelu_tanh", "gelu") else "mlp_swiglu"]
    if model.get("final_softcap"):
        req.append("logit_softcap")
    return _finish(family, model, layers, req)


def lower_qwen2(cfg: dict[str, Any]) -> dict[str, Any]:
    return _lower_classic(cfg, "qwen2")


def lower_qwen3(cfg: dict[str, Any]) -> dict[str, Any]:
    return _lower_classic(cfg, "qwen3")


def lower_llama(cfg: dict[str, Any]) -> dict[str, Any]:
    return _lower_classic(cfg, "llama")


def lower_mistral(cfg: dict[str, Any]) -> dict[str, Any]:
    return _lower_classic(cfg, "mistral")


def lower_gemma2(cfg: dict[str, Any]) -> dict[str, Any]:
    return _lower_classic(cfg, "gemma2")


def lower_gemma3(cfg: dict[str, Any]) -> dict[str, Any]:
    return _lower_classic(cfg, "gemma3")


def lower_phi3(cfg: dict[str, Any]) -> dict[str, Any]:
    return _lower_classic(cfg, "phi3")


# ---- Gemma 4 ---------------------------------------------------------------

def lower_gemma4(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = _text_cfg(cfg)
    model = _base_model(tc, family="gemma4")
    model["act"] = "gelu_tanh"
    model["norm"] = "rmsnorm_gemma"
    model["post_norms"] = True
    model["qk_norm"] = True
    model["sliding_window"] = int(tc.get("sliding_window") or 512)
    model["sw_pattern"] = 0  # per-layer via layer_types / alternating
    model["embed_scale"] = math.sqrt(float(model["hidden"])) if model["hidden"] else 0.0
    model["final_softcap"] = float(tc.get("final_logit_softcapping") or 30.0)
    n = int(model["layers"])
    kv_share = int(tc.get("num_kv_shared_layers") or 0)
    double_wide = bool(tc.get("use_double_wide_mlp"))
    # First (n - kv_share) layers own KV; later layers share from the last owner.
    share_from = max(0, n - kv_share - 1) if kv_share > 0 else -1
    layers: list[dict[str, Any]] = []
    inter = int(model["inter"])
    for i in range(n):
        ops: list[dict[str, Any]] = []
        attn: dict[str, Any] = {"op": "attn_gqa"}
        # Alternate SW / global when sliding_window set (gemma4 pattern).
        if model["sliding_window"] and (i % 2 == 0):
            attn["sliding_window"] = model["sliding_window"]
        ops.append(attn)
        if kv_share > 0 and i > share_from:
            ops.append({"op": "kv_share", "kv_share_from": share_from})
        # E2B: later / even layers may use double-wide MLP.
        use_dw = double_wide and (i >= n // 2 or i == n - 1)
        if use_dw:
            ops.append({"op": "mlp_double_wide", "inter": inter * 2})
        else:
            ops.append({"op": "mlp_gelu", "inter": inter})
        ops.append({"op": "ple_gate"})
        layers.append({"ops": ops})
    req = ["attn_gqa", "mlp_gelu", "mlp_double_wide", "kv_share", "ple_gate", "logit_softcap"]
    return _finish(
        "gemma4", model, layers, req,
        weight_prefix="model.language_model.",
        text_only=True,
        notes="text-only; vision/audio towers stripped at convert",
    )


# ---- Qwen 3.5 / 3.6 hybrid -------------------------------------------------

def lower_qwen35(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = _text_cfg(cfg)
    family = "qwen3_5_moe" if (
        tc.get("num_experts") or tc.get("n_routed_experts") or
        str(cfg.get("model_type", "")).endswith("_moe")
    ) else "qwen3_5"
    model = _base_model(tc, family=family)
    model["qk_norm"] = True
    model["partial_rotary"] = float(tc.get("partial_rotary_factor") or 0.25)
    layer_types = list(tc.get("layer_types") or cfg.get("layer_types") or [])
    n = int(model["layers"])
    if not layer_types:
        # Default 3 linear + 1 full pattern.
        layer_types = [("linear_attention" if (i % 4) != 3 else "full_attention") for i in range(n)]
    while len(layer_types) < n:
        layer_types.append(layer_types[-1] if layer_types else "full_attention")
    n_experts = int(tc.get("num_experts") or tc.get("n_routed_experts") or 0)
    top_k = int(tc.get("num_experts_per_tok") or tc.get("moe_topk") or 8)
    layers: list[dict[str, Any]] = []
    req = {"mlp_swiglu"}
    for i in range(n):
        ops: list[dict[str, Any]] = []
        lt = str(layer_types[i]).lower()
        if "linear" in lt:
            ops.append({"op": "attn_linear_gdn"})
            req.add("attn_linear_gdn")
        else:
            ops.append({"op": "attn_gqa"})
            req.add("attn_gqa")
        if n_experts > 0:
            ops.append({
                "op": "moe_routed",
                "n_experts": n_experts,
                "top_k": top_k,
                "shared_expert": 1,
                "router": "softmax",
            })
            req.add("moe_routed")
        else:
            ops.append({"op": "mlp_swiglu"})
        layers.append({"ops": ops})
    return _finish(
        family, model, layers, list(req),
        weight_prefix="model.language_model.",
        text_only=True,
        notes="hybrid linear+full attention; text-only VL strip",
    )


# ---- Llama 4 ---------------------------------------------------------------

def lower_llama4(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = _text_cfg(cfg)
    model = _base_model(tc, family="llama4")
    model["qk_norm"] = bool(tc.get("use_qk_norm", True))
    n = int(model["layers"])
    n_experts = int(tc.get("num_local_experts") or tc.get("num_experts") or 16)
    top_k = int(tc.get("num_experts_per_tok") or 1)
    chunk = int(tc.get("attention_chunk_size") or 8192)
    layers: list[dict[str, Any]] = []
    for i in range(n):
        ops: list[dict[str, Any]] = [
            {"op": "attn_chunked", "chunk_size": chunk},
        ]
        # Interleaved MoE (every other layer typical for Scout).
        if n_experts > 0 and (i % 2 == 1):
            ops.append({
                "op": "moe_routed",
                "n_experts": n_experts,
                "top_k": top_k,
                "router": "softmax",
            })
        else:
            ops.append({"op": "mlp_swiglu"})
        layers.append({"ops": ops})
    return _finish(
        "llama4", model, layers, ["attn_chunked", "moe_routed", "mlp_swiglu"],
        weight_prefix="language_model.model." if "text_config" in cfg else "",
        text_only=True,
        notes="MoE + chunked attention; gated HF access may be required",
    )


# ---- Kimi K2.x (DeepSeek-V3-like MLA MoE) ----------------------------------

def lower_kimi(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = _text_cfg(cfg)
    model = _base_model(tc, family="kimi_k25")
    n = int(model["layers"] or tc.get("num_hidden_layers") or 0)
    n_experts = int(tc.get("n_routed_experts") or tc.get("num_experts") or 384)
    top_k = int(tc.get("num_experts_per_tok") or 8)
    first_dense = int(tc.get("first_k_dense_replace") or 3)
    layers: list[dict[str, Any]] = []
    for i in range(n):
        ops: list[dict[str, Any]] = [{"op": "attn_mla"}]
        if i < first_dense:
            ops.append({"op": "mlp_swiglu"})
        else:
            ops.append({
                "op": "moe_routed",
                "n_experts": n_experts,
                "top_k": top_k,
                "shared_expert": 1,
                "router": "sigmoid",
            })
        layers.append({"ops": ops})
    return _finish(
        "kimi_k25", model, layers, ["attn_mla", "moe_routed", "mlp_swiglu"],
        weight_prefix="",
        text_only=True,
        notes="MLA+MoE streamed experts; desktop needs large SSD/RAM",
    )


# ---- DeepSeek V4 -----------------------------------------------------------

def lower_deepseek_v4(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = _text_cfg(cfg)
    model = _base_model(tc, family="deepseek_v4")
    n = int(model["layers"])
    n_experts = int(tc.get("n_routed_experts") or tc.get("num_experts") or 256)
    top_k = int(tc.get("num_experts_per_tok") or 6)
    sw = int(tc.get("sliding_window") or 128)
    layers: list[dict[str, Any]] = []
    for i in range(n):
        ops: list[dict[str, Any]] = [
            {"op": "attn_csa_hca", "sliding_window": sw},
            {
                "op": "moe_routed",
                "n_experts": n_experts,
                "top_k": top_k,
                "shared_expert": 1,
                "router": "hash" if i < 3 else "sigmoid",
            },
        ]
        layers.append({"ops": ops})
    return _finish(
        "deepseek_v4", model, layers, ["attn_csa_hca", "moe_routed"],
        text_only=True,
        notes="CSA/HCA approx via sliding GQA + streamed MoE",
    )


# ---- MiniMax M3 ------------------------------------------------------------

def lower_minimax(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = _text_cfg(cfg)
    model = _base_model(tc, family="minimax_m3")
    n = int(model["layers"])
    n_experts = int(tc.get("num_local_experts") or tc.get("num_experts") or 128)
    top_k = int(tc.get("num_experts_per_tok") or 4)
    layers: list[dict[str, Any]] = []
    for i in range(n):
        ops: list[dict[str, Any]] = [
            {"op": "attn_msa", "chunk_size": 128},
            {
                "op": "moe_routed",
                "n_experts": n_experts,
                "top_k": top_k,
                "shared_expert": 1,
                "router": "softmax",
            },
        ]
        layers.append({"ops": ops})
    return _finish(
        "minimax_m3", model, layers, ["attn_msa", "moe_routed"],
        weight_prefix="",
        text_only=True,
        notes="MSA approx via block window GQA; text-only",
    )


# ---- Mistral Large 3 (params.json / MLA MoE) -------------------------------

def lower_mistral_large3(cfg: dict[str, Any]) -> dict[str, Any]:
    """Accept either HF-style config or Mistral params.json fields."""
    tc = _text_cfg(cfg)
    # params.json uses dim / n_layers / n_heads naming.
    if "hidden_size" not in tc and "dim" in tc:
        tc = dict(tc)
        tc["hidden_size"] = tc.get("dim")
        tc["num_hidden_layers"] = tc.get("n_layers")
        tc["num_attention_heads"] = tc.get("n_heads")
        tc["num_key_value_heads"] = tc.get("n_kv_heads") or tc.get("n_heads")
        tc["intermediate_size"] = tc.get("ffn_dim") or tc.get("intermediate_size")
        tc["vocab_size"] = tc.get("vocab_size") or tc.get("n_vocab")
        tc["head_dim"] = tc.get("head_dim") or (
            int(tc["hidden_size"]) // int(tc["num_attention_heads"])
            if tc.get("num_attention_heads") else 0
        )
    model = _base_model(tc, family="mistral_large3")
    n = int(model["layers"])
    n_experts = int(tc.get("n_routed_experts") or tc.get("num_experts") or 128)
    top_k = int(tc.get("num_experts_per_tok") or 4)
    first_dense = int(tc.get("first_k_dense_replace") or 3)
    layers: list[dict[str, Any]] = []
    for i in range(n):
        ops: list[dict[str, Any]] = [{"op": "attn_mla"}]
        if i < first_dense:
            ops.append({"op": "mlp_swiglu"})
        else:
            ops.append({
                "op": "moe_routed",
                "n_experts": n_experts,
                "top_k": top_k,
                "shared_expert": 1,
                "router": "sigmoid",
            })
        layers.append({"ops": ops})
    return _finish(
        "mistral_large3", model, layers, ["attn_mla", "moe_routed", "mlp_swiglu"],
        text_only=True,
        notes="MLA MoE; ingest via params.json when config.json absent",
    )


# ---- GLM MoE (existing engine path marker) ---------------------------------

def lower_glm_moe(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = _text_cfg(cfg)
    model = _base_model(tc, family="glm_moe_dsa")
    n = int(model["layers"] or tc.get("num_hidden_layers") or 0)
    n_experts = int(tc.get("n_routed_experts") or 0)
    top_k = int(tc.get("num_experts_per_tok") or 8)
    first_dense = int(tc.get("first_k_dense_replace") or 1)
    layers: list[dict[str, Any]] = []
    for i in range(max(n, 1)):
        ops: list[dict[str, Any]] = [{"op": "attn_mla"}]
        if i < first_dense:
            ops.append({"op": "mlp_swiglu"})
        else:
            ops.append({
                "op": "moe_routed",
                "n_experts": n_experts,
                "top_k": top_k,
                "shared_expert": 1,
                "router": "sigmoid",
            })
        layers.append({"ops": ops})
    return _finish(
        "glm_moe_dsa", model, layers, ["attn_mla", "moe_routed", "mlp_swiglu"],
        text_only=True,
        notes="executed by engine.c MoE path; WMIR records the contract",
    )


FAMILY_LOWERERS: dict[str, LowerFn] = {
    "qwen2": lower_qwen2,
    "qwen3": lower_qwen3,
    "llama": lower_llama,
    "mistral": lower_mistral,
    "gemma2": lower_gemma2,
    "gemma3": lower_gemma3,
    "phi3": lower_phi3,
    "gemma4": lower_gemma4,
    "gemma4_text": lower_gemma4,
    "gemma4_unified": lower_gemma4,
    "qwen3_5": lower_qwen35,
    "qwen3.5": lower_qwen35,
    "qwen3_5_text": lower_qwen35,
    "qwen3_5_moe": lower_qwen35,
    "llama4": lower_llama4,
    "llama4_text": lower_llama4,
    "kimi_k25": lower_kimi,
    "kimi_k2": lower_kimi,
    "deepseek_v4": lower_deepseek_v4,
    "minimax_m3": lower_minimax,
    "minimax_m3_vl": lower_minimax,
    "mistral_large3": lower_mistral_large3,
    "glm_moe_dsa": lower_glm_moe,
}


def can_lower(cfg: dict[str, Any]) -> bool:
    mt = _mt(cfg)
    top = str(cfg.get("model_type") or "").lower()
    if mt in FAMILY_LOWERERS or top in FAMILY_LOWERERS:
        return True
    # Mistral params.json without model_type
    if cfg.get("dim") and cfg.get("n_layers") and (
        cfg.get("n_routed_experts") or cfg.get("moe")
    ):
        return True
    return False


def lower_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Lower any supported HF / params.json config to a WMIR block."""
    mt = _mt(cfg)
    top = str(cfg.get("model_type") or "").lower()
    for key in (mt, top):
        fn = FAMILY_LOWERERS.get(key)
        if fn:
            return fn(cfg)
    if cfg.get("dim") and cfg.get("n_layers"):
        return lower_mistral_large3(cfg)
    raise RuntimeError(
        f"wmir: no lowerer for model_type={top!r} text={mt!r}. "
        f"Supported: {', '.join(sorted(set(FAMILY_LOWERERS)))}"
    )
