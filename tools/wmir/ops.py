"""Kernel registry — ops the Windhover C runtime can execute."""

from __future__ import annotations

# Must stay in sync with wmir_kernel_supported() in engine/runtime/wmir.h
KERNEL_OPS: frozenset[str] = frozenset({
    "rms_norm",
    "rms_norm_gemma",
    "attn_gqa",
    "attn_mla",
    "attn_linear_gdn",
    "attn_chunked",
    "attn_csa_hca",
    "attn_msa",
    "mlp_swiglu",
    "mlp_gelu",
    "mlp_double_wide",
    "moe_routed",
    "kv_share",
    "embed",
    "lm_head",
    "logit_softcap",
    "ple_gate",
})


def missing_ops(required: list[str] | set[str]) -> list[str]:
    return sorted({op for op in required if op not in KERNEL_OPS})


def required_ops_ok(required: list[str] | set[str]) -> bool:
    return not missing_ops(required)
