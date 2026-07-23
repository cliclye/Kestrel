"""Emit WMIR dict fragments for kestrel.json."""

from __future__ import annotations

from typing import Any


def synthesize_dense_layers(desc: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-layer ops for classic dense families (qwen/llama/gemma2/3/phi)."""
    layers = int(desc.get("layers") or 0)
    sw = int(desc.get("sliding_window") or 0)
    pattern = int(desc.get("sw_pattern") or 0)
    act = desc.get("act") or "silu"
    mlp_op = "mlp_gelu" if act in ("gelu_tanh", "gelu") else "mlp_swiglu"
    out: list[dict[str, Any]] = []
    for i in range(layers):
        ops: list[dict[str, Any]] = []
        attn: dict[str, Any] = {"op": "attn_gqa"}
        if sw > 0:
            is_sw = ((i % pattern) != (pattern - 1)) if pattern > 0 else True
            if is_sw:
                attn["sliding_window"] = sw
        ops.append(attn)
        ops.append({"op": mlp_op})
        out.append({"ops": ops})
    return out


def build_wmir_block(
    family: str,
    model: dict[str, Any],
    layers: list[dict[str, Any]],
    required_ops: list[str],
    *,
    text_only: bool = True,
    weight_prefix: str = "",
    notes: str | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "version": 1,
        "text_only": bool(text_only),
        "family": family,
        "model": model,
        "layers": layers,
        "required_ops": sorted(set(required_ops)),
    }
    if weight_prefix:
        block["weight_prefix"] = weight_prefix
    if notes:
        block["notes"] = notes
    return block
