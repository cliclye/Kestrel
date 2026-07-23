"""Windhover Model IR (WMIR) — HF → layer-typed graph for the C engine."""

from .ops import KERNEL_OPS, required_ops_ok, missing_ops
from .lower import lower_config, can_lower, FAMILY_LOWERERS
from .emit import build_wmir_block, synthesize_dense_layers

__all__ = [
    "KERNEL_OPS",
    "required_ops_ok",
    "missing_ops",
    "lower_config",
    "can_lower",
    "FAMILY_LOWERERS",
    "build_wmir_block",
    "synthesize_dense_layers",
]
