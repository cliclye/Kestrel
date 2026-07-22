#!/usr/bin/env python3
"""Render README benchmark SVG from docs/full_bench.json (measured data only).

Public framing: same laptop · without Windhover vs with Windhover.
(JSON still uses keys stock/kestrel for the two binaries under test.)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "docs" / "full_bench.json"
OUT = ROOT / "docs" / "screenshots" / "bench-without-vs-with-kestrel.svg"
# Keep old path as a copy/symlink-friendly duplicate name for existing links
OUT_LEGACY = ROOT / "docs" / "screenshots" / "bench-stock-vs-kestrel.svg"


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render(data: dict) -> str:
    base_pos = data["stock"]["pos_per_s_batch_means"]["mean"]
    kest_pos = data["kestrel"]["pos_per_s_batch_means"]["mean"]
    base_wall = data["stock"]["batch_wall_s"]["mean"]
    kest_wall = data["kestrel"]["batch_wall_s"]["mean"]
    base_rss = (data["stock"].get("max_rss_mb") or {}).get("mean")
    kest_rss = (data["kestrel"].get("max_rss_mb") or {}).get("mean")
    d_pos = data["comparison"]["throughput_pos_per_s"]["delta_pct"]
    d_wall = data["comparison"]["batch_wall_s"]["delta_pct"]
    ts = data["meta"].get("timestamp_utc", "")[:19].replace("T", " ") + " UTC"
    batches = data["meta"].get("batches")
    batch_size = data["meta"].get("batch_size")

    wall_base_idx = 100.0
    wall_kest_idx = (base_wall / kest_wall) * 100.0 if kest_wall else 0.0

    w, h = 920, 520
    pad_l = 72

    def pair(
        x0: float,
        y0: float,
        bw: float,
        max_h: float,
        a: float,
        b: float,
        fmt: str = "{:,.0f}",
        label_a: str = "Without",
        label_b: str = "With Kestrel",
    ):
        m = max(a, b) or 1.0
        ha = max_h * (a / m)
        hb = max_h * (b / m)
        gap = 12
        xa, xb = x0, x0 + bw + gap
        la, lb = fmt.format(a), fmt.format(b)
        return "\n".join(
            [
                f'<rect x="{xa:.1f}" y="{y0 + max_h - ha:.1f}" width="{bw}" height="{ha:.1f}" rx="5" fill="#8a9096"/>',
                f'<rect x="{xb:.1f}" y="{y0 + max_h - hb:.1f}" width="{bw}" height="{hb:.1f}" rx="5" fill="#6fbf94"/>',
                f'<text x="{xa + bw/2:.1f}" y="{y0 + max_h + 18:.1f}" text-anchor="middle" class="tick">{_esc(label_a)}</text>',
                f'<text x="{xb + bw/2:.1f}" y="{y0 + max_h + 18:.1f}" text-anchor="middle" class="tick">{_esc(label_b)}</text>',
                f'<text x="{xa + bw/2:.1f}" y="{y0 + max_h - ha - 8:.1f}" text-anchor="middle" class="val">{_esc(la)}</text>',
                f'<text x="{xb + bw/2:.1f}" y="{y0 + max_h - hb - 8:.1f}" text-anchor="middle" class="val">{_esc(lb)}</text>',
            ]
        )

    rss_panel = ""
    if base_rss is not None and kest_rss is not None:
        rss_panel = f"""
  <rect x="640" y="64" width="256" height="280" rx="10" fill="#ffffff" stroke="#d5ddd7"/>
  <text x="660" y="88" class="panel">Peak RSS (MB)</text>
  <text x="870" y="88" text-anchor="end" class="foot">lower is better</text>
  {pair(690, 120, 48, 160, base_rss, kest_rss, "{:.1f}", "Without", "With")}
"""
        pos_panel_w = 300
        wall_x = 340
    else:
        pos_panel_w = 420
        wall_x = 460

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img"
     aria-label="Same laptop: without Kestrel engine vs with Kestrel engine on glm_tiny">
  <defs>
    <style>
      .title {{ font: 700 20px ui-sans-serif, system-ui, sans-serif; fill: #1c2420; }}
      .sub {{ font: 400 12px ui-sans-serif, system-ui, sans-serif; fill: #5c6b63; }}
      .panel {{ font: 600 13px ui-sans-serif, system-ui, sans-serif; fill: #2a3530; }}
      .tick {{ font: 500 11px ui-sans-serif, system-ui, sans-serif; fill: #5c6b63; }}
      .val {{ font: 600 12px ui-monospace, Menlo, monospace; fill: #1c2420; }}
      .legend {{ font: 500 12px ui-sans-serif, system-ui, sans-serif; fill: #2a3530; }}
      .foot {{ font: 400 11px ui-sans-serif, system-ui, sans-serif; fill: #7a8780; }}
      .delta {{ font: 700 13px ui-monospace, Menlo, monospace; fill: #3d8f63; }}
    </style>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f4f6f4"/>
      <stop offset="100%" stop-color="#e8ece9"/>
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#bg)" rx="12"/>
  <text x="{pad_l}" y="28" class="title">Same laptop · without Kestrel vs with Kestrel</text>
  <text x="{pad_l}" y="48" class="sub">SYNTHETIC glm_tiny oracle only (not GLM-5.2 / Kimi) · {batches}×{batch_size} · 32/32 · { _esc(ts) }</text>

  <rect x="24" y="64" width="{pos_panel_w}" height="280" rx="10" fill="#ffffff" stroke="#d5ddd7"/>
  <text x="44" y="88" class="panel">Prefill throughput (pos/s)</text>
  <text x="{24 + pos_panel_w - 16}" y="88" text-anchor="end" class="delta">+{d_pos:.0f}%</text>
  {pair(70 if pos_panel_w < 400 else 100, 120, 70 if pos_panel_w > 350 else 52, 160, base_pos, kest_pos)}

  <rect x="{wall_x}" y="64" width="280" height="280" rx="10" fill="#ffffff" stroke="#d5ddd7"/>
  <text x="{wall_x + 20}" y="88" class="panel">Batch wall speed index</text>
  <text x="{wall_x + 264}" y="88" text-anchor="end" class="delta">{abs(d_wall):.0f}% faster</text>
  {pair(wall_x + 55, 120, 55, 160, wall_base_idx, wall_kest_idx, label_a="Without", label_b="With")}
  <text x="{wall_x + 140}" y="320" text-anchor="middle" class="foot">baseline wall / Kestrel wall × 100</text>
{rss_panel}

  <rect x="24" y="360" width="872" height="100" rx="10" fill="#ffffff" stroke="#d5ddd7"/>
  <text x="44" y="386" class="panel">What this measures</text>
  <text x="44" y="408" class="foot">• Same Mac, same fixture, same prompts — baseline local MoE engine path vs windhover-engine</text>
  <text x="44" y="426" class="foot">• Interleaved A/B batches · warmup discarded · Welch on batch means · both sides 32/32 oracle</text>
  <text x="44" y="444" class="foot">• glm_tiny is synthetic (~2MB). Frontier MoE (GLM-5.2 / Kimi): tools/real_model_bench.py after download.</text>

  <g transform="translate(44, 488)">
    <rect width="12" height="12" rx="2" fill="#8a9096"/>
    <text x="18" y="11" class="legend">Without Kestrel (baseline engine)</text>
    <rect x="250" width="12" height="12" rx="2" fill="#6fbf94"/>
    <text x="268" y="11" class="legend">With Windhover (windhover-engine)</text>
    <text x="520" y="11" class="foot">Source: docs/full_bench.json · verify: python3 tools/verify_bench.py</text>
  </g>
</svg>
'''
    return svg


def main() -> int:
    data = json.loads(BENCH.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    svg = render(data)
    OUT.write_text(svg)
    OUT_LEGACY.write_text(svg)
    print(f"wrote {OUT}")
    print(f"wrote {OUT_LEGACY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
