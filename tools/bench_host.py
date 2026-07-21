"""Shared host metadata for Kestrel bench scripts (macOS + Linux)."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def host_info() -> dict:
    out: dict = {
        "platform": sys.platform,
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "logical_cpu": str(os.cpu_count() or 0),
    }
    # macOS sysctl
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
    # Linux /proc
    if "cpu_brand" not in out:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name") or line.lower().startswith("hardware"):
                    out["cpu_brand"] = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    if "mem_bytes" not in out:
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    out["mem_bytes"] = str(kb * 1024)
                    break
        except Exception:
            pass
    if "physical_cpu" not in out:
        try:
            out["physical_cpu"] = str(
                len({
                    l.split(":")[1].strip()
                    for l in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines()
                    if l.startswith("physical id")
                })
                or out.get("logical_cpu", "0")
            )
        except Exception:
            out["physical_cpu"] = out.get("logical_cpu", "0")
    if "mem_bytes" in out:
        try:
            out["mem_gb"] = round(int(out["mem_bytes"]) / (1024**3), 1)
        except Exception:
            pass
    return out
