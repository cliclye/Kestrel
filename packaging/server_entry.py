#!/usr/bin/env python3
"""Frozen entrypoint for the Windhover desktop sidecar (PyInstaller).

Runs `windhover app` with ROOT = bundle (MEIPASS) and engine beside the exe
(Tauri externalBin layout).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    frozen = getattr(sys, "frozen", False)
    if frozen:
        bundle = Path(getattr(sys, "_MEIPASS"))
        exe_dir = Path(sys.executable).resolve().parent
    else:
        bundle = Path(__file__).resolve().parents[1]
        exe_dir = bundle / "engine"

    # Windows: UTF-8 stdio before importing windhover (download progress uses Unicode).
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if stream is None:
                continue
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass
        # Wrap so a failed reconfigure still cannot raise UnicodeEncodeError.
        try:
            # Prefer windhover's safe wrapper once imported; until then, soft-replace.
            for name in ("stdout", "stderr"):
                stream = getattr(sys, name, None)
                if stream is None:
                    continue

                class _Soft:
                    def __init__(self, inner):
                        self._inner = inner

                    def write(self, s):
                        try:
                            return self._inner.write(s)
                        except UnicodeEncodeError:
                            enc = getattr(self._inner, "encoding", None) or "ascii"
                            safe = s.encode(enc, errors="replace").decode(enc, errors="replace")
                            return self._inner.write(safe)

                    def flush(self):
                        return self._inner.flush()

                    def __getattr__(self, n):
                        return getattr(self._inner, n)

                setattr(sys, name, _Soft(stream))
        except Exception:
            pass

    # CI / packaging: verify Library-download + engine-chat deps inside the freeze.
    if "--sidecar-selfcheck" in sys.argv:
        try:
            import json
            import tempfile

            sys.path.insert(0, str(bundle))
            sys.path.insert(0, str(bundle / "tools"))

            import huggingface_hub
            from huggingface_hub import snapshot_download  # noqa: F401
            import numpy  # noqa: F401
            import safetensors  # noqa: F401
            import kestrel_pack  # noqa: F401

            # Chat must prefer windhover-engine without torch/transformers.
            try:
                import torch  # noqa: F401

                torch_ok = True
            except ImportError:
                torch_ok = False

            try:
                import bundled_windhover as wh  # type: ignore
            except ImportError:
                wh = None

            # Prefer the live `windhover` source — packaging/bundled_windhover.py can be
            # a stale copy left from an earlier PyInstaller run during local checks.
            wh_src = bundle / "windhover"
            if wh_src.is_file():
                import importlib.machinery
                import importlib.util

                loader = importlib.machinery.SourceFileLoader("wh_selfcheck", str(wh_src))
                spec = importlib.util.spec_from_loader(loader.name, loader)
                if spec is not None and spec.loader is not None:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    wh = mod

            if wh is not None:
                tmp = Path(tempfile.mkdtemp(prefix="wh-selfcheck-"))
                (tmp / "config.json").write_text(
                    json.dumps(
                        {
                            "model_type": "qwen2",
                            "architectures": ["Qwen2ForCausalLM"],
                            "hidden_size": 64,
                            "num_hidden_layers": 1,
                            "num_attention_heads": 4,
                            "num_key_value_heads": 4,
                            "intermediate_size": 128,
                            "vocab_size": 128,
                        }
                    ),
                    encoding="utf-8",
                )
                if not wh._dense_loadtime_ok(tmp):
                    raise RuntimeError("_dense_loadtime_ok failed for qwen2 fixture")
                # Without 30MB weights, mode is blocked — that's fine. Routing helper must exist.
                if not callable(wh._chat_mode) or not callable(wh._ensure_engine_pack):
                    raise RuntimeError("chat routing helpers missing")

            print(
                "sidecar-selfcheck ok "
                f"huggingface_hub={huggingface_hub.__version__} "
                f"numpy={numpy.__version__} "
                f"torch_bundled={torch_ok}"
            )
            if torch_ok:
                print(
                    "sidecar-selfcheck WARN: torch unexpectedly present in freeze "
                    "(sidecar should stay torch-free)",
                    file=sys.stderr,
                )
            return 0
        except Exception as e:
            print(f"sidecar-selfcheck FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    os.environ.setdefault("WINDHOVER_ROOT", str(bundle))

    for name in ("windhover-engine.exe", "windhover-engine"):
        cand = exe_dir / name
        if cand.is_file():
            os.environ["WINDHOVER_ENGINE"] = str(cand)
            break
        # Tauri sometimes nests resources one level up from the sidecar
        alt = exe_dir.parent / name
        if alt.is_file():
            os.environ["WINDHOVER_ENGINE"] = str(alt)
            break

    # Ensure tools/ is importable from the bundle
    sys.path.insert(0, str(bundle))
    sys.path.insert(0, str(bundle / "tools"))

    # Avoid interactive first-run pulls in the frozen GUI path unless needed.
    os.environ.setdefault("WINDHOVER_APP_NO_AUTOPULL", "1")

    sys.argv = [
        "windhover",
        "app",
        "--host",
        os.environ.get("WINDHOVER_HOST", "127.0.0.1"),
        "--port",
        os.environ.get("WINDHOVER_PORT", "8000"),
    ]

    # Prefer the .py copy that PyInstaller traces (see windhover-server.spec).
    try:
        import bundled_windhover as mod  # type: ignore
    except ImportError:
        wh = bundle / "windhover"
        if not wh.is_file():
            print(f"windhover-server: missing {wh}", file=sys.stderr)
            return 1
        import importlib.machinery
        import importlib.util

        loader = importlib.machinery.SourceFileLoader("windhover_cli", str(wh))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        if spec is None or spec.loader is None:
            print("windhover-server: failed to load windhover", file=sys.stderr)
            return 1
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    if hasattr(mod, "ROOT"):
        mod.ROOT = bundle
        mod.ENGINE_DIR = bundle / "engine"
        if hasattr(mod, "_engine_bin"):
            mod.ENGINE_BIN = mod._engine_bin()
        mod.CATALOG_PATH = bundle / "app" / "public" / "catalog.json"
        if not mod.CATALOG_PATH.is_file():
            mod.CATALOG_PATH = bundle / "app" / "dist" / "catalog.json"
        # Re-bind agent_workspace against the bundle tools/ dir
        sys.path.insert(0, str(bundle / "tools"))
        try:
            import agent_workspace as _agent_ws  # type: ignore

            mod._agent_ws = _agent_ws
        except ImportError:
            pass
    return int(mod.main())


if __name__ == "__main__":
    raise SystemExit(main())
