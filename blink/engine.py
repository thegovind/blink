from __future__ import annotations

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "space" / "blink.py"
_spec = importlib.util.spec_from_file_location("_blink_space_engine", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load blink engine from {_path}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in dir(_mod):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_mod, _name)
