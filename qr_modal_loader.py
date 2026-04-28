from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path


def load_module_from_file(module_name: str, file_path: str | Path):
    file_path = Path(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__builtins__"] = builtins
    spec.loader.exec_module(module)
    return module
