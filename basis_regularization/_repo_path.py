"""Locate another local checkout for standalone developer scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def repo_path(here: Path, name: str) -> Path:
    here = Path(here).resolve()
    root = next((p for p in (here, *here.parents) if (p / ".pyauto-root").is_file()),
                here.parent)
    resolver = root / "PyAutoBrain" / "agents" / "_repo_paths.py"
    if resolver.is_file():
        spec = importlib.util.spec_from_file_location("_pyauto_repo_paths", resolver)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.repo_path(root, name, required=True)
    flat = root / name
    if (flat / ".git").exists():
        return flat
    if any((family / name).exists() for family in root.iterdir()
           if family.is_dir() and not (family / ".git").exists()):
        raise RuntimeError(f"Grouped checkout {name} needs PyAutoBrain repo resolver")
    raise FileNotFoundError(f"Missing checkout {name}: {flat}")
