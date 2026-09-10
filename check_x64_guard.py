"""
Check that no script imports ``jax`` before the PyAuto config layer.

``autonerves/jax_wrapper.py`` sets ``JAX_ENABLE_X64=True``,
``XLA_FLAGS=--xla_disable_hlo_passes=constant_folding`` and
``JAX_COMPILATION_CACHE_DIR`` **at import time**. JAX reads all three when it is
first imported, so a script that imports ``jax`` before anything that pulls in
the config layer silently runs in float32 with constant folding left on — its
printed precision and timing claims are then measuring a different program from
the one it says it is.

A file is guarded when, before its first ``jax`` import, it imports either
``jax_wrapper`` directly or any top-level PyAuto package (which pulls the
wrapper in transitively).

Run from the repo root::

    python check_x64_guard.py          # exit 1 and one line per offender
    python check_x64_guard.py --list   # also list the guarded files
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PYAUTO_PACKAGES = {
    "autolens",
    "autogalaxy",
    "autoarray",
    "autofit",
    "autoconf",
    "autonerves",
}

GUARD_LINE = "from autolens import jax_wrapper  # noqa: F401 — must be first"


def _root_name(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def _first_lines(path: Path) -> tuple[int | None, int | None]:
    """Return ``(first jax import line, first guard import line)``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None, None

    jax_line: int | None = None
    guard_line: int | None = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [_root_name(a.name) for a in node.names]
            leaves = [a.name.rsplit(".", 1)[-1] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            # A relative import has no module; it cannot be jax or a PyAuto package.
            if node.level:
                continue
            names = [_root_name(node.module or "")]
            leaves = [a.name for a in node.names]
        else:
            continue

        if "jax" in names and jax_line is None:
            jax_line = node.lineno
        if (
            PYAUTO_PACKAGES.intersection(names) or "jax_wrapper" in leaves
        ) and guard_line is None:
            guard_line = node.lineno

    return jax_line, guard_line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="also list the guarded files"
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    offenders: list[tuple[Path, int]] = []
    guarded: list[Path] = []

    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts or path.name == Path(__file__).name:
            continue
        jax_line, guard_line = _first_lines(path)
        if jax_line is None:
            continue
        if guard_line is not None and guard_line < jax_line:
            guarded.append(path.relative_to(root))
        else:
            offenders.append((path.relative_to(root), jax_line))

    if args.list:
        for path in guarded:
            print(f"guarded  {path}")

    for path, jax_line in offenders:
        print(f"UNGUARDED  {path}:{jax_line}  imports jax with no config layer before it")

    total = len(guarded) + len(offenders)
    if offenders:
        print(
            f"\n{len(offenders)} of {total} jax-importing files are unguarded. "
            f"Add as the first import (after any __future__ import):\n\n    {GUARD_LINE}\n"
        )
        return 1

    print(f"x64 guard: OK ({total} jax-importing files, all guarded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
