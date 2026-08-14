"""Architectural invariant: src/rscene/core/ imports only numpy, scipy and the
standard library.

`core` is the layer everything else in the pipeline (cli, io) depends on. It
has no automated protection today against accidentally picking up a heavy or
non-deterministic dependency (e.g. laspy, trimesh) during a future change --
this test is that protection.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

_CORE_DIR = Path(__file__).parent.parent.parent / "src" / "rscene" / "core"

_ALLOWED_THIRD_PARTY = {"numpy", "scipy"}


def _imported_module_roots(path: Path) -> set[str]:
    """Every top-level module name a file imports, ignoring relative imports."""
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue    # relative import within the package -- always fine
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_core_layer_imports_only_numpy_scipy_and_stdlib():
    allowed = _ALLOWED_THIRD_PARTY | set(sys.stdlib_module_names)

    violations: dict[str, set[str]] = {}
    py_files = sorted(_CORE_DIR.rglob("*.py"))
    assert py_files, f"expected .py files under {_CORE_DIR}"

    for path in py_files:
        roots = _imported_module_roots(path)
        bad = roots - allowed
        if bad:
            violations[str(path.relative_to(_CORE_DIR))] = bad

    assert not violations, (
        "src/rscene/core/ must import only numpy, scipy and the standard "
        f"library; found disallowed imports: {violations}"
    )
