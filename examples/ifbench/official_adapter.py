"""Loader for GEPA's vendored IFBench registry.

The AllenAI IFBench package only ships verifiers for the ~58 *new-style*
IFBench constraints. The IFBench training data mixes those with IFEval-era
instructions (e.g. ``length_constraints:nth_paragraph_first_word``). GEPA's
artifact repo ships a merged registry under
``gepa_artifact/benchmarks/IFBench/utils_ifbench/`` that unions both and
exposes an ``INSTRUCTION_DICT`` keyed by every id appearing in the pool.

This module fetches those modules on first use, caches them under
``~/.cache/levi/ifbench/gepa_vendored/``, makes the cache importable, and
returns the merged registry module.
"""

from __future__ import annotations

import importlib
import os
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path
from types import ModuleType

GEPA_RAW_BASE = os.getenv(
    "IFBENCH_GEPA_RAW_BASE",
    "https://raw.githubusercontent.com/gepa-ai/gepa-artifact/main/gepa_artifact/benchmarks/IFBench",
)
VENDOR_CACHE_DIR = Path(
    os.getenv(
        "IFBENCH_GEPA_VENDOR_DIR",
        str(Path.home() / ".cache" / "levi" / "ifbench" / "gepa_vendored"),
    )
)
PACKAGE_NAME = "gepa_vendored_ifbench"

# Files to fetch, mapped to their path inside the local package.
_FILES = {
    "utils_ifbench/instructions.py": "utils_ifbench/instructions.py",
    "utils_ifbench/instructions_ifeval.py": "utils_ifbench/instructions_ifeval.py",
    "utils_ifbench/instructions_registry.py": "utils_ifbench/instructions_registry.py",
    "utils_ifbench/instructions_registry_ifeval.py": "utils_ifbench/instructions_registry_ifeval.py",
    "utils_ifbench/instructions_util.py": "utils_ifbench/instructions_util.py",
    "utils_ifbench/instructions_util_ifeval.py": "utils_ifbench/instructions_util_ifeval.py",
}


def _download_file(remote_rel: str, local_path: Path) -> None:
    url = f"{GEPA_RAW_BASE}/{remote_rel}"
    with urllib.request.urlopen(url) as response:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(response.read())


def _ensure_vendored_package() -> Path:
    """Create ``<cache>/<PACKAGE_NAME>/`` with every vendored file."""
    package_root = VENDOR_CACHE_DIR / PACKAGE_NAME
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").touch(exist_ok=True)
    utils_pkg = package_root / "utils_ifbench"
    utils_pkg.mkdir(parents=True, exist_ok=True)
    (utils_pkg / "__init__.py").touch(exist_ok=True)

    for remote_rel, local_rel in _FILES.items():
        local_path = package_root / local_rel
        if local_path.exists() and local_path.stat().st_size > 0:
            continue
        _download_file(remote_rel, local_path)

    return package_root


@lru_cache(maxsize=1)
def load_gepa_registry() -> ModuleType:
    """Return GEPA's merged ``instructions_registry`` module."""
    package_root = _ensure_vendored_package()
    parent = str(package_root.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    importlib.invalidate_caches()
    return importlib.import_module(f"{PACKAGE_NAME}.utils_ifbench.instructions_registry")
