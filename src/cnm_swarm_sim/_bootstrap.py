"""Runtime preparation for binary dependencies installed by conda on Windows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_HANDLES: list[object] = []


def prepare_windows_dll_path() -> None:
    """Expose conda-style DLL folders when running from a standard venv."""

    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidates = [Path(sys.prefix) / "Library" / "bin"]
    for candidate in candidates:
        if candidate.is_dir():
            handle = os.add_dll_directory(str(candidate))
            _DLL_HANDLES.append(handle)

