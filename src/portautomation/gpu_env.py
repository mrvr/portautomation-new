"""Configure NVIDIA library paths before TensorFlow import."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIGURED = False


def discover_nvidia_lib_dirs() -> list[str]:
    """Find lib directories shipped with NVIDIA pip packages in the active venv."""
    lib_dirs: list[str] = []
    seen: set[str] = set()

    candidates: list[Path] = []
    for entry in sys.path:
        base = Path(entry)
        if base.name == "site-packages" or (base / "nvidia").exists():
            candidates.append(base)

    for base in candidates:
        nvidia_root = base / "nvidia"
        if not nvidia_root.exists():
            continue
        for lib_dir in nvidia_root.glob("*/lib"):
            resolved = str(lib_dir.resolve())
            if lib_dir.is_dir() and resolved not in seen:
                seen.add(resolved)
                lib_dirs.append(resolved)

    return lib_dirs


def setup_nvidia_library_path() -> list[str]:
    """Add NVIDIA pip package libraries to LD_LIBRARY_PATH."""
    global _CONFIGURED
    lib_dirs = discover_nvidia_lib_dirs()
    if not lib_dirs:
        return lib_dirs

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    merged = lib_dirs + ([existing] if existing else [])
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(merged))
    _CONFIGURED = True
    logger.debug("Configured LD_LIBRARY_PATH with %s NVIDIA lib dirs", len(lib_dirs))
    return lib_dirs


def preload_nvidia_libraries() -> None:
    """Preload NVIDIA shared libraries for the current Python process."""
    lib_dirs = setup_nvidia_library_path()
    if not lib_dirs:
        return

    preferred = (
        "libcudart.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcudnn.so.8",
        "libcufft.so.11",
        "libcurand.so.10",
        "libcusolver.so.11",
        "libcusparse.so.12",
    )

    for lib_dir in lib_dirs:
        for lib_name in preferred:
            for candidate in Path(lib_dir).glob(lib_name):
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue

    logger.debug("Preloaded NVIDIA runtime libraries from %s directories", len(lib_dirs))


def ensure_gpu_environment() -> list[str]:
    """Prepare GPU libraries for TensorFlow in the current process."""
    os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "0")
    lib_dirs = setup_nvidia_library_path()
    preload_nvidia_libraries()
    return lib_dirs
