"""Launch the Port Automation web control panel."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from portautomation.gpu_env import ensure_gpu_environment

ensure_gpu_environment()


def main() -> None:
    port = int(os.environ.get("PORT", "8081"))
    uvicorn.run(
        "portautomation.webapp.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
