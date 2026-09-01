"""Manage the training pipeline subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from portautomation.config import PROJECT_ROOT
from portautomation.gpu_env import discover_nvidia_lib_dirs


@dataclass
class AppStatus:
    state: str = "idle"
    model: str = "both"
    pid: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    logs: list[str] = field(default_factory=list)
    message: str = "Ready to start training."


class ProcessManager:
    """Start and stop the main training application."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or PROJECT_ROOT
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._status = AppStatus()
        self._log_buffer: deque[str] = deque(maxlen=500)

    def _snapshot_status(self) -> AppStatus:
        """Build a status snapshot. Caller must hold _lock."""
        return AppStatus(
            state=self._status.state,
            model=self._status.model,
            pid=self._status.pid,
            started_at=self._status.started_at,
            finished_at=self._status.finished_at,
            exit_code=self._status.exit_code,
            logs=list(self._log_buffer),
            message=self._status.message,
        )

    @property
    def status(self) -> AppStatus:
        with self._lock:
            self._sync_process_state()
            return self._snapshot_status()

    def _python_executable(self) -> str:
        venv_python = self.project_root / ".venv" / "bin" / "python"
        return str(venv_python) if venv_python.exists() else sys.executable

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(self.project_root / "src")
        nvidia_libs = discover_nvidia_lib_dirs()
        if nvidia_libs:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(
                nvidia_libs + ([existing] if existing else [])
            )
        return env

    def _sync_process_state(self) -> None:
        if self._process is None:
            return
        return_code = self._process.poll()
        if return_code is None:
            return
        if self._status.state == "running":
            self._status.finished_at = time.time()
            self._status.exit_code = return_code
            self._status.state = "completed" if return_code == 0 else "failed"
            self._status.message = (
                "Training completed successfully."
                if return_code == 0
                else f"Training failed with exit code {return_code}."
            )
        self._process = None

    def start(self, model: str = "both") -> AppStatus:
        with self._lock:
            self._sync_process_state()
            if self._process and self._process.poll() is None:
                self._status.message = "Training is already running."
                return self._snapshot_status()

            if model not in {"cnn", "mobilenet", "both"}:
                self._status.message = f"Invalid model selection: {model}"
                return self._snapshot_status()

            command = [self._python_executable(), "-u", "main.py", "--model", model]
            self._log_buffer.clear()
            self._log_buffer.append(f"$ {' '.join(command)}")

            try:
                self._process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=self._build_env(),
                )
            except OSError as exc:
                self._status = AppStatus(
                    state="failed",
                    model=model,
                    message=f"Failed to start training process: {exc}",
                )
                self._log_buffer.append(str(exc))
                return self._snapshot_status()

            self._status = AppStatus(
                state="running",
                model=model,
                pid=self._process.pid,
                started_at=time.time(),
                message=f"Training started with model={model}.",
            )
            self._reader_thread = threading.Thread(
                target=self._read_output,
                daemon=True,
                name="training-log-reader",
            )
            self._reader_thread.start()
            return self._snapshot_status()

    def stop(self) -> AppStatus:
        with self._lock:
            self._sync_process_state()
            if not self._process or self._process.poll() is not None:
                self._status.state = "idle"
                self._status.message = "No running process to stop."
                return self._snapshot_status()

            self._log_buffer.append("Stopping training process...")
            self._process.terminate()
            try:
                self._process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)

            self._status.state = "stopped"
            self._status.finished_at = time.time()
            self._status.exit_code = self._process.returncode
            self._status.message = "Training stopped by user."
            self._log_buffer.append("Training process stopped.")
            self._process = None
            return self._snapshot_status()

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        for line in process.stdout:
            cleaned = line.rstrip()
            if cleaned:
                with self._lock:
                    self._log_buffer.append(cleaned)

        process.wait()
        with self._lock:
            if self._status.state == "running":
                self._status.finished_at = time.time()
                self._status.exit_code = process.returncode
                self._status.state = "completed" if process.returncode == 0 else "failed"
                self._status.message = (
                    "Training completed successfully."
                    if process.returncode == 0
                    else f"Training failed with exit code {process.returncode}."
                )
            self._process = None

    def get_logs(self) -> list[str]:
        with self._lock:
            return list(self._log_buffer)

    def to_dict(self) -> dict:
        status = self.status
        payload = asdict(status)
        payload["is_running"] = status.state == "running"
        if status.started_at and status.state == "running":
            payload["elapsed_seconds"] = round(time.time() - status.started_at, 1)
        elif status.started_at and status.finished_at:
            payload["elapsed_seconds"] = round(status.finished_at - status.started_at, 1)
        else:
            payload["elapsed_seconds"] = None
        return payload
