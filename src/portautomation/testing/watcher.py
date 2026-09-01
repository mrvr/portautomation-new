"""Watch source and test files and rerun pytest on changes."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from portautomation.config import PROJECT_ROOT
from portautomation.testing.generator import generate_tests
from portautomation.testing.runner import TestRunner

logger = logging.getLogger(__name__)

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:  # pragma: no cover
    WATCHDOG_AVAILABLE = False


class _DebouncedHandler(FileSystemEventHandler):
    def __init__(self, callback, debounce_seconds: float = 1.5) -> None:
        super().__init__()
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix not in {".py", ".ini"}:
            return
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            return

        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self.callback)
            self._timer.daemon = True
            self._timer.start()


class TestWatcher:
    """Background watcher that regenerates and reruns tests on file changes."""

    def __init__(self, runner: TestRunner, project_root: Path | None = None) -> None:
        self.runner = runner
        self.project_root = project_root or PROJECT_ROOT
        self._observer: Observer | None = None
        self._enabled = False
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def available(self) -> bool:
        return WATCHDOG_AVAILABLE

    def start(self) -> None:
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog is not installed; auto-test watcher disabled")
            return
        if self._enabled:
            return

        handler = _DebouncedHandler(self._on_change)
        self._observer = Observer()
        for folder in ("src", "tests"):
            watch_path = self.project_root / folder
            if watch_path.exists():
                self._observer.schedule(handler, str(watch_path), recursive=True)
        self._observer.start()
        self._enabled = True
        logger.info("Test watcher started")

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
        self._enabled = False
        logger.info("Test watcher stopped")

    def _on_change(self) -> None:
        if self.runner.is_running:
            return
        logger.info("Changes detected; regenerating and running tests")
        try:
            generate_tests()
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to regenerate tests: %s", exc)
        self.runner.run(trigger="auto")
