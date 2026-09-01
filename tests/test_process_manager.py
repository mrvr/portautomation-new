"""Tests for process manager start/stop behavior."""

import threading
import time

from portautomation.webapp.process_manager import ProcessManager


def test_start_returns_running_status_without_deadlock():
    manager = ProcessManager()
    result_holder: list = []

    def _start() -> None:
        result_holder.append(manager.start(model="both"))

    thread = threading.Thread(target=_start)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive(), "start() deadlocked"
    assert result_holder, "start() did not return"
    assert result_holder[0].state == "running"
    assert result_holder[0].pid is not None
    assert manager.get_logs()

    manager.stop()
