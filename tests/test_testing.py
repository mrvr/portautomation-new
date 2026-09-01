from pathlib import Path

from portautomation.testing.generator import generate_tests
from portautomation.testing.runner import TestRunner as UnitTestRunner
from portautomation.webapp.process_manager import ProcessManager


def test_generate_tests_writes_file(tmp_path: Path):
    output = tmp_path / "tests" / "test_auto_generated.py"
    path = generate_tests(output_path=output)
    content = path.read_text(encoding="utf-8")
    assert "test_import_portautomation_config" in content
    assert "MODULES" in content


def test_process_manager_idle_status():
    manager = ProcessManager()
    status = manager.to_dict()
    assert status["state"] == "idle"
    assert status["is_running"] is False


def test_runner_initial_state():
    runner = UnitTestRunner()
    payload = runner.to_dict()
    assert payload["status"] == "idle"
    assert payload["is_running"] is False
