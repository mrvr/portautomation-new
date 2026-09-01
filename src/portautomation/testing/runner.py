"""Run pytest and collect structured results."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from portautomation.config import PROJECT_ROOT


@dataclass
class TestCaseResult:
    name: str
    classname: str
    status: str
    time: float
    message: str = ""


@dataclass
class TestRunResult:
    status: str = "idle"
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total: int = 0
    duration_seconds: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    trigger: str = "manual"
    tests: list[TestCaseResult] = field(default_factory=list)
    output: str = ""
    message: str = "No test run yet."


class TestRunner:
    """Execute pytest and expose the latest results."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or PROJECT_ROOT
        self.junit_path = self.project_root / "outputs" / "test_results.xml"
        self._lock = threading.Lock()
        self._running = False
        self._latest = TestRunResult()

    @property
    def latest(self) -> TestRunResult:
        with self._lock:
            return self._latest

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def _python_executable(self) -> str:
        venv_python = self.project_root / ".venv" / "bin" / "python"
        return str(venv_python) if venv_python.exists() else sys.executable

    def start_async(self, trigger: str = "manual") -> TestRunResult:
        """Mark tests as running and execute pytest in a background thread."""
        with self._lock:
            if self._running:
                self._latest.message = "Tests are already running."
                return self._latest
            self._running = True
            self._latest = TestRunResult(
                status="running",
                started_at=time.time(),
                trigger=trigger,
                message=f"Running tests ({trigger})...",
            )
            snapshot = self._latest

        def _worker() -> None:
            try:
                result = self._execute_pytest(trigger)
            except Exception as exc:  # pragma: no cover
                result = TestRunResult(
                    status="failed",
                    started_at=snapshot.started_at,
                    finished_at=time.time(),
                    trigger=trigger,
                    output=str(exc),
                    message=f"Test run failed: {exc}",
                )
            with self._lock:
                self._running = False
                self._latest = result

        threading.Thread(target=_worker, daemon=True, name=f"pytest-{trigger}").start()
        return snapshot

    def run(self, trigger: str = "manual") -> TestRunResult:
        """Run pytest synchronously."""
        self.start_async(trigger=trigger)
        while self.is_running:
            time.sleep(0.1)
        return self.latest

    def _execute_pytest(self, trigger: str) -> TestRunResult:
        self.junit_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        env = {
            **os.environ.copy(),
            "PYTHONPATH": str(self.project_root / "src"),
        }
        command = [
            self._python_executable(),
            "-m",
            "pytest",
            "tests/",
            f"--junitxml={self.junit_path}",
            "-q",
        ]

        completed = subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        finished = time.time()
        output = (completed.stdout or "") + (completed.stderr or "")
        parsed = self._parse_junit(self.junit_path)
        parsed.duration_seconds = round(finished - started, 2)
        parsed.started_at = started
        parsed.finished_at = finished
        parsed.trigger = trigger
        parsed.output = output[-12000:]
        parsed.status = "passed" if completed.returncode == 0 else "failed"
        parsed.message = (
            f"All {parsed.total} tests passed."
            if completed.returncode == 0
            else f"{parsed.failed + parsed.errors} of {parsed.total} tests failed."
        )
        return parsed

    def _parse_junit(self, path: Path) -> TestRunResult:
        if not path.exists():
            return TestRunResult(
                status="failed",
                output="JUnit report not produced.",
                message="JUnit report not produced.",
            )

        root = ET.parse(path).getroot()
        testsuite = root if root.tag == "testsuite" else root.find("testsuite")
        if testsuite is None:
            return TestRunResult(
                status="failed",
                output="Invalid JUnit XML.",
                message="Invalid JUnit XML.",
            )

        result = TestRunResult(
            passed=int(testsuite.attrib.get("tests", 0))
            - int(testsuite.attrib.get("failures", 0))
            - int(testsuite.attrib.get("errors", 0))
            - int(testsuite.attrib.get("skipped", 0)),
            failed=int(testsuite.attrib.get("failures", 0)),
            errors=int(testsuite.attrib.get("errors", 0)),
            skipped=int(testsuite.attrib.get("skipped", 0)),
            total=int(testsuite.attrib.get("tests", 0)),
            duration_seconds=float(testsuite.attrib.get("time", 0.0)),
        )

        for case in testsuite.findall("testcase"):
            status = "passed"
            message = ""
            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")
            if failure is not None:
                status = "failed"
                message = failure.attrib.get("message", failure.text or "")
            elif error is not None:
                status = "error"
                message = error.attrib.get("message", error.text or "")
            elif skipped is not None:
                status = "skipped"
                message = skipped.attrib.get("message", skipped.text or "")

            result.tests.append(
                TestCaseResult(
                    name=case.attrib.get("name", "unknown"),
                    classname=case.attrib.get("classname", ""),
                    status=status,
                    time=float(case.attrib.get("time", 0.0)),
                    message=message.strip(),
                )
            )
        return result

    def to_dict(self) -> dict:
        with self._lock:
            payload = asdict(self._latest)
            payload["is_running"] = self._running
            return payload

    def output_text(self) -> str:
        return self.latest.output
