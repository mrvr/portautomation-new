"""FastAPI web application for training control and test dashboard."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from portautomation.config import PROJECT_ROOT
from portautomation.device import configure_devices, device_dict
from portautomation.testing.generator import generate_tests
from portautomation.testing.runner import TestRunner
from portautomation.testing.watcher import TestWatcher
from portautomation.webapp.process_manager import ProcessManager

STATIC_DIR = Path(__file__).resolve().parent / "static"

process_manager = ProcessManager()
test_runner = TestRunner()
test_watcher = TestWatcher(test_runner)


class StartRequest(BaseModel):
    model: Literal["cnn", "mobilenet", "both"] = "both"


class WatcherRequest(BaseModel):
    enabled: bool


class GenerateTestsRequest(BaseModel):
    run_after: bool = True


def _action_response(success: bool, message: str, data: dict) -> dict:
    return {"success": success, "message": message, "data": data}


@asynccontextmanager
async def lifespan(_: FastAPI):
    device = configure_devices()
    logger = logging.getLogger("portautomation.webapp")
    logger.info("Webapp compute device: %s", device.message)
    generate_tests()
    test_watcher.start()
    test_runner.start_async(trigger="startup")
    yield
    test_watcher.stop()
    process_manager.stop()


app = FastAPI(title="Port Automation Control Panel", version="1.1.0", lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/app/status")
def app_status() -> dict:
    payload = process_manager.to_dict()
    payload["device"] = device_dict()
    return _action_response(True, "Application status fetched.", payload)


@app.get("/api/device")
def device_status() -> dict:
    return _action_response(True, "Device status fetched.", device_dict())


@app.get("/api/app/logs")
def app_logs() -> dict:
    logs = process_manager.get_logs()
    return _action_response(True, "Application logs fetched.", {"logs": logs, "count": len(logs)})


@app.post("/api/app/start")
def app_start(request: StartRequest) -> dict:
    status = process_manager.start(model=request.model)
    success = status.state == "running"
    message = status.message if success else status.message
    return _action_response(success, message, process_manager.to_dict())


@app.post("/api/app/stop")
def app_stop() -> dict:
    status = process_manager.stop()
    success = status.state in {"stopped", "idle", "completed", "failed"}
    return _action_response(success, status.message, process_manager.to_dict())


@app.get("/api/tests/status")
def tests_status() -> dict:
    return _action_response(True, "Test status fetched.", test_runner.to_dict())


@app.get("/api/tests/output")
def tests_output() -> dict:
    payload = test_runner.to_dict()
    payload["output"] = test_runner.output_text()
    return _action_response(True, "Test output fetched.", payload)


@app.post("/api/tests/run")
def tests_run() -> dict:
    if test_runner.is_running:
        raise HTTPException(status_code=409, detail="Tests are already running.")
    test_runner.start_async(trigger="manual")
    return _action_response(True, "Test run started.", test_runner.to_dict())


@app.post("/api/tests/generate")
def tests_generate(request: GenerateTestsRequest | None = None) -> dict:
    request = request or GenerateTestsRequest()
    path = generate_tests()
    if request.run_after:
        if test_runner.is_running:
            return _action_response(
                True,
                f"Generated tests at {path.name}; test run already in progress.",
                {
                    "generated_file": str(path.relative_to(PROJECT_ROOT)),
                    "tests": test_runner.to_dict(),
                },
            )
        test_runner.start_async(trigger="generate")
        message = f"Generated tests at {path.name} and started test run."
    else:
        message = f"Generated tests at {path.name}."

    return _action_response(
        True,
        message,
        {
            "generated_file": str(path.relative_to(PROJECT_ROOT)),
            "tests": test_runner.to_dict(),
        },
    )


@app.post("/api/tests/watch")
def tests_watch(request: WatcherRequest) -> dict:
    if request.enabled:
        test_watcher.start()
    else:
        test_watcher.stop()
    return _action_response(
        True,
        "Auto-test watcher updated.",
        {"enabled": test_watcher.enabled, "available": test_watcher.available},
    )


@app.get("/api/tests/watch")
def tests_watch_status() -> dict:
    return _action_response(
        True,
        "Auto-test watcher status fetched.",
        {"enabled": test_watcher.enabled, "available": test_watcher.available},
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
