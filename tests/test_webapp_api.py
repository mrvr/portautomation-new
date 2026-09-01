from fastapi.testclient import TestClient

from portautomation.webapp.app import app

client = TestClient(app)


def test_app_status_endpoint():
    response = client.get("/api/app/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["state"] == "idle"


def test_tests_status_endpoint():
    response = client.get("/api/tests/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert "total" in payload["data"]


def test_generate_tests_endpoint():
    response = client.post("/api/tests/generate", json={"run_after": False})
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert "generated_file" in payload["data"]


def test_run_tests_endpoint():
    response = client.post("/api/tests/run")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True


def test_app_logs_endpoint():
    response = client.get("/api/app/logs")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert "logs" in payload["data"]
