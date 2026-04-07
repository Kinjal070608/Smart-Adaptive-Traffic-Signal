from fastapi.testclient import TestClient

from server import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_reset_post_and_step_flow():
    response = client.post("/reset", json={"task_name": "easy", "seed": 0})
    assert response.status_code == 200
    data = response.json()
    assert data["observation"]["step"] == 0
    assert data["done"] is False
    assert data["info"]["task"] == "easy"

    step_response = client.post("/step", json={"task_name": "easy", "action": {"phase": "EW"}})
    assert step_response.status_code == 200
    step_data = step_response.json()
    assert step_data["observation"]["step"] == 1
    assert step_data["observation"]["phase"] == "EW"
    assert "reward" in step_data and isinstance(step_data["reward"]["value"], float)
    assert "info" in step_data and "score" in step_data["info"]


def test_state_endpoint_returns_current_state():
    client.post("/reset", json={"task_name": "medium", "seed": 1})
    response = client.get("/state", params={"task_name": "medium"})
    assert response.status_code == 200
    state = response.json()["state"]
    assert state["task"] == "medium"
    assert state["phase"] in {"NS", "EW"}
