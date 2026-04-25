"""Tests for the FastAPI server using FastAPI's TestClient (no real server)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from market_env import server as server_module
from market_env.environment import MarketEnvironment


@pytest.fixture
def client():
    # Reset the env between tests so episode_ids don't leak across cases.
    server_module.env = MarketEnvironment()
    return TestClient(server_module.app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /tasks
# ---------------------------------------------------------------------------

class TestTasks:
    def test_tasks_returns_list(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Each task should have the expected keys
        sample = data[0]
        for key in ("task_id", "difficulty", "seed", "bot_config"):
            assert key in sample


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_with_no_body_uses_defaults(self, client):
        r = client.post("/reset")
        assert r.status_code == 200
        obs = r.json()
        assert "episode_id" in obs
        assert obs["turn"] == 0
        assert obs["true_value"] is None

    def test_reset_with_body_overrides_defaults(self, client):
        r = client.post("/reset", json={"seed": 7, "difficulty": "easy"})
        assert r.status_code == 200
        # Reset twice with same seed → identical signals
        r2 = client.post("/reset", json={"seed": 7, "difficulty": "easy"})
        assert r.json()["private_signals"] == r2.json()["private_signals"]

    def test_reset_with_unknown_task_returns_400(self, client):
        r = client.post("/reset", json={"task_id": "nope"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_with_hold_advances_turn(self, client):
        ep = client.post("/reset", json={"episode_length": 5}).json()
        r = client.post("/step", json={
            "episode_id": ep["episode_id"],
            "action": {"action_type": "hold"},
        })
        assert r.status_code == 200
        body = r.json()
        assert body["observation"]["turn"] == 1
        assert body["reward"] == 0.0
        assert body["done"] is False

    def test_step_unknown_episode_returns_404(self, client):
        r = client.post("/step", json={
            "episode_id": "doesnotexist",
            "action": {"action_type": "hold"},
        })
        assert r.status_code == 404

    def test_step_after_done_returns_400(self, client):
        ep = client.post("/reset", json={"episode_length": 1}).json()
        client.post("/step", json={
            "episode_id": ep["episode_id"],
            "action": {"action_type": "hold"},
        })
        # Now done; another step should 400
        r = client.post("/step", json={
            "episode_id": ep["episode_id"],
            "action": {"action_type": "hold"},
        })
        assert r.status_code == 400

    def test_step_buy_action_accepted(self, client):
        ep = client.post("/reset", json={"episode_length": 5}).json()
        r = client.post("/step", json={
            "episode_id": ep["episode_id"],
            "action": {"action_type": "buy", "price": 50.0, "quantity": 5},
        })
        assert r.status_code == 200
        assert r.json()["info"]["action_status"] == "accepted"

    def test_step_invalid_action_payload_returns_422(self, client):
        """Pydantic should reject an action with a non-finite price up front."""
        ep = client.post("/reset", json={"episode_length": 5}).json()
        r = client.post("/step", json={
            "episode_id": ep["episode_id"],
            "action": {"action_type": "buy", "price": -5.0, "quantity": 5},
        })
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /state
# ---------------------------------------------------------------------------

class TestState:
    def test_state_returns_episode_status(self, client):
        ep = client.post("/reset").json()
        r = client.get("/state", params={"episode_id": ep["episode_id"]})
        assert r.status_code == 200
        st = r.json()
        for key in ("episode_id", "turn", "max_turns", "done", "positions"):
            assert key in st

    def test_state_unknown_episode_returns_404(self, client):
        r = client.get("/state", params={"episode_id": "doesnotexist"})
        assert r.status_code == 404
