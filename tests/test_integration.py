"""End-to-end integration tests covering env + reward + server + client."""
from __future__ import annotations

import statistics

import pytest
from fastapi.testclient import TestClient

from client import MarketClient
from market_env import server as server_module
from market_env.bots import InformedBot
from market_env.environment import (
    EpisodeAlreadyDone,
    EpisodeNotFound,
    MarketEnvironment,
)
from market_env.models import MarketAction


# ---------------------------------------------------------------------------
# Direct env API: full episode life cycle
# ---------------------------------------------------------------------------

class TestFullEpisodeCycle:
    def test_reset_then_50_steps_then_done(self):
        env = MarketEnvironment()
        obs = env.reset(seed=42, difficulty="medium", episode_length=50)
        episode_id = obs.episode_id
        last_done = False
        last_reward = 0.0
        last_info = {}
        for _ in range(50):
            obs, reward, done, info = env.step(
                episode_id, MarketAction(action_type="hold"),
            )
            last_done, last_reward, last_info = done, reward, info
        assert last_done is True
        # Episode-end reward must be a finite number in [-1, 1]
        assert -1.0 <= last_reward <= 1.0
        assert "reward_breakdown" in last_info
        assert obs.true_value is not None

    def test_invalid_episode_id_raises(self):
        env = MarketEnvironment()
        with pytest.raises(EpisodeNotFound):
            env.step("not-a-real-id", MarketAction(action_type="hold"))

    def test_action_on_completed_episode_raises(self):
        env = MarketEnvironment()
        obs = env.reset(episode_length=2)
        env.step(obs.episode_id, MarketAction(action_type="hold"))
        env.step(obs.episode_id, MarketAction(action_type="hold"))
        with pytest.raises(EpisodeAlreadyDone):
            env.step(obs.episode_id, MarketAction(action_type="hold"))

    def test_malformed_action_treated_as_hold(self):
        """A 'buy' action with price=None must not crash; the env downgrades to hold."""
        env = MarketEnvironment()
        obs = env.reset(episode_length=5)
        bad_action = MarketAction(action_type="buy", price=None, quantity=None)
        new_obs, reward, done, info = env.step(obs.episode_id, bad_action)
        # Did not crash; reward is 0 mid-episode; episode still ongoing
        assert reward == 0.0
        assert not done


# ---------------------------------------------------------------------------
# Ground truth: an informed agent should profit on average
# ---------------------------------------------------------------------------

class TestInformedAgentBaseline:
    def test_informed_agent_makes_positive_pnl_on_average(self):
        """An agent that knows the true value should beat the market on average.

        Setup:
            trainable agent driven by InformedBot (has noisy true_value)
            opponent: just a MarketMakerBot (provides liquidity)
            10 seeds, easy difficulty for clearer signals.

        Expectation:
            Mean normalized P&L > 0 across the sample.
        """
        env = MarketEnvironment()
        pnls = []
        for seed in range(10):
            obs = env.reset(
                seed=seed,
                difficulty="easy",
                bot_config="liquidity_only",   # only a MarketMaker on the other side
                episode_length=50,
            )
            # Cheat: peek at true_value to seed the InformedBot
            ep_state = env._episodes[obs.episode_id]
            true_value = ep_state.scenario.true_value

            informed = InformedBot("agent_1", seed=seed, edge=0.20, qty=10)
            informed.set_true_value(true_value)

            done = False
            info = {}
            while not done:
                action = informed.act(obs)
                obs, _, done, info = env.step(obs.episode_id, action)

            pnls.append(info["reward_breakdown"]["pnl_normalized"])

        # On average, an informed agent must profit. Use mean > 0 as a soft signal;
        # a fair market with informed trader vs market maker should show edge.
        assert statistics.mean(pnls) > 0.0, f"informed agent did not profit on average: {pnls}"


# ---------------------------------------------------------------------------
# HTTP layer end-to-end via TestClient + the real MarketClient
# ---------------------------------------------------------------------------

class TestHttpEndToEnd:
    @pytest.fixture
    def client(self):
        # Reset env for isolation between tests
        server_module.env = MarketEnvironment()
        return TestClient(server_module.app)

    def test_full_cycle_via_http(self, client):
        """reset → step × N → done, all over HTTP, status codes intact."""
        ep = client.post("/reset", json={"episode_length": 10}).json()
        eid = ep["episode_id"]
        for i in range(10):
            r = client.post("/step", json={
                "episode_id": eid,
                "action": {"action_type": "hold"},
            })
            assert r.status_code == 200, r.text
        body = r.json()
        assert body["done"] is True
        assert body["observation"]["true_value"] is not None

    def test_market_client_class_against_test_server(self, client):
        """The MarketClient class should work against a TestClient-served app.

        We hand-build the client by injecting the FastAPI TestClient — it
        speaks the same wire format as a real httpx.Client, so the HTTP
        contract is the same.
        """
        m = MarketClient.__new__(MarketClient)
        m.base_url = ""
        m._http = client     # TestClient is httpx-compatible

        assert m.health() == {"status": "ok"}
        tasks = m.list_tasks()
        assert len(tasks) > 0

        obs = m.reset(seed=42, difficulty="easy", episode_length=5)
        assert obs.turn == 0

        last_done = False
        for _ in range(5):
            obs, reward, done, info = m.step(
                obs.episode_id, MarketAction(action_type="hold"),
            )
            last_done = done
        assert last_done is True
        assert obs.true_value is not None

        st = m.state(obs.episode_id)
        assert st["done"] is True

    def test_market_client_constructor_and_context_manager(self):
        """Cover the user-facing init/enter/exit/close paths."""
        from client.client import MarketClient
        # __init__ with custom base_url; trailing slash should be stripped.
        with MarketClient(base_url="http://localhost:9999/", timeout=1.0) as m:
            assert m.base_url == "http://localhost:9999"
        # __exit__ → close() should have been called; calling again is safe.
        m.close()

    def test_market_client_raises_on_http_error(self, client):
        """Server errors are surfaced as MarketClientError with status code."""
        from client.client import MarketClient, MarketClientError
        m = MarketClient.__new__(MarketClient)
        m.base_url = ""
        m._http = client
        with pytest.raises(MarketClientError) as exc_info:
            m.step("doesnotexist", MarketAction(action_type="hold"))
        assert exc_info.value.status_code == 404
        assert "doesnotexist" in exc_info.value.detail
