"""Tests for the multi-session MarketEnvironment (Python API only — no HTTP)."""
from __future__ import annotations

import pytest

from market_env.bots import InformedBot
from market_env.environment import (
    BOT_FACTORIES,
    EpisodeAlreadyDone,
    EpisodeNotFound,
    MarketEnvironment,
)
from market_env.models import MarketAction, MarketObservation


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_market_observation(self):
        env = MarketEnvironment()
        obs = env.reset()
        assert isinstance(obs, MarketObservation)
        assert obs.turn == 0
        assert obs.episode_id != ""

    def test_reset_creates_distinct_episode_ids(self):
        env = MarketEnvironment()
        ids = {env.reset().episode_id for _ in range(5)}
        assert len(ids) == 5

    def test_reset_with_seed_is_reproducible(self):
        env = MarketEnvironment()
        obs1 = env.reset(seed=42, difficulty="easy")
        obs2 = env.reset(seed=42, difficulty="easy")
        # Different episode_ids but same scenario contents
        assert obs1.episode_id != obs2.episode_id
        assert obs1.private_signals == obs2.private_signals

    def test_reset_assigns_signals_to_trainable_agent(self):
        env = MarketEnvironment()
        obs = env.reset(trainable_agent_id="agent_1")
        # agent_1's specialty: earnings + competitor
        assert set(obs.private_signals) == {"earnings", "competitor"}

    def test_reset_true_value_hidden_until_done(self):
        env = MarketEnvironment()
        obs = env.reset()
        assert obs.true_value is None

    def test_reset_with_invalid_bot_config_raises(self):
        env = MarketEnvironment()
        with pytest.raises(ValueError, match="bot_config"):
            env.reset(bot_config="nonexistent")

    def test_reset_with_unknown_task_id_raises(self):
        env = MarketEnvironment()
        with pytest.raises(ValueError, match="task_id"):
            env.reset(task_id="does_not_exist")

    def test_reset_with_known_task_id_loads_config(self):
        env = MarketEnvironment()
        obs = env.reset(task_id="eval_00")
        # eval set is medium + eval composition
        assert obs.turn == 0


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_advances_turn_counter(self):
        env = MarketEnvironment()
        obs = env.reset()
        new_obs, _, _, _ = env.step(obs.episode_id, MarketAction(action_type="hold"))
        assert new_obs.turn == 1

    def test_step_returns_zero_reward_mid_episode(self):
        env = MarketEnvironment()
        obs = env.reset(episode_length=10)
        _, reward, done, _ = env.step(obs.episode_id, MarketAction(action_type="hold"))
        assert reward == 0.0
        assert not done

    def test_step_done_at_max_turns(self):
        env = MarketEnvironment()
        obs = env.reset(episode_length=3)
        for _ in range(3):
            obs2, reward, done, info = env.step(
                obs.episode_id, MarketAction(action_type="hold"),
            )
        assert done is True
        assert "reward_breakdown" in info

    def test_step_reveals_true_value_at_done(self):
        env = MarketEnvironment()
        obs = env.reset(episode_length=2)
        for _ in range(2):
            obs, _, done, _ = env.step(obs.episode_id, MarketAction(action_type="hold"))
        assert done
        assert obs.true_value is not None

    def test_step_unknown_episode_id_raises(self):
        env = MarketEnvironment()
        with pytest.raises(EpisodeNotFound):
            env.step("nonexistent", MarketAction(action_type="hold"))

    def test_step_after_done_raises(self):
        env = MarketEnvironment()
        obs = env.reset(episode_length=1)
        env.step(obs.episode_id, MarketAction(action_type="hold"))
        with pytest.raises(EpisodeAlreadyDone):
            env.step(obs.episode_id, MarketAction(action_type="hold"))

    def test_step_buy_action_executes(self):
        env = MarketEnvironment()
        obs = env.reset(episode_length=10)
        action = MarketAction(action_type="buy", price=50.0, quantity=5)
        new_obs, _, _, info = env.step(obs.episode_id, action)
        assert info["action_status"] == "accepted"

    def test_step_buy_with_missing_price_treated_as_hold(self):
        """Malformed buy/sell action does not crash; downgrades to hold."""
        env = MarketEnvironment()
        obs = env.reset(episode_length=10)
        # MarketAction validator allows price=None when not buy/sell, but the env
        # must still tolerate this case gracefully.
        action = MarketAction(action_type="buy", price=None, quantity=None)
        new_obs, reward, done, info = env.step(obs.episode_id, action)
        assert reward == 0.0
        assert not done

    def test_step_cancel_executes_for_valid_order(self):
        # Use bot_config="empty" so no opposing bot can fill our resting order.
        env = MarketEnvironment()
        obs = env.reset(episode_length=10, bot_config="empty")
        place = MarketAction(action_type="buy", price=40.0, quantity=5)
        new_obs, _, _, _ = env.step(obs.episode_id, place)
        assert len(new_obs.open_orders) == 1
        oid = new_obs.open_orders[0].order_id
        cancel = MarketAction(action_type="cancel", order_id=oid)
        after, _, _, info = env.step(obs.episode_id, cancel)
        assert info["action_status"] == "accepted"
        assert after.open_orders == []

    def test_step_cancel_without_order_id_is_a_hold(self):
        env = MarketEnvironment()
        obs = env.reset(episode_length=5)
        cancel = MarketAction(action_type="cancel", order_id=None)
        _, reward, done, _ = env.step(obs.episode_id, cancel)
        assert reward == 0.0
        assert not done

    def test_step_quantity_exceeding_engine_max_rejected(self):
        """Pydantic accepts quantity=10001 but the order book rejects it.

        The env catches the ValueError and marks the action rejected
        without crashing the episode.
        """
        from market_env.order_book import MAX_QUANTITY
        env = MarketEnvironment()
        obs = env.reset(episode_length=5)
        huge = MarketAction(
            action_type="buy", price=50.0, quantity=MAX_QUANTITY + 1,
        )
        _, _, done, info = env.step(obs.episode_id, huge)
        assert info["action_status"] == "rejected"
        assert "MAX_QUANTITY" in info.get("rejection_reason", "")
        assert not done


# ---------------------------------------------------------------------------
# state()
# ---------------------------------------------------------------------------

class TestState:
    def test_state_returns_dict_with_expected_keys(self):
        env = MarketEnvironment()
        obs = env.reset()
        st = env.state(obs.episode_id)
        for key in ("episode_id", "turn", "max_turns", "done", "positions",
                    "best_bid", "best_ask", "mid_price", "true_value"):
            assert key in st

    def test_state_unknown_episode_raises(self):
        env = MarketEnvironment()
        with pytest.raises(EpisodeNotFound):
            env.state("nonexistent")

    def test_state_true_value_hidden_until_done(self):
        env = MarketEnvironment()
        obs = env.reset(episode_length=2)
        st1 = env.state(obs.episode_id)
        assert st1["true_value"] is None

        for _ in range(2):
            env.step(obs.episode_id, MarketAction(action_type="hold"))
        st2 = env.state(obs.episode_id)
        assert st2["true_value"] is not None
        assert st2["done"]


# ---------------------------------------------------------------------------
# list_tasks()
# ---------------------------------------------------------------------------

class TestListTasks:
    def test_list_tasks_includes_eval_set(self):
        env = MarketEnvironment()
        tasks = env.list_tasks()
        eval_ids = [t["task_id"] for t in tasks if t["task_id"].startswith("eval_")]
        assert len(eval_ids) == 50
        assert "eval_00" in eval_ids
        assert "eval_49" in eval_ids

    def test_list_tasks_includes_difficulty_demos(self):
        env = MarketEnvironment()
        tasks = env.list_tasks()
        ids = [t["task_id"] for t in tasks]
        assert "demo_easy" in ids
        assert "demo_medium" in ids
        assert "demo_hard" in ids

    def test_eval_tasks_use_eval_bot_composition(self):
        env = MarketEnvironment()
        tasks = env.list_tasks()
        evals = [t for t in tasks if t["task_id"].startswith("eval_")]
        assert all(t["bot_config"] == "eval" for t in evals)


# ---------------------------------------------------------------------------
# Bot factories
# ---------------------------------------------------------------------------

class TestBotFactories:
    def test_default_factory_excludes_informed_bot(self):
        from market_env.scenario import ScenarioGenerator
        scenario = ScenarioGenerator(seed=0).sample()
        bots = BOT_FACTORIES["default"](scenario, seed=0)
        assert "informed_bot" not in bots
        assert len(bots) == 4

    def test_eval_factory_includes_informed_bot(self):
        from market_env.scenario import ScenarioGenerator
        scenario = ScenarioGenerator(seed=0).sample()
        bots = BOT_FACTORIES["eval"](scenario, seed=0)
        assert "informed_bot" in bots
        assert isinstance(bots["informed_bot"], InformedBot)

    def test_empty_factory_has_no_bots(self):
        from market_env.scenario import ScenarioGenerator
        scenario = ScenarioGenerator(seed=0).sample()
        assert BOT_FACTORIES["empty"](scenario, seed=0) == {}


# ---------------------------------------------------------------------------
# Conservation laws (sanity invariants of a well-behaved env)
# ---------------------------------------------------------------------------

class TestConservation:
    def test_full_episode_preserves_total_cash_and_zero_net_shares(self):
        env = MarketEnvironment()
        obs = env.reset(episode_length=10, bot_config="default")
        episode_id = obs.episode_id
        for _ in range(10):
            env.step(episode_id, MarketAction(action_type="hold"))
        st = env.state(episode_id)
        total_cash = sum(p["cash"] for p in st["positions"].values())
        total_shares = sum(p["shares_held"] for p in st["positions"].values())
        from market_env.reward import INITIAL_CASH
        n_agents = len(st["positions"])
        assert total_shares == 0
        assert total_cash == pytest.approx(INITIAL_CASH * n_agents, abs=1e-6)
