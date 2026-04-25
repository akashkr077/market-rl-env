"""
Multi-session market environment.

Stage 1 design (M5):
    1 trainable agent + 4 scripted bots (Random, Momentum, MeanReversion, MarketMaker).
    InformedBot is held back as eval-only opponent.

The environment is multi-session: a single MarketEnvironment instance manages
many concurrent episodes keyed by episode_id. The FastAPI server is a stateless
wrapper on top of this; multiple HTTP clients can interleave reset/step calls
against different episodes without confusion.

Public Python API:
    env.reset(...) → MarketObservation               # creates new session
    env.step(episode_id, action) → (obs, r, done, info)
    env.state(episode_id) → dict
    env.list_tasks() → list[dict]
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from market_env.bots import (
    Bot,
    InformedBot,
    MarketMakerBot,
    MeanReversionBot,
    MomentumBot,
    RandomBot,
)
from market_env.models import (
    INITIAL_CASH,
    MarketAction,
    MarketObservation,
    Position,
    open_order_view_from_order,
)
from market_env.order_book import OrderBook
from market_env.reward import (
    AgentStats,
    RewardBreakdown,
    compute_reward,
)
from market_env.scenario import (
    DEFAULT_EPISODE_LENGTH,
    MarketScenario,
    ScenarioGenerator,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EnvironmentError(Exception):
    """Base for environment-level errors."""


class EpisodeNotFound(EnvironmentError):
    pass


class EpisodeAlreadyDone(EnvironmentError):
    pass


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    episode_id: str
    scenario: MarketScenario
    book: OrderBook
    bots: dict[str, Bot]
    positions: dict[str, Position]
    stats: dict[str, AgentStats]
    trainable_agent_id: str
    turn: int = 0
    done: bool = False
    rewards: dict[str, RewardBreakdown] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bot composition factories
# ---------------------------------------------------------------------------

def _default_bot_factory(scenario: MarketScenario, seed: int) -> dict[str, Bot]:
    """Stage 1 training opponents — deliberately excludes InformedBot."""
    return {
        "random_bot":   RandomBot("random_bot", seed=seed + 100),
        "momentum_bot": MomentumBot("momentum_bot", seed=seed + 101),
        "mean_rev_bot": MeanReversionBot("mean_rev_bot", anchor=scenario.initial_mid),
        "mm_bot":       MarketMakerBot("mm_bot", anchor=scenario.initial_mid),
    }


def _eval_bot_factory(scenario: MarketScenario, seed: int) -> dict[str, Bot]:
    """Eval composition — adds InformedBot as the hard benchmark opponent."""
    bots = _default_bot_factory(scenario, seed)
    informed = InformedBot("informed_bot", seed=seed + 200)
    informed.set_true_value(scenario.true_value)
    bots["informed_bot"] = informed
    return bots


def _empty_bot_factory(scenario: MarketScenario, seed: int) -> dict[str, Bot]:
    """No opponents (used for unit tests of the trainable agent in isolation)."""
    return {}


def _liquidity_only_bot_factory(scenario: MarketScenario, seed: int) -> dict[str, Bot]:
    """Just a market maker — used for the InformedBot ground-truth test."""
    return {"mm_bot": MarketMakerBot("mm_bot", anchor=scenario.initial_mid)}


BOT_FACTORIES = {
    "default":        _default_bot_factory,
    "eval":           _eval_bot_factory,
    "empty":          _empty_bot_factory,
    "liquidity_only": _liquidity_only_bot_factory,
}


# ---------------------------------------------------------------------------
# Predefined task list
# ---------------------------------------------------------------------------

def _list_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    # 50-scenario eval set (seeds 0–49, medium difficulty, eval bot composition)
    for i in range(50):
        tasks.append({
            "task_id": f"eval_{i:02d}",
            "difficulty": "medium",
            "seed": i,
            "bot_config": "eval",
        })
    # Difficulty demos
    for diff in ("easy", "medium", "hard"):
        tasks.append({
            "task_id": f"demo_{diff}",
            "difficulty": diff,
            "seed": 42,
            "bot_config": "default",
        })
    return tasks


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MarketEnvironment:
    """OpenEnv-compliant multi-agent market environment."""

    def __init__(self) -> None:
        self._episodes: dict[str, EpisodeState] = {}

    def list_tasks(self) -> list[dict[str, Any]]:
        return _list_tasks()

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: int = 42,
        difficulty: str = "medium",
        bot_config: str = "default",
        trainable_agent_id: str = "agent_1",
        episode_length: int = DEFAULT_EPISODE_LENGTH,
    ) -> MarketObservation:
        """Create a new episode and return the trainable agent's first observation."""
        if task_id is not None:
            task = next((t for t in _list_tasks() if t["task_id"] == task_id), None)
            if task is None:
                raise ValueError(f"unknown task_id: {task_id!r}")
            seed = task["seed"]
            difficulty = task["difficulty"]
            bot_config = task["bot_config"]

        if bot_config not in BOT_FACTORIES:
            raise ValueError(
                f"unknown bot_config: {bot_config!r} "
                f"(must be one of {list(BOT_FACTORIES)})"
            )

        scenario = ScenarioGenerator(seed=seed).sample(
            difficulty=difficulty,
            episode_length=episode_length,
            agent_ids=[trainable_agent_id],
        )

        bots = BOT_FACTORIES[bot_config](scenario, seed)

        all_agents = [trainable_agent_id] + list(bots.keys())
        positions = {a: Position(agent_id=a) for a in all_agents}
        stats = {a: AgentStats() for a in all_agents}

        episode_id = uuid.uuid4().hex[:12]
        state = EpisodeState(
            episode_id=episode_id,
            scenario=scenario,
            book=OrderBook(),
            bots=bots,
            positions=positions,
            stats=stats,
            trainable_agent_id=trainable_agent_id,
        )
        self._episodes[episode_id] = state

        return self._build_observation_for(state, trainable_agent_id)

    def step(
        self, episode_id: str, action: MarketAction
    ) -> tuple[MarketObservation, float, bool, dict[str, Any]]:
        state = self._get_state(episode_id)
        if state.done:
            raise EpisodeAlreadyDone(f"episode {episode_id} is already done")

        info: dict[str, Any] = {
            "trades_this_step": 0,
            "action_status": "accepted",
        }
        state.book.set_turn(state.turn)

        # 1. Trainable agent's action
        try:
            n_trades = self._apply_action(state, state.trainable_agent_id, action)
            info["trades_this_step"] += n_trades
        except ValueError as e:
            info["action_status"] = "rejected"
            info["rejection_reason"] = str(e)
            # Treat rejection as a hold for stats purposes
            state.stats[state.trainable_agent_id].holds += 1

        # 2. Scripted bots
        for agent_id, bot in state.bots.items():
            obs = self._build_observation_for(state, agent_id)
            bot_action = bot.act(obs)
            try:
                n_trades = self._apply_action(state, agent_id, bot_action)
                info["trades_this_step"] += n_trades
            except ValueError:
                pass  # bot actions are well-formed; defensive only

        state.turn += 1

        # 3. Termination check
        if state.turn >= state.scenario.episode_length:
            state.done = True
            self._finalize(state)

        # 4. Step reward — sparse, only at episode end
        if state.done:
            breakdown = state.rewards[state.trainable_agent_id]
            reward = breakdown.total
            info["reward_breakdown"] = asdict(breakdown)
            info["true_value"] = state.scenario.true_value
            info["all_agent_rewards"] = {
                a: asdict(b) for a, b in state.rewards.items()
            }
        else:
            reward = 0.0

        return (
            self._build_observation_for(state, state.trainable_agent_id),
            reward,
            state.done,
            info,
        )

    def state(self, episode_id: str) -> dict[str, Any]:
        st = self._get_state(episode_id)
        snap = st.book.get_snapshot(depth=10)
        return {
            "episode_id": st.episode_id,
            "scenario_id": st.scenario.scenario_id,
            "turn": st.turn,
            "max_turns": st.scenario.episode_length,
            "done": st.done,
            "trainable_agent_id": st.trainable_agent_id,
            "positions": {
                a: {"shares_held": p.shares_held, "cash": p.cash}
                for a, p in st.positions.items()
            },
            "best_bid": snap.bids[0].price if snap.bids else None,
            "best_ask": snap.asks[0].price if snap.asks else None,
            "mid_price": snap.mid_price,
            "true_value": st.scenario.true_value if st.done else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_state(self, episode_id: str) -> EpisodeState:
        st = self._episodes.get(episode_id)
        if st is None:
            raise EpisodeNotFound(f"unknown episode_id: {episode_id!r}")
        return st

    def _apply_action(
        self, state: EpisodeState, agent_id: str, action: MarketAction
    ) -> int:
        """Apply an action; update positions and stats. Returns trade count.

        Malformed buy/sell (missing price/qty) silently downgrades to hold —
        no crash, no reward penalty (parse-failure penalty is the training
        loop's job, applied via stats.parse_failures).
        """
        stats = state.stats[agent_id]

        if action.action_type == "hold":
            stats.holds += 1
            return 0

        if action.action_type == "cancel":
            if action.order_id:
                state.book.cancel_order(agent_id, action.order_id)
                stats.orders_cancelled += 1
            else:
                stats.holds += 1
            return 0

        # buy / sell
        if action.price is None or action.quantity is None:
            stats.holds += 1
            return 0

        result = state.book.place_limit_order(
            agent_id, action.action_type, action.price, action.quantity,
        )
        stats.orders_placed += 1

        for trade in result.trades:
            if trade.buyer_id in state.positions:
                state.positions[trade.buyer_id].apply_trade(
                    "buy", trade.price, trade.quantity,
                )
            if trade.seller_id in state.positions:
                state.positions[trade.seller_id].apply_trade(
                    "sell", trade.price, trade.quantity,
                )

        # Track max position after the action
        cur_pos = abs(state.positions[agent_id].shares_held)
        if cur_pos > stats.max_abs_position:
            stats.max_abs_position = cur_pos

        return len(result.trades)

    def _build_observation_for(
        self, state: EpisodeState, agent_id: str
    ) -> MarketObservation:
        snap = state.book.get_snapshot(depth=5)
        recent_trades = state.book.get_recent_trades(n=10)
        position = state.positions[agent_id]
        open_orders = [
            open_order_view_from_order(o)
            for o in state.book.get_open_orders_for_agent(agent_id)
        ]
        signals = state.scenario.agent_signals.get(agent_id, {})

        mark_price = (
            snap.mid_price if snap.mid_price > 0 else state.scenario.initial_mid
        )
        unrealized = position.shares_held * mark_price

        return MarketObservation(
            order_book=snap,
            recent_trades=recent_trades,
            agent_id=agent_id,
            shares_held=position.shares_held,
            cash=position.cash,
            realized_pnl=position.realized_pnl,
            unrealized_pnl=unrealized,
            open_orders=open_orders,
            private_signals=signals,
            signal_names=list(signals.keys()),
            turn=state.turn,
            max_turns=state.scenario.episode_length,
            episode_id=state.episode_id,
            true_value=state.scenario.true_value if state.done else None,
        )

    def _finalize(self, state: EpisodeState) -> None:
        for agent_id in state.positions:
            breakdown = compute_reward(
                cash_final=state.positions[agent_id].cash,
                shares_final=state.positions[agent_id].shares_held,
                true_value=state.scenario.true_value,
                stats=state.stats[agent_id],
            )
            state.rewards[agent_id] = breakdown
