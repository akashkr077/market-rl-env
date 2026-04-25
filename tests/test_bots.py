"""Tests for the scripted baseline bots."""
from __future__ import annotations

from typing import Optional

import pytest

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
)
from market_env.order_book import (
    OrderBook,
    OrderBookLevel,
    OrderBookSnapshot,
    TradeRecord,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_observation(
    bids: list[tuple[float, int]] = (),
    asks: list[tuple[float, int]] = (),
    recent_prices: list[float] = (),
    agent_id: str = "test",
    private_signals: Optional[dict[str, float]] = None,
    turn: int = 1,
) -> MarketObservation:
    """Build a MarketObservation from compact level/trade specs."""
    bid_levels = [OrderBookLevel(price=p, quantity=q, num_orders=1) for p, q in bids]
    ask_levels = [OrderBookLevel(price=p, quantity=q, num_orders=1) for p, q in asks]
    best_bid = bid_levels[0].price if bid_levels else None
    best_ask = ask_levels[0].price if ask_levels else None
    if best_bid is not None and best_ask is not None:
        mid, spread = (best_bid + best_ask) / 2, best_ask - best_bid
    elif best_bid is not None:
        mid, spread = best_bid, 0.0
    elif best_ask is not None:
        mid, spread = best_ask, 0.0
    else:
        mid, spread = 0.0, 0.0

    snap = OrderBookSnapshot(bids=bid_levels, asks=ask_levels, mid_price=mid, spread=spread)
    trades = [
        TradeRecord(
            trade_id=f"T{i}", price=p, quantity=10,
            buyer_id="x", seller_id="y", aggressor_side="buy", turn=i,
        )
        for i, p in enumerate(recent_prices)
    ]
    return MarketObservation(
        order_book=snap, recent_trades=trades, agent_id=agent_id,
        private_signals=private_signals or {}, turn=turn, max_turns=50,
        episode_id="test_ep",
    )


# ---------------------------------------------------------------------------
# RandomBot
# ---------------------------------------------------------------------------

class TestRandomBot:
    def test_action_probability_roughly_respected(self):
        bot = RandomBot("test", seed=42, action_prob=0.6)
        obs = make_observation(bids=[(49.0, 10)], asks=[(51.0, 10)])
        non_hold = sum(
            1 for _ in range(2000)
            if bot.act(obs).action_type != "hold"
        )
        # Expected ~1200 ± a bit; very wide tolerance because RNG variance
        assert 1000 < non_hold < 1400

    def test_orders_centered_on_mid_price(self):
        bot = RandomBot("test", seed=42, price_jitter=3.0, action_prob=1.0)
        obs = make_observation(bids=[(49.0, 10)], asks=[(51.0, 10)])
        prices = [
            bot.act(obs).price for _ in range(2000)
            if (a := bot.act(obs)).price is not None
        ]
        assert 49.5 < (sum(prices) / len(prices)) < 50.5

    def test_quantities_within_bounds(self):
        bot = RandomBot("test", seed=42, action_prob=1.0, max_qty=30)
        obs = make_observation(bids=[(49.0, 10)], asks=[(51.0, 10)])
        for _ in range(200):
            a = bot.act(obs)
            if a.quantity is not None:
                assert 1 <= a.quantity <= 30


# ---------------------------------------------------------------------------
# MomentumBot
# ---------------------------------------------------------------------------

class TestMomentumBot:
    def test_buys_on_clear_uptrend(self):
        bot = MomentumBot("mo", seed=42, threshold=3)
        obs = make_observation(
            bids=[(49.0, 10)], asks=[(51.0, 10)],
            recent_prices=[49.0, 49.5, 50.0, 50.5, 51.0],   # 4 ups
        )
        action = bot.act(obs)
        assert action.action_type == "buy"
        assert action.price >= 51.0   # at-or-above the ask

    def test_sells_on_clear_downtrend(self):
        bot = MomentumBot("mo", seed=42, threshold=3)
        obs = make_observation(
            bids=[(49.0, 10)], asks=[(51.0, 10)],
            recent_prices=[51.0, 50.5, 50.0, 49.5, 49.0],
        )
        action = bot.act(obs)
        assert action.action_type == "sell"
        assert action.price <= 49.0

    def test_holds_with_too_few_trades(self):
        bot = MomentumBot("mo", seed=42)
        obs = make_observation(
            bids=[(49.0, 10)], asks=[(51.0, 10)],
            recent_prices=[50.0],
        )
        assert bot.act(obs).action_type == "hold"

    def test_holds_when_trend_is_mixed(self):
        bot = MomentumBot("mo", seed=42, threshold=3)
        obs = make_observation(
            bids=[(49.0, 10)], asks=[(51.0, 10)],
            recent_prices=[50.0, 50.1, 50.0, 50.1, 50.0],   # flat / oscillating
        )
        assert bot.act(obs).action_type == "hold"


# ---------------------------------------------------------------------------
# MeanReversionBot
# ---------------------------------------------------------------------------

class TestMeanReversionBot:
    def test_sells_when_mid_above_anchor_plus_threshold(self):
        bot = MeanReversionBot("mr", anchor=50.0, threshold=1.0)
        obs = make_observation(bids=[(52.0, 10)], asks=[(53.0, 10)])
        action = bot.act(obs)
        assert action.action_type == "sell"

    def test_buys_when_mid_below_anchor_minus_threshold(self):
        bot = MeanReversionBot("mr", anchor=50.0, threshold=1.0)
        obs = make_observation(bids=[(47.0, 10)], asks=[(48.0, 10)])
        action = bot.act(obs)
        assert action.action_type == "buy"

    def test_holds_within_threshold(self):
        bot = MeanReversionBot("mr", anchor=50.0, threshold=1.0)
        obs = make_observation(bids=[(49.5, 10)], asks=[(50.5, 10)])
        assert bot.act(obs).action_type == "hold"

    def test_holds_on_empty_book(self):
        bot = MeanReversionBot("mr")
        assert bot.act(make_observation()).action_type == "hold"


# ---------------------------------------------------------------------------
# InformedBot
# ---------------------------------------------------------------------------

class TestInformedBot:
    def test_holds_without_estimate_set(self):
        bot = InformedBot("inf", seed=42)
        obs = make_observation(bids=[(49.0, 10)], asks=[(51.0, 10)])
        assert bot.act(obs).action_type == "hold"

    def test_buys_when_ask_well_below_estimate(self):
        bot = InformedBot("inf", seed=42, edge=0.50)
        bot.set_true_value(60.0)
        obs = make_observation(bids=[(49.0, 10)], asks=[(50.0, 10)])
        action = bot.act(obs)
        assert action.action_type == "buy"
        assert action.price == 50.0

    def test_sells_when_bid_well_above_estimate(self):
        bot = InformedBot("inf", seed=42, edge=0.50)
        bot.set_true_value(40.0)
        obs = make_observation(bids=[(50.0, 10)], asks=[(51.0, 10)])
        action = bot.act(obs)
        assert action.action_type == "sell"
        assert action.price == 50.0

    def test_holds_when_no_edge_available(self):
        bot = InformedBot("inf", seed=42, edge=0.50, noise_std=0.0)
        bot.set_true_value(50.0)
        obs = make_observation(bids=[(49.9, 10)], asks=[(50.1, 10)])
        assert bot.act(obs).action_type == "hold"

    def test_reset_clears_estimate(self):
        bot = InformedBot("inf", seed=42)
        bot.set_true_value(60.0)
        bot.reset()
        obs = make_observation(bids=[(49.0, 10)], asks=[(50.0, 10)])
        assert bot.act(obs).action_type == "hold"


# ---------------------------------------------------------------------------
# MarketMakerBot
# ---------------------------------------------------------------------------

class TestMarketMakerBot:
    def test_alternates_buy_and_sell_each_turn(self):
        bot = MarketMakerBot("mm")
        obs = make_observation(bids=[(49.0, 10)], asks=[(51.0, 10)])
        sides = [bot.act(obs).action_type for _ in range(6)]
        assert sides == ["buy", "sell", "buy", "sell", "buy", "sell"]

    def test_quotes_at_mid_plus_minus_half_spread(self):
        bot = MarketMakerBot("mm", half_spread=0.30)
        obs = make_observation(bids=[(49.0, 10)], asks=[(51.0, 10)])  # mid=50
        a1 = bot.act(obs)
        a2 = bot.act(obs)
        # Expect buy at 49.70 and sell at 50.30
        prices = sorted([a1.price, a2.price])
        assert prices[0] == pytest.approx(49.70)
        assert prices[1] == pytest.approx(50.30)

    def test_uses_anchor_when_book_empty(self):
        bot = MarketMakerBot("mm", anchor=50.0, half_spread=0.30)
        empty_obs = make_observation()
        a1 = bot.act(empty_obs)
        a2 = bot.act(empty_obs)
        prices = sorted([a1.price, a2.price])
        assert prices[0] == pytest.approx(49.70)
        assert prices[1] == pytest.approx(50.30)

    def test_reset_restarts_alternation(self):
        bot = MarketMakerBot("mm")
        obs = make_observation(bids=[(49.0, 10)], asks=[(51.0, 10)])
        first = bot.act(obs).action_type   # "buy"
        bot.act(obs)                       # "sell"
        bot.reset()
        post_reset = bot.act(obs).action_type
        assert post_reset == first


# ---------------------------------------------------------------------------
# Common interface — every bot can be polled in a uniform loop
# ---------------------------------------------------------------------------

class TestCommonInterface:
    @pytest.fixture
    def all_bots(self) -> list[Bot]:
        return [
            RandomBot("r", seed=0),
            MomentumBot("mo", seed=0),
            MeanReversionBot("mr"),
            InformedBot("inf", seed=0),
            MarketMakerBot("mm"),
        ]

    def test_every_bot_returns_a_market_action(self, all_bots):
        obs = make_observation(bids=[(49.0, 10)], asks=[(51.0, 10)])
        for bot in all_bots:
            action = bot.act(obs)
            assert isinstance(action, MarketAction)

    def test_every_bot_can_be_reset(self, all_bots):
        for bot in all_bots:
            bot.reset()  # must not raise


# ---------------------------------------------------------------------------
# Integration: all 5 bots playing one full episode against the order book
# ---------------------------------------------------------------------------

class TestEpisodeIntegration:
    def test_full_episode_runs_to_completion_without_error(self):
        """Plumbing test: 50-turn episode with 5 bots completes cleanly and
        produces at least some trades."""
        from market_env.scenario import ScenarioGenerator

        scenario = ScenarioGenerator(seed=42).sample(difficulty="easy")
        book = OrderBook()

        bots: dict[str, Bot] = {
            "agent_1": InformedBot("agent_1", seed=1),
            "agent_2": RandomBot("agent_2", seed=2),
            "agent_3": MomentumBot("agent_3", seed=3),
            "mm_bot":  MarketMakerBot("mm_bot"),
            "mr_bot":  MeanReversionBot("mr_bot"),
        }
        bots["agent_1"].set_true_value(scenario.true_value)

        positions = {a: Position(agent_id=a) for a in bots}

        for turn in range(scenario.episode_length):
            book.set_turn(turn)
            for agent_id, bot in bots.items():
                snap = book.get_snapshot(depth=5)
                obs = MarketObservation(
                    order_book=snap,
                    recent_trades=book.get_recent_trades(n=10),
                    agent_id=agent_id,
                    cash=positions[agent_id].cash,
                    shares_held=positions[agent_id].shares_held,
                    private_signals=scenario.agent_signals.get(agent_id, {}),
                    turn=turn, max_turns=scenario.episode_length,
                    episode_id=scenario.scenario_id,
                )
                action = bot.act(obs)

                if action.action_type == "hold":
                    continue
                if action.action_type == "cancel":
                    if action.order_id:
                        book.cancel_order(agent_id, action.order_id)
                    continue
                if action.action_type in ("buy", "sell"):
                    if action.price is None or action.quantity is None:
                        continue
                    try:
                        result = book.place_limit_order(
                            agent_id, action.action_type, action.price, action.quantity,
                        )
                    except ValueError:
                        continue
                    # Each trade affects BOTH the buyer and the seller positions,
                    # not just the aggressor (who is just the agent placing the action).
                    for trade in result.trades:
                        if trade.buyer_id in positions:
                            positions[trade.buyer_id].apply_trade("buy", trade.price, trade.quantity)
                        if trade.seller_id in positions:
                            positions[trade.seller_id].apply_trade("sell", trade.price, trade.quantity)

        all_trades = book.get_recent_trades(n=10_000)
        # There should be SOME trades — InformedBot vs MarketMakerBot will cross
        assert len(all_trades) > 0
        # Cash conservation: across all agents, net cash + net share value
        # (at any consistent mark price) should equal sum of starting cash.
        total_cash = sum(p.cash for p in positions.values())
        total_shares = sum(p.shares_held for p in positions.values())
        # Net shares must be zero (every buy is matched by a sell)
        assert total_shares == 0
        # Net cash must equal initial total (zero-sum at any mark)
        assert total_cash == pytest.approx(INITIAL_CASH * len(bots), abs=1e-6)
