"""Lightweight tests for the Pydantic models layer."""
from __future__ import annotations

import math
import pytest

from market_env.models import (
    INITIAL_CASH,
    MarketAction,
    MarketObservation,
    OpenOrderView,
    Position,
    open_order_view_from_order,
)
from market_env.order_book import (
    Order,
    OrderBookSnapshot,
    TradeRecord,
)


# ---------------------------------------------------------------------------
# MarketAction validation
# ---------------------------------------------------------------------------

class TestMarketActionValidation:
    def test_hold_action_is_minimal(self):
        a = MarketAction(action_type="hold")
        assert a.action_type == "hold"
        assert a.price is None
        assert a.quantity is None

    def test_buy_action_with_valid_fields(self):
        a = MarketAction(action_type="buy", price=50.0, quantity=10)
        assert a.action_type == "buy"
        assert a.price == 50.0
        assert a.quantity == 10

    def test_invalid_action_type_rejected(self):
        with pytest.raises(Exception):
            MarketAction(action_type="trade", price=50.0, quantity=10)

    def test_nan_price_rejected(self):
        with pytest.raises(Exception, match="finite"):
            MarketAction(action_type="buy", price=float("nan"), quantity=10)

    def test_inf_price_rejected(self):
        with pytest.raises(Exception, match="finite"):
            MarketAction(action_type="buy", price=float("inf"), quantity=10)

    def test_negative_price_rejected(self):
        with pytest.raises(Exception, match="positive"):
            MarketAction(action_type="buy", price=-1.0, quantity=10)

    def test_zero_quantity_rejected(self):
        with pytest.raises(Exception, match="positive"):
            MarketAction(action_type="buy", price=50.0, quantity=0)

    def test_bool_quantity_rejected(self):
        with pytest.raises(Exception, match="int"):
            MarketAction(action_type="buy", price=50.0, quantity=True)

    def test_cancel_action_with_none_price_and_quantity(self):
        """Cancel actions legitimately have None for price and quantity."""
        a = MarketAction(action_type="cancel", order_id="X1")
        assert a.price is None
        assert a.quantity is None
        assert a.order_id == "X1"

    def test_explicit_none_price_and_quantity_pass_validators(self):
        """Validators must accept explicit None (used by hold/cancel actions)."""
        a = MarketAction(action_type="hold", price=None, quantity=None)
        assert a.price is None
        assert a.quantity is None


# ---------------------------------------------------------------------------
# MarketObservation accepts internal dataclasses (Pydantic v2 compat check)
# ---------------------------------------------------------------------------

class TestMarketObservation:
    def test_constructs_with_dataclass_fields(self):
        snap = OrderBookSnapshot(bids=[], asks=[], mid_price=50.0, spread=0.0)
        obs = MarketObservation(
            order_book=snap, agent_id="a1", turn=0, max_turns=50,
            episode_id="ep_1",
        )
        assert obs.agent_id == "a1"
        assert obs.cash == INITIAL_CASH
        assert obs.true_value is None
        assert obs.visible_other_orders == []

    def test_serializes_to_json(self):
        """Critical: the observation must be JSON-serializable for the API."""
        snap = OrderBookSnapshot(bids=[], asks=[], mid_price=50.0, spread=0.0)
        trade = TradeRecord(
            trade_id="T1", price=50.0, quantity=5, buyer_id="b", seller_id="s",
            aggressor_side="buy", turn=0,
        )
        obs = MarketObservation(
            order_book=snap, recent_trades=[trade], agent_id="a1",
            private_signals={"earnings": 1.5}, turn=3, max_turns=50,
            episode_id="ep_1",
        )
        js = obs.model_dump_json()
        assert "ep_1" in js
        assert "earnings" in js
        assert "T1" in js


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class TestPosition:
    def test_buy_decreases_cash_and_increases_shares(self):
        p = Position(agent_id="a1")
        p.apply_trade("buy", price=50.0, quantity=10)
        assert p.shares_held == 10
        assert p.cash == INITIAL_CASH - 500.0

    def test_sell_increases_cash_and_decreases_shares(self):
        p = Position(agent_id="a1")
        p.apply_trade("sell", price=50.0, quantity=10)
        assert p.shares_held == -10
        assert p.cash == INITIAL_CASH + 500.0

    def test_mark_to_value(self):
        p = Position(agent_id="a1", shares_held=10, cash=5000.0)
        assert p.mark_to_value(50.0) == 5500.0
        assert p.mark_to_value(60.0) == 5600.0

    def test_pnl_is_zero_at_initial_state(self):
        p = Position(agent_id="a1")
        assert p.pnl(50.0) == 0.0

    def test_pnl_after_round_trip_at_correct_value(self):
        """Buy 10 at 50, sell 10 at 55 → P&L = $50 regardless of true_value."""
        p = Position(agent_id="a1")
        p.apply_trade("buy", 50.0, 10)
        p.apply_trade("sell", 55.0, 10)
        assert p.shares_held == 0
        assert p.pnl(true_value=50.0) == 50.0
        assert p.pnl(true_value=999.0) == 50.0   # no shares → mark_price irrelevant


# ---------------------------------------------------------------------------
# OpenOrderView conversion helper
# ---------------------------------------------------------------------------

class TestOpenOrderView:
    def test_view_strips_internal_fields(self):
        order = Order(
            order_id="X1", agent_id="a1", side="buy", price=50.0,
            quantity=10, seq=42, filled=3,
        )
        view = open_order_view_from_order(order)
        assert view.order_id == "X1"
        assert view.side == "buy"
        assert view.price == 50.0
        assert view.quantity == 10
        assert view.filled == 3
        assert view.remaining == 7
        # Confirm internal fields not exposed
        assert not hasattr(view, "agent_id")
        assert not hasattr(view, "seq")
