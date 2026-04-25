"""
Pydantic models for the public market environment API.

These define the contract between agents and the environment.
Internal engine types (Order, OrderBookSnapshot, TradeRecord) live in
order_book.py as plain dataclasses. Pydantic v2 supports stdlib dataclasses
as fields directly, so we embed them here without duplicate definitions.

The two boundary types added here are:
- OpenOrderView: a trimmed public view of an agent's resting order
  (no agent_id since the agent already knows it's their own; no seq number).
- MarketAction:  the action submitted by an agent each turn.
- MarketObservation: what the agent sees each turn.
- Position:      the agent's running cash/shares state.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_env.order_book import (
    Order,
    OrderBookSnapshot,
    TradeRecord,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ActionType = Literal["buy", "sell", "cancel", "hold"]
Side = Literal["buy", "sell"]

INITIAL_CASH: float = 10_000.0


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class OpenOrderView(BaseModel):
    """Public view of one of an agent's resting orders."""
    order_id: str
    side: Side
    price: float
    quantity: int          # original quantity placed
    filled: int            # already filled
    remaining: int         # still resting


class MarketAction(BaseModel):
    """An action submitted by an agent for one turn.

    - buy/sell: price and quantity required
    - cancel:   order_id required
    - hold:     no other fields needed
    """

    action_type: ActionType
    price: Optional[float] = None
    quantity: Optional[int] = None
    order_id: Optional[str] = None
    reasoning: Optional[str] = None  # logged but not used in reward

    @field_validator("price")
    @classmethod
    def _validate_price(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if math.isnan(v) or math.isinf(v):
            raise ValueError("price must be finite")
        if v <= 0:
            raise ValueError("price must be positive")
        return v

    @field_validator("quantity", mode="before")
    @classmethod
    def _validate_quantity(cls, v):
        # mode="before" runs prior to Pydantic's int coercion, so we can
        # see and reject bool inputs (which would otherwise become 1/0).
        if v is None:
            return v
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("quantity must be an int")
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v


class MarketObservation(BaseModel):
    """Per-turn observation passed to each agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Private state — fixed for the episode, only this agent sees its own signals
    private_signals: dict[str, float] = Field(default_factory=dict)
    signal_names: list[str] = Field(default_factory=list)

    # Public market state — same for all agents this turn
    order_book: OrderBookSnapshot
    recent_trades: list[TradeRecord] = Field(default_factory=list)

    # Agent's own position
    agent_id: str
    shares_held: int = 0
    cash: float = INITIAL_CASH
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    open_orders: list[OpenOrderView] = Field(default_factory=list)

    # Stage-2-only field: anonymized view of other agents' resting orders.
    # Always empty in Stage 1. Defined here from the start to avoid a schema
    # migration when we introduce self-play.
    visible_other_orders: list[OpenOrderView] = Field(default_factory=list)

    # Episode meta
    turn: int
    max_turns: int
    episode_id: str

    # Revealed only at episode end (None during the episode)
    true_value: Optional[float] = None


class Position(BaseModel):
    """Agent's running cash and share position."""

    agent_id: str
    shares_held: int = 0
    cash: float = INITIAL_CASH
    realized_pnl: float = 0.0

    def mark_to_value(self, mark_price: float) -> float:
        """Return cash + shares marked at the given price."""
        return self.cash + self.shares_held * mark_price

    def apply_trade(self, side: Side, price: float, quantity: int) -> None:
        """Update cash and share count for a single executed trade."""
        if side == "buy":
            self.cash -= price * quantity
            self.shares_held += quantity
        else:  # sell
            self.cash += price * quantity
            self.shares_held -= quantity

    def pnl(self, true_value: float) -> float:
        """Final P&L: mark-to-true-value minus initial cash."""
        return self.mark_to_value(true_value) - INITIAL_CASH


def open_order_view_from_order(order: Order) -> OpenOrderView:
    """Convert an internal Order to the public OpenOrderView."""
    return OpenOrderView(
        order_id=order.order_id,
        side=order.side,
        price=order.price,
        quantity=order.quantity,
        filled=order.filled,
        remaining=order.remaining,
    )
