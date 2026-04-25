"""
Scripted baseline bots for Stage 1 training and evaluation.

All bots implement the same interface:
    bot.act(observation: MarketObservation) -> MarketAction
    bot.reset() -> None       # called at episode start

Five archetypes:
    RandomBot        — random orders ±$3 around mid (baseline liquidity, no info)
    MomentumBot      — buys on uptrend, sells on downtrend (loses money in mean-revert)
    MeanReversionBot — bets against $1+ deviations from initial mid
    InformedBot      — observes true_value + Gaussian noise, trades aggressively
                       toward the estimate. The "smart money" baseline.
    MarketMakerBot   — alternates posting bid then ask each turn at ±half_spread

InformedBot exposes set_true_value(v) which the environment must call at episode
start. The bot does not read true_value from the observation — that field is
None during the episode.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from market_env.models import MarketAction, MarketObservation


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Bot(ABC):
    """Common interface for all scripted bots."""

    def __init__(self, agent_id: str, seed: Optional[int] = None) -> None:
        self.agent_id = agent_id
        self._rng = random.Random(seed)

    @abstractmethod
    def act(self, observation: MarketObservation) -> MarketAction:
        """Decide on an action given the current observation."""
        ...

    def reset(self) -> None:
        """Reset internal state at the start of an episode. No-op by default."""


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

class RandomBot(Bot):
    """Submits random buy/sell orders around the mid price."""

    def __init__(
        self,
        agent_id: str,
        seed: Optional[int] = None,
        action_prob: float = 0.6,
        price_jitter: float = 3.0,
        max_qty: int = 30,
    ) -> None:
        super().__init__(agent_id, seed)
        self.action_prob = action_prob
        self.price_jitter = price_jitter
        self.max_qty = max_qty

    def act(self, observation: MarketObservation) -> MarketAction:
        if self._rng.random() > self.action_prob:
            return MarketAction(action_type="hold")
        side = self._rng.choice(["buy", "sell"])
        mid = observation.order_book.mid_price or 50.0
        offset = self._rng.uniform(-self.price_jitter, self.price_jitter)
        price = round(max(0.01, mid + offset), 2)
        qty = self._rng.randint(1, self.max_qty)
        return MarketAction(action_type=side, price=price, quantity=qty)


class MomentumBot(Bot):
    """Buys when recent trades trend up, sells when they trend down.

    Counts up-moves and down-moves in the last `lookback` trades. If at least
    3 out of (lookback - 1) consecutive moves are in one direction, take a
    position in that direction.
    """

    def __init__(
        self,
        agent_id: str,
        seed: Optional[int] = None,
        lookback: int = 5,
        threshold: int = 3,
        edge: float = 0.10,
        qty: int = 10,
    ) -> None:
        super().__init__(agent_id, seed)
        self.lookback = lookback
        self.threshold = threshold
        self.edge = edge
        self.qty = qty

    def act(self, observation: MarketObservation) -> MarketAction:
        trades = observation.recent_trades[-self.lookback:]
        if len(trades) < self.threshold:
            return MarketAction(action_type="hold")
        prices = [t.price for t in trades]
        ups = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i - 1])
        downs = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i - 1])
        ob = observation.order_book

        if ups >= self.threshold and ob.asks:
            return MarketAction(
                action_type="buy",
                price=round(ob.asks[0].price + self.edge, 2),
                quantity=self.qty,
            )
        if downs >= self.threshold and ob.bids:
            return MarketAction(
                action_type="sell",
                price=round(ob.bids[0].price - self.edge, 2),
                quantity=self.qty,
            )
        return MarketAction(action_type="hold")


class MeanReversionBot(Bot):
    """Bets against significant deviations from a reference anchor price."""

    def __init__(
        self,
        agent_id: str,
        seed: Optional[int] = None,
        anchor: float = 50.0,
        threshold: float = 1.0,
        qty: int = 10,
    ) -> None:
        super().__init__(agent_id, seed)
        self.anchor = anchor
        self.threshold = threshold
        self.qty = qty

    def act(self, observation: MarketObservation) -> MarketAction:
        ob = observation.order_book
        mid = ob.mid_price
        if mid <= 0:
            return MarketAction(action_type="hold")
        if mid > self.anchor + self.threshold and ob.bids:
            return MarketAction(
                action_type="sell",
                price=round(ob.bids[0].price, 2),
                quantity=self.qty,
            )
        if mid < self.anchor - self.threshold and ob.asks:
            return MarketAction(
                action_type="buy",
                price=round(ob.asks[0].price, 2),
                quantity=self.qty,
            )
        return MarketAction(action_type="hold")


class InformedBot(Bot):
    """Observes the true value with Gaussian noise and trades toward the estimate.

    The environment must call set_true_value(v) at episode start to inject the
    noisy estimate. The bot does not read observation.true_value (which is None
    during the episode anyway).
    """

    def __init__(
        self,
        agent_id: str,
        seed: Optional[int] = None,
        noise_std: float = 0.40,
        edge: float = 0.50,
        qty: int = 10,
    ) -> None:
        super().__init__(agent_id, seed)
        self.noise_std = noise_std
        self.edge = edge
        self.qty = qty
        self._estimate: Optional[float] = None

    def set_true_value(self, true_value: float) -> None:
        """Called once at episode start to set this bot's noisy estimate."""
        self._estimate = true_value + self._rng.gauss(0.0, self.noise_std)

    def reset(self) -> None:
        self._estimate = None

    def act(self, observation: MarketObservation) -> MarketAction:
        if self._estimate is None:
            return MarketAction(action_type="hold")
        ob = observation.order_book
        if ob.asks and ob.asks[0].price < self._estimate - self.edge:
            return MarketAction(
                action_type="buy",
                price=round(ob.asks[0].price, 2),
                quantity=self.qty,
            )
        if ob.bids and ob.bids[0].price > self._estimate + self.edge:
            return MarketAction(
                action_type="sell",
                price=round(ob.bids[0].price, 2),
                quantity=self.qty,
            )
        return MarketAction(action_type="hold")


class MarketMakerBot(Bot):
    """Posts limit quotes at ±half_spread around mid (or anchor if book is empty).

    Alternates between buy and sell each turn since the action schema permits
    only one action per turn. Provides baseline liquidity but loses to informed
    traders via adverse selection.
    """

    def __init__(
        self,
        agent_id: str,
        seed: Optional[int] = None,
        anchor: float = 50.0,
        half_spread: float = 0.30,
        qty: int = 10,
    ) -> None:
        super().__init__(agent_id, seed)
        self.anchor = anchor
        self.half_spread = half_spread
        self.qty = qty
        self._tick: int = 0

    def act(self, observation: MarketObservation) -> MarketAction:
        ob = observation.order_book
        center = ob.mid_price if ob.mid_price > 0 else self.anchor
        self._tick += 1
        if self._tick % 2 == 1:
            return MarketAction(
                action_type="buy",
                price=round(center - self.half_spread, 2),
                quantity=self.qty,
            )
        return MarketAction(
            action_type="sell",
            price=round(center + self.half_spread, 2),
            quantity=self.qty,
        )

    def reset(self) -> None:
        self._tick = 0
