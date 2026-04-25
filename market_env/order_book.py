"""
Continuous double-auction limit order book with price-time priority.

Core invariants:
- Price-time priority: best price wins; ties broken by earliest sequence number.
- Trades execute at the resting order's price (standard market convention).
- Partial fills leave the remainder resting on the book.
- Cancelled and filled orders are lazily removed from the heap on access.
- Prices are rounded to TICK_SIZE at entry. This prevents float equality bugs
  in snapshot aggregation when LLM outputs produce noisy prices like 49.9999.
- Self-trades are prevented by default: an agent's aggressive order does not
  match against their own resting orders. Same-agent resting orders are
  temporarily set aside during matching and restored afterward. This blocks
  wash-trading as a reward-hacking vector during RL training.
"""

from __future__ import annotations

import heapq
import math
import uuid
from dataclasses import dataclass
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TICK_SIZE: float = 0.01           # price rounding granularity (penny ticks)
MAX_PRICE: float = 1000.0         # sanity bound to reject runaway model output
MAX_QUANTITY: int = 10_000        # sanity bound on order quantity

Side = Literal["buy", "sell"]
OrderStatus = Literal["resting", "partial", "filled"]


def round_to_tick(price: float, tick: float = TICK_SIZE) -> float:
    """Round a price to the nearest tick. Ensures stable price-level grouping."""
    return round(price / tick) * tick


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Order:
    order_id: str
    agent_id: str
    side: Side
    price: float
    quantity: int
    seq: int               # monotonically increasing insertion counter
    filled: int = 0
    cancelled: bool = False

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled

    @property
    def is_done(self) -> bool:
        return self.cancelled or self.remaining <= 0


@dataclass
class TradeRecord:
    trade_id: str
    price: float
    quantity: int
    buyer_id: str
    seller_id: str
    aggressor_side: Side    # which side initiated the trade
    turn: int


@dataclass
class OrderResult:
    order_id: str
    status: OrderStatus
    filled_quantity: int
    trades: list[TradeRecord]


@dataclass
class CancelResult:
    order_id: str
    success: bool
    error: Optional[str] = None


@dataclass
class OrderBookLevel:
    price: float
    quantity: int           # total resting quantity at this price level
    num_orders: int         # number of distinct resting orders at this level


@dataclass
class OrderBookSnapshot:
    bids: list[OrderBookLevel]    # descending by price
    asks: list[OrderBookLevel]    # ascending by price
    mid_price: float
    spread: float


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------

class OrderBook:
    """
    Continuous double-auction limit order book.

    Heap layout:
        bids: (-price, seq, order_id)   — min-heap top = highest price, earliest
        asks: ( price, seq, order_id)   — min-heap top = lowest price, earliest

    Args:
        allow_self_trade: If False (default), an agent cannot match against
            their own resting orders. Those orders are temporarily held aside
            during matching and restored to the book afterward. Keeping this
            at False is important for RL training — wash trades would let an
            agent inflate volume/participation signals without real trading.
    """

    def __init__(self, allow_self_trade: bool = False) -> None:
        self.allow_self_trade: bool = allow_self_trade
        self._bids: list[tuple] = []
        self._asks: list[tuple] = []
        self._orders: dict[str, Order] = {}
        self._recent_trades: list[TradeRecord] = []
        self._seq: int = 0
        self._trade_counter: int = 0
        self._turn: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_turn(self, turn: int) -> None:
        """Set the current turn; trades executed after this are tagged with it."""
        self._turn = turn

    def place_limit_order(
        self,
        agent_id: str,
        side: Side,
        price: float,
        quantity: int,
    ) -> OrderResult:
        """
        Place a limit order. Matches immediately against crossing opposing orders;
        any unfilled remainder rests on the book.

        Prices are rounded to the nearest tick (TICK_SIZE) at entry.

        Raises:
            ValueError: on any malformed input (empty agent_id, unknown side,
                NaN/Inf/non-positive/out-of-range price, non-int/non-positive/
                out-of-range quantity).
        """
        self._validate_input(agent_id, side, price, quantity)
        price = round_to_tick(price)

        self._seq += 1
        order = Order(
            order_id=uuid.uuid4().hex[:8],
            agent_id=agent_id,
            side=side,
            price=price,
            quantity=quantity,
            seq=self._seq,
        )
        self._orders[order.order_id] = order

        trades = self._match(order)

        if order.remaining > 0:
            if side == "buy":
                heapq.heappush(self._bids, (-price, order.seq, order.order_id))
            else:
                heapq.heappush(self._asks, (price, order.seq, order.order_id))

        if order.filled == 0:
            status: OrderStatus = "resting"
        elif order.remaining > 0:
            status = "partial"
        else:
            status = "filled"

        return OrderResult(
            order_id=order.order_id,
            status=status,
            filled_quantity=order.filled,
            trades=trades,
        )

    def cancel_order(self, agent_id: str, order_id: str) -> CancelResult:
        """Cancel a resting order. Agents can only cancel their own orders."""
        order = self._orders.get(order_id)
        if order is None:
            return CancelResult(order_id=order_id, success=False, error="order not found")
        if order.agent_id != agent_id:
            return CancelResult(order_id=order_id, success=False, error="not your order")
        if order.is_done:
            return CancelResult(order_id=order_id, success=False, error="order already done")
        order.cancelled = True
        return CancelResult(order_id=order_id, success=True)

    def get_snapshot(self, depth: int = 5) -> OrderBookSnapshot:
        """Return top-N aggregated bid and ask levels, plus mid and spread."""
        bid_map: dict[float, list[int]] = {}    # price -> [total_qty, count]
        ask_map: dict[float, list[int]] = {}

        for order in self._orders.values():
            if order.is_done:
                continue
            target = bid_map if order.side == "buy" else ask_map
            agg = target.setdefault(order.price, [0, 0])
            agg[0] += order.remaining
            agg[1] += 1

        bids = [
            OrderBookLevel(price=p, quantity=q, num_orders=n)
            for p, (q, n) in sorted(bid_map.items(), reverse=True)
        ][:depth]
        asks = [
            OrderBookLevel(price=p, quantity=q, num_orders=n)
            for p, (q, n) in sorted(ask_map.items())
        ][:depth]

        best_bid = bids[0].price if bids else None
        best_ask = asks[0].price if asks else None

        if best_bid is not None and best_ask is not None:
            mid, spread = (best_bid + best_ask) / 2, best_ask - best_bid
        elif best_bid is not None:
            mid, spread = best_bid, 0.0
        elif best_ask is not None:
            mid, spread = best_ask, 0.0
        else:
            mid, spread = 0.0, 0.0

        return OrderBookSnapshot(bids=bids, asks=asks, mid_price=mid, spread=spread)

    def get_recent_trades(self, n: int = 10) -> list[TradeRecord]:
        """Return the most recent n trades (oldest first)."""
        return self._recent_trades[-n:]

    def get_order(self, order_id: str) -> Optional[Order]:
        """Look up an order by ID regardless of status."""
        return self._orders.get(order_id)

    def get_open_orders_for_agent(self, agent_id: str) -> list[Order]:
        """Return all resting (non-done) orders belonging to the given agent."""
        return [
            o for o in self._orders.values()
            if o.agent_id == agent_id and not o.is_done
        ]

    def __len__(self) -> int:
        """Number of active (non-done) orders currently on the book."""
        return sum(1 for o in self._orders.values() if not o.is_done)

    def __repr__(self) -> str:
        snap = self.get_snapshot(depth=1)
        best_bid = snap.bids[0].price if snap.bids else None
        best_ask = snap.asks[0].price if snap.asks else None
        return (
            f"OrderBook(active={len(self)}, best_bid={best_bid}, "
            f"best_ask={best_ask}, turn={self._turn})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(agent_id: str, side: str, price: float, quantity: int) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if side not in ("buy", "sell"):
            raise ValueError(f"invalid side: {side!r} (must be 'buy' or 'sell')")
        if not isinstance(price, (int, float)) or isinstance(price, bool):
            raise ValueError(f"price must be a number, got {type(price).__name__}")
        if math.isnan(price) or math.isinf(price):
            raise ValueError(f"price must be a finite number, got {price!r}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if price > MAX_PRICE:
            raise ValueError(f"price {price} exceeds MAX_PRICE={MAX_PRICE}")
        if isinstance(quantity, bool) or not isinstance(quantity, int):
            raise ValueError(f"quantity must be an int, got {type(quantity).__name__}")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if quantity > MAX_QUANTITY:
            raise ValueError(f"quantity {quantity} exceeds MAX_QUANTITY={MAX_QUANTITY}")

    def _skip_dead(self, heap: list) -> None:
        """Lazy deletion: pop done/cancelled/missing orders from the heap top."""
        while heap:
            order = self._orders.get(heap[0][2])
            if order is None or order.is_done:
                heapq.heappop(heap)
            else:
                break

    def _best_ask(self) -> Optional[Order]:
        self._skip_dead(self._asks)
        return self._orders[self._asks[0][2]] if self._asks else None

    def _best_bid(self) -> Optional[Order]:
        self._skip_dead(self._bids)
        return self._orders[self._bids[0][2]] if self._bids else None

    def _next_trade_id(self) -> str:
        self._trade_counter += 1
        return f"T{self._trade_counter:06d}"

    def _match(self, aggressive: Order) -> list[TradeRecord]:
        """
        Match the aggressive order against the opposing side.

        Self-trade prevention (when enabled): same-agent resting orders are
        popped from the heap into a temporary buffer during matching and
        pushed back afterward, so they remain on the book but cannot match.
        """
        trades: list[TradeRecord] = []
        heap = self._asks if aggressive.side == "buy" else self._bids
        get_best = self._best_ask if aggressive.side == "buy" else self._best_bid
        skipped: list[tuple] = []

        def crosses(resting: Order) -> bool:
            if aggressive.side == "buy":
                return resting.price <= aggressive.price
            return resting.price >= aggressive.price

        while aggressive.remaining > 0:
            resting = get_best()
            if resting is None or not crosses(resting):
                break

            if not self.allow_self_trade and resting.agent_id == aggressive.agent_id:
                # Temporarily set aside; restore after matching completes.
                skipped.append(heapq.heappop(heap))
                continue

            qty = min(aggressive.remaining, resting.remaining)
            if aggressive.side == "buy":
                buyer, seller = aggressive.agent_id, resting.agent_id
            else:
                buyer, seller = resting.agent_id, aggressive.agent_id

            trade = TradeRecord(
                trade_id=self._next_trade_id(),
                price=resting.price,
                quantity=qty,
                buyer_id=buyer,
                seller_id=seller,
                aggressor_side=aggressive.side,
                turn=self._turn,
            )
            trades.append(trade)
            self._recent_trades.append(trade)
            aggressive.filled += qty
            resting.filled += qty
            if resting.is_done:
                heapq.heappop(heap)

        # Restore self-match skipped orders to the heap
        for entry in skipped:
            heapq.heappush(heap, entry)

        return trades
