import random
import time

import pytest

from market_env.order_book import (
    MAX_PRICE,
    MAX_QUANTITY,
    TICK_SIZE,
    OrderBook,
    round_to_tick,
)


# ---------------------------------------------------------------------------
# Basic matching
# ---------------------------------------------------------------------------

class TestBasicMatching:
    def test_crossing_buy_executes_at_resting_ask_price(self):
        """Buy at 51 against resting ask at 50 → trade executes at 50 (resting price)."""
        book = OrderBook()
        book.place_limit_order("seller", "sell", price=50.0, quantity=10)
        result = book.place_limit_order("buyer", "buy", price=51.0, quantity=10)

        assert result.status == "filled"
        assert result.filled_quantity == 10
        assert len(result.trades) == 1
        assert result.trades[0].price == 50.0
        assert result.trades[0].quantity == 10
        assert result.trades[0].buyer_id == "buyer"
        assert result.trades[0].seller_id == "seller"
        assert result.trades[0].aggressor_side == "buy"

    def test_crossing_sell_executes_at_resting_bid_price(self):
        """Sell at 49 against resting bid at 50 → trade executes at 50 (resting price)."""
        book = OrderBook()
        book.place_limit_order("buyer", "buy", price=50.0, quantity=10)
        result = book.place_limit_order("seller", "sell", price=49.0, quantity=10)

        assert result.status == "filled"
        assert result.filled_quantity == 10
        assert result.trades[0].price == 50.0
        assert result.trades[0].aggressor_side == "sell"

    def test_non_crossing_orders_both_rest(self):
        """Buy at 49, sell at 51 — no match, both rest on the book."""
        book = OrderBook()
        r1 = book.place_limit_order("buyer", "buy", price=49.0, quantity=5)
        r2 = book.place_limit_order("seller", "sell", price=51.0, quantity=5)

        assert r1.status == "resting"
        assert r2.status == "resting"
        assert r1.trades == []
        assert r2.trades == []

        snap = book.get_snapshot()
        assert snap.bids[0].price == 49.0
        assert snap.asks[0].price == 51.0

    def test_empty_book_order_rests_without_error(self):
        """Placing an order on an empty book rests cleanly."""
        book = OrderBook()
        result = book.place_limit_order("a1", "buy", price=50.0, quantity=10)

        assert result.status == "resting"
        assert result.filled_quantity == 0
        assert result.trades == []


# ---------------------------------------------------------------------------
# Partial fills
# ---------------------------------------------------------------------------

class TestPartialFills:
    def test_buy_larger_than_resting_ask_leaves_remainder_on_book(self):
        book = OrderBook()
        book.place_limit_order("seller", "sell", price=50.0, quantity=5)
        result = book.place_limit_order("buyer", "buy", price=50.0, quantity=10)

        assert result.status == "partial"
        assert result.filled_quantity == 5

        snap = book.get_snapshot()
        assert snap.asks == []
        assert snap.bids[0].price == 50.0
        assert snap.bids[0].quantity == 5

    def test_resting_ask_partially_filled_has_correct_remaining(self):
        book = OrderBook()
        sell_result = book.place_limit_order("seller", "sell", price=50.0, quantity=10)
        book.place_limit_order("buyer", "buy", price=50.0, quantity=3)

        order = book.get_order(sell_result.order_id)
        assert order.filled == 3
        assert order.remaining == 7

    def test_large_buy_consumes_multiple_asks_in_price_order(self):
        book = OrderBook()
        book.place_limit_order("s1", "sell", price=50.0, quantity=5)
        book.place_limit_order("s2", "sell", price=50.5, quantity=5)
        book.place_limit_order("s3", "sell", price=51.0, quantity=5)
        result = book.place_limit_order("buyer", "buy", price=52.0, quantity=12)

        assert result.filled_quantity == 12
        assert len(result.trades) == 3
        assert result.trades[0].price == 50.0
        assert result.trades[1].price == 50.5
        assert result.trades[2].price == 51.0
        snap = book.get_snapshot()
        assert snap.asks[0].price == 51.0
        assert snap.asks[0].quantity == 3

    def test_buy_that_clears_book_rests_unfilled_remainder(self):
        book = OrderBook()
        book.place_limit_order("s1", "sell", price=50.0, quantity=3)
        result = book.place_limit_order("buyer", "buy", price=50.0, quantity=10)

        assert result.status == "partial"
        assert result.filled_quantity == 3
        snap = book.get_snapshot()
        assert snap.asks == []
        assert snap.bids[0].quantity == 7

    def test_large_aggressor_fills_multiple_orders_at_same_price(self):
        """Sell of 10 fills two resting bids of 5 each at the same price, earliest first."""
        book = OrderBook()
        book.place_limit_order("b1", "buy", price=50.0, quantity=5)
        book.place_limit_order("b2", "buy", price=50.0, quantity=5)

        result = book.place_limit_order("seller", "sell", price=50.0, quantity=10)

        assert result.filled_quantity == 10
        assert len(result.trades) == 2
        assert result.trades[0].buyer_id == "b1"   # earlier seq
        assert result.trades[1].buyer_id == "b2"
        assert book.get_snapshot().bids == []      # both bids consumed


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestCancellation:
    def test_cancel_removes_order_from_snapshot(self):
        book = OrderBook()
        result = book.place_limit_order("a1", "buy", price=50.0, quantity=10)

        cancel = book.cancel_order("a1", result.order_id)
        assert cancel.success

        snap = book.get_snapshot()
        assert snap.bids == []

    def test_cancelled_ask_is_skipped_during_matching(self):
        book = OrderBook()
        sell = book.place_limit_order("seller", "sell", price=50.0, quantity=10)
        book.cancel_order("seller", sell.order_id)

        buy = book.place_limit_order("buyer", "buy", price=51.0, quantity=10)
        assert buy.status == "resting"
        assert buy.filled_quantity == 0

    def test_cancel_wrong_agent_fails(self):
        book = OrderBook()
        result = book.place_limit_order("agent1", "buy", price=50.0, quantity=10)

        cancel = book.cancel_order("agent2", result.order_id)
        assert not cancel.success
        assert "not your order" in cancel.error

    def test_cancel_nonexistent_order_fails(self):
        book = OrderBook()
        cancel = book.cancel_order("a1", "does-not-exist")
        assert not cancel.success
        assert "not found" in cancel.error

    def test_cancel_fully_filled_order_fails(self):
        book = OrderBook()
        sell = book.place_limit_order("seller", "sell", price=50.0, quantity=5)
        book.place_limit_order("buyer", "buy", price=50.0, quantity=5)

        cancel = book.cancel_order("seller", sell.order_id)
        assert not cancel.success
        assert "already done" in cancel.error

    def test_multiple_cancels_same_order_fails_second_time(self):
        book = OrderBook()
        result = book.place_limit_order("a1", "sell", price=50.0, quantity=5)
        book.cancel_order("a1", result.order_id)
        second = book.cancel_order("a1", result.order_id)
        assert not second.success


# ---------------------------------------------------------------------------
# Price-time priority
# ---------------------------------------------------------------------------

class TestPriceTimePriority:
    def test_higher_bid_fills_before_lower_bid(self):
        book = OrderBook()
        book.place_limit_order("b_low", "buy", price=49.0, quantity=5)
        book.place_limit_order("b_high", "buy", price=50.0, quantity=5)

        result = book.place_limit_order("seller", "sell", price=49.0, quantity=5)

        assert result.trades[0].buyer_id == "b_high"
        assert result.trades[0].price == 50.0

    def test_lower_ask_fills_before_higher_ask(self):
        book = OrderBook()
        book.place_limit_order("s_high", "sell", price=51.0, quantity=5)
        book.place_limit_order("s_low", "sell", price=50.0, quantity=5)

        result = book.place_limit_order("buyer", "buy", price=51.0, quantity=5)

        assert result.trades[0].seller_id == "s_low"
        assert result.trades[0].price == 50.0

    def test_time_priority_same_price_bids(self):
        book = OrderBook()
        book.place_limit_order("b1", "buy", price=50.0, quantity=5)
        book.place_limit_order("b2", "buy", price=50.0, quantity=5)

        result = book.place_limit_order("seller", "sell", price=50.0, quantity=5)

        assert len(result.trades) == 1
        assert result.trades[0].buyer_id == "b1"

    def test_time_priority_same_price_asks(self):
        book = OrderBook()
        book.place_limit_order("s1", "sell", price=50.0, quantity=5)
        book.place_limit_order("s2", "sell", price=50.0, quantity=5)

        result = book.place_limit_order("buyer", "buy", price=50.0, quantity=5)

        assert len(result.trades) == 1
        assert result.trades[0].seller_id == "s1"


# ---------------------------------------------------------------------------
# Snapshot depth and aggregation
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_depth_parameter_limits_levels_returned(self):
        book = OrderBook()
        for i, p in enumerate([49.0, 48.0, 47.0, 46.0]):
            book.place_limit_order(f"b{i}", "buy", price=p, quantity=5)
        for i, p in enumerate([51.0, 52.0, 53.0, 54.0]):
            book.place_limit_order(f"s{i}", "sell", price=p, quantity=5)

        snap = book.get_snapshot(depth=2)
        assert len(snap.bids) == 2
        assert len(snap.asks) == 2
        assert snap.bids[0].price == 49.0
        assert snap.bids[1].price == 48.0
        assert snap.asks[0].price == 51.0
        assert snap.asks[1].price == 52.0

    def test_orders_at_same_price_aggregate_into_one_level(self):
        book = OrderBook()
        book.place_limit_order("b1", "buy", price=50.0, quantity=5)
        book.place_limit_order("b2", "buy", price=50.0, quantity=7)

        snap = book.get_snapshot()
        assert len(snap.bids) == 1
        assert snap.bids[0].quantity == 12
        assert snap.bids[0].num_orders == 2

    def test_mid_price_and_spread_computed_correctly(self):
        book = OrderBook()
        book.place_limit_order("buyer", "buy", price=49.0, quantity=5)
        book.place_limit_order("seller", "sell", price=51.0, quantity=5)

        snap = book.get_snapshot()
        assert snap.mid_price == 50.0
        assert snap.spread == 2.0

    def test_empty_book_snapshot_has_zero_mid_and_spread(self):
        snap = OrderBook().get_snapshot()
        assert snap.bids == []
        assert snap.asks == []
        assert snap.mid_price == 0.0
        assert snap.spread == 0.0

    def test_bids_only_snapshot(self):
        book = OrderBook()
        book.place_limit_order("b1", "buy", price=50.0, quantity=5)
        snap = book.get_snapshot()
        assert snap.mid_price == 50.0
        assert snap.spread == 0.0

    def test_cancelled_orders_excluded_from_snapshot(self):
        book = OrderBook()
        r = book.place_limit_order("b1", "buy", price=50.0, quantity=10)
        book.place_limit_order("b2", "buy", price=50.0, quantity=5)
        book.cancel_order("b1", r.order_id)

        snap = book.get_snapshot()
        assert snap.bids[0].quantity == 5
        assert snap.bids[0].num_orders == 1


# ---------------------------------------------------------------------------
# Recent trades
# ---------------------------------------------------------------------------

class TestRecentTrades:
    def test_executed_trades_appear_in_recent_trades(self):
        book = OrderBook()
        book.place_limit_order("seller", "sell", price=50.0, quantity=10)
        book.place_limit_order("buyer", "buy", price=50.0, quantity=10)

        trades = book.get_recent_trades(n=10)
        assert len(trades) == 1
        assert trades[0].price == 50.0
        assert trades[0].quantity == 10

    def test_recent_trades_n_limit_respected(self):
        book = OrderBook()
        for i in range(5):
            book.place_limit_order(f"s{i}", "sell", price=50.0, quantity=1)
            book.place_limit_order(f"b{i}", "buy", price=50.0, quantity=1)

        assert len(book.get_recent_trades(n=3)) == 3
        assert len(book.get_recent_trades(n=10)) == 5

    def test_turn_recorded_on_trade(self):
        book = OrderBook()
        book.set_turn(7)
        book.place_limit_order("seller", "sell", price=50.0, quantity=5)
        book.place_limit_order("buyer", "buy", price=50.0, quantity=5)

        assert book.get_recent_trades()[0].turn == 7

    def test_no_trades_on_empty_book(self):
        assert OrderBook().get_recent_trades() == []


# ---------------------------------------------------------------------------
# Self-trade prevention
# ---------------------------------------------------------------------------

class TestSelfTradePrevention:
    def test_self_trade_prevented_by_default(self):
        """An agent's aggressive buy does not match their own resting sell."""
        book = OrderBook()
        book.place_limit_order("agent1", "sell", price=50.0, quantity=10)
        buy = book.place_limit_order("agent1", "buy", price=50.0, quantity=10)

        assert buy.status == "resting"
        assert buy.filled_quantity == 0
        assert buy.trades == []

        # Both orders remain on book at $50
        snap = book.get_snapshot()
        assert snap.bids[0].price == 50.0
        assert snap.bids[0].quantity == 10
        assert snap.asks[0].price == 50.0
        assert snap.asks[0].quantity == 10

    def test_self_trade_allowed_with_flag(self):
        """When allow_self_trade=True, same-agent orders match normally."""
        book = OrderBook(allow_self_trade=True)
        book.place_limit_order("agent1", "sell", price=50.0, quantity=10)
        buy = book.place_limit_order("agent1", "buy", price=50.0, quantity=10)

        assert buy.status == "filled"
        assert buy.filled_quantity == 10
        assert buy.trades[0].buyer_id == "agent1"
        assert buy.trades[0].seller_id == "agent1"

    def test_self_trade_prevention_skips_self_order_but_matches_others(self):
        """Self-trade prevention skips over self-order and matches next non-self order."""
        book = OrderBook()
        # agent1's own sell at $50 (best price, would normally fill first)
        book.place_limit_order("agent1", "sell", price=50.0, quantity=5)
        # agent2's sell at $50.50
        book.place_limit_order("agent2", "sell", price=50.50, quantity=5)

        # agent1 buys at $51 — should skip own $50 sell and fill agent2's $50.50
        result = book.place_limit_order("agent1", "buy", price=51.0, quantity=5)

        assert result.filled_quantity == 5
        assert len(result.trades) == 1
        assert result.trades[0].seller_id == "agent2"
        assert result.trades[0].price == 50.50

    def test_self_match_orders_remain_on_book_after_matching(self):
        """Self-match-skipped orders are restored to the book (not cancelled)."""
        book = OrderBook()
        r = book.place_limit_order("agent1", "sell", price=50.0, quantity=10)
        book.place_limit_order("agent1", "buy", price=51.0, quantity=10)

        # The original sell must still rest on the book, unchanged
        order = book.get_order(r.order_id)
        assert order.remaining == 10
        assert not order.is_done

        snap = book.get_snapshot()
        assert snap.asks[0].price == 50.0
        assert snap.asks[0].quantity == 10


# ---------------------------------------------------------------------------
# Tick size / price rounding
# ---------------------------------------------------------------------------

class TestTickRounding:
    def test_price_rounded_to_tick_on_entry(self):
        """Non-tick-aligned prices are rounded to the nearest cent."""
        book = OrderBook()
        r = book.place_limit_order("a1", "buy", price=49.999, quantity=5)
        assert book.get_order(r.order_id).price == 50.0

    def test_rounded_prices_aggregate_at_same_level(self):
        """Orders at 49.999 and 50.0 both land at $50.00 in the snapshot."""
        book = OrderBook()
        book.place_limit_order("b1", "buy", price=49.999, quantity=5)
        book.place_limit_order("b2", "buy", price=50.0, quantity=5)

        snap = book.get_snapshot()
        assert len(snap.bids) == 1
        assert snap.bids[0].price == 50.0
        assert snap.bids[0].quantity == 10
        assert snap.bids[0].num_orders == 2

    def test_round_to_tick_utility(self):
        assert round_to_tick(49.994) == 49.99
        assert round_to_tick(49.995) == pytest.approx(50.00, abs=1e-9)
        assert round_to_tick(50.001) == 50.00
        assert round_to_tick(TICK_SIZE) == TICK_SIZE


# ---------------------------------------------------------------------------
# Agent queries
# ---------------------------------------------------------------------------

class TestAgentQueries:
    def test_get_open_orders_returns_agent_resting_orders(self):
        book = OrderBook()
        book.place_limit_order("a1", "buy", price=49.0, quantity=5)
        book.place_limit_order("a1", "sell", price=51.0, quantity=5)
        book.place_limit_order("a2", "buy", price=49.5, quantity=5)

        a1_open = book.get_open_orders_for_agent("a1")
        assert len(a1_open) == 2
        assert all(o.agent_id == "a1" for o in a1_open)

    def test_get_open_orders_excludes_filled(self):
        book = OrderBook()
        book.place_limit_order("a1", "sell", price=50.0, quantity=10)
        book.place_limit_order("a2", "buy", price=50.0, quantity=10)   # fills a1's sell

        assert book.get_open_orders_for_agent("a1") == []
        assert book.get_open_orders_for_agent("a2") == []

    def test_get_open_orders_excludes_cancelled(self):
        book = OrderBook()
        r = book.place_limit_order("a1", "buy", price=50.0, quantity=5)
        book.cancel_order("a1", r.order_id)
        assert book.get_open_orders_for_agent("a1") == []

    def test_get_open_orders_for_unknown_agent_returns_empty(self):
        book = OrderBook()
        book.place_limit_order("a1", "buy", price=50.0, quantity=5)
        assert book.get_open_orders_for_agent("nobody") == []


# ---------------------------------------------------------------------------
# Meta operations (__len__, __repr__)
# ---------------------------------------------------------------------------

class TestMetaOperations:
    def test_len_counts_only_active_orders(self):
        book = OrderBook()
        assert len(book) == 0

        book.place_limit_order("a1", "buy", price=49.0, quantity=5)
        assert len(book) == 1

        book.place_limit_order("a1", "sell", price=51.0, quantity=5)
        assert len(book) == 2

        # A cross-agent trade fills one order; the other remains
        book.place_limit_order("a2", "buy", price=51.0, quantity=5)   # fills a1's sell
        assert len(book) == 1   # only the $49 buy remains

    def test_repr_shows_book_state(self):
        book = OrderBook()
        book.place_limit_order("b1", "buy", price=49.0, quantity=5)
        book.place_limit_order("s1", "sell", price=51.0, quantity=5)
        r = repr(book)
        assert "OrderBook" in r
        assert "49" in r
        assert "51" in r


# ---------------------------------------------------------------------------
# Stress / invariant test
# ---------------------------------------------------------------------------

class TestStress:
    def test_1000_random_operations_no_crash_or_invariant_violation(self):
        """1000 random place/cancel operations maintain all book invariants."""
        rng = random.Random(42)
        book = OrderBook()
        live: list[tuple[str, str]] = []     # (agent_id, order_id)

        for _ in range(1000):
            if live and rng.random() < 0.3:
                agent_id, order_id = rng.choice(live)
                book.cancel_order(agent_id, order_id)
            else:
                agent = f"a{rng.randint(1, 5)}"
                side = rng.choice(["buy", "sell"])
                price = round(rng.uniform(45.0, 55.0), 2)
                qty = rng.randint(1, 20)
                result = book.place_limit_order(agent, side, price, qty)
                assert result.order_id in book._orders
                if result.status in ("resting", "partial"):
                    live.append((agent, result.order_id))

        # Invariant: every resting level has positive quantity and order count
        snap = book.get_snapshot(depth=1000)
        for level in snap.bids + snap.asks:
            assert level.quantity > 0
            assert level.num_orders > 0

        # Invariant: bids descending, asks ascending
        bid_prices = [lvl.price for lvl in snap.bids]
        ask_prices = [lvl.price for lvl in snap.asks]
        assert bid_prices == sorted(bid_prices, reverse=True)
        assert ask_prices == sorted(ask_prices)

        # Invariant: best bid < best ask (when both sides populated)
        if snap.bids and snap.asks:
            assert snap.bids[0].price < snap.asks[0].price

    def test_1000_orders_complete_in_under_1_second_with_matching(self):
        """Performance: 1000 operations across 10 agents (real matching) under 1s."""
        rng = random.Random(0)
        book = OrderBook()
        start = time.perf_counter()
        for i in range(1000):
            book.place_limit_order(
                f"agent{i % 10}",                         # 10 agents → real matching
                rng.choice(["buy", "sell"]),
                round(rng.uniform(48.0, 52.0), 2),
                rng.randint(1, 10),
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"1000 orders took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_side_raises_value_error(self):
        with pytest.raises(ValueError, match="invalid side"):
            OrderBook().place_limit_order("a1", "hold", price=50.0, quantity=10)

    def test_zero_quantity_raises_value_error(self):
        with pytest.raises(ValueError, match="quantity"):
            OrderBook().place_limit_order("a1", "buy", price=50.0, quantity=0)

    def test_negative_quantity_raises_value_error(self):
        with pytest.raises(ValueError, match="quantity"):
            OrderBook().place_limit_order("a1", "buy", price=50.0, quantity=-5)

    def test_non_int_quantity_rejected(self):
        with pytest.raises(ValueError, match="int"):
            OrderBook().place_limit_order("a1", "buy", price=50.0, quantity=10.5)

    def test_bool_quantity_rejected(self):
        """bool is a subclass of int in Python — must be explicitly rejected."""
        with pytest.raises(ValueError, match="int"):
            OrderBook().place_limit_order("a1", "buy", price=50.0, quantity=True)

    def test_zero_price_raises_value_error(self):
        with pytest.raises(ValueError, match="price"):
            OrderBook().place_limit_order("a1", "buy", price=0.0, quantity=10)

    def test_negative_price_raises_value_error(self):
        with pytest.raises(ValueError, match="price"):
            OrderBook().place_limit_order("a1", "buy", price=-1.0, quantity=10)

    def test_non_numeric_price_rejected(self):
        with pytest.raises(ValueError, match="price"):
            OrderBook().place_limit_order("a1", "buy", price="50", quantity=10)

    def test_nan_price_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            OrderBook().place_limit_order("a1", "buy", price=float("nan"), quantity=5)

    def test_positive_inf_price_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            OrderBook().place_limit_order("a1", "buy", price=float("inf"), quantity=5)

    def test_negative_inf_price_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            OrderBook().place_limit_order("a1", "buy", price=float("-inf"), quantity=5)

    def test_exceeding_max_price_rejected(self):
        with pytest.raises(ValueError, match="MAX_PRICE"):
            OrderBook().place_limit_order("a1", "buy", price=MAX_PRICE + 1, quantity=5)

    def test_exceeding_max_quantity_rejected(self):
        with pytest.raises(ValueError, match="MAX_QUANTITY"):
            OrderBook().place_limit_order("a1", "buy", price=50.0, quantity=MAX_QUANTITY + 1)

    def test_empty_agent_id_rejected(self):
        with pytest.raises(ValueError, match="agent_id"):
            OrderBook().place_limit_order("", "buy", price=50.0, quantity=5)

    def test_non_string_agent_id_rejected(self):
        with pytest.raises(ValueError, match="agent_id"):
            OrderBook().place_limit_order(None, "buy", price=50.0, quantity=5)
