"""Unit tests for the reward computation."""
from __future__ import annotations

import pytest

from market_env.reward import (
    EMPTY_ACTION_PENALTY,
    EMPTY_ACTION_THRESHOLD,
    INITIAL_CASH,
    PARSE_FAILURE_PENALTY,
    PARTICIPATION_BONUS,
    PARTICIPATION_THRESHOLD,
    PNL_SCALE,
    POSITION_LIMIT,
    POSITION_LIMIT_PENALTY,
    QUOTE_STUFFING_PENALTY,
    QUOTE_STUFFING_THRESHOLD,
    AgentStats,
    compute_reward,
)


def _stats(**kwargs) -> AgentStats:
    return AgentStats(**kwargs)


# ---------------------------------------------------------------------------
# Raw P&L math
# ---------------------------------------------------------------------------

class TestRawPnL:
    def test_zero_position_zero_cash_change_means_zero_pnl(self):
        b = compute_reward(INITIAL_CASH, 0, 50.0, _stats())
        assert b.raw_pnl == 0.0

    def test_long_position_marked_to_true_value(self):
        # Bought 10 shares at $50 → cash -500, then mark at true_value=$60 → +$100 P&L
        b = compute_reward(INITIAL_CASH - 500, 10, 60.0, _stats())
        assert b.raw_pnl == pytest.approx(100.0)

    def test_short_position_profits_when_true_value_below_sell_price(self):
        # Sold 10 at $50 → cash +500, then mark at true_value=$40 → +$100 P&L
        b = compute_reward(INITIAL_CASH + 500, -10, 40.0, _stats())
        assert b.raw_pnl == pytest.approx(100.0)

    def test_zero_true_value_safe_normalization(self):
        b = compute_reward(INITIAL_CASH, 0, 0.0, _stats())
        # Should not divide by zero
        assert b.pnl_normalized == 0.0


# ---------------------------------------------------------------------------
# Normalization and scaling
# ---------------------------------------------------------------------------

class TestNormalizationAndScale:
    def test_pnl_normalized_uses_true_value_times_100(self):
        # raw_pnl=$500, true_value=$50 → normalized = 500 / 5000 = 0.10
        b = compute_reward(INITIAL_CASH + 500, 0, 50.0, _stats())
        assert b.pnl_normalized == pytest.approx(0.10)

    def test_pnl_scaled_is_normalized_times_5(self):
        b = compute_reward(INITIAL_CASH + 500, 0, 50.0, _stats())
        assert b.pnl_scaled == pytest.approx(0.10 * PNL_SCALE)


# ---------------------------------------------------------------------------
# Participation bonus
# ---------------------------------------------------------------------------

class TestParticipationBonus:
    def test_bonus_applied_when_threshold_met(self):
        b = compute_reward(INITIAL_CASH, 0, 50.0, _stats(orders_placed=PARTICIPATION_THRESHOLD))
        assert b.participation_bonus == PARTICIPATION_BONUS

    def test_no_bonus_below_threshold(self):
        b = compute_reward(
            INITIAL_CASH, 0, 50.0,
            _stats(orders_placed=PARTICIPATION_THRESHOLD - 1),
        )
        assert b.participation_bonus == 0.0


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------

class TestPenalties:
    def test_quote_stuffing_penalty_at_threshold_plus_one(self):
        b = compute_reward(
            INITIAL_CASH, 0, 50.0,
            _stats(orders_cancelled=QUOTE_STUFFING_THRESHOLD + 1),
        )
        assert b.quote_stuffing_penalty == QUOTE_STUFFING_PENALTY

    def test_no_quote_stuffing_at_threshold(self):
        b = compute_reward(
            INITIAL_CASH, 0, 50.0,
            _stats(orders_cancelled=QUOTE_STUFFING_THRESHOLD),
        )
        assert b.quote_stuffing_penalty == 0.0

    def test_position_limit_penalty(self):
        b = compute_reward(
            INITIAL_CASH, 0, 50.0,
            _stats(max_abs_position=POSITION_LIMIT + 1),
        )
        assert b.position_limit_penalty == POSITION_LIMIT_PENALTY

    def test_empty_action_penalty(self):
        b = compute_reward(
            INITIAL_CASH, 0, 50.0,
            _stats(holds=EMPTY_ACTION_THRESHOLD + 1),
        )
        assert b.empty_action_penalty == EMPTY_ACTION_PENALTY

    def test_parse_failure_penalty_scales_linearly(self):
        b = compute_reward(
            INITIAL_CASH, 0, 50.0,
            _stats(parse_failures=4),
        )
        assert b.parse_failure_penalty == pytest.approx(4 * PARSE_FAILURE_PENALTY)


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

class TestClamping:
    def test_total_clamped_to_plus_one_for_huge_profit(self):
        b = compute_reward(50_000.0, 0, 50.0, _stats(orders_placed=10))
        assert b.total == 1.0
        assert b.total_unclamped > 1.0

    def test_total_clamped_to_minus_one_for_huge_loss(self):
        b = compute_reward(-50_000.0, 0, 50.0, _stats())
        assert b.total == -1.0

    def test_total_inside_bounds_when_unclamped_in_range(self):
        b = compute_reward(INITIAL_CASH + 200, 0, 50.0, _stats(orders_placed=5))
        # Normalized = 200 / 5000 = 0.04, scaled = 0.20, + 0.01 bonus = 0.21
        assert -1.0 < b.total < 1.0
        assert b.total == pytest.approx(b.total_unclamped)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

class TestComposition:
    def test_winning_active_agent_above_baseline(self):
        """A profitable active agent scores higher than a profitable inactive one."""
        active = compute_reward(
            INITIAL_CASH + 200, 0, 50.0, _stats(orders_placed=10),
        )
        inactive = compute_reward(
            INITIAL_CASH + 200, 0, 50.0, _stats(orders_placed=0, holds=40),
        )
        assert active.total > inactive.total

    def test_pnl_dominates_auxiliary_signals(self):
        """The ×5 scale ensures big profit moves the reward more than bonuses do."""
        big_profit = compute_reward(
            INITIAL_CASH + 1000, 0, 50.0, _stats(),
        )
        small_profit_with_bonus = compute_reward(
            INITIAL_CASH + 100, 0, 50.0, _stats(orders_placed=5),
        )
        assert big_profit.total > small_profit_with_bonus.total
