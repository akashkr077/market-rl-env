"""
Reward computation for the market environment.

Formula (after calibration fixes — see planning.md issues 5 & 6):

    total = clamp(pnl_normalized * PNL_SCALE
                  + participation_bonus
                  - penalties,
                  -1.0, +1.0)

where:
    raw_pnl              = (cash_final - INITIAL_CASH) + shares_final * true_value
    pnl_normalized       = raw_pnl / (true_value * 100)     ← max position dollar scale

    participation_bonus  = +0.01  if num_orders_placed >= 3
    quote_stuffing_pen   = +0.10  if num_cancels > 15
    position_limit_pen   = +0.05  if max_abs_position > 150 at any point
    empty_action_pen     = +0.02  if num_holds > 30 (out of 50 turns)
    parse_failure_pen    = +0.05  per parse failure (incremented externally)

The PNL_SCALE = 5 is critical — see planning.md issue 5. Without scaling,
auxiliaries (each ~0.01-0.10) would dwarf the typical normalized P&L (~0.05-0.20)
and the agent would optimize for participation instead of profit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

INITIAL_CASH: float = 10_000.0

# ---------------------------------------------------------------------------
# Reward weights (calibrated; do not change without rerunning calibration)
# ---------------------------------------------------------------------------

PNL_SCALE: float = 5.0
PARTICIPATION_BONUS: float = 0.01
PARTICIPATION_THRESHOLD: int = 3
QUOTE_STUFFING_THRESHOLD: int = 15
QUOTE_STUFFING_PENALTY: float = 0.10
POSITION_LIMIT: int = 150
POSITION_LIMIT_PENALTY: float = 0.05
EMPTY_ACTION_THRESHOLD: int = 30
EMPTY_ACTION_PENALTY: float = 0.02
PARSE_FAILURE_PENALTY: float = 0.05


# ---------------------------------------------------------------------------
# Per-agent running stats (mutable, owned by the environment)
# ---------------------------------------------------------------------------

@dataclass
class AgentStats:
    orders_placed: int = 0
    orders_cancelled: int = 0
    holds: int = 0
    parse_failures: int = 0       # incremented by the training loop, not the env
    max_abs_position: int = 0


# ---------------------------------------------------------------------------
# Reward output
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    raw_pnl: float
    pnl_normalized: float
    pnl_scaled: float                     # pnl_normalized * PNL_SCALE
    participation_bonus: float
    quote_stuffing_penalty: float
    position_limit_penalty: float
    empty_action_penalty: float
    parse_failure_penalty: float
    total_unclamped: float
    total: float                          # clamped to [-1, 1]


def compute_reward(
    cash_final: float,
    shares_final: int,
    true_value: float,
    stats: AgentStats,
    initial_cash: float = INITIAL_CASH,
) -> RewardBreakdown:
    """End-of-episode reward for one agent."""
    raw_pnl = (cash_final - initial_cash) + shares_final * true_value

    if true_value > 0:
        pnl_normalized = raw_pnl / (true_value * 100)
    else:
        pnl_normalized = 0.0
    pnl_scaled = pnl_normalized * PNL_SCALE

    participation_bonus = (
        PARTICIPATION_BONUS
        if stats.orders_placed >= PARTICIPATION_THRESHOLD
        else 0.0
    )
    quote_stuffing_penalty = (
        QUOTE_STUFFING_PENALTY
        if stats.orders_cancelled > QUOTE_STUFFING_THRESHOLD
        else 0.0
    )
    position_limit_penalty = (
        POSITION_LIMIT_PENALTY
        if stats.max_abs_position > POSITION_LIMIT
        else 0.0
    )
    empty_action_penalty = (
        EMPTY_ACTION_PENALTY
        if stats.holds > EMPTY_ACTION_THRESHOLD
        else 0.0
    )
    parse_failure_penalty = PARSE_FAILURE_PENALTY * stats.parse_failures

    total_unclamped = (
        pnl_scaled
        + participation_bonus
        - quote_stuffing_penalty
        - position_limit_penalty
        - empty_action_penalty
        - parse_failure_penalty
    )
    total = max(-1.0, min(1.0, total_unclamped))

    return RewardBreakdown(
        raw_pnl=raw_pnl,
        pnl_normalized=pnl_normalized,
        pnl_scaled=pnl_scaled,
        participation_bonus=participation_bonus,
        quote_stuffing_penalty=quote_stuffing_penalty,
        position_limit_penalty=position_limit_penalty,
        empty_action_penalty=empty_action_penalty,
        parse_failure_penalty=parse_failure_penalty,
        total_unclamped=total_unclamped,
        total=total,
    )
