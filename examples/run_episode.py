"""Manual validation — run a 50-turn episode with 5 scripted bots.

This is a sanity-check harness, not the production environment. The real
OpenEnv-compliant Environment lives in M3 (market_env/environment.py).

Usage:
    python -m examples.run_episode --seed 42 --difficulty medium
"""

from __future__ import annotations

import argparse
from typing import Optional

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
from market_env.scenario import ScenarioGenerator


def build_observation(
    book: OrderBook,
    agent_id: str,
    position: Position,
    turn: int,
    max_turns: int,
    episode_id: str,
    private_signals: Optional[dict[str, float]],
) -> MarketObservation:
    snap = book.get_snapshot(depth=5)
    open_orders = [
        open_order_view_from_order(o)
        for o in book.get_open_orders_for_agent(agent_id)
    ]
    mark_price = snap.mid_price if snap.mid_price > 0 else 50.0
    unrealized = position.shares_held * mark_price
    return MarketObservation(
        order_book=snap,
        recent_trades=book.get_recent_trades(n=10),
        agent_id=agent_id,
        cash=position.cash,
        shares_held=position.shares_held,
        unrealized_pnl=unrealized,
        open_orders=open_orders,
        turn=turn,
        max_turns=max_turns,
        episode_id=episode_id,
        private_signals=private_signals or {},
    )


def apply_action(
    book: OrderBook,
    agent_id: str,
    action: MarketAction,
    positions: dict[str, Position],
) -> int:
    """Apply an action and update ALL affected positions; return trade count."""
    if action.action_type == "hold":
        return 0
    if action.action_type == "cancel":
        if action.order_id:
            book.cancel_order(agent_id, action.order_id)
        return 0
    if action.action_type in ("buy", "sell"):
        if action.price is None or action.quantity is None:
            return 0
        try:
            result = book.place_limit_order(
                agent_id, action.action_type, action.price, action.quantity
            )
        except ValueError as e:
            print(f"  [warn] {agent_id} action rejected: {e}")
            return 0
        # Each trade updates BOTH the buyer and seller positions
        for trade in result.trades:
            if trade.buyer_id in positions:
                positions[trade.buyer_id].apply_trade("buy", trade.price, trade.quantity)
            if trade.seller_id in positions:
                positions[trade.seller_id].apply_trade("sell", trade.price, trade.quantity)
        return len(result.trades)
    return 0


def run_episode(seed: int = 42, difficulty: str = "medium", verbose: bool = True) -> dict:
    scenario = ScenarioGenerator(seed=seed).sample(difficulty=difficulty)
    book = OrderBook()

    bots: dict[str, Bot] = {
        "agent_1": InformedBot("agent_1", seed=seed + 1),
        "agent_2": RandomBot("agent_2", seed=seed + 2),
        "agent_3": MomentumBot("agent_3", seed=seed + 3),
        "mm_bot":  MarketMakerBot("mm_bot"),
        "mr_bot":  MeanReversionBot("mr_bot"),
    }
    if isinstance(bots["agent_1"], InformedBot):
        bots["agent_1"].set_true_value(scenario.true_value)

    positions: dict[str, Position] = {a: Position(agent_id=a) for a in bots}

    if verbose:
        print(f"=== Scenario {scenario.scenario_id} ===")
        print(f"  difficulty:  {scenario.difficulty}")
        print(f"  true_value:  ${scenario.true_value:.2f}")
        print(f"  components:  {scenario.components}")
        print()
        print("  Agent private signals:")
        for aid, sigs in scenario.agent_signals.items():
            print(f"    {aid}: {sigs}")
        print()

    total_trades = 0
    for turn in range(scenario.episode_length):
        book.set_turn(turn)
        for agent_id, bot in bots.items():
            obs = build_observation(
                book, agent_id, positions[agent_id], turn,
                scenario.episode_length, scenario.scenario_id,
                scenario.agent_signals.get(agent_id, {}),
            )
            action = bot.act(obs)
            total_trades += apply_action(book, agent_id, action, positions)

    final_pnl = {a: p.pnl(scenario.true_value) for a, p in positions.items()}

    if verbose:
        snap = book.get_snapshot(depth=3)
        print("=== End of episode ===")
        print(f"  total trades:  {total_trades}")
        print(f"  final book:    "
              f"bid={snap.bids[0].price if snap.bids else 'none'}  "
              f"ask={snap.asks[0].price if snap.asks else 'none'}  "
              f"mid={snap.mid_price:.2f}")
        print()
        print(f"  Final P&L  (true value = ${scenario.true_value:.2f},  "
              f"starting cash = ${INITIAL_CASH:.0f}):")
        for agent_id in sorted(final_pnl, key=lambda a: -final_pnl[a]):
            pos = positions[agent_id]
            print(f"    {agent_id:8s}  "
                  f"cash=${pos.cash:9.2f}  "
                  f"shares={pos.shares_held:+5d}  "
                  f"pnl=${final_pnl[agent_id]:+8.2f}")

    return {
        "scenario": scenario,
        "positions": positions,
        "pnl": final_pnl,
        "total_trades": total_trades,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"], default="medium"
    )
    args = parser.parse_args()
    run_episode(seed=args.seed, difficulty=args.difficulty)
