"""
Theory-of-Mind probe system for M6 evaluation.

Four probes test whether profit-maximizing RL training produces implicit
theory-of-mind in a multi-agent market setting:

  Probe 1 — Price Efficiency:  |mid_price − true_value| per turn.
  Probe 2 — Order Flow Probe:  logistic probe on hidden states → true_value side.
  Probe 3 — Opponent Signal:   logistic probe on hidden states → opponent signal.
  Probe 4 — Behavioral Adaptation: correlation(book_imbalance, aggressiveness).

Probes 1 & 4 run on CPU using scripted-bot policies.
Probes 2 & 3 are GPU-only skeletons that require model hidden states.

Usage:
    python -m training.tom_probes --tasks 50 --output-dir assets/results
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from market_env.bots import (
    InformedBot,
    MarketMakerBot,
    MeanReversionBot,
    MomentumBot,
    RandomBot,
)
from market_env.environment import MarketEnvironment
from market_env.models import MarketAction, MarketObservation
from training.rollout import Policy, bot_policy

try:
    import warnings as _warnings
    from scipy.stats import pearsonr as _pearsonr

    def pearsonr(x, y):
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            return _pearsonr(x, y)

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hold_policy(obs: MarketObservation) -> tuple[MarketAction, bool]:
    return MarketAction(action_type="hold"), True


def _make_random_policy(seed: int = 0) -> Policy:
    return bot_policy(RandomBot("agent_1", seed=seed))


def _make_mm_policy(anchor: float = 50.0) -> Policy:
    return bot_policy(MarketMakerBot("agent_1", anchor=anchor))


def _make_momentum_policy(seed: int = 0) -> Policy:
    return bot_policy(MomentumBot("agent_1", seed=seed))


def _make_meanrev_policy(anchor: float = 50.0) -> Policy:
    return bot_policy(MeanReversionBot("agent_1", anchor=anchor))


# ---------------------------------------------------------------------------
# Probe 1 — Price Efficiency over Turns
# ---------------------------------------------------------------------------

def _run_episode_tracking_midprice(
    env: MarketEnvironment,
    policy: Policy,
    *,
    seed: int,
    difficulty: str = "medium",
    episode_length: int = 50,
    bot_config: str = "eval",
    trainable_agent_id: str = "agent_1",
    true_value_override: Optional[float] = None,
) -> tuple[list[float], float]:
    """Run one episode and return (mid_prices_per_turn, true_value).

    Uses a manual step loop instead of run_episode so we can capture
    the order-book mid_price at every turn.
    """
    obs = env.reset(
        seed=seed,
        difficulty=difficulty,
        episode_length=episode_length,
        bot_config=bot_config,
        trainable_agent_id=trainable_agent_id,
    )
    true_value = env._episodes[obs.episode_id].scenario.true_value

    if true_value_override is not None:
        true_value = true_value_override

    mid_prices: list[float] = []
    done = False
    while not done:
        mid = obs.order_book.mid_price
        mid_prices.append(mid if mid > 0 else float("nan"))
        action, _ = policy(obs)
        obs, _, done, _ = env.step(obs.episode_id, action)

    return mid_prices, true_value


def _run_informed_episode_tracking_midprice(
    env: MarketEnvironment,
    *,
    seed: int,
    difficulty: str = "medium",
    episode_length: int = 50,
    bot_config: str = "eval",
) -> tuple[list[float], float]:
    """Informed-bot variant — needs true_value injection."""
    obs = env.reset(
        seed=seed,
        difficulty=difficulty,
        episode_length=episode_length,
        bot_config=bot_config,
        trainable_agent_id="agent_1",
    )
    true_value = env._episodes[obs.episode_id].scenario.true_value

    teacher = InformedBot("agent_1", seed=seed, edge=0.20, qty=10)
    teacher.set_true_value(true_value)

    mid_prices: list[float] = []
    done = False
    while not done:
        mid = obs.order_book.mid_price
        mid_prices.append(mid if mid > 0 else float("nan"))
        action = teacher.act(obs)
        obs, _, done, _ = env.step(obs.episode_id, action)

    return mid_prices, true_value


def probe_price_efficiency(
    n_tasks: int = 50,
    output_dir: Path = Path("assets/results"),
) -> dict:
    """Probe 1: track |mid_price − true_value| per turn for each policy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = MarketEnvironment()
    tasks = env.list_tasks()
    eval_tasks = [t for t in tasks if t["task_id"].startswith("eval_")][:n_tasks]

    policies: dict[str, Policy] = {
        "RandomBot": _make_random_policy(seed=999),
        "MomentumBot": _make_momentum_policy(seed=999),
        "MeanReversionBot": _make_meanrev_policy(anchor=50.0),
        "MarketMakerBot": _make_mm_policy(anchor=50.0),
        "HoldBaseline": _hold_policy,
    }

    results: dict[str, list[list[float]]] = {}

    for policy_name, policy in policies.items():
        print(f"  [probe1] {policy_name} ...", end=" ", flush=True)
        t0 = time.time()
        all_gaps: list[list[float]] = []
        for task in eval_tasks:
            mid_prices, tv = _run_episode_tracking_midprice(
                env, policy,
                seed=task["seed"],
                difficulty=task["difficulty"],
                bot_config=task["bot_config"],
            )
            gaps = [abs(m - tv) if not math.isnan(m) else float("nan")
                    for m in mid_prices]
            all_gaps.append(gaps)
        results[policy_name] = all_gaps
        print(f"done ({time.time() - t0:.1f}s)")

    # InformedBot (special)
    print("  [probe1] InformedBot ...", end=" ", flush=True)
    t0 = time.time()
    informed_gaps: list[list[float]] = []
    for task in eval_tasks:
        mid_prices, tv = _run_informed_episode_tracking_midprice(
            env,
            seed=task["seed"],
            difficulty=task["difficulty"],
            bot_config=task["bot_config"],
        )
        gaps = [abs(m - tv) if not math.isnan(m) else float("nan")
                for m in mid_prices]
        informed_gaps.append(gaps)
    results["InformedBot"] = informed_gaps
    print(f"done ({time.time() - t0:.1f}s)")

    # Compute per-turn means
    max_len = max(len(g) for gs in results.values() for g in gs)
    summary: dict[str, list[Optional[float]]] = {}
    for name, all_gaps in results.items():
        per_turn: list[Optional[float]] = []
        for t in range(max_len):
            vals = [g[t] for g in all_gaps if t < len(g) and not math.isnan(g[t])]
            per_turn.append(round(sum(vals) / len(vals), 4) if vals else None)
        summary[name] = per_turn

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, curve in summary.items():
        turns = list(range(len(curve)))
        values = [v if v is not None else float("nan") for v in curve]
        ax.plot(turns, values, label=name, linewidth=1.5)
    ax.set_xlabel("Turn")
    ax.set_ylabel("|Mid Price − True Value|")
    ax.set_title("Probe 1: Price Efficiency over Turns")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "price_efficiency.png", dpi=150)
    plt.close(fig)
    print(f"  [probe1] saved {output_dir / 'price_efficiency.png'}")

    # --- JSON ---
    json_data = {
        "probe": "price_efficiency",
        "n_tasks": len(eval_tasks),
        "per_turn_mean_gap": summary,
    }
    json_path = output_dir / "price_efficiency.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  [probe1] saved {json_path}")

    return json_data


# ---------------------------------------------------------------------------
# Probe 4 — Behavioral Adaptation (Imbalance vs Aggressiveness)
# ---------------------------------------------------------------------------

def _compute_imbalance(obs: MarketObservation) -> Optional[float]:
    """Order-book imbalance: (bid_vol − ask_vol) / (bid_vol + ask_vol)."""
    bid_vol = sum(level.quantity for level in obs.order_book.bids)
    ask_vol = sum(level.quantity for level in obs.order_book.asks)
    total = bid_vol + ask_vol
    if total == 0:
        return None
    return (bid_vol - ask_vol) / total


def _compute_aggressiveness(obs: MarketObservation, action: MarketAction) -> float:
    """How far above best-bid (buy) or below best-ask (sell) the order is.

    Positive = aggressive crossing. Zero for hold/cancel.
    """
    if action.action_type == "buy" and action.price is not None:
        if obs.order_book.bids:
            return action.price - obs.order_book.bids[0].price
        return 0.0
    if action.action_type == "sell" and action.price is not None:
        if obs.order_book.asks:
            return obs.order_book.asks[0].price - action.price
        return 0.0
    return 0.0


def _collect_imbalance_aggressiveness(
    env: MarketEnvironment,
    policy: Policy,
    eval_tasks: list[dict],
    *,
    bot_config_override: Optional[str] = None,
    informed: bool = False,
) -> tuple[list[float], list[float]]:
    """Run episodes and collect (imbalance, aggressiveness) pairs."""
    imbalances: list[float] = []
    aggressivenesses: list[float] = []

    for task in eval_tasks:
        bc = bot_config_override or task["bot_config"]
        obs = env.reset(
            seed=task["seed"],
            difficulty=task["difficulty"],
            bot_config=bc,
            trainable_agent_id="agent_1",
        )
        true_value = env._episodes[obs.episode_id].scenario.true_value

        if informed:
            teacher = InformedBot("agent_1", seed=task["seed"], edge=0.20, qty=10)
            teacher.set_true_value(true_value)

        done = False
        while not done:
            imb = _compute_imbalance(obs)
            if informed:
                action = teacher.act(obs)
                parse_ok = True
            else:
                action, parse_ok = policy(obs)

            if imb is not None:
                agg = _compute_aggressiveness(obs, action)
                imbalances.append(imb)
                aggressivenesses.append(agg)

            obs, _, done, _ = env.step(obs.episode_id, action)

    return imbalances, aggressivenesses


def probe_behavioral_adaptation(
    n_tasks: int = 50,
    output_dir: Path = Path("assets/results"),
) -> dict:
    """Probe 4: correlation between order-book imbalance and aggressiveness."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = MarketEnvironment()
    tasks = env.list_tasks()
    eval_tasks = [t for t in tasks if t["task_id"].startswith("eval_")][:n_tasks]

    policies: dict[str, tuple[Policy, bool]] = {
        "RandomBot": (_make_random_policy(seed=999), False),
        "MomentumBot": (_make_momentum_policy(seed=999), False),
        "MeanReversionBot": (_make_meanrev_policy(anchor=50.0), False),
        "MarketMakerBot": (_make_mm_policy(anchor=50.0), False),
        "HoldBaseline": (_hold_policy, False),
        "InformedBot": (_hold_policy, True),  # placeholder; handled via informed flag
    }

    summary: dict[str, dict] = {}

    for name, (policy, is_informed) in policies.items():
        print(f"  [probe4] {name} ...", end=" ", flush=True)
        t0 = time.time()
        imbs, aggs = _collect_imbalance_aggressiveness(
            env, policy, eval_tasks, informed=is_informed,
        )
        entry: dict = {
            "n_observations": len(imbs),
            "correlation": None,
            "p_value": None,
            "reads_book_pressure": False,
        }

        if HAS_SCIPY and len(imbs) > 2:
            corr, pval = pearsonr(imbs, aggs)
            if math.isnan(float(corr)):
                entry["note"] = "constant input — correlation undefined"
            else:
                entry["correlation"] = round(float(corr), 4)
                entry["p_value"] = float(pval)
                entry["reads_book_pressure"] = bool(abs(corr) > 0.3 and pval < 0.01)
        elif not HAS_SCIPY:
            entry["note"] = "scipy not available — correlation skipped"

        summary[name] = entry
        elapsed = time.time() - t0
        corr_str = (f"r={entry['correlation']:.3f}" if entry["correlation"] is not None
                     else "n/a")
        print(f"done ({elapsed:.1f}s)  {corr_str}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    policy_order = list(policies.keys())

    for idx, name in enumerate(policy_order):
        ax = axes_flat[idx]
        policy, is_informed = policies[name]
        imbs, aggs = _collect_imbalance_aggressiveness(
            env, policy, eval_tasks[:min(10, len(eval_tasks))],
            informed=is_informed,
        )
        ax.scatter(imbs, aggs, alpha=0.15, s=8, edgecolors="none")
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        info = summary[name]
        if info["correlation"] is not None:
            ax.text(
                0.05, 0.95,
                f"r={info['correlation']:.3f}",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
            )

    for ax in axes_flat[len(policy_order):]:
        ax.set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Order Book Imbalance")
    for ax in axes[:, 0]:
        ax.set_ylabel("Aggressiveness")

    fig.suptitle("Probe 4: Behavioral Adaptation — Imbalance vs Aggressiveness",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "behavioral_adaptation.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  [probe4] saved {output_dir / 'behavioral_adaptation.png'}")

    # --- JSON ---
    json_data = {
        "probe": "behavioral_adaptation",
        "n_tasks": len(eval_tasks),
        "policies": summary,
    }
    json_path = output_dir / "behavioral_adaptation.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  [probe4] saved {json_path}")

    return json_data


# ---------------------------------------------------------------------------
# Probe 2 — Hidden State Probe: Order Flow Prediction (GPU skeleton)
# ---------------------------------------------------------------------------

def probe_order_flow_prediction(
    model=None,
    tokenizer=None,
    n_tasks: int = 50,
    output_dir: Path = Path("assets/results"),
) -> Optional[dict]:
    """Probe 2: predict whether true_value > 50 from transformer hidden states.

    Trains a linear probe (sklearn LogisticRegression) on the last-token
    hidden state extracted from the model at each turn of eval episodes.

    The label is binary: 1 if true_value > 50 (stock is underpriced at the
    start), 0 otherwise. High accuracy means the model's internal
    representations encode information about the hidden true value — evidence
    of implicit order-flow reading.

    Requires:
        - A HuggingFace causal LM (model) with .forward() that returns
          hidden_states when output_hidden_states=True
        - The matching tokenizer
        - torch and sklearn installed
        - Ideally a CUDA GPU for speed

    Returns:
        dict with probe accuracy, or None if deps unavailable.
    """
    if model is None or tokenizer is None:
        print("  [probe2] *** SKIPPED — requires model and tokenizer (GPU) ***")
        print("           Pass a loaded HF model to probe_order_flow_prediction()")
        print("           to train a logistic probe on last-token hidden states.")
        return None

    try:
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError as e:
        print(f"  [probe2] *** SKIPPED — missing dependency: {e} ***")
        return None

    from training.prompts import SYSTEM_PROMPT, format_observation

    env = MarketEnvironment()
    tasks = env.list_tasks()
    eval_tasks = [t for t in tasks if t["task_id"].startswith("eval_")][:n_tasks]

    device = next(model.parameters()).device
    hidden_states_list: list = []
    labels: list[int] = []

    print(f"  [probe2] collecting hidden states from {len(eval_tasks)} episodes ...")
    model.eval()
    for task in eval_tasks:
        obs = env.reset(
            seed=task["seed"],
            difficulty=task["difficulty"],
            bot_config=task["bot_config"],
            trainable_agent_id="agent_1",
        )
        true_value = env._episodes[obs.episode_id].scenario.true_value
        label = 1 if true_value > 50.0 else 0

        done = False
        while not done:
            user_msg = format_observation(obs)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            last_hidden = outputs.hidden_states[-1][0, -1, :]  # last layer, last token
            hidden_states_list.append(last_hidden.cpu().numpy())
            labels.append(label)

            action, _ = _hold_policy(obs)
            obs, _, done, _ = env.step(obs.episode_id, action)

    import numpy as np
    X = np.stack(hidden_states_list)
    y = np.array(labels)

    print(f"  [probe2] fitting logistic probe on {X.shape[0]} samples, "
          f"dim={X.shape[1]} ...")
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    accuracy = float(scores.mean())
    std = float(scores.std())

    print(f"  [probe2] accuracy = {accuracy:.3f} ± {std:.3f}")

    result = {
        "probe": "order_flow_prediction",
        "n_samples": len(labels),
        "accuracy_mean": round(accuracy, 4),
        "accuracy_std": round(std, 4),
        "cv_folds": 5,
        "chance_level": 0.5,
    }

    json_path = output_dir / "probe_order_flow.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [probe2] saved {json_path}")

    return result


# ---------------------------------------------------------------------------
# Probe 3 — Hidden State Probe: Opponent Signal Inference (GPU skeleton)
# ---------------------------------------------------------------------------

def probe_opponent_signal(
    model=None,
    tokenizer=None,
    n_tasks: int = 50,
    output_dir: Path = Path("assets/results"),
) -> Optional[dict]:
    """Probe 3: predict opponent's private signal direction from hidden states.

    The scenario assigns different signal components to different agents.
    Agent_1 (the trainable agent) sees {earnings, competitor}. Agent_2 sees
    {macro, insider}. This probe tests whether the model's hidden states
    encode information about agent_2's signals — which can only be inferred
    from observing agent_2's trading behavior.

    Approach:
        1. Run eval episodes, extracting the last-token hidden state at
           each turn from the model.
        2. Label = 1 if sum of agent_2's signal components > 0, else 0.
           (i.e., is agent_2 net bullish or bearish?)
        3. Train sklearn LogisticRegression on (hidden_state → label).
        4. Report 5-fold CV accuracy.

    Above-chance accuracy (>> 0.5) indicates the model has learned to infer
    information it never directly observes — strong evidence of implicit
    theory-of-mind.

    Requires:
        - A HuggingFace causal LM (model) with output_hidden_states support
        - The matching tokenizer
        - torch and sklearn installed
        - Ideally a CUDA GPU

    Returns:
        dict with probe accuracy, or None if deps unavailable.
    """
    if model is None or tokenizer is None:
        print("  [probe3] *** SKIPPED — requires model and tokenizer (GPU) ***")
        print("           Pass a loaded HF model to probe_opponent_signal()")
        print("           to train a logistic probe on last-token hidden states.")
        return None

    try:
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError as e:
        print(f"  [probe3] *** SKIPPED — missing dependency: {e} ***")
        return None

    from training.prompts import SYSTEM_PROMPT, format_observation

    env = MarketEnvironment()
    tasks = env.list_tasks()
    eval_tasks = [t for t in tasks if t["task_id"].startswith("eval_")][:n_tasks]

    device = next(model.parameters()).device
    hidden_states_list: list = []
    labels: list[int] = []

    print(f"  [probe3] collecting hidden states from {len(eval_tasks)} episodes ...")
    model.eval()
    for task in eval_tasks:
        obs = env.reset(
            seed=task["seed"],
            difficulty=task["difficulty"],
            bot_config=task["bot_config"],
            trainable_agent_id="agent_1",
        )
        scenario = env._episodes[obs.episode_id].scenario
        agent2_signals = scenario.agent_signals.get("agent_2", {})
        opponent_net = sum(agent2_signals.values())
        label = 1 if opponent_net > 0 else 0

        done = False
        while not done:
            user_msg = format_observation(obs)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            last_hidden = outputs.hidden_states[-1][0, -1, :]
            hidden_states_list.append(last_hidden.cpu().numpy())
            labels.append(label)

            action, _ = _hold_policy(obs)
            obs, _, done, _ = env.step(obs.episode_id, action)

    import numpy as np
    X = np.stack(hidden_states_list)
    y = np.array(labels)

    print(f"  [probe3] fitting logistic probe on {X.shape[0]} samples, "
          f"dim={X.shape[1]} ...")
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    accuracy = float(scores.mean())
    std = float(scores.std())

    print(f"  [probe3] accuracy = {accuracy:.3f} ± {std:.3f}")

    result = {
        "probe": "opponent_signal_inference",
        "n_samples": len(labels),
        "accuracy_mean": round(accuracy, 4),
        "accuracy_std": round(std, 4),
        "cv_folds": 5,
        "chance_level": 0.5,
    }

    json_path = output_dir / "probe_opponent_signal.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [probe3] saved {json_path}")

    return result


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_all_probes(
    n_tasks: int = 50,
    output_dir: Path = Path("assets/results"),
    checkpoint: Optional[str] = None,
) -> dict:
    """Run all Theory-of-Mind probes and return combined results.

    Probes 1 & 4 always run (CPU-safe, scripted bots only).
    Probes 2 & 3 run only when a checkpoint is provided and GPU deps exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    print("=" * 60)
    print("  Theory-of-Mind Probes")
    print("=" * 60)

    # Probe 1
    print("\n--- Probe 1: Price Efficiency ---")
    results["price_efficiency"] = probe_price_efficiency(n_tasks, output_dir)

    # Probe 4
    print("\n--- Probe 4: Behavioral Adaptation ---")
    results["behavioral_adaptation"] = probe_behavioral_adaptation(n_tasks, output_dir)

    # Probes 2 & 3 (GPU)
    model = None
    tokenizer = None
    if checkpoint:
        try:
            from training.evaluate import _make_trained_policy, _detect_base_model
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            ckpt = Path(checkpoint)
            if ckpt.exists():
                base_name = _detect_base_model(ckpt)
                has_cuda = torch.cuda.is_available()
                dtype = torch.bfloat16 if has_cuda else torch.float32
                tokenizer = AutoTokenizer.from_pretrained(base_name)
                model = AutoModelForCausalLM.from_pretrained(
                    base_name, torch_dtype=dtype,
                    device_map="auto" if has_cuda else "cpu",
                )
                model = PeftModel.from_pretrained(model, str(ckpt))
                model.eval()
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print(f"\n[tom_probes] loaded model from {ckpt}")
        except Exception as e:
            print(f"\n[tom_probes] could not load model: {e}")
            model = None
            tokenizer = None

    print("\n--- Probe 2: Order Flow Prediction (hidden state) ---")
    p2 = probe_order_flow_prediction(model, tokenizer, n_tasks, output_dir)
    if p2:
        results["order_flow_prediction"] = p2

    print("\n--- Probe 3: Opponent Signal Inference (hidden state) ---")
    p3 = probe_opponent_signal(model, tokenizer, n_tasks, output_dir)
    if p3:
        results["opponent_signal_inference"] = p3

    print("\n" + "=" * 60)
    print("  All probes complete.")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Theory-of-Mind probes for M6 evaluation",
    )
    parser.add_argument(
        "--tasks", type=int, default=50,
        help="Number of eval tasks per probe (default: 50)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("assets/results"),
        help="Directory for output plots and JSON",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained LoRA adapter (enables Probes 2 & 3)",
    )
    args = parser.parse_args()

    run_all_probes(
        n_tasks=args.tasks,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
