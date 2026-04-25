"""
M6 Evaluation harness — proves whether the trained policy beats baselines.

Runs a fixed set of eval episodes (50 by default, seeds 0–49, medium
difficulty, eval bot composition including InformedBot) and reports
metrics for every policy:

    Avg P&L  |  Avg Reward  |  Win Rate  |  Parse Fail  |  Participation

Policies evaluated:
    RandomBot        — random noise trader (lower bound)
    MarketMakerBot   — spread-quoting liquidity provider
    HoldBaseline     — does nothing (anchor at 0 P&L)
    InformedBot      — cheating teacher with noisy true-value access (upper bound)
    Trained Stage 1  — loaded from LoRA checkpoint (skip if no checkpoint given)

Usage:
    # Baselines only (no checkpoint needed)
    python -m training.evaluate

    # With trained checkpoint
    python -m training.evaluate --checkpoint /path/to/grpo-checkpoint

    # Quick smoke test (5 tasks)
    python -m training.evaluate --tasks 5

Outputs:
    assets/results/evaluation_stage1.json
    assets/results/evaluation_table.md
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    from scipy.stats import mannwhitneyu as _mannwhitneyu
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from market_env.bots import InformedBot, MarketMakerBot, RandomBot
from market_env.environment import MarketEnvironment
from market_env.models import MarketAction, MarketObservation

from training.prompts import (
    SYSTEM_PROMPT,
    format_observation,
    parse_action,
    serialize_action,
)
from training.rollout import Policy, Trajectory, bot_policy, run_episode


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

def _hold_policy(obs: MarketObservation) -> tuple[MarketAction, bool]:
    """Always hold — the do-nothing anchor."""
    return MarketAction(action_type="hold"), True


def _make_random_policy(seed: int = 0) -> Policy:
    bot = RandomBot("agent_1", seed=seed)
    return bot_policy(bot)


def _make_mm_policy(anchor: float = 50.0) -> Policy:
    bot = MarketMakerBot("agent_1", anchor=anchor)
    return bot_policy(bot)


# ---------------------------------------------------------------------------
# Trained-model policy (placeholder — loads when checkpoint is available)
# ---------------------------------------------------------------------------

def _make_trained_policy(checkpoint_path: str) -> Optional[Policy]:
    """Load a LoRA checkpoint and return a Policy that runs inference.

    Returns None if the checkpoint cannot be loaded (missing deps,
    bad path, GPU unavailable, etc.) so the evaluator can gracefully skip.
    """
    try:
        from peft import PeftModel  # noqa: F811
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("[evaluate] transformers/peft not installed — skipping trained policy")
        return None

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        print(f"[evaluate] checkpoint not found: {ckpt} — skipping trained policy")
        return None

    print(f"[evaluate] loading checkpoint from {ckpt} ...")
    base_model_name = _detect_base_model(ckpt)

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    except Exception as exc:
        print(f"[evaluate] failed to load tokenizer for {base_model_name}: {exc}")
        return None

    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if has_cuda else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto" if has_cuda else "cpu",
        )
        model = PeftModel.from_pretrained(model, str(ckpt))
    except Exception as exc:
        print(f"[evaluate] failed to load model: {exc}")
        print("[evaluate] (hint: bnb-4bit checkpoints require a CUDA GPU; "
              "run this on Colab or a GPU machine)")
        return None

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if has_cuda else "cpu"

    def _policy(obs: MarketObservation) -> tuple[MarketAction, bool]:
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
            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return parse_action(generated)

    print("[evaluate] trained policy loaded successfully")
    return _policy


# Unsloth bnb-4bit names → standard HF names for CPU/non-CUDA fallback.
_UNSLOTH_TO_HF: dict[str, str] = {
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit": "Qwen/Qwen2.5-3B-Instruct",
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit": "Qwen/Qwen2.5-1.5B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit": "Qwen/Qwen2.5-7B-Instruct",
}


def _detect_base_model(ckpt_path: Path) -> str:
    """Read base_model_name_or_path from adapter_config.json.

    If the adapter was trained with an unsloth bnb-4bit model and CUDA
    is not available, remap to the standard HF model so we can still
    load on CPU (at higher memory cost).
    """
    import torch

    config_file = ckpt_path / "adapter_config.json"
    base = "Qwen/Qwen2.5-3B-Instruct"
    if config_file.exists():
        with open(config_file) as f:
            cfg = json.load(f)
        base = cfg.get("base_model_name_or_path", base)

    if not torch.cuda.is_available() and base in _UNSLOTH_TO_HF:
        remapped = _UNSLOTH_TO_HF[base]
        print(f"[evaluate] no CUDA — remapping {base} → {remapped}")
        base = remapped

    return base


# ---------------------------------------------------------------------------
# Per-episode evaluation (handles InformedBot's true_value injection)
# ---------------------------------------------------------------------------

def _run_informed_baseline(
    env: MarketEnvironment,
    seed: int,
    difficulty: str,
    bot_config: str,
    episode_length: int = 50,
) -> Trajectory:
    """Run InformedBot as the trainable agent (upper-bound reference).

    Requires peeking at the scenario's true_value after reset — same
    pattern as generate_sft_data.py.
    """
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

    traj = Trajectory(episode_id=obs.episode_id, seed=seed, difficulty=difficulty)

    done = False
    reward = 0.0
    info: dict = {}
    while not done:
        action = teacher.act(obs)
        from training.rollout import Turn
        traj.turns.append(
            Turn(
                turn_index=obs.turn,
                prompt=format_observation(obs),
                action=action,
                action_text=serialize_action(action),
                parse_ok=True,
            )
        )
        obs, reward, done, info = env.step(obs.episode_id, action)

    traj.final_reward = float(reward)
    traj.true_value = obs.true_value
    traj.reward_breakdown = info.get("reward_breakdown", {})
    return traj


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class PolicyMetrics:
    name: str
    n_episodes: int = 0
    avg_pnl: float = 0.0
    avg_reward: float = 0.0
    win_rate: float = 0.0
    parse_fail_rate: float = 0.0
    participation_rate: float = 0.0
    pnl_ci_low: float = 0.0
    pnl_ci_high: float = 0.0
    trajectories: list[dict] = field(default_factory=list)


def _bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean of *values*."""
    if not values:
        return 0.0, 0.0
    rng = random.Random(rng_seed)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_resamples)
    )
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_resamples)]
    hi = means[int((1 - alpha) * n_resamples) - 1]
    return lo, hi


def _compute_metrics(name: str, trajs: list[Trajectory]) -> PolicyMetrics:
    n = len(trajs)
    if n == 0:
        return PolicyMetrics(name=name)

    pnls = [t.reward_breakdown.get("raw_pnl", 0.0) for t in trajs]
    rewards = [t.final_reward for t in trajs]
    wins = sum(1 for p in pnls if p > 0)
    parse_fails = [t.parse_failure_rate for t in trajs]

    participation_rates = []
    for t in trajs:
        if t.turns:
            active = sum(1 for turn in t.turns if turn.action.action_type != "hold")
            participation_rates.append(active / len(t.turns))
        else:
            participation_rates.append(0.0)

    ci_low, ci_high = _bootstrap_ci(pnls)

    per_episode = []
    for t in trajs:
        per_episode.append({
            "seed": t.seed,
            "difficulty": t.difficulty,
            "pnl": t.reward_breakdown.get("raw_pnl", 0.0),
            "reward": t.final_reward,
            "true_value": t.true_value,
            "parse_fail_rate": t.parse_failure_rate,
        })

    return PolicyMetrics(
        name=name,
        n_episodes=n,
        avg_pnl=sum(pnls) / n,
        avg_reward=sum(rewards) / n,
        win_rate=wins / n,
        parse_fail_rate=sum(parse_fails) / n,
        participation_rate=sum(participation_rates) / n,
        pnl_ci_low=ci_low,
        pnl_ci_high=ci_high,
        trajectories=per_episode,
    )


# ---------------------------------------------------------------------------
# Statistical significance (Mann–Whitney U)
# ---------------------------------------------------------------------------

def _run_significance_tests(results: list[PolicyMetrics]) -> list[dict]:
    """Pairwise Mann-Whitney U tests on per-episode PnL distributions."""
    if not _HAS_SCIPY:
        print("[evaluate] scipy not installed — skipping significance tests")
        return []

    pnl_by_policy: dict[str, list[float]] = {}
    for r in results:
        pnl_by_policy[r.name] = [ep["pnl"] for ep in r.trajectories]

    tests: list[dict] = []
    names = list(pnl_by_policy.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = pnl_by_policy[names[i]], pnl_by_policy[names[j]]
            if not a or not b:
                continue
            try:
                stat, p = _mannwhitneyu(a, b, alternative="two-sided")
                tests.append({
                    "pair": f"{names[i]} vs {names[j]}",
                    "u_stat": round(float(stat), 2),
                    "p_value": round(float(p), 6),
                    "significant": bool(p < 0.05),
                })
            except Exception as exc:
                print(f"[evaluate] Mann-Whitney failed for {names[i]} vs {names[j]}: {exc}")
    return tests


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _format_table(results: list[PolicyMetrics]) -> str:
    header = (
        f"{'Policy':<22} {'Avg P&L':>10} {'CI Low':>10} {'CI High':>10} "
        f"{'Avg Reward':>12} {'Win Rate':>10} {'Parse Fail':>12} "
        f"{'Participation':>15}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        lines.append(
            f"{r.name:<22} {r.avg_pnl:>10.3f} {r.pnl_ci_low:>10.3f} "
            f"{r.pnl_ci_high:>10.3f} {r.avg_reward:>12.3f} "
            f"{r.win_rate:>9.0%} {r.parse_fail_rate:>11.1%} "
            f"{r.participation_rate:>14.0%}"
        )
    return "\n".join(lines)


def _format_markdown_table(results: list[PolicyMetrics]) -> str:
    lines = [
        "# Stage 1 Evaluation Results",
        "",
        "| Policy | Avg P&L | 95% CI Low | 95% CI High | Avg Reward "
        "| Win Rate | Parse Fail | Participation |",
        "|--------|--------:|-----------:|------------:|-----------:"
        "|---------:|-----------:|--------------:|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} "
            f"| {r.avg_pnl:.3f} "
            f"| {r.pnl_ci_low:.3f} "
            f"| {r.pnl_ci_high:.3f} "
            f"| {r.avg_reward:.3f} "
            f"| {r.win_rate:.0%} "
            f"| {r.parse_fail_rate:.1%} "
            f"| {r.participation_rate:.0%} |"
        )
    lines.append("")
    lines.append(f"*{results[0].n_episodes} eval episodes per policy, "
                 f"medium difficulty, eval bot composition (incl. InformedBot)*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

_TRAINING_LOG_CANDIDATES = [
    Path("training/runs/stage1_2026-04-25/training_log.json"),
    Path("assets/results/training_log.json"),
]


def _generate_plots(
    results: list[PolicyMetrics],
    output_dir: Path,
    training_log_path: Optional[str] = None,
) -> None:
    """Create evaluation charts and save to *output_dir*."""
    if not _HAS_MPL:
        print("[evaluate] matplotlib not installed — skipping plots")
        return

    # --- Plot 1: baseline_comparison.png ---
    names = [r.name for r in results]
    avg_pnls = [r.avg_pnl for r in results]
    ci_lo = [r.pnl_ci_low for r in results]
    ci_hi = [r.pnl_ci_high for r in results]
    errors = [
        [avg - lo for avg, lo in zip(avg_pnls, ci_lo)],
        [hi - avg for avg, hi in zip(avg_pnls, ci_hi)],
    ]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in avg_pnls]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.8)))
    ax.barh(names, avg_pnls, xerr=errors, color=colors,
            edgecolor="white", capsize=4)
    ax.set_xlabel("Average P&L")
    ax.set_title("Stage 1 Evaluation: Average P&L by Policy")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    bar_path = output_dir / "baseline_comparison.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] saved {bar_path}")

    # --- Plot 2: reward_curve_stage1.png (from training log) ---
    log_path = _resolve_training_log(training_log_path)
    if log_path is None:
        return

    try:
        with open(log_path) as f:
            log_data = json.load(f)
    except Exception as exc:
        print(f"[evaluate] could not read training log {log_path}: {exc}")
        return

    entries = log_data if isinstance(log_data, list) else log_data.get("log", [])
    if not entries:
        print("[evaluate] training log empty — skipping reward curve")
        return

    steps = [e.get("step", i) for i, e in enumerate(entries)]
    rewards = [e.get("reward", 0.0) for e in entries]

    window = min(20, len(rewards))
    rolling = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        rolling.append(sum(rewards[start:i + 1]) / (i - start + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rewards, alpha=0.35, linewidth=0.8, label="Per-step reward")
    ax.plot(steps, rolling, linewidth=2, label=f"Rolling mean (w={window})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")
    ax.set_title("GRPO Stage 1 Reward Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    curve_path = output_dir / "reward_curve_stage1.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] saved {curve_path}")


def _resolve_training_log(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        print(f"[evaluate] training log not found: {p} — skipping reward curve")
        return None
    for candidate in _TRAINING_LOG_CANDIDATES:
        if candidate.exists():
            return candidate
    print("[evaluate] no training log found — skipping reward curve")
    return None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    n_tasks: int = 50,
    checkpoint: Optional[str] = None,
    output_dir: Path = Path("assets/results"),
    plots: bool = True,
    training_log_path: Optional[str] = None,
) -> list[PolicyMetrics]:
    """Run the full Stage 1 evaluation suite."""
    output_dir.mkdir(parents=True, exist_ok=True)

    env = MarketEnvironment()
    tasks = env.list_tasks()
    eval_tasks = [t for t in tasks if t["task_id"].startswith("eval_")][:n_tasks]

    print(f"[evaluate] running {len(eval_tasks)} eval tasks per policy\n")

    # --- Policies to evaluate ---
    policies: dict[str, Optional[Policy]] = {
        "RandomBot": _make_random_policy(seed=999),
        "MarketMakerBot": _make_mm_policy(anchor=50.0),
        "HoldBaseline": _hold_policy,
    }

    trained_policy = None
    if checkpoint:
        trained_policy = _make_trained_policy(checkpoint)
        if trained_policy is not None:
            policies["Trained Stage 1"] = trained_policy

    all_results: list[PolicyMetrics] = []

    # Evaluate standard policies (Random, MM, Hold, Trained)
    for policy_name, policy in policies.items():
        print(f"  evaluating {policy_name} ...", end=" ", flush=True)
        t0 = time.time()
        trajs: list[Trajectory] = []

        for task in eval_tasks:
            traj = run_episode(
                env,
                policy,
                seed=task["seed"],
                difficulty=task["difficulty"],
                bot_config=task["bot_config"],
                trainable_agent_id="agent_1",
            )
            trajs.append(traj)

        elapsed = time.time() - t0
        metrics = _compute_metrics(policy_name, trajs)
        all_results.append(metrics)
        print(f"done ({elapsed:.1f}s)  avg_pnl={metrics.avg_pnl:.3f}  "
              f"win_rate={metrics.win_rate:.0%}")

    # InformedBot — special handling (needs true_value injection)
    print("  evaluating InformedBot ...", end=" ", flush=True)
    t0 = time.time()
    informed_trajs: list[Trajectory] = []
    for task in eval_tasks:
        traj = _run_informed_baseline(
            env,
            seed=task["seed"],
            difficulty=task["difficulty"],
            bot_config=task["bot_config"],
        )
        informed_trajs.append(traj)

    elapsed = time.time() - t0
    informed_metrics = _compute_metrics("InformedBot", informed_trajs)
    all_results.append(informed_metrics)
    print(f"done ({elapsed:.1f}s)  avg_pnl={informed_metrics.avg_pnl:.3f}  "
          f"win_rate={informed_metrics.win_rate:.0%}")

    # Sort: InformedBot (upper bound) first, then by avg_pnl descending
    all_results.sort(key=lambda r: r.avg_pnl, reverse=True)

    # --- Print summary ---
    print()
    print(_format_table(all_results))
    print()

    # --- Significance tests ---
    sig_tests = _run_significance_tests(all_results)
    if sig_tests:
        print()
        print("  Significance tests (Mann-Whitney U, p < 0.05):")
        for t in sig_tests:
            marker = "*" if t["significant"] else " "
            print(f"    [{marker}] {t['pair']:>40s}  U={t['u_stat']:<10.1f} "
                  f"p={t['p_value']:.4f}")
        print()

    # --- Save outputs ---
    json_path = output_dir / "evaluation_stage1.json"
    json_data = {
        "n_eval_tasks": len(eval_tasks),
        "policies": [
            {
                "name": r.name,
                "n_episodes": r.n_episodes,
                "avg_pnl": round(r.avg_pnl, 4),
                "pnl_ci_low": round(r.pnl_ci_low, 4),
                "pnl_ci_high": round(r.pnl_ci_high, 4),
                "avg_reward": round(r.avg_reward, 4),
                "win_rate": round(r.win_rate, 4),
                "parse_fail_rate": round(r.parse_fail_rate, 4),
                "participation_rate": round(r.participation_rate, 4),
                "per_episode": r.trajectories,
            }
            for r in all_results
        ],
        "significance_tests": sig_tests,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"[evaluate] saved {json_path}")

    md_path = output_dir / "evaluation_table.md"
    md_path.write_text(_format_markdown_table(all_results))
    print(f"[evaluate] saved {md_path}")

    # --- Plots ---
    if plots:
        _generate_plots(all_results, output_dir, training_log_path)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 evaluation: baselines + trained policy",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained LoRA adapter (skip if not available yet)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=50,
        help="Number of eval tasks to run (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/results"),
        help="Where to write evaluation_stage1.json and evaluation_table.md",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate matplotlib plots after evaluation (default: True)",
    )
    parser.add_argument(
        "--training-log",
        type=str,
        default=None,
        help="Path to training_log.json for reward curve (auto-detected if omitted)",
    )
    args = parser.parse_args()

    evaluate(
        n_tasks=args.tasks,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        plots=args.plots,
        training_log_path=args.training_log,
    )


if __name__ == "__main__":
    main()
