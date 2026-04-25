"""
Market scenario generation for the multi-agent trading environment.

A scenario is the full configuration of one episode:
- A hidden true value for the stock (the agents must infer it)
- A decomposition into 4 named signal components
- An assignment of which signals each agent observes, with noise
- A starting mid price (always $50.00 — fixed for normalization)
- An episode length (default 50 turns)
- Difficulty level (controls signal magnitude and noise)

Information asymmetry by design:
  agent_1 sees [earnings, competitor]   — earnings specialist, low noise
  agent_2 sees [macro, insider]         — macro specialist, low noise
  agent_3 sees [all four]               — diversified, but high noise

No agent has complete information. Profit requires inferring what others know
from how they trade — that is the theory-of-mind hypothesis the env is designed
to test.

Reproducibility: ScenarioGenerator(seed=42) is fully deterministic. Calling
.sample() repeatedly on the same generator advances state, but two generators
with the same seed produce the same sequence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_MID: float = 50.00              # starting mid price for every scenario
TRUE_VALUE_FLOOR: float = 40.00         # clamp range — keeps prices sane
TRUE_VALUE_CEILING: float = 60.00
DEFAULT_EPISODE_LENGTH: int = 50

Difficulty = Literal["easy", "medium", "hard"]

SIGNAL_COMPONENTS: tuple[str, ...] = ("earnings", "competitor", "macro", "insider")

# (component_magnitude_range, signal_noise_std_range) per difficulty
DIFFICULTY_CONFIG: dict[str, dict[str, tuple[float, float]]] = {
    "easy":   {"mag_range": (2.0, 5.0), "noise_std": (0.10, 0.20)},
    "medium": {"mag_range": (1.0, 3.0), "noise_std": (0.40, 0.60)},
    "hard":   {"mag_range": (0.5, 1.5), "noise_std": (0.80, 1.20)},
}

# Which components each agent can observe.
# agent_3 (diversified) sees everything but with elevated noise (see DIVERSIFIED_NOISE_MULT).
AGENT_SIGNAL_VISIBILITY: dict[str, tuple[str, ...]] = {
    "agent_1": ("earnings", "competitor"),
    "agent_2": ("macro", "insider"),
    "agent_3": SIGNAL_COMPONENTS,
}

DIVERSIFIED_NOISE_MULT: float = 2.5     # agent_3's noise is multiplied by this


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class MarketScenario:
    scenario_id: str
    true_value: float                              # hidden ground truth
    components: dict[str, float]                   # name -> magnitude
    agent_signals: dict[str, dict[str, float]]     # agent_id -> {component: noisy_obs}
    initial_mid: float = INITIAL_MID
    episode_length: int = DEFAULT_EPISODE_LENGTH
    difficulty: Difficulty = "medium"
    seed: int = 0


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """Generates reproducible market scenarios.

    Args:
        seed: RNG seed. Two generators with the same seed produce the same
            sequence of scenarios.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self._sample_count = 0

    def sample(
        self,
        difficulty: Difficulty = "medium",
        agent_ids: Optional[list[str]] = None,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
    ) -> MarketScenario:
        """Sample one scenario."""
        if difficulty not in DIFFICULTY_CONFIG:
            raise ValueError(
                f"unknown difficulty: {difficulty!r} "
                f"(must be one of {list(DIFFICULTY_CONFIG)})"
            )
        if agent_ids is None:
            agent_ids = list(AGENT_SIGNAL_VISIBILITY.keys())

        cfg = DIFFICULTY_CONFIG[difficulty]
        mag_lo, mag_hi = cfg["mag_range"]
        noise_lo, noise_hi = cfg["noise_std"]

        self._sample_count += 1

        # 1. Generate the four signal components (each in [-mag_hi, +mag_hi])
        components: dict[str, float] = {}
        for name in SIGNAL_COMPONENTS:
            magnitude = self._rng.uniform(mag_lo, mag_hi)
            sign = self._rng.choice([-1.0, 1.0])
            components[name] = round(sign * magnitude, 2)

        # 2. Compute the (clamped) true value
        raw_value = INITIAL_MID + sum(components.values())
        true_value = max(TRUE_VALUE_FLOOR, min(TRUE_VALUE_CEILING, raw_value))
        true_value = round(true_value, 2)

        # 3. Build per-agent noisy observations of their visible components
        agent_signals: dict[str, dict[str, float]] = {}
        for agent_id in agent_ids:
            visible = AGENT_SIGNAL_VISIBILITY.get(agent_id, ())
            base_noise = self._rng.uniform(noise_lo, noise_hi)
            noise_std = (
                base_noise * DIVERSIFIED_NOISE_MULT
                if agent_id == "agent_3"
                else base_noise
            )
            obs: dict[str, float] = {}
            for component in visible:
                noise = self._rng.gauss(0.0, noise_std)
                obs[component] = round(components[component] + noise, 2)
            agent_signals[agent_id] = obs

        return MarketScenario(
            scenario_id=f"sc_seed{self.seed}_n{self._sample_count}_{difficulty}",
            true_value=true_value,
            components=components,
            agent_signals=agent_signals,
            difficulty=difficulty,
            episode_length=episode_length,
            seed=self.seed,
        )
