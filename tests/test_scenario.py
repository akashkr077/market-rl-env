"""Tests for the market scenario generator."""
from __future__ import annotations

import statistics

import pytest

from market_env.scenario import (
    AGENT_SIGNAL_VISIBILITY,
    DEFAULT_EPISODE_LENGTH,
    DIFFICULTY_CONFIG,
    INITIAL_MID,
    SIGNAL_COMPONENTS,
    TRUE_VALUE_CEILING,
    TRUE_VALUE_FLOOR,
    MarketScenario,
    ScenarioGenerator,
)


# ---------------------------------------------------------------------------
# Reproducibility — the central guarantee
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_produces_same_first_scenario(self):
        s1 = ScenarioGenerator(seed=42).sample()
        s2 = ScenarioGenerator(seed=42).sample()
        assert s1.true_value == s2.true_value
        assert s1.components == s2.components
        assert s1.agent_signals == s2.agent_signals
        assert s1.scenario_id == s2.scenario_id

    def test_same_seed_produces_same_sequence(self):
        g1 = ScenarioGenerator(seed=7)
        g2 = ScenarioGenerator(seed=7)
        for _ in range(3):
            assert g1.sample() == g2.sample()

    def test_different_seeds_produce_different_scenarios(self):
        s1 = ScenarioGenerator(seed=42).sample()
        s2 = ScenarioGenerator(seed=43).sample()
        # At least one of these should differ; in practice many will
        assert (s1.true_value, s1.components) != (s2.true_value, s2.components)

    def test_consecutive_samples_have_distinct_ids(self):
        g = ScenarioGenerator(seed=42)
        ids = {g.sample().scenario_id for _ in range(5)}
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# True value bounds
# ---------------------------------------------------------------------------

class TestTrueValueBounds:
    def test_true_value_within_clamp_range(self):
        for seed in range(50):
            for diff in ("easy", "medium", "hard"):
                s = ScenarioGenerator(seed=seed).sample(difficulty=diff)
                assert TRUE_VALUE_FLOOR <= s.true_value <= TRUE_VALUE_CEILING

    def test_initial_mid_is_constant(self):
        s = ScenarioGenerator(seed=0).sample()
        assert s.initial_mid == INITIAL_MID == 50.00

    def test_episode_length_default(self):
        s = ScenarioGenerator(seed=0).sample()
        assert s.episode_length == DEFAULT_EPISODE_LENGTH

    def test_episode_length_overridable(self):
        s = ScenarioGenerator(seed=0).sample(episode_length=20)
        assert s.episode_length == 20


# ---------------------------------------------------------------------------
# Signal assignment matrix
# ---------------------------------------------------------------------------

class TestSignalAssignment:
    def test_agent_1_only_sees_earnings_and_competitor(self):
        s = ScenarioGenerator(seed=42).sample()
        assert set(s.agent_signals["agent_1"].keys()) == {"earnings", "competitor"}

    def test_agent_1_does_not_see_macro_or_insider(self):
        s = ScenarioGenerator(seed=42).sample()
        assert "macro" not in s.agent_signals["agent_1"]
        assert "insider" not in s.agent_signals["agent_1"]

    def test_agent_2_only_sees_macro_and_insider(self):
        s = ScenarioGenerator(seed=42).sample()
        assert set(s.agent_signals["agent_2"].keys()) == {"macro", "insider"}

    def test_agent_2_does_not_see_earnings_or_competitor(self):
        s = ScenarioGenerator(seed=42).sample()
        assert "earnings" not in s.agent_signals["agent_2"]
        assert "competitor" not in s.agent_signals["agent_2"]

    def test_agent_3_sees_all_signals(self):
        s = ScenarioGenerator(seed=42).sample()
        assert set(s.agent_signals["agent_3"].keys()) == set(SIGNAL_COMPONENTS)

    def test_custom_agent_ids_excluded_get_no_signals(self):
        s = ScenarioGenerator(seed=42).sample(agent_ids=["agent_1", "noise_bot"])
        assert "agent_1" in s.agent_signals
        # noise_bot is not in AGENT_SIGNAL_VISIBILITY → empty signal set
        assert s.agent_signals["noise_bot"] == {}
        # agent_2 was not requested → not present
        assert "agent_2" not in s.agent_signals


# ---------------------------------------------------------------------------
# Signal noise scales with difficulty
# ---------------------------------------------------------------------------

class TestNoiseByDifficulty:
    def test_easy_noise_smaller_than_hard_noise(self):
        """Aggregate observation error is smaller on Easy than on Hard."""
        easy_errors = []
        hard_errors = []
        for seed in range(100):
            se = ScenarioGenerator(seed=seed).sample(difficulty="easy")
            sh = ScenarioGenerator(seed=seed).sample(difficulty="hard")
            for c in se.agent_signals["agent_1"]:
                easy_errors.append(abs(se.agent_signals["agent_1"][c] - se.components[c]))
            for c in sh.agent_signals["agent_1"]:
                hard_errors.append(abs(sh.agent_signals["agent_1"][c] - sh.components[c]))
        assert statistics.mean(easy_errors) < statistics.mean(hard_errors)

    def test_diversified_agent_has_higher_noise_than_specialist(self):
        """agent_3 (diversified) sees noisier signals than agent_1 (specialist)."""
        a1_errors = []
        a3_earnings_errors = []
        for seed in range(200):
            s = ScenarioGenerator(seed=seed).sample(difficulty="medium")
            a1_errors.append(abs(s.agent_signals["agent_1"]["earnings"] - s.components["earnings"]))
            a3_earnings_errors.append(abs(s.agent_signals["agent_3"]["earnings"] - s.components["earnings"]))
        assert statistics.mean(a3_earnings_errors) > statistics.mean(a1_errors)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_unknown_difficulty_raises(self):
        with pytest.raises(ValueError, match="difficulty"):
            ScenarioGenerator(seed=42).sample(difficulty="impossible")

    def test_known_difficulties_match_config(self):
        assert set(DIFFICULTY_CONFIG.keys()) == {"easy", "medium", "hard"}

    def test_signal_components_count(self):
        assert len(SIGNAL_COMPONENTS) == 4
