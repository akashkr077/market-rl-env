# Planning Notes — Multi-Agent Market RL Environment
# OpenEnv Hackathon Round 2

---

## 1. The Thesis

One sentence: **An agent that only gets rewarded for profit, but must infer other agents' hidden information to profit, will — if trained long enough — develop implicit theory-of-mind.**

This is not a claim about finance. It is a claim about cognition. Markets are the vehicle. Theory-of-mind (ToM) is the scientific contribution. Every design decision should serve that single sentence. If a feature does not help prove or demonstrate ToM emergence, it gets cut.

Why this is compelling to judges:
- Domain is underexplored in RL/LLM training
- The reward signal is objective (money), not hand-crafted for ToM
- ToM is measurable post-hoc with probe tests
- A researcher could write a real paper about this
- No one has trained an LLM to do this via GRPO

---

## 2. Why This Environment Is Novel

No existing OpenEnv environment trains theory-of-mind via economic competition. The nearest academic work is:
- Lewis signaling games — but those require explicit communication channels
- OpenAI Hide & Seek — physical inference, not informational
- MARL market simulations — don't train LLMs via GRPO, don't measure ToM
- Emergent communication in multi-agent RL — agents need to communicate; here they cannot

What is new:
- Information asymmetry as the core challenge, not physical strategy
- Pure profit signal as the only training signal — zero auxiliary ToM supervision
- LLM agents reasoning from natural-language text observations
- ToM measured post-hoc via probing, not assumed or supervised
- Staged curriculum that makes training tractable and results interpretable

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HF Space  (OpenEnv Compliant)                    │
│                                                                          │
│  ┌──────────────────┐    ┌────────────────────────┐    ┌─────────────┐ │
│  │  Scenario        │───▶│  Order Book Engine      │───▶│  Reward     │ │
│  │  Generator       │    │  (limit orders,         │    │  Calculator │ │
│  │  (true value,    │    │   price-time priority,  │    │  (P&L +     │ │
│  │   signal decomp) │    │   matching, cancels)    │    │   risk adj) │ │
│  └──────────────────┘    └────────────────────────┘    └─────────────┘ │
│                                      ▲                                   │
│                    ┌─────────────────┼─────────────────┐                │
│                    │                 │                 │                 │
│             Agent 1 (LLM)    Agent 2 (LLM)     Noise Bots               │
│             Private: earnings  Private: macro    Scripted:               │
│             + competitor       + insider         random,                 │
│             signals            signals           momentum, MM            │
│                    │                 │                                   │
│             Text observation  Text observation                           │
│             → Text action     → Text action                              │
└─────────────────────────────────────────────────────────────────────────┘
                              ▲
                              │  GRPO rollouts + gradient updates
               ┌──────────────────────────────────┐
               │    Google Colab + HF Credits      │
               │  Stage 1: A100 Colab notebook    │
               │  Stage 2: A100 Colab notebook    │
               │  Judges can re-run the notebook  │
               │  (judging requirement, not opt.) │
               └──────────────────────────────────┘
```

---

## 4. Core Component Design

### 4.1 Order Book Engine

Standard continuous double-auction with price-time priority. Textbook market microstructure — implement it once, test it thoroughly, never touch it again.

**Data structures:**
- Bids: sorted descending by price, then ascending by timestamp
- Asks: sorted ascending by price, then ascending by timestamp
- Each order: `order_id`, `agent_id`, `side`, `price`, `quantity`, `timestamp`

**Operations:**
```
place_limit_order(agent_id, side, price, qty) → OrderResult
cancel_order(agent_id, order_id) → CancelResult
get_book_snapshot(depth=5) → OrderBookSnapshot
get_recent_trades(n=10) → list[TradeRecord]
```

**Matching algorithm:**
1. New buy at price P: iterate asks ascending, fill against all asks ≤ P
2. New sell at price P: iterate bids descending, fill against all bids ≥ P
3. Partial fills leave remainder resting on book
4. Full fills remove order from book entirely

**What we do NOT implement:**
- Market orders (agents must specify a price — prevents gaming)
- Stop-loss or iceberg orders (unnecessary complexity)
- Multiple stocks (one is enough)
- Circuit breakers (not needed for 50-turn episodes)

**Unit test targets:**
- Two crossing orders execute at resting price
- Partial fill leaves correct remainder
- Cancel removes from book, subsequent match skips it
- Book snapshot reflects all resting orders exactly
- Price-time priority: two orders at same price, earlier one fills first
- Agent can only cancel their own orders

### 4.2 Scenario Generator

Generates a complete market configuration for one episode. Fully deterministic given a seed — reproducible for evaluation, diverse for training.

```python
@dataclass
class MarketScenario:
    scenario_id: str
    true_value: float           # Hidden ground truth, e.g. $50.00 ± delta
    components: dict            # {"earnings": +3.2, "competitor": -1.8, ...}
    agent_signals: dict         # {agent_id: {component: observed ± noise}}
    noise_trader_configs: list  # One config per scripted bot
    initial_mid: float          # Starting reference price (always $50.00)
    episode_length: int         # Number of turns (default: 50)
    difficulty: str             # "easy" | "medium" | "hard"
```

**True value construction:**
- Base price: $50.00 (fixed — simplifies normalization)
- Four signal components, each drawn from N(0, σ²):
  - `earnings`: σ = $2.00
  - `competitor`: σ = $1.50
  - `macro`: σ = $1.00
  - `insider`: σ = $2.50
- `true_value = 50.00 + sum(components)` — clamped to [$40, $60]

**Signal assignment:**
| Agent | Signals Visible | Noise (std dev) |
|-------|-----------------|-----------------|
| Agent 1 (earnings spec) | earnings, competitor | $0.20 |
| Agent 2 (macro spec) | macro, insider | $0.20 |
| Agent 3 (diversified) | all four | $0.80 |
| Noise bots | none | — |

No agent has complete information. Profit requires inferring what others know.

**Difficulty curriculum:**
- Easy: component magnitudes $2–$5, noise $0.10–$0.20 — signals are clear
- Medium: magnitudes $1–$3, noise $0.40–$0.60 — some ambiguity
- Hard: magnitudes $0.50–$1.50, noise $0.80–$1.20 — heavy inference required

### 4.3 Scripted Baseline Bots

Five archetypes for Stage 1 training. Each is deterministic given the scenario config.

**RandomBot**
Orders uniformly at random within ±$3 of mid price, quantities 1–30 shares, 60% chance of acting each turn. Provides liquidity with zero information content. Any trained agent should beat this easily.

**MomentumBot**
Looks at last 5 trade prices. If 3+ are increasing, buys at ask+$0.10. If 3+ are decreasing, sells at bid-$0.10. Buys high, sells low in a mean-reverting market — a useful punching bag.

**MeanReversionBot**
If `mid > initial_mid + $1.00`, bets on reversion (sells). If `mid < initial_mid - $1.00`, buys. Correct in an uninformed market but adversely selected by informed agents — a realistic opponent.

**InformedBot (semi-scripted)**
Observes the true value + Gaussian noise ($0.40 std dev). Buys aggressively when `bid < true_value_estimate - $0.50`. Sells aggressively when `ask > true_value_estimate + $0.50`. This is the hardest scripted opponent. A trained agent that beats InformedBot is genuinely learning something.

**MarketMakerBot**
Posts limit buy at `mid - $0.30` and limit sell at `mid + $0.30`, quantities 10 shares, every turn. Earns the spread but gets adversely selected by informed traders. Provides baseline liquidity and makes the order book look realistic.

### 4.4 Pydantic Data Models

```python
class TradeRecord(BaseModel):
    trade_id: str
    price: float
    quantity: int
    buyer_id: str
    seller_id: str
    turn: int

class OrderBookLevel(BaseModel):
    price: float
    quantity: int
    num_orders: int

class OrderBookSnapshot(BaseModel):
    bids: list[OrderBookLevel]   # Top 5, descending price
    asks: list[OrderBookLevel]   # Top 5, ascending price
    mid_price: float
    spread: float

class OpenOrder(BaseModel):
    order_id: str
    side: Literal["buy", "sell"]
    price: float
    quantity: int
    filled: int
    turn_placed: int

class MarketObservation(BaseModel):
    # Private state (fixed for episode, never changes)
    private_signals: dict[str, float]   # {"earnings": 2.8, "competitor": -1.2}
    signal_names: list[str]             # Which components this agent can see

    # Public market state
    order_book: OrderBookSnapshot
    recent_trades: list[TradeRecord]    # Last 10 executions

    # Own position
    shares_held: int                    # Positive = long, negative = short
    cash: float                         # Available cash (starts at $10,000)
    realized_pnl: float                 # From completed trades this episode
    unrealized_pnl: float               # Mark-to-mid on current position
    open_orders: list[OpenOrder]        # Agent's resting limit orders

    # Other agents' observable footprint (public information — not their signals)
    # Stage 1: always empty list. Stage 2: populated with other agents' resting orders.
    # Using the base class for both stages avoids a schema migration mid-project.
    visible_other_orders: list[OpenOrder] = []   # order_id, side, price, qty — agent_id is anonymized

    # Episode meta
    turn: int
    max_turns: int
    episode_id: str

    # Revealed at episode end only
    true_value: float | None = None

class MarketAction(BaseModel):
    action_type: Literal["buy", "sell", "cancel", "hold"]
    price: float | None = None          # Required for buy/sell
    quantity: int | None = None         # Required for buy/sell (1–100)
    order_id: str | None = None         # Required for cancel
    reasoning: str | None = None        # Optional — logged but not used in reward

class EpisodeReward(BaseModel):
    total: float                        # Clamped to [-1, 1]
    raw_pnl: float
    pnl_normalized: float
    participation_bonus: float
    penalties: float
    breakdown: dict[str, float]
```

### 4.5 Reward Function

**Primary reward: End-of-episode P&L**

```
raw_pnl = (cash_final - cash_initial) + (shares_final × true_value)
pnl_normalized = raw_pnl / (true_value × initial_shares_capacity)
```

Mark all positions to true value at episode end. Pure, objective, cannot be gamed without actually profiting.

**Auxiliary rewards (added after Stage 1 shows convergence):**

| Component | Weight | Condition |
|-----------|--------|-----------|
| Participation bonus | +0.01 | Agent placed ≥ 3 orders during episode |
| Risk-adjusted P&L | ×0.8 multiplier | P&L / max_intra_episode_drawdown |
| Quote stuffing penalty | -0.10 | Cancelled > 15 orders in episode |
| Position limit penalty | -0.05 per unit | \|shares_held\| > 150 |
| Parse failure penalty | -0.05 per occurrence | Model output could not be parsed as valid JSON action |
| Empty-action penalty | -0.02 | Held > 30 of 50 turns |

**Final episode reward:**
```
total_reward = (pnl_normalized × 5)    ← scale up before clamping
             + participation_bonus
             - penalties

total_reward = clamp(total_reward, -1.0, 1.0)
```

**Why normalization matters for GRPO:** GRPO computes advantages across a batch of rollouts. If raw P&L varies from -$500 to +$500 across scenarios, gradient magnitudes swing wildly. Normalizing to [-1, 1] per scenario keeps training stable.

**Reward magnitude calibration (do this before M5):** Run 100 episodes with only scripted bots (no trained agent). Record the distribution of raw P&L. Compute `mean(|pnl_normalized × 5|)` — this is your typical reward magnitude. Auxiliary bonuses and penalties should each be ≤ 10% of this number. If participation_bonus (+0.01) is > 10% of typical magnitude, reduce it further. If it's < 5%, you can leave it as-is. Do not skip this step — miscalibrated auxiliaries cause the agent to optimize for the wrong thing from step 1.

**Why parse-failure penalty is critical:** Without it, unparseable output silently becomes "hold," which may have the same reward as an intentional hold. During the first 500 training steps, the model frequently outputs free-form English — all parsed as "hold," all receive near-zero reward, gradient is flat. The -0.05 parse penalty gives the model a strong early signal to produce valid JSON before it learns anything else.

**Anti-gaming checklist:**
- Server enforces: agents cannot read other agents' private_signals fields
- Server enforces: true_value is None in all observations until episode end
- Order cancellation rate logged per agent per episode
- Position limits hard-capped at server level (orders rejected if they'd exceed limit)
- Agent cannot cancel another agent's orders
- Agent cannot place orders while they have a pending unprocessed action

### 4.6 Theory-of-Mind Measurement

This is the scientific contribution. It must be measured, not assumed.

**Proxy Metric 1: Price Efficiency**
```
efficiency_t = 1 - |mid_price_t - true_value| / |initial_mid - true_value|
```
Measures: how quickly does market price converge to true value when trained agents trade?
Compare: random agents → trained agents. Faster convergence = agents are revealing information through order flow.

**Proxy Metric 2: Order Flow Prediction Probe**
After training, freeze the model. Construct a probe dataset:
- Input: order book snapshots + trade history (public only, no private signals)
- Label: was the true value above or below initial mid price?
- Train a linear probe on the model's hidden states
If probe accuracy > 60% → the model's representations encode inferred market state.

**Proxy Metric 3: Opponent Signal Inference — Linear Probe**
Do NOT prompt the trained model directly with the question. Direct prompting is gameable: the model can learn to pattern-match order flow surface features without actually using them for trading decisions, giving a false positive.

Correct approach: freeze the model, extract hidden states from the last transformer layer when the model processes each observation, then train a lightweight linear classifier (logistic regression, sklearn) on top of those hidden states:
- Input: hidden state vector at the final observation token
- Label: was Agent 2's private signal estimate positive or negative?
- Train on 500 episodes, evaluate on 200 held-out episodes
- Metric: accuracy vs 50% random baseline

If the linear probe achieves > 60% accuracy on held-out episodes → the model's internal representations encode information about other agents' private signals, even without being explicitly trained to do so. That is the ToM claim.

**Proxy Metric 4: Behavioral Adaptation Signature**
Does the agent trade more aggressively (tighter prices, larger quantities) when the order book is thick on one side?
Measure: correlation between order_book_imbalance and agent_order_aggressiveness across 1000 episodes.
If correlation > 0.3 and p < 0.01 → agent is reading book pressure as an informational signal.

These four probes together constitute a defensible empirical claim about ToM emergence. They are not the training signal. They are diagnostic tools.

---

## 5. Training Strategy

### Stage 1: Single-Agent RL vs Scripted Bots (Weeks 3–5)

The de-risked foundation. One trainable agent plays against 4 scripted bots. This is not multi-agent RL. It is single-agent RL in a populated environment. Far more stable. Submittable on its own.

**Setup:**
- Model: Qwen2.5-3B-Instruct (via Unsloth 4-bit quantization)
- Trainer: GRPO via HF TRL `GRPOTrainer`
- Opponents: RandomBot × 1, MomentumBot × 1, MeanReversionBot × 1, MarketMakerBot × 1
- **InformedBot is evaluation-only, never a training opponent.** Its direct access to true_value would dominate the market and prevent the trained agent from ever getting positive reward — killing the gradient signal entirely. It appears only in held-out evaluation as the hard benchmark.
- Episodes per batch: 16
- Learning rate: 5e-7 (conservative for instruct model fine-tuning)
- Max new tokens per action: 128 (price, quantity, reasoning)
- Episodes per training run: 3000

**Curriculum schedule:**
- Steps 0–600: Easy scenarios only (strong signals, low noise)
- Steps 600–1500: Mix 60% easy, 40% medium
- Steps 1500–2500: Mix 30% easy, 50% medium, 20% hard
- Steps 2500+: Full distribution

**Convergence criteria:**
- Average reward over last 100 episodes > average reward of first 100 episodes × 1.5
- Win rate vs InformedBot > 40% (InformedBot has an inherent edge from true value access)
- Agent places orders on ≥ 80% of turns (participation)

**What to monitor during training:**
- Overall reward curve (smoothed with window=50)
- Separate columns: participation_bonus, pnl_normalized, penalty_total
- Sample 5 episode transcripts every 200 steps — read them, don't just plot them
- Order cancellation rate (watch for quote stuffing emergence)

### Stage 2: Population-Based Self-Play (Weeks 6–7, conditional on Stage 1 success)

**Only proceed if:** Stage 1 agent beats InformedBot on held-out scenarios at > 45% win rate.

**Setup:**
- 4 concurrent agent copies (same architecture, different random seeds)
- No scripted bots — pure 4-agent competition
- Each agent's observation includes other agents' order IDs (not their signals)
- Population-Based Training (PBT): every 300 steps, evaluate all 4 copies on 20 held-out scenarios
- Bottom 2 copies replaced by perturbed weights of top 2 (LR perturbation ±50%)
- Run on HF compute credits (Colab A100 Pro, same notebook format as Stage 1)

**What makes Stage 2 different from Stage 1:**
- Agents can no longer rely on fixed bot behavior patterns
- Opponents adapt, so agent must generalize beyond pattern matching
- Emergent specialization: one agent may become aggressive, another conservative
- This is where ToM as a strategy (not just a coincidence) becomes plausible

**Failure mode:** If rewards collapse or all agents converge to identical strategies, fall back to Stage 1 results and spend the remaining time on evaluation + demo polish. There is no shame in this — the judges care about demonstrated training improvement, not which stage you reached.

### Why GRPO, Not PPO

- No value model: one less hyperparameter knob, one less failure mode
- GRPO works by scoring a batch of rollouts relative to each other — the "advantage" is just how much better one rollout is than the average of its batch
- This fits our episodic structure perfectly: one episode = one rollout = one reward scalar
- Memory footprint with Unsloth + 4-bit: fits comfortably on a single A100 80GB
- TRL's `GRPOTrainer` has built-in OpenEnv environment hooks as of latest release

### Prompt Format

System prompt (fixed):
```
You are a trading agent in a continuous double-auction market.
You have private information about some components of the stock's true value.
You cannot see other agents' private information.
You will make money by trading intelligently.

Your private signals: {signal_name}: {observed_value}

Current state:
- Order book: {order_book_formatted}
- Recent trades: {recent_trades_formatted}
- Your position: {shares_held} shares, ${cash:.2f} cash
- Unrealized P&L: ${unrealized_pnl:.2f}
- Open orders: {open_orders_formatted}
- Turn: {turn}/{max_turns}

Respond with a JSON action:
{"action_type": "buy"|"sell"|"cancel"|"hold", "price": float, "quantity": int, "reasoning": "..."}
```

Action parsing: Regex-extract JSON from model output. If malformed, treat as "hold" and log the failure. Do not crash on malformed output.

---

## 6. Compute Plan

**Primary resource:** HF compute credits (allocated onsite April 25–26) + Google Colab Pro
**No HPC.** All training lives in `notebooks/train_colab.ipynb` so judges can re-run it — this is a hard judging requirement, not optional.

**Hardware allocation:**
| Resource | Use Case | When |
|----------|----------|------|
| Colab A100 (HF credits) | Stage 1 main training (Qwen2.5-3B-Instruct, GRPO, 3000 steps) | Week 4–5 |
| Colab A100 (HF credits) | Stage 2 self-play or ablations | Week 6–7 |
| Local CPU | All development, unit tests, FastAPI, Docker build/test | Weeks 1–3 |

**Not using GPU for:**
- Order book engine, scenario generator, scripted bots → pure Python, CPU only
- FastAPI server → no GPU needed
- Docker build and local test → local machine only

**Colab notebook structure (`notebooks/train_colab.ipynb`):**
```
Cell 1: pip install all dependencies (pinned versions)
Cell 2: Mount Google Drive for checkpoint saving
Cell 3: Connect to HF Space environment (or run env locally in Colab)
Cell 4: Define rollout function + prompt formatter + action parser
Cell 5: GRPO training loop (configurable steps for quick judge re-runs)
Cell 6: Plot reward curve inline
Cell 7: Run evaluation on held-out scenarios
Cell 8: Save final checkpoint to Drive + HF Hub
```

**Checkpoint strategy:**
- Save every 200 steps to Google Drive (persistent across Colab sessions)
- Also push best checkpoint to HF Hub model repo after training
- Keep best-3 checkpoints by held-out reward

**Expected runtimes on Colab A100:**
- Stage 1 (3000 steps, batch 16, Qwen2.5-3B-Instruct): ~8–10 hours
- Stage 2 (2000 steps, 4 agents): ~18–20 hours (split across sessions if needed)
- Evaluation (50 scenarios × 4 baselines): ~1–2 hours

---

## 7. OpenEnv Compliance

Full compliance with the latest OpenEnv release. Pattern-match against Round 1 Git Merge Resolver for structure — the plumbing is identical.

**Required checklist:**
- [ ] `Environment` base class properly subclassed
- [ ] `reset(task_id=None)` → `MarketObservation` (JSON-serializable)
- [ ] `step(action: MarketAction)` → `(MarketObservation, float, bool, dict)`
- [ ] `state()` → current episode state as JSON
- [ ] `/reset` POST endpoint
- [ ] `/step` POST endpoint
- [ ] `/state` GET endpoint
- [ ] `/tasks` GET endpoint (returns list of available scenario configs)
- [ ] `/health` GET endpoint (returns 200 OK)
- [ ] WebSocket `/ws` endpoint (optional — adds polish)
- [ ] `openenv.yaml` manifest with correct frontmatter
- [ ] `Dockerfile` at repo root (NOT in a subdirectory — learned from Round 1)
- [ ] Client code in `client/` with zero imports from `server/` or `market_env/`
- [ ] No reserved tool names (reset, step, state, close) used as MCP tool names
- [ ] HF Space deployed, `/health` returns 200
- [ ] HF Space sleep timeout set to 48h (so it doesn't sleep during judging)

**openenv.yaml frontmatter (required or HF breaks):**
```yaml
---
title: Multi-Agent Market RL
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: apache-2.0
tags:
  - openenv
  - multi-agent
  - reinforcement-learning
  - theory-of-mind
  - finance
---
```

---

## 8. Evaluation Framework

### Held-Out Test Set

50 market scenarios generated with a fixed seed (seed=42), never used during training. Locked before training begins and never changed.

**Evaluation matrix:**

| Metric | Random | Prompted Qwen2.5-3B | Trained Stage 1 | Trained Stage 2 |
|--------|--------|---------------------|-----------------|-----------------|
| Avg P&L / episode | — | — | — | — |
| Win rate vs InformedBot | — | — | — | — |
| Price efficiency (turn 25) | — | — | — | — |
| Price efficiency (turn 50) | — | — | — | — |
| ToM Proxy 2 (probe acc) | — | — | — | — |
| ToM Proxy 3 (opponent inf) | — | — | — | — |
| Participation rate | — | — | — | — |

**Statistical rigor:**
- Run each baseline 5× with different random seeds
- Report mean ± std for each metric
- Use Mann-Whitney U test for win rate significance (non-parametric, correct for this)
- Bootstrap confidence intervals for P&L metrics

**Baselines:**
1. **Random baseline**: uniform random orders, no information used
2. **Prompted Qwen2.5-3B**: same model, same prompt, zero GRPO training
3. **Trained Stage 1**: your best Stage 1 checkpoint
4. **Trained Stage 2** (if applicable): best self-play checkpoint
5. **InformedBot** (upper bound reference): has true_value + noise, scripted

### ToM Probe Evaluation (20 probe scenarios)
- Each probe: agent receives only public market data, no private signals
- Task: predict whether true_value > $50.00 or < $50.00 (binary classification)
- Metric: accuracy vs majority-vote baseline (should be ~50%)
- If trained agent achieves > 60%: statistically significant ToM evidence

---

## 9. Demo Plan

The demo is the weapon. 30% of judging is storytelling.

**Recorded video (~90 seconds, uploaded to YouTube, linked from README — do NOT commit to repo):**
```
0:00–0:15  Setup shot: "Four agents, each with different private info. None can see the others'."
0:15–0:45  Episode replay: order book updating, thought bubbles showing each agent's private 
           signals + last action. P&L counters running.
0:45–1:00  True value reveal. Final P&L breakdown. Winner shown.
1:00–1:30  Before/after: same scenario with baseline model vs trained model.
           Trained agent makes significantly more money. One number, one chart.
```

**HF Space Gradio interface:**
- Trigger one episode run, watch it play out in ~15 seconds
- Show order book depth chart updating each turn
- Show agent P&L lines over time
- Final reveal screen
- No interactive controls needed — just press "Run Episode"

**Blog post title options:**
- "Teaching LLMs to Read Minds Through Market Data: Emergent Theory-of-Mind from Pure Profit Signal"
- "Theory of Mind for Free: What Happens When You Put LLMs in a Stock Market"
- "Can an LLM Learn What Other Agents Know by Watching How They Trade?"

**Blog structure:**
1. Hook: "We gave four LLMs money and watched them try to steal each other's secrets."
2. Problem: LLMs can't currently infer hidden information from behavioral signals.
3. Environment: one paragraph, one diagram.
4. Training: one paragraph on GRPO, one on why profit signal is enough.
5. Results: one table, one reward curve, one ToM probe chart.
6. Conclusion: what this means for training social reasoning in LLMs.

**Slide deck (for onsite presentation):**
- 8 slides max
- Slide 1: Thesis sentence, one diagram
- Slide 2: Why markets, why ToM
- Slide 3: How the environment works (one diagram)
- Slide 4: The reward function (one equation)
- Slide 5: Training approach (Stage 1 → Stage 2)
- Slide 6: Results table (fill in during Week 6–7)
- Slide 7: ToM probe evidence
- Slide 8: Demo screenshot + QR code to HF Space

---

## 10. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GRPO doesn't converge on Stage 1 | Medium | High | Try Qwen2.5-1.5B, simpler curriculum, smaller batch |
| Stage 2 self-play mode collapses | Medium | Medium | Fall back to Stage 1 — it is independently submittable |
| HF Space sleeps during judging | Low | High | Set sleep_timeout=172800 (48h) in Space settings |
| Colab session times out mid-training | Medium | Medium | Save checkpoints to Drive every 200 steps; resume from checkpoint |
| HF compute credits insufficient | Low | Medium | Qwen2.5-1.5B fallback halves compute; Colab free tier covers eval |
| Reward hacking via order book bugs | Medium | High | Comprehensive unit tests before any training; audit engine first |
| Docker build fails on HF Spaces | Low | Medium | Test locally with `docker build .` before pushing; pin all deps |
| GRPO memory OOM on Colab A100 | Low | Medium | Unsloth 4-bit + gradient checkpointing covers 3B model easily |
| ToM probes show no signal | Medium | Low | Stage 1 results still stand; ToM is a bonus, not the floor |

---

## 11. What Not To Build

These are tempting. Don't touch them.

- **Real market data** — deterministic synthetic is cleaner, reproducible, no data license issues
- **Complex order types** — stop-loss, iceberg, FOK, IOC — unnecessary
- **Multiple stocks** — one stock is enough for the research question
- **LLM-generated news** — deterministic signal generator is more controlled
- **Visualization during training** — only build viz for the final demo
- **GUI dashboards** — Gradio only, and only at the end
- **Continuous-time market** — discrete turns is sufficient and simpler
- **More than 5 agent types** — InformedBot is the hard ceiling for Stage 1
- **Real options pricing or derivatives** — not relevant to the thesis
- **Centralized authority agent** — the environment itself enforces rules

---

## 12. Project File Structure

```
market-rl-env/
│
├── Dockerfile                    # MUST be at root for HF Spaces
├── pyproject.toml                # Package metadata, dependencies
├── openenv.yaml                  # OpenEnv manifest (required)
├── README.md                     # Tells the story; links all materials
│
├── market_env/
│   ├── __init__.py
│   ├── models.py                 # Pydantic: MarketAction, MarketObservation,
│   │                             #   EpisodeReward, MarketScenario, etc.
│   ├── order_book.py             # Order book engine (core, heavily tested)
│   ├── scenario.py               # Scenario generator + curriculum
│   ├── bots.py                   # 5 scripted baseline bots
│   ├── reward.py                 # P&L calculation, aux rewards, penalties
│   ├── environment.py            # OpenEnv Environment subclass
│   └── server.py                 # FastAPI app: /reset /step /state /tasks /health
│
├── client/
│   ├── __init__.py
│   └── client.py                 # OpenEnv client — ZERO imports from market_env/
│
├── training/
│   ├── train_stage1.py           # GRPO single-agent vs scripted bots
│   ├── train_stage2.py           # Population-based self-play
│   ├── evaluate.py               # Held-out evaluation harness
│   ├── tom_probes.py             # Theory-of-mind probe tests
│   └── curriculum.py             # Scenario difficulty scheduler
│
├── notebooks/
│   ├── train_colab.ipynb         # Colab training notebook (judging requirement)
│   └── analysis.ipynb            # Results analysis + plot generation
│
├── tests/
│   ├── test_order_book.py        # Unit: matching, cancels, partial fills, priority
│   ├── test_scenario.py          # Unit: signal generation, assignment, curriculum
│   ├── test_reward.py            # Unit: P&L calc, penalties, edge cases
│   ├── test_bots.py              # Unit: each bot behaves as designed
│   └── test_integration.py       # Full episode cycle: reset → 50 steps → reward
│
├── assets/
│   └── results/
│       ├── reward_curve_stage1.png
│       ├── reward_curve_stage2.png
│       ├── price_efficiency.png
│       ├── tom_probe_accuracy.png
│       └── baseline_comparison.png    # All plots committed to repo, not just Colab
│
└── demo/
    └── gradio_app.py             # HF Space UI — built last, not first
```

---

## 13. Pre-Onsite Checklist (April 25–26)

For the onsite, you need a clear pitch and a plan for the HF compute credits you'll receive.

- [ ] Slide deck (8 slides, see Section 9)
- [ ] One paragraph explaining the thesis to a non-technical judge
- [ ] Order book engine code written (even if not fully tested)
- [ ] Scenario generator working in a Python shell
- [ ] Can demo: `python -c "from market_env.models import MarketAction; print('ok')"`
- [ ] Know exactly what you'll use HF compute credits for: Stage 1 GRPO training (~8h A100), and Stage 2 self-play if Stage 1 converges (~18h A100)

The question judges will ask: "What makes this environment different from anything that already exists?" Answer: "Theory-of-mind emergence from pure profit signal, in an LLM, measured empirically. No existing environment does this."

---

## 14. Decision Points

Three explicit checkpoints where you decide whether to go forward or adjust:

**End of Week 2: Core engine works**
Can you run a 50-turn episode with 5 scripted bots and get a meaningful order book? If the engine is buggy or slow, fix it before writing training code. A broken environment produces garbage training signal.

**End of Week 5: Stage 1 converging**
Is the reward curve going up? Is the agent placing orders? Is it beating RandomBot consistently? If yes → decide whether to push Stage 2. If no → try simpler model, simpler curriculum, or add SFT warm-start before GRPO.

**End of Week 7: Training is locked**
Whatever you have at this point is what you submit. Stop training, start polishing evaluation, demo, README, and blog. No new training runs in the final week.
