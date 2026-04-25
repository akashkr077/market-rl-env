# Milestones — Multi-Agent Market RL Environment
# OpenEnv Hackathon Round 2 | 8-Week Plan

Timeline: April 24, 2026 → June 22, 2026
Onsite event: April 25–26, 2026 (present concept, receive HF compute credits)
Compute: HF credits (A100 Colab) + Google Colab Pro — no HPC, all training in runnable notebooks

---

## Pre-Onsite Sprint (April 24–25)
### Goal: Have a clear pitch and a working scaffold to demo

**Tasks:**
- [ ] Finalize thesis in one sentence (it is: "Profit signal alone, with information asymmetry, produces theory-of-mind")
- [ ] Write 8-slide pitch deck
- [ ] Scaffold project directory (`pyproject.toml`, `openenv.yaml`, `market_env/__init__.py`)
- [ ] Write `market_env/models.py` — all Pydantic models (MarketObservation, MarketAction, EpisodeReward, MarketScenario)
- [ ] Verify: `from market_env.models import MarketAction` works without error
- [ ] Know your answer to: "What existing benchmark does this beat?" (None — this is a new capability)

**Done when:** Slide deck is ready. Models import cleanly. You can explain the architecture to a stranger in 90 seconds.

**Note:** You do not need training results for the onsite. You need a compelling idea and a working scaffold. The judges are awarding compute credits, not final results.

---

## M1 — Order Book Engine
### Target: May 1 (Week 1, post-onsite)
### The core of everything. Build it once, test it thoroughly, never revisit.

**Tasks:**
- [ ] Implement `market_env/order_book.py`
  - [ ] `OrderBook` class with bids (max-heap) and asks (min-heap)
  - [ ] `place_limit_order(agent_id, side, price, qty)` → fills immediately if crossing, else rests
  - [ ] `cancel_order(agent_id, order_id)` → error if not found or wrong agent
  - [ ] `get_snapshot(depth=5)` → `OrderBookSnapshot` (top-N bid/ask levels)
  - [ ] `get_recent_trades(n=10)` → list of `TradeRecord`
  - [ ] Price-time priority matching: best price first, then earliest timestamp
  - [ ] Partial fill support: remainder stays on book after partial fill
- [ ] Write `tests/test_order_book.py` — ALL of the following:
  - [ ] Two crossing orders execute at resting order's price
  - [ ] Non-crossing orders both rest on book
  - [ ] Partial fill: correct remainder, correct fill amount
  - [ ] Cancel removes order from book, subsequent match skips it
  - [ ] Price-time priority: two bids at same price, earlier one fills first
  - [ ] Agent can only cancel their own orders (raise error otherwise)
  - [ ] Book snapshot depth is respected (only top-N levels returned)
  - [ ] 1000 random order operations without crash or invariant violation
  - [ ] Empty book: no match, no error

**Done when:** `pytest tests/test_order_book.py` passes, 100% coverage on `order_book.py`. You can run 1000 orders through the book in under 1 second.

**Do not touch:** Training code, models, FastAPI — those come later.

---

## M2 — Scenario Generator + Scripted Bots
### Target: May 7 (Week 2)
### The environment's "world" — what agents play in.

**Tasks:**
- [ ] Implement `market_env/scenario.py`
  - [ ] `MarketScenario` dataclass (true_value, components, agent_signals, noise_configs, difficulty)
  - [ ] `ScenarioGenerator` class with `.sample(difficulty, seed)` method
  - [ ] True value = $50.00 + sum(4 signal components drawn from N(0, σ²))
  - [ ] Signal assignment matrix: Agent 1 sees [earnings, competitor], Agent 2 sees [macro, insider], Agent 3 sees all with high noise
  - [ ] Difficulty controls: Easy (σ_signal=0.10, component_mag=$2–$5), Medium (σ=0.40, mag=$1–$3), Hard (σ=0.80, mag=$0.5–$1.5)
  - [ ] Reproducible: `ScenarioGenerator(seed=42).sample()` always returns same scenario
- [ ] Implement `market_env/bots.py`
  - [ ] `RandomBot`: uniform random orders ±$3 around mid, quantities 1–30, 60% action probability
  - [ ] `MomentumBot`: 3-trade lookback, buys on uptrend, sells on downtrend
  - [ ] `MeanReversionBot`: bets against $1+ deviations from initial mid
  - [ ] `InformedBot`: observes true_value + N(0, $0.40), trades toward value aggressively
  - [ ] `MarketMakerBot`: posts ±$0.30 quotes both sides, quantity 10
  - [ ] All bots implement common `Bot.act(observation) → MarketAction` interface
- [ ] Write `tests/test_scenario.py`
  - [ ] True value is always in [$40, $60]
  - [ ] Agent 1 never receives macro/insider components
  - [ ] Agent 2 never receives earnings/competitor components
  - [ ] Signal noise is correct magnitude per difficulty
  - [ ] Seeded generator is reproducible
- [ ] Write `tests/test_bots.py`
  - [ ] RandomBot: mean order price is close to mid
  - [ ] InformedBot: buys when mid << true_value, sells when mid >> true_value
  - [ ] MarketMakerBot: always quotes both sides
- [ ] Manual validation: Run a 50-turn episode with 5 scripted bots. Print the order book and trade history. Verify it looks like a real market.

**Done when:** Can run `python -c "from market_env.scenario import ScenarioGenerator; from market_env.bots import InformedBot; ..."` and simulate a full episode. `pytest tests/test_scenario.py tests/test_bots.py` passes.

---

## M3 — Environment Core + FastAPI Server
### Target: May 12 (Week 3)
### The OpenEnv-compliant wrapper. Pure plumbing, pattern-match Round 1.

**Tasks:**
- [ ] Implement `market_env/environment.py`
  - [ ] `MarketEnvironment(Environment)` subclassing OpenEnv base class
  - [ ] `reset(task_id=None)` → creates new episode, returns `MarketObservation`
  - [ ] `step(action: MarketAction)` → advances one turn, returns `(MarketObservation, float, bool, dict)`
    - [ ] Turn ends when: all agents have acted (or held) for this turn
    - [ ] Episode ends when: `turn == max_turns` or all agents have no shares + no cash to trade
    - [ ] On episode end: reveal true_value, compute rewards, return done=True
  - [ ] `state()` → JSON dict of current episode state
  - [ ] Session management: `episode_id` → `EpisodeState` dict in memory
  - [ ] Error handling: invalid episode_id, action on ended episode, malformed action
- [ ] Implement `market_env/reward.py`
  - [ ] `compute_episode_reward(episode_state) → EpisodeReward`
  - [ ] Raw P&L: cash_final - cash_initial + shares_final × true_value
  - [ ] Normalized P&L: raw_pnl / (true_value × 100)
  - [ ] Participation bonus: +0.05 if agent placed ≥ 3 orders
  - [ ] Quote stuffing penalty: -0.10 if cancelled > 15 orders
  - [ ] Position limit penalty: -0.05 if |shares_held| > 150 at any point
  - [ ] Clamped total to [-1, 1]
- [ ] Implement `market_env/server.py`
  - [ ] `POST /reset` → calls `env.reset()`, returns `MarketObservation`
  - [ ] `POST /step` → calls `env.step(action)`, returns observation + reward + done
  - [ ] `GET /state` → calls `env.state()`
  - [ ] `GET /tasks` → returns list of available scenario configs
  - [ ] `GET /health` → 200 OK
- [ ] Implement `client/client.py`
  - [ ] `MarketClient` class with `reset()`, `step()`, `state()` methods
  - [ ] Zero imports from `market_env/` (server internals)
  - [ ] Uses `httpx` or `requests` to call the FastAPI endpoints
- [ ] Write `tests/test_integration.py`
  - [ ] Reset → 50 steps → episode ends → reward returned
  - [ ] Invalid episode_id raises correct error
  - [ ] Action on completed episode raises correct error
  - [ ] Malformed action treated as hold (no crash)
  - [ ] Ground truth reward: InformedBot (1 agent) vs empty market → positive P&L

**Done when:** `pytest tests/` passes. Can manually run `uvicorn market_env.server:app` and `curl http://localhost:8000/health` returns `{"status": "ok"}`. Full `reset → step × 50 → done` cycle completes without error from Python shell.

---

## M4 — Docker + HF Space v1 Deployment
### Target: May 16 (Week 3, end)
### Deploy early to catch plumbing bugs before training. A deployed v1 is a safety net.

**Tasks:**
- [ ] Write `Dockerfile` (at repo root — never in a subdirectory)
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY pyproject.toml .
  RUN pip install -e .
  COPY . .
  EXPOSE 7860
  CMD ["uvicorn", "market_env.server:app", "--host", "0.0.0.0", "--port", "7860"]
  ```
- [ ] Write `openenv.yaml` with correct frontmatter (title, emoji, sdk: docker, tags: openenv)
- [ ] Local Docker test: `docker build . -t market-rl && docker run -p 7860:7860 market-rl`
  - [ ] `curl http://localhost:7860/health` → 200
  - [ ] `curl -X POST http://localhost:7860/reset` → valid MarketObservation JSON
  - [ ] `curl -X POST http://localhost:7860/step` → valid response
- [ ] Push to HF Space
  - [ ] Create new Space (Docker SDK)
  - [ ] `git push` to Space repo
  - [ ] Wait for build — monitor build logs
  - [ ] Verify Space shows "Running" status
  - [ ] `curl https://<space-url>/health` → 200
- [ ] Set HF Space sleep_timeout to 48h (so it doesn't sleep during judging)
- [ ] Pin all package versions in `pyproject.toml` (avoids HF build breaking on future releases)

**Done when:** HF Space shows "Running". `/health` returns 200 from the public URL. `/reset` returns a valid `MarketObservation`. Document the Space URL.

**Important:** Do NOT start training until this milestone is complete. A broken deployment wastes training time.

---

## M5 — Colab Notebook + Stage 1 Training
### Target: May 26 (Weeks 4–5)
### First real training run. Most important milestone for judging.

### Week 4 — SFT Warm-Start + Notebook Setup (May 16–19)

**Step 0: SFT warm-start (default, not optional)**
Before any GRPO, run a short supervised fine-tune to teach the model to produce valid JSON actions reliably. Without this, Qwen2.5-3B will output free-form English for the first ~500 GRPO steps, parse-failure penalties pile up, and gradient signal is near-zero.

- [ ] Run 500 episodes with InformedBot as the "teacher" agent in the training environment
- [ ] Save `(formatted_observation, action_json)` pairs to `training/sft_data.jsonl`
- [ ] Fine-tune Qwen2.5-3B-Instruct for 1 epoch on this data using HF TRL `SFTTrainer`
- [ ] Verify: after SFT, ≥ 90% of model outputs parse as valid JSON on 50 test observations
- [ ] Save SFT checkpoint to Drive — GRPO starts from this checkpoint, not from base model
- [ ] Expected time: ~1–2 hours on Colab A100

**Notebook setup tasks (`notebooks/train_colab.ipynb`):**
- [ ] Cell 1: Install all dependencies with pinned versions
  ```
  !pip install unsloth==... trl==... transformers==... pydantic openenv
  ```
- [ ] Cell 2: Mount Google Drive for checkpoint saving
- [ ] Cell 3: Install + start the market environment (either connect to HF Space URL or run FastAPI in Colab subprocess)
- [ ] Cell 4: Rollout function — calls `env.reset()`, loops `env.step()`, collects trajectory + reward
- [ ] Cell 5: Prompt formatter — converts `MarketObservation` to model-readable text string
- [ ] Cell 6: Action parser — regex-extracts JSON from model output, falls back to "hold" on failure
- [ ] Cell 7: Curriculum scheduler — Easy steps 0–600, mixed to 1500, full to 2500+
- [ ] Cell 8: GRPO training loop using HF TRL `GRPOTrainer`
  - [ ] Reward normalization in [-1, 1] before passing to trainer
  - [ ] Log every 50 steps: average_reward, pnl, participation_rate, cancel_rate
  - [ ] Save checkpoint to Google Drive every 200 steps, keep best-3
- [ ] Cell 9: Inline reward curve plot (matplotlib, labeled axes)
- [ ] Validation run (200 steps, batch=8) before full run:
  - [ ] Verify loop doesn't crash
  - [ ] Verify actions are being parsed (not all "hold")
  - [ ] Verify memory fits in A100 VRAM

### Week 5 — Full Training Run (May 19–26)

- [ ] Full Stage 1 run: 3000 steps, batch=16, Qwen2.5-3B-Instruct, Colab A100 (HF credits)
- [ ] Save reward curve plot as `assets/results/reward_curve_stage1.png` and commit to repo
- [ ] Check every few hours: is reward going up? Any reward hacking signatures?
- [ ] Read 5 episode transcripts manually at step 500, 1000, 2000 — don't just watch the plot
- [ ] If reward stalls: switch to Qwen2.5-1.5B, simplify curriculum, or reduce batch
- [ ] Push best checkpoint to HF Hub model repo

**Done when:** Reward curve shows clear upward trend over baseline. Agent participates on ≥ 70% of turns. At least one checkpoint beats RandomBot on 10 held-out scenarios. Notebook runs cell-by-cell without errors.

**Checkpoint:** If Stage 1 is not improving by step 800 on the first run, stop and diagnose before spending more compute. Common causes: action parsing failures (all "hold"), reward scale too large, curriculum too hard too early.

---

## M6 — Evaluation + Theory-of-Mind Probes
### Target: June 2 (Week 6) — THE CRITICAL DECISION POINT
### Lock in what you have. Decide whether to go to Stage 2.

**Tasks:**
- [ ] Implement `training/evaluate.py`
  - [ ] Load any checkpoint and evaluate on the held-out 50-scenario test set
  - [ ] Output: average P&L, win rate vs each bot, price efficiency at turns 25 and 50
  - [ ] Saves results as JSON + formatted table
- [ ] Implement `training/tom_probes.py`
  - [ ] Probe 1: Price efficiency — plot `|mid - true_value|` over turns for each baseline
  - [ ] Probe 2: Order flow prediction — given only public data, can model predict value direction?
  - [ ] Probe 3: Opponent signal inference — can model predict if Agent 2 has positive/negative info?
  - [ ] Probe 4: Behavioral adaptation — correlate order_book_imbalance with agent_aggressiveness
- [ ] Run full evaluation on all baselines:
  - [ ] Random baseline (5 seeds)
  - [ ] Prompted Qwen2.5-3B, no training (5 seeds)
  - [ ] Trained Stage 1 best checkpoint (5 seeds)
- [ ] Generate all plots (`assets/results/*.png`) — labeled axes, units, readable in 5 seconds
- [ ] Fill in the evaluation matrix table from planning.md
- [ ] Compute statistical significance (Mann-Whitney U for win rates, bootstrap CI for P&L)

**Decision: Stage 2 or not?**

Go to Stage 2 if ALL of the following:
- [ ] Trained agent beats InformedBot on ≥ 40% of held-out episodes
- [ ] Reward curve shows clear upward trend (not just noise)
- [ ] There are ≥ 2 weeks left in the timeline

Stay with Stage 1 if ANY of the following:
- Agent barely beats RandomBot
- Less than 2 weeks remain
- HF compute credits are running low (save remaining for evaluation)

If staying with Stage 1: spend the remaining time on more training steps, better ablations, better demo, and polishing the blog. A great Stage 1 with compelling evaluation beats a broken Stage 2.

**Done when:** Evaluation matrix is fully populated with numbers. Plots are committed to `assets/results/`. Decision on Stage 2 is made and documented.

---

## M7A — Stage 2: Population-Based Self-Play (conditional)
### Target: June 14 (Weeks 6–7) — only if M6 decision says yes
### Where theory-of-mind as an emergent property becomes a defensible claim.

**Tasks:**
- [ ] Implement `training/train_stage2.py`
  - [ ] 4 concurrent agent copies (same architecture, different seeds)
  - [ ] Each episode: 4 trained agents compete on same scenario (no scripted bots)
  - [ ] Population-Based Training loop: evaluate every 300 steps, replace bottom 2
  - [ ] Mutation: perturb LR and temperature of top-2 copies to generate replacements
  - [ ] Log per-agent rewards separately (not just average)
- [ ] Run on Colab A100 (HF compute credits) — same notebook format as Stage 1
  - [ ] Expected: ~20 hours for 2000 steps (split across Colab sessions if needed, resume from Drive checkpoint)
  - [ ] Monitor every few hours: are agents diversifying in strategy? Or all converging?
- [ ] Stage 2 evaluation:
  - [ ] Take best Stage 2 checkpoint
  - [ ] Evaluate on same 50 held-out scenarios
  - [ ] Add Stage 2 column to evaluation matrix
  - [ ] Specifically: does Stage 2 agent beat Stage 1 agent?
- [ ] Stage 2 ToM analysis:
  - [ ] Does Stage 2 agent score higher on ToM Probe 3 (opponent signal inference)?
  - [ ] Is price efficiency higher with 4 Stage 2 agents vs 4 Stage 1 agents?
- [ ] Generate Stage 2 reward curve plot and commit to `assets/results/`

**Done when:** Stage 2 training run is complete. Best Stage 2 checkpoint is saved to Google Drive and pushed to HF Hub. Evaluation matrix has Stage 2 column filled.

---

## M7B — Stage 1 Polish (alternative if M6 decision is no)
### Target: June 14 (Weeks 6–7) — if Stage 2 is not attempted
### Make Stage 1 results as strong as possible.

**Tasks:**
- [ ] Run additional training steps on best Stage 1 hyperparameters (up to 5000 total)
- [ ] Ablation: train with auxiliary rewards enabled vs disabled (compare performance)
- [ ] Ablation: train with vs without curriculum (show curriculum helps)
- [ ] Ablation: 1.5B vs 3B model (show scale matters)
- [ ] Run evaluation on all ablations, add to results table
- [ ] Improve action parser to handle more malformed outputs gracefully
- [ ] Write `notebooks/analysis.ipynb` — all plots generated programmatically from saved logs

**Done when:** Best Stage 1 checkpoint is finalized. All ablation results are documented. Plots are committed.

---

## M8 — Demo + Blog + README + Submission
### Target: June 22 (Week 8)
### Ship everything. Presentation quality matters as much as results.

### Demo (June 14–16)
- [ ] Implement `demo/gradio_app.py`
  - [ ] "Run Episode" button → runs one full episode (≤ 15 seconds with cached model)
  - [ ] Order book depth chart updating each turn (bar chart, bids vs asks)
  - [ ] Agent P&L line chart over episode turns
  - [ ] Final results card: each agent's P&L, true value revealed, winner highlighted
  - [ ] Second tab: "Before vs After" — same scenario with baseline model vs trained model
- [ ] Record demo video (screen recording, ~90 seconds)
  - [ ] Upload to YouTube (unlisted or public) — do NOT commit to repo, file is too large
  - [ ] Add YouTube link to README and to `blog.md`

### Blog Post (June 15–17)
- [ ] Write HF blog post as a markdown file placed **in the environment repo** (per hackathon tip)
  - [ ] Filename: `blog.md` at repo root
  - [ ] Title: "Theory of Mind for Free: What Happens When You Put LLMs in a Stock Market"
  - [ ] Hook: one paragraph, one diagram
  - [ ] Problem: one paragraph
  - [ ] Environment: one paragraph + one code snippet (the MarketObservation)
  - [ ] Training: one paragraph on GRPO
  - [ ] Results: one table, one reward curve image embedded
  - [ ] Conclusion: why this matters for training social reasoning
  - [ ] Link to HF Space, GitHub repo, Colab notebook
- [ ] Optionally also publish to HuggingFace blog (same markdown, copy-paste)

### Colab Training Notebook (June 16–18)
- [ ] `notebooks/train_colab.ipynb` — must be runnable by judges on a T4 or A100
  - [ ] Installs all dependencies in first cell (pinned versions, no user input required)
  - [ ] Connects to the deployed HF Space as the environment
  - [ ] Configurable `MAX_STEPS` variable at top — set to 50 for judge test-run, 3000 for real
  - [ ] Runs end-to-end: install → connect → train → plot → save checkpoint
  - [ ] Plots reward curve inline with labeled axes
  - [ ] Saves model checkpoint to Google Drive (with fallback to `/tmp` if Drive not mounted)
  - [ ] Markdown cells explaining each section — judges read this, make it clear
- [ ] Share notebook as **publicly accessible Google Colab link** (File → Share → Anyone with link)
- [ ] Add Colab link to README (per hackathon tip — link not .ipynb is the preferred format)

### README (June 18–20)
- [ ] Section 1: What is this? (one diagram, one paragraph)
- [ ] Section 2: Why it's interesting (the thesis sentence, the ToM claim)
- [ ] Section 3: How to run it locally (3 commands max)
- [ ] Section 4: How to run training (link to Colab notebook)
- [ ] Section 5: Results (embed the evaluation table + reward curve image)
- [ ] Section 6: ToM evidence (embed probe accuracy chart)
- [ ] Section 7: Links (HF Space, blog post, demo video, Colab notebook, paper-style writeup)
- [ ] Frontmatter: the openenv.yaml tags must also appear in README for HF discoverability

### Final checks (June 20–22)
- [ ] All plots embedded in README with one-line captions
- [ ] All external links work (blog, video, Colab)
- [ ] HF Space is "Running" — verify `/health` returns 200
- [ ] `pytest tests/` passes on a clean clone
- [ ] `docker build . && docker run -p 7860:7860 market-rl` — verify local build works
- [ ] Colab notebook runs end-to-end without errors (test this yourself)
- [ ] Submit environment URL on hackathon platform

**Done when:** URL submitted. Space shows Running. README has all links. Blog is published. Video is linked. Colab notebook is public.

---

## Submission Checklist (Final)

### Non-negotiable (missing any = serious disadvantage):
- [ ] OpenEnv latest release used (not monkey-patched, properly subclassed)
- [ ] Working training script in Colab (Unsloth + HF TRL)
- [ ] Evidence of actual training: reward curve plot + evaluation table from real run
- [ ] Short writeup: HF blog post or YouTube video < 2 minutes
- [ ] HF Space deployed and accessible (public URL returns 200)
- [ ] README with problem motivation, environment explanation, results, all external links

### Strong submission extras:
- [ ] Multiple baselines compared (random, prompted, trained)
- [ ] Statistical significance reported (not just raw numbers)
- [ ] ToM probe results (the differentiator)
- [ ] Before/after demo (same scenario, baseline vs trained)
- [ ] Reward curve comparison (Stage 1 vs Stage 2 if applicable)
- [ ] Colab notebook actually runnable by a judge (test this)

### Polish items (if time allows):
- [ ] Weights committed to HF Hub (or linked)
- [ ] Wandb run link included in README
- [ ] Interactive Gradio demo in the Space
- [ ] Separate evaluation results notebook with all analysis reproducible

---

## Timeline Summary

| Week | Dates | Primary Focus | Compute | Key Deliverable |
|------|-------|---------------|---------|-----------------|
| Pre | Apr 24–26 | Onsite pitch + scaffold | CPU only | Slides, models.py |
| 1 | Apr 27 – May 1 | Order book engine | CPU only | M1: Tested order book |
| 2 | May 2–7 | Scenario gen + bots | CPU only | M2: Full episode simulation |
| 3 | May 8–16 | OpenEnv wrapper + deploy | CPU only | M3 + M4: HF Space live |
| 4 | May 16–19 | Colab notebook + validation run | Colab A100 (HF credits, short) | M5 start: Loop verified |
| 5 | May 19–26 | Full Stage 1 training | Colab A100 (HF credits, ~10h) | M5: Reward curve going up |
| 6 | May 27 – Jun 2 | Evaluation + ToM probes + decision | Colab A100 (eval, ~2h) | M6: Results table filled |
| 7 | Jun 3–14 | Stage 2 OR Stage 1 polish | Colab A100 (HF credits, ~20h) | M7: Best checkpoint locked |
| 8 | Jun 15–22 | Demo + blog + README + submit | CPU only | M8: Submitted |

---

## Post-Mortem Notes (to fill in as you go)

Use this section during the project to record decisions, surprises, and time spent. Future-you will thank present-you.

```
Week 1: 
Week 2: 
Week 3: 
Week 4: 
Week 5: 
Week 6: 
Week 7: 
Week 8: 
```

---

actual lessons from Round 1 to carry forward:
- Docker root path issue cost 30 min: Dockerfile MUST be at repo root
- Tasks and graders took 2× longer than expected: same will apply to scripted bots + reward function
- Deploy early (M4 is early) — deploying at the end is where problems hide
- The inference script is simpler than the environment — don't neglect environment testing
