"""
Generate notebooks/train_colab.ipynb from this script.

Run:  python -m notebooks._build_notebook
Why a builder: editing raw .ipynb JSON is awful, but the deliverable per
the milestone is a runnable .ipynb. So we keep the source as a readable
Python script and regenerate the notebook on demand.
"""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
    "accelerator": "GPU",
}

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# =============================================================================
# Title + intro
# =============================================================================

md("""
# Stage 1 Training — Multi-Agent Market RL

End-to-end trainer for the Stage 1 LLM agent. Two phases:

1. **SFT warm-start** (~15 min) — fine-tune Qwen2.5-3B on 25k InformedBot
   demonstrations so it learns the JSON action format.
2. **GRPO** (~2 hr at 300 steps) — reinforce profitable actions using a per-action
   oracle reward (uses the hidden true_value at training time, never at
   evaluation).

**Hardware:** A10G small (24 GB VRAM) is the recommended tier — fits
Qwen2.5-3B in 4-bit + LoRA with headroom.

**Configurable knobs** are at the top of the cells they apply to. The
`MAX_GRPO_STEPS` knob in the GRPO cell is the main one — set it to 50
for a smoke run (≈2 min) or 3000 for the full run.

**Outputs:** SFT + GRPO LoRA adapters saved to your Google Drive at
`/MyDrive/market-rl-stage1/`. Push to HF Hub at the end if `HF_TOKEN`
is set.
""")

# =============================================================================
# Cell 1 — install
# =============================================================================

md("## 1. Install dependencies")

code("""\
%%capture
# Pinned to versions that work with Unsloth's GRPO path.
# If anything fails, the most likely culprit is a transformers/trl drift
# — try `!pip install -U unsloth trl` and re-run.
!pip install -q --upgrade pip
!pip install -q "unsloth"
!pip install -q --no-deps "trl>=0.13.0" "peft>=0.13.0"
!pip install -q "datasets>=3.0" "wandb>=0.18" "matplotlib>=3.9"
!pip install -q "pydantic==2.12.5" "fastapi==0.136.1" "uvicorn[standard]==0.43.0" "httpx==0.28.1"
""")

# =============================================================================
# Cell 2 — clone the env code
# =============================================================================

md("""## 2. Clone the env repo

Pulls `market_env/`, `client/`, and `training/`. We run the env
locally inside this Colab process for speed (per-step latency matters
for GRPO rollouts).
""")

code("""\
import os, sys, subprocess

REPO_URL = "https://github.com/PrathameshWable/market-rl-env.git"  # change if you have a different repo
REPO_DIR = "/content/market-rl-env"

if not os.path.isdir(REPO_DIR):
    subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# Sanity check — these must import cleanly
from market_env.environment import MarketEnvironment
from training.prompts import SYSTEM_PROMPT, format_observation, parse_action, serialize_action
print("env + training package import OK")
""")

# =============================================================================
# Cell 3 — secrets, drive, wandb
# =============================================================================

md("""## 3. Secrets, Drive, wandb

Set these as **Colab secrets** (left sidebar → key icon → Add secret):
- `WANDB_API_KEY` — from https://wandb.ai/authorize
- `HF_TOKEN` — from https://huggingface.co/settings/tokens (write scope, only needed if you want to push the checkpoint to HF Hub)
""")

code("""\
import os
from google.colab import drive, userdata

drive.mount("/content/drive")
os.makedirs("/content/drive/MyDrive/market-rl-stage1", exist_ok=True)

# Best-effort: if a secret isn't set, we just disable the corresponding integration.
try:
    os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
    WANDB_ENABLED = True
except Exception:
    print("WANDB_API_KEY not set — wandb logging disabled")
    WANDB_ENABLED = False

try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    HF_PUSH_ENABLED = True
except Exception:
    print("HF_TOKEN not set — won't push checkpoint to HF Hub")
    HF_PUSH_ENABLED = False

if WANDB_ENABLED:
    import wandb
    wandb.login()
    WANDB_ENTITY = "prathameshwable155-wandb"
    WANDB_PROJECT = "market-rl-stage1"
""")

# =============================================================================
# Cell 4 — generate SFT data
# =============================================================================

md("""## 4. Generate SFT warm-start data

Plays InformedBot (cheats — knows the noisy true_value) as the trainable
agent for 500 episodes. ~45 sec on Colab CPU. Faster than downloading
the 46 MB dataset and avoids LFS hassle.
""")

code("""\
from pathlib import Path
from training.generate_sft_data import generate

SFT_PATH = Path("training/sft_data.jsonl")
stats = generate(n_episodes=500, out_path=SFT_PATH, episode_length=50)
print(stats)
""")

# =============================================================================
# Cell 5 — load model with Unsloth
# =============================================================================

md("""## 5. Load Qwen2.5-3B in 4-bit with LoRA

Unsloth handles the 4-bit quantization and LoRA wrapping. ~3 min on first
run (downloads ~2 GB).
""")

code("""\
import torch
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"Model loaded. Trainable params: {model.get_nb_trainable_parameters()}")
""")

# =============================================================================
# Cell 6 — SFT training
# =============================================================================

md("""## 6. SFT warm-start

Trains for 1 epoch on the 25k chat examples. Goal: ≥90% parse rate after
this so GRPO has a sane starting point.
""")

code("""\
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

ds = load_dataset("json", data_files=str(SFT_PATH), split="train")

def render_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False,
    )
    return {"text": text}

ds = ds.map(render_chat, remove_columns=["messages"])

sft_config = SFTConfig(
    output_dir="/content/sft_out",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="no",
    bf16=True,
    max_seq_length=MAX_SEQ_LEN,
    report_to=("wandb" if WANDB_ENABLED else "none"),
    run_name="sft-warmstart",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=sft_config,
)

trainer.train()
trainer.save_model("/content/drive/MyDrive/market-rl-stage1/sft-checkpoint")
print("SFT done. Adapter saved to Drive.")
""")

# =============================================================================
# Cell 7 — SFT eval
# =============================================================================

md("""## 7. SFT eval — parse rate

If this is below 0.85, GRPO will struggle. Diagnostics: re-run cell 6
with more epochs, or check that the system prompt examples are valid JSON.
""")

code("""\
from market_env.environment import MarketEnvironment

FastLanguageModel.for_inference(model)

env = MarketEnvironment()
N_TEST = 50
n_ok = 0

for i in range(N_TEST):
    obs = env.reset(seed=10_000 + i, difficulty="medium", episode_length=5)
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs)},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    _, ok = parse_action(response)
    n_ok += int(ok)

parse_rate = n_ok / N_TEST
print(f"SFT parse rate over {N_TEST} held-out observations: {parse_rate:.1%}")
""")

# =============================================================================
# Cell 8 — GRPO setup explanation
# =============================================================================

md("""## 8. GRPO — explanation

We use a **per-action oracle reward** rather than full episode returns:

For each (observation, generated_action) pair, the reward function:
1. Parses the action JSON. Fail to parse → reward `-1.0`.
2. If buy or sell, computes the *expected* per-share P&L given the
   true_value of that scenario (we know it at training time):
   - `buy at price P`  →  reward ∝ `(true_value − P) / 5`
   - `sell at price P` →  reward ∝ `(P − true_value) / 5`
3. Hold gets a small negative shaping term so the model learns to act.
4. Cancel gets a tiny positive (it's a valid format action).
5. Final reward is clipped to `[-1, 1]`.

This lets us use TRL's standard `GRPOTrainer` (one prompt → K completions
→ K scalars) without needing to fork the env state. The full-episode
evaluation in M6 measures whether the policy actually wins games — that's
the real test, not this surrogate.
""")

# =============================================================================
# Cell 9 — GRPO training
# =============================================================================

code("""\
import json
import random
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# ------------- knobs ------------------------------------------------------
MAX_GRPO_STEPS = 300           # set to 50 for a smoke test
GRPO_BATCH_SIZE = 4            # prompts per step
GRPO_GROUP_SIZE = 4            # completions per prompt (K in GRPO)
PROMPTS_DATASET_SIZE = 1500    # diverse (obs, true_value) pairs
# -------------------------------------------------------------------------

# ----- Build a prompt dataset by running the env with mixed policies -----
# Each prompt carries the hidden true_value as metadata for the reward fn.
from market_env.bots import RandomBot, MarketMakerBot
from training.rollout import bot_policy, run_episode

def collect_prompts(n: int) -> list[dict]:
    env = MarketEnvironment()
    rng = random.Random(0)
    out = []
    seed = 0
    while len(out) < n:
        diff = rng.choice(["easy", "medium"])
        bot = RandomBot("agent_1", seed=seed) if rng.random() < 0.5 else MarketMakerBot("agent_1", seed=seed)
        traj = run_episode(env, bot_policy(bot), seed=seed, difficulty=diff, episode_length=50)
        # Every Nth turn becomes a training prompt
        for i in range(0, len(traj.turns), 5):
            out.append({
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": traj.turns[i].prompt}],
                    tokenize=False, add_generation_prompt=True,
                ),
                "true_value": traj.true_value,
            })
            if len(out) >= n:
                break
        seed += 1
    return out

prompts = collect_prompts(PROMPTS_DATASET_SIZE)
prompt_ds = Dataset.from_list(prompts)
print(f"Built {len(prompts)} GRPO prompts.")


# ----- Reward function: per-action oracle ---------------------------------
HOLD_PENALTY = -0.05
CANCEL_BONUS = 0.02
PARSE_FAIL_PENALTY = -1.0
PNL_SCALE = 5.0     # divide raw $/share P&L by this before clipping

def oracle_reward(prompts, completions, true_value, **_):
    rewards = []
    for completion, tv in zip(completions, true_value):
        action, ok = parse_action(completion if isinstance(completion, str) else completion[0])
        if not ok:
            rewards.append(PARSE_FAIL_PENALTY)
            continue
        if action.action_type == "hold":
            rewards.append(HOLD_PENALTY)
            continue
        if action.action_type == "cancel":
            rewards.append(CANCEL_BONUS)
            continue
        if action.price is None:
            rewards.append(PARSE_FAIL_PENALTY)
            continue
        if action.action_type == "buy":
            r = (tv - action.price) / PNL_SCALE
        else:  # sell
            r = (action.price - tv) / PNL_SCALE
        rewards.append(max(-1.0, min(1.0, r)))
    return rewards


# ----- Train --------------------------------------------------------------
FastLanguageModel.for_training(model)

grpo_config = GRPOConfig(
    output_dir="/content/grpo_out",
    max_steps=MAX_GRPO_STEPS,
    per_device_train_batch_size=GRPO_BATCH_SIZE,
    num_generations=GRPO_GROUP_SIZE,
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    bf16=True,
    max_prompt_length=1024,
    max_completion_length=80,
    temperature=0.9,
    report_to=("wandb" if WANDB_ENABLED else "none"),
    run_name="grpo-stage1",
)

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=oracle_reward,
    args=grpo_config,
    train_dataset=prompt_ds,
)

grpo_trainer.train()
grpo_trainer.save_model("/content/drive/MyDrive/market-rl-stage1/grpo-checkpoint")
print("GRPO done. Adapter saved to Drive.")
""")

# =============================================================================
# Cell 10 — reward curve plot + save
# =============================================================================

md("""## 9. Reward curve + checkpoint sanity check""")

code("""\
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Pull rewards from the trainer's log history
rewards = [
    log["reward"]
    for log in grpo_trainer.state.log_history
    if "reward" in log
]
steps = [
    log["step"]
    for log in grpo_trainer.state.log_history
    if "reward" in log
]

plt.figure(figsize=(10, 4))
plt.plot(steps, rewards, alpha=0.4, label="step reward")
# 50-step rolling mean
if len(rewards) >= 50:
    import numpy as np
    smooth = np.convolve(rewards, np.ones(50)/50, mode="valid")
    plt.plot(steps[49:], smooth, color="red", label="50-step rolling mean")
plt.xlabel("GRPO step")
plt.ylabel("mean reward")
plt.title("Stage 1 GRPO reward curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

OUT_DIR = Path("/content/drive/MyDrive/market-rl-stage1")
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_DIR / "reward_curve_stage1.png", dpi=150)
plt.show()

# Also save the raw log for offline analysis
with open(OUT_DIR / "training_log.json", "w") as fh:
    json.dump(grpo_trainer.state.log_history, fh, indent=2)
print("Plot + log saved to Drive.")
""")

# =============================================================================
# Cell 11 — smoke test against env
# =============================================================================

md("""## 10. Smoke test — does the trained model actually trade well?

Run 5 episodes against the default 4-bot opponent set and report the
trained agent's mean normalized P&L. A positive number means the model
is making money; negative means it's worse than holding.

Full evaluation (50 held-out scenarios + statistical significance) lives
in M6 / `training/evaluate.py`.
""")

code("""\
import statistics
from training.rollout import run_episode

FastLanguageModel.for_inference(model)

def llm_policy(obs):
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs)},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_action(text)

env = MarketEnvironment()
pnls = []
for seed in range(5):
    traj = run_episode(env, llm_policy, seed=20_000 + seed, difficulty="medium")
    pnls.append(traj.reward_breakdown.get("pnl_normalized", 0.0))
    print(f"  seed {seed}: pnl_normalized={pnls[-1]:+.4f}, parse_fail_rate={traj.parse_failure_rate:.1%}")

print(f"\\nMean normalized P&L: {statistics.mean(pnls):+.4f}")
print(f"(positive = beating the buy-and-hold baseline)")
""")

# =============================================================================
# (Optional) Push to HF Hub
# =============================================================================

md("""## 11. (Optional) Push the LoRA adapter to HF Hub

Skipped if `HF_TOKEN` wasn't set as a Colab secret. Useful for sharing
the trained model with judges via a single URL.
""")

code("""\
if HF_PUSH_ENABLED:
    repo_id = "Prathamesh0292/market-rl-stage1"
    grpo_trainer.push_to_hub(repo_id, private=False)
    print(f"Pushed to https://huggingface.co/{repo_id}")
else:
    print("HF_TOKEN not set; skipping push. Set it as a Colab secret to enable.")
""")


# =============================================================================
# Build the notebook
# =============================================================================

nb.cells = cells
out = Path(__file__).parent / "train_colab.ipynb"
with out.open("w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print(f"Wrote {out} ({len(cells)} cells)")
