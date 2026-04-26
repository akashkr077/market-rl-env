"""
Streamlit dashboard for Market RL Environment — Stage 1 Results.

Run:  streamlit run dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS = Path(__file__).parent / "assets" / "results"

st.set_page_config(
    page_title="Market RL — Stage 1 Results",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_json(name: str) -> dict:
    with open(RESULTS / name) as f:
        return json.load(f)


def load_eval() -> pd.DataFrame:
    raw = load_json("evaluation_stage1.json")
    rows = []
    for policy, m in raw.items():
        rows.append({
            "Policy": policy,
            "Avg P&L ($)": round(m["avg_pnl"], 1),
            "Avg Reward": round(m["avg_reward"], 3),
            "Win Rate": round(m["win_rate"] * 100, 1),
            "Parse Fail (%)": round(m["parse_fail_rate"] * 100, 1),
            "Participation (%)": round(m["participation"] * 100, 1),
        })
    df = pd.DataFrame(rows)
    order = ["InformedBot", "MarketMakerBot", "HoldBaseline", "Trained Stage 1", "RandomBot"]
    df["_sort"] = df["Policy"].map({n: i for i, n in enumerate(order)})
    return df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def load_checkpoints() -> pd.DataFrame:
    raw = load_json("checkpoint_comparison.json")
    rows = []
    for ckpt, m in raw.items():
        rows.append({
            "Checkpoint": ckpt,
            "Avg P&L ($)": round(m["avg_pnl"], 1),
            "Avg Reward": round(m["avg_reward"], 3),
            "Win Rate (%)": round(m["win_rate"] * 100, 1),
            "Parse Fail (%)": round(m["parse_fail_rate"] * 100, 1),
            "Participation (%)": round(m["participation"] * 100, 1),
        })
    return pd.DataFrame(rows)


def load_probes() -> dict:
    return load_json("hidden_state_probes.json")


def load_behavior() -> dict:
    return load_json("behavioral_adaptation.json")


def load_price_eff() -> dict:
    return load_json("price_efficiency.json")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Market RL Env")
    st.caption("OpenEnv Hackathon Round 2 — April 2026")
    st.divider()
    st.markdown("""
    **Links**
    - [HF Space](https://huggingface.co/spaces/Prathamesh0292/market-rl-env)
    - [GitHub](https://github.com/PrathameshWable/market-rl-env)
    - [Trained Adapter](https://huggingface.co/Prathamesh0292/market-rl-stage1)
    - [Colab Notebook](https://colab.research.google.com/drive/1dVUBw60a5JrGvVYdcL3wdZVQ1QGfXnre?usp=sharing)
    """)
    st.divider()
    st.markdown("**Stage 1 Summary**")
    st.metric("Best Checkpoint", "GRPO-250")
    st.metric("Best P&L", "−$23.7")
    st.metric("Parse Fail Rate", "0%")
    st.metric("Participation", "100%")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Theory of Mind for Free")
st.markdown(
    "**What happens when you put LLMs in a stock market with hidden information?**  \n"
    "We trained Qwen2.5-1.5B with SFT + GRPO in an information-asymmetric "
    "continuous double-auction market. 5 agents, 4 signal components, 50 turns per episode."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Introduction",
    "Evaluation Results",
    "Checkpoint Progression",
    "ToM Probes",
    "All Charts",
    "Key Findings",
])

# ========================== TAB 1: Introduction ============================

with tab1:
    st.header("The Environment")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        A **continuous double-auction limit order book** where 5 agents trade a
        single asset over 50 turns.

        - **1 trainable agent** (Qwen2.5-1.5B LLM)
        - **4 scripted bots** — MarketMaker, Momentum, MeanReversion, Random

        The asset has a hidden **true value** composed of 4 signal components.
        Each agent sees only a noisy subset of signals. Nobody sees the full picture.

        ```
        true_value ≈ $50 + earnings + competitor + macro + insider
        ```

        The trainable agent outputs a JSON action each turn:
        `{"action_type": "buy", "price": 51.5, "quantity": 10}`

        Reward = normalized P&L at episode end.
        """)

    with col2:
        st.markdown("#### Training Pipeline")
        st.markdown("""
        | Phase | What | Time |
        |-------|------|------|
        | **SFT** | Learn JSON format from InformedBot demos | ~15 min |
        | **GRPO** | Optimize for profit (300 steps) | ~2 hr |

        **Reward shaping:**
        - Parse fail → −1.00
        - Hold → −0.05
        - Buy at P → clip((true\\_value − P) / 5, −1, +1)
        - Sell at P → clip((P − true\\_value) / 5, −1, +1)
        """)

    st.info(
        "**Key insight:** `true_value` is used for reward at training time only. "
        "The model never sees it during evaluation — it must infer from signals and order flow."
    )

# ========================== TAB 2: Evaluation ==============================

with tab2:
    st.header("Held-Out Evaluation — 50 Episodes")

    df_eval = load_eval()

    # Bar chart
    fig = px.bar(
        df_eval, x="Avg P&L ($)", y="Policy", orientation="h",
        color="Avg P&L ($)",
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
        text="Avg P&L ($)",
    )
    fig.update_traces(textposition="outside", texttemplate="%{text:+.1f}")
    fig.update_layout(
        height=350, yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        title="Average P&L by Policy",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    trained = df_eval[df_eval["Policy"] == "Trained Stage 1"].iloc[0]
    c1.metric("Trained P&L", f"${trained['Avg P&L ($)']:+.1f}")
    c2.metric("Win Rate", f"{trained['Win Rate']:.0f}%")
    c3.metric("Parse Fail", f"{trained['Parse Fail (%)']:.0f}%")
    c4.metric("Participation", f"{trained['Participation (%)']:.0f}%")

    # Full table
    st.dataframe(
        df_eval.style.format({
            "Avg P&L ($)": "{:+.1f}",
            "Avg Reward": "{:+.3f}",
            "Win Rate": "{:.1f}%",
            "Parse Fail (%)": "{:.1f}%",
            "Participation (%)": "{:.1f}%",
        }).highlight_max(subset=["Avg P&L ($)", "Win Rate"], color="#2ecc7144")
        .highlight_min(subset=["Parse Fail (%)"], color="#2ecc7144"),
        use_container_width=True,
        hide_index=True,
    )

# ========================== TAB 3: Checkpoints =============================

with tab3:
    st.header("Checkpoint Progression — SFT → GRPO")

    df_ckpt = load_checkpoints()

    # 4-panel chart
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Avg P&L ($)", "Avg Reward", "Win Rate (%)", "Participation (%)"
    ])

    metrics = ["Avg P&L ($)", "Avg Reward", "Win Rate (%)", "Participation (%)"]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for metric, (r, c), color in zip(metrics, positions, colors):
        bar_colors = [
            "#f39c12" if ckpt == "GRPO-250" else color
            for ckpt in df_ckpt["Checkpoint"]
        ]
        fig.add_trace(
            go.Bar(
                x=df_ckpt["Checkpoint"], y=df_ckpt[metric],
                marker_color=bar_colors,
                text=df_ckpt[metric].apply(
                    lambda v: f"{v:+.1f}" if abs(v) > 1 else f"{v:+.3f}"
                ),
                textposition="outside",
                showlegend=False,
            ),
            row=r, col=c,
        )

    fig.update_layout(height=550, title_text="GRPO-250 is the sweet spot")
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.dataframe(
        df_ckpt.style.format({
            "Avg P&L ($)": "{:+.1f}",
            "Avg Reward": "{:+.3f}",
            "Win Rate (%)": "{:.1f}%",
            "Parse Fail (%)": "{:.1f}%",
            "Participation (%)": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        "**GRPO-250** is the best checkpoint: −$23.7 P&L, +0.05 reward, 55% win rate. "
        "GRPO-300 degraded — classic RL overfitting."
    )

# ========================== TAB 4: ToM Probes ==============================

with tab4:
    st.header("Theory-of-Mind Probes")

    # Probe 1: Price Efficiency
    st.subheader("Probe 1 — Price Efficiency")
    st.markdown("Does trading pull market prices toward the hidden true value? Lower gap = better.")

    pe_data = load_price_eff()
    pe_turns = pe_data["per_turn_mean_gap"]

    fig = go.Figure()
    bot_colors = {
        "RandomBot": "#e74c3c", "MarketMakerBot": "#3498db",
        "HoldBaseline": "#95a5a6", "InformedBot": "#2ecc71",
        "MomentumBot": "#e67e22", "MeanReversionBot": "#9b59b6",
    }
    for bot, gaps in pe_turns.items():
        clean = [g for g in gaps if g is not None]
        fig.add_trace(go.Scatter(
            x=list(range(1, len(clean) + 1)), y=clean,
            name=bot, mode="lines",
            line=dict(width=2.5, color=bot_colors.get(bot, "#888")),
        ))
    fig.update_layout(
        height=400,
        xaxis_title="Turn", yaxis_title="|Mid Price − True Value| ($)",
        title="Price Gap Over Time (lower = better price discovery)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Probes 2 & 3: Hidden State
    st.subheader("Probes 2 & 3 — Hidden State Probes")
    st.markdown(
        "Can a linear classifier predict hidden information from the model's "
        "internal representations? Chance level = 50%."
    )

    probes = load_probes()

    col1, col2 = st.columns(2)
    with col1:
        p2 = probes["probe2_true_value"]
        acc2 = p2["accuracy"] * 100
        st.metric("Probe 2: True Value Direction", f"{acc2:.1f}%", f"+{acc2-50:.1f}pp vs chance")
        fig2 = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=[v * 100 for v in p2["per_fold"]],
            marker_color="#e67e22",
            text=[f"{v*100:.0f}%" for v in p2["per_fold"]],
            textposition="outside",
        ))
        fig2.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Chance")
        fig2.update_layout(height=300, yaxis_range=[0, 110], title="Probe 2 Per-Fold Accuracy")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        p3 = probes["probe3_signal"]
        acc3 = p3["accuracy"] * 100
        st.metric("Probe 3: Signal Direction", f"{acc3:.1f}%", f"+{acc3-50:.1f}pp vs chance")
        fig3 = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=[v * 100 for v in p3["per_fold"]],
            marker_color="#3498db",
            text=[f"{v*100:.0f}%" for v in p3["per_fold"]],
            textposition="outside",
        ))
        fig3.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Chance")
        fig3.update_layout(height=300, yaxis_range=[0, 110], title="Probe 3 Per-Fold Accuracy")
        st.plotly_chart(fig3, use_container_width=True)

    verdict2 = "SIGNAL FOUND" if acc2 > 60 else "No signal"
    verdict3 = "SIGNAL FOUND" if acc3 > 60 else "No signal"
    st.success(
        f"**Probe 2** ({acc2:.1f}%): {verdict2} — model encodes true value direction.  \n"
        f"**Probe 3** ({acc3:.1f}%): {verdict3} — model encodes private signal direction."
    )

    st.divider()

    # Probe 4: Behavioral Adaptation
    st.subheader("Probe 4 — Behavioral Adaptation")
    st.markdown("Does the agent adjust aggressiveness based on order book imbalance?")

    behav = load_behavior()
    rows = []
    for bot, d in behav["policies"].items():
        if d["correlation"] is not None:
            rows.append({
                "Bot": bot,
                "Correlation (r)": round(d["correlation"], 4),
                "p-value": round(d["p_value"], 4) if d["p_value"] else None,
                "Significant": "Yes" if d["p_value"] and d["p_value"] < 0.05 else "No",
                "N": d["n_observations"],
            })
    df_behav = pd.DataFrame(rows)
    st.dataframe(df_behav, use_container_width=True, hide_index=True)

    sig_bots = df_behav[df_behav["Significant"] == "Yes"]["Bot"].tolist()
    if sig_bots:
        st.info(f"**Statistically significant adaptation:** {', '.join(sig_bots)}")

# ========================== TAB 5: All Charts ==============================

with tab5:
    st.header("All Generated Charts")

    pngs = sorted(RESULTS.glob("*.png"))
    if not pngs:
        st.warning("No PNG files found in assets/results/")
    else:
        cols_per_row = 2
        for i in range(0, len(pngs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(pngs):
                    with col:
                        st.image(
                            str(pngs[idx]),
                            caption=pngs[idx].name,
                            use_container_width=True,
                        )

# ========================== TAB 6: Key Findings ============================

with tab6:
    st.header("Key Findings")

    # Big numbers
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SFT → GRPO-250", "+$281", "P&L improvement")
    c2.metric("Parse Failures", "0%", "Across all checkpoints")
    c3.metric("Hidden State Probes", "72–74%", "Above 50% chance")
    c4.metric("GRPO Peak", "Step 250", "Degraded at 300")

    st.divider()

    st.subheader("GRPO Overfitting at 300 Steps")
    st.markdown("""
    The checkpoint progression tells a clear story:

    | Checkpoint | Avg P&L | Key Change |
    |---|---|---|
    | **SFT** | −$304.6 | Learned format, not strategy |
    | **GRPO-200** | −$161.2 | +$143 improvement, 100% participation |
    | **GRPO-250** | −$23.7 | Sweet spot — nearly break-even |
    | **GRPO-300** | −$168.2 | Collapsed — classic RL overfitting |

    GRPO-250 had positive average reward (+0.05) and 55% win rate.
    GRPO-300 overfit to training scenarios and lost generalization.
    """)

    st.divider()

    st.subheader("What the Probes Tell Us")
    st.markdown("""
    - **Probe 1 (Price Efficiency):** InformedBot drives prices closest to
      true value. RandomBot surprisingly decent — random orders still move prices.
    - **Probe 2 (True Value Direction):** 72.2% accuracy — the model's hidden
      states encode information about the unobserved true value. This is above chance.
    - **Probe 3 (Signal Direction):** 74.4% accuracy — the model has learned to
      represent its own private signal information internally.
    - **Probe 4 (Behavioral Adaptation):** MarketMakerBot shows statistically
      significant adaptation (r=−0.054, p=0.007).
    """)

    st.divider()

    st.subheader("Stage 2 Roadmap")
    st.markdown("""
    1. **Resume from GRPO-250** — the best checkpoint, not the final one
    2. **More steps with early stopping** — 500–1000 steps, evaluate every 25
    3. **Self-play** — replace scripted bots with trained model copies
    4. **Larger model** — Qwen2.5-7B for better generalization
    """)

    st.divider()

    st.markdown(
        "> **A profit signal alone, in an information-asymmetric market, is sufficient "
        "to train an LLM agent that learns valid action formatting (0% parse failure), "
        "actively participates (100% vs 77% SFT), and improves P&L by +$281 over its "
        "SFT starting point.**"
    )
