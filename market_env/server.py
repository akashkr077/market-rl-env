"""
FastAPI HTTP layer for the market environment.

Exposes the standard OpenEnv endpoints on top of MarketEnvironment.

Endpoints:
    GET  /health        — liveness probe
    GET  /tasks         — list available task configurations
    POST /reset         — start a new episode, return initial observation
    POST /step          — advance an episode by one turn
    GET  /state         — query the current state of an episode

Run locally:
    uvicorn market_env.server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from market_env.environment import (
    EpisodeAlreadyDone,
    EpisodeNotFound,
    MarketEnvironment,
)
from market_env.models import MarketAction, MarketObservation


app = FastAPI(
    title="Multi-Agent Market RL Environment",
    description=(
        "OpenEnv-compliant trading environment for theory-of-mind LLM training. "
        "Agents trade on a continuous double-auction order book with asymmetric "
        "private information; profit signal is the primary reward."
    ),
    version="0.1.0",
)

# Single env instance shared across all requests; manages multiple sessions internally.
env = MarketEnvironment()


# ---------------------------------------------------------------------------
# Request/response wrappers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: int = 42
    difficulty: str = "medium"
    bot_config: str = "default"
    trainable_agent_id: str = "agent_1"
    episode_length: int = 50


class StepRequest(BaseModel):
    episode_id: str
    action: MarketAction


class StepResponse(BaseModel):
    observation: MarketObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks() -> list[dict[str, Any]]:
    return env.list_tasks()


@app.post("/reset", response_model=MarketObservation)
def reset(req: Optional[ResetRequest] = None) -> MarketObservation:
    if req is None:
        req = ResetRequest()
    try:
        return env.reset(
            task_id=req.task_id,
            seed=req.seed,
            difficulty=req.difficulty,
            bot_config=req.bot_config,
            trainable_agent_id=req.trainable_agent_id,
            episode_length=req.episode_length,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    try:
        obs, reward, done, info = env.step(req.episode_id, req.action)
    except EpisodeNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EpisodeAlreadyDone as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def get_state(episode_id: str = Query(...)) -> dict[str, Any]:
    try:
        return env.state(episode_id)
    except EpisodeNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
