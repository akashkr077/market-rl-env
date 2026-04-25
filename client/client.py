"""
HTTP client for the market environment server.

Usage:
    from client import MarketClient
    from market_env.models import MarketAction

    client = MarketClient("http://localhost:7860")
    obs = client.reset(seed=42)
    while True:
        action = MarketAction(action_type="hold")
        obs, reward, done, info = client.step(obs.episode_id, action)
        if done:
            print("final reward:", reward)
            break

The client imports Pydantic models from market_env.models (pure data
definitions only). It does NOT import any server-side modules
(environment, server, order_book, scenario, bots, reward).
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from market_env.models import MarketAction, MarketObservation


class MarketClientError(Exception):
    """Raised when the server returns an error response."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class MarketClient:
    """Synchronous HTTP client for the market environment."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    # --- context manager so callers can do `with MarketClient(...) as c:` ---
    def __enter__(self) -> "MarketClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    def health(self) -> dict[str, str]:
        return self._get("/health")

    def list_tasks(self) -> list[dict[str, Any]]:
        return self._get("/tasks")

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: int = 42,
        difficulty: str = "medium",
        bot_config: str = "default",
        trainable_agent_id: str = "agent_1",
        episode_length: int = 50,
    ) -> MarketObservation:
        body = {
            "task_id": task_id,
            "seed": seed,
            "difficulty": difficulty,
            "bot_config": bot_config,
            "trainable_agent_id": trainable_agent_id,
            "episode_length": episode_length,
        }
        data = self._post("/reset", body)
        return MarketObservation(**data)

    def step(
        self, episode_id: str, action: MarketAction
    ) -> tuple[MarketObservation, float, bool, dict[str, Any]]:
        body = {
            "episode_id": episode_id,
            "action": action.model_dump(),
        }
        data = self._post("/step", body)
        return (
            MarketObservation(**data["observation"]),
            float(data["reward"]),
            bool(data["done"]),
            data.get("info", {}),
        )

    def state(self, episode_id: str) -> dict[str, Any]:
        return self._get("/state", params={"episode_id": episode_id})

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        r = self._http.get(path, params=params)
        return self._parse(r)

    def _post(self, path: str, body: dict) -> Any:
        r = self._http.post(path, json=body)
        return self._parse(r)

    @staticmethod
    def _parse(response: httpx.Response) -> Any:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise MarketClientError(response.status_code, detail)
        return response.json()
