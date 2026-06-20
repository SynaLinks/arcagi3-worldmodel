"""HTTP + mock clients for the ARC-AGI-3 game API.

The ARC-AGI-3 server exposes a small REST API (see https://docs.arcprize.org):

    Base URL (online):  https://three.arcprize.org
    Base URL (local):   http://localhost:8001
    Auth header:        X-API-Key: <your key>

    GET  /api/games                 -> [{"game_id": "...", "title": "..."}, ...]
    POST /api/scorecard/open        -> {"card_id": "..."}
    POST /api/scorecard/close       -> {... scorecard summary ...}
    POST /api/cmd/RESET             -> FrameData
    POST /api/cmd/ACTION1 .. ACTION5-> FrameData   (simple actions)
    POST /api/cmd/ACTION6           -> FrameData   (complex: needs x, y in 0..63)

`guid` ties subsequent ACTIONs to the session opened by RESET. The server also
relies on a load-balancer cookie (AWSALB*) for session affinity, so we reuse a
single `requests.Session` for the lifetime of a client.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from data_models import GRID, NUM_COLORS, FrameData, GameState

# Note: this is a library module — it reads ARC_API_KEY from the environment but
# does NOT load a .env file itself. The entry-point app (train.py, the example
# scripts, or your own program) is responsible for calling load_dotenv().


class ArcClient:
    """Interface implemented by the HTTP and mock clients."""

    def list_games(self) -> list[str]:
        raise NotImplementedError

    def open_scorecard(self, tags: Optional[list[str]] = None) -> str:
        raise NotImplementedError

    def get_scorecard(self, card_id: str) -> dict[str, Any]:
        """Read a scorecard summary WITHOUT closing it (for progress logging)."""
        raise NotImplementedError

    def close_scorecard(self, card_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def reset(self, game_id: str, card_id: str, guid: str = "") -> FrameData:
        raise NotImplementedError

    def action(
        self,
        action_id: int,
        game_id: str,
        guid: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> FrameData:
        raise NotImplementedError

    def close(self) -> None:
        pass


class HttpArcClient(ArcClient):
    """Talks to a real ARC-AGI-3 server over HTTP."""

    def __init__(
        self,
        root_url: str = "https://three.arcprize.org",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        import requests  # imported lazily so the mock path needs no dependency

        self.root_url = root_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-API-Key": api_key or os.getenv("ARC_API_KEY", ""),
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        r = self.session.post(
            f"{self.root_url}{path}", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def list_games(self) -> list[str]:
        r = self.session.get(f"{self.root_url}/api/games", timeout=self.timeout)
        r.raise_for_status()
        return [g["game_id"] for g in r.json()]

    def open_scorecard(self, tags: Optional[list[str]] = None) -> str:
        data = self._post("/api/scorecard/open", {"tags": tags or []})
        return data.get("card_id", "")

    def get_scorecard(self, card_id: str) -> dict[str, Any]:
        r = self.session.get(f"{self.root_url}/api/scorecard/{card_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def close_scorecard(self, card_id: str) -> dict[str, Any]:
        return self._post("/api/scorecard/close", {"card_id": card_id})

    def reset(self, game_id: str, card_id: str, guid: str = "") -> FrameData:
        payload: dict[str, Any] = {"game_id": game_id, "card_id": card_id}
        if guid:
            payload["guid"] = guid
        return FrameData.from_json(self._post("/api/cmd/RESET", payload))

    def action(
        self,
        action_id: int,
        game_id: str,
        guid: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> FrameData:
        payload: dict[str, Any] = {"game_id": game_id, "guid": guid}
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if action_id == 6:  # complex action carries coordinates
            payload["x"] = int(x or 0)
            payload["y"] = int(y or 0)
        data = self._post(f"/api/cmd/ACTION{action_id}", payload)
        return FrameData.from_json(data)

    def close(self) -> None:
        self.session.close()


class MockArcClient(ArcClient):
    """A self-contained fake game so the env can be exercised offline.

    Toy rules: a single "cursor" lives on a GRID x GRID board. ACTION1-4 move
    it (up/down/left/right) leaving a colour trail; ACTION5 cycles the paint
    colour; ACTION6 paints at (x, y). Reaching the bottom-right corner WINs.
    It exists only for tests / examples / `check_env` without an API key.
    """

    def __init__(self, win_levels: int = 1, seed: int = 0) -> None:
        import numpy as np

        self._win_levels = win_levels
        self._seed = seed
        self._board = np.zeros((GRID, GRID), dtype=np.uint8)
        self._pos = [0, 0]
        self._color = 1
        self._guid = ""
        self._state = GameState.NOT_PLAYED
        self._levels = 0

    def list_games(self) -> list[str]:
        return ["mock-grid-v0"]

    def open_scorecard(self, tags: Optional[list[str]] = None) -> str:
        return "mock-card"

    def get_scorecard(self, card_id: str) -> dict[str, Any]:
        return {"card_id": card_id, "levels_completed": self._levels}

    def close_scorecard(self, card_id: str) -> dict[str, Any]:
        return {"card_id": card_id, "levels_completed": self._levels}

    def _snapshot(self, action_id: int) -> FrameData:
        return FrameData(
            game_id="mock-grid-v0",
            guid=self._guid,
            frame=[self._board.tolist()],
            state=self._state,
            levels_completed=self._levels,
            win_levels=self._win_levels,
            action_input={"id": action_id},
            available_actions=[1, 2, 3, 4, 5, 6],
        )

    def reset(self, game_id: str, card_id: str, guid: str = "") -> FrameData:
        self._board[:] = 0
        self._pos = [0, 0]
        self._color = 1
        self._guid = f"mock-{self._seed}-{id(self) % 100000}"
        self._state = GameState.NOT_FINISHED
        self._board[0, 0] = self._color
        return self._snapshot(0)

    def action(
        self,
        action_id: int,
        game_id: str,
        guid: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> FrameData:
        r, c = self._pos
        if action_id == 1:  # up
            r = max(0, r - 1)
        elif action_id == 2:  # down
            r = min(GRID - 1, r + 1)
        elif action_id == 3:  # left
            c = max(0, c - 1)
        elif action_id == 4:  # right
            c = min(GRID - 1, c + 1)
        elif action_id == 5:  # activate
            self._color = 1 + (self._color % (NUM_COLORS - 1))
        elif action_id == 6:  # click
            py = min(GRID - 1, max(0, int(y or 0)))
            px = min(GRID - 1, max(0, int(x or 0)))
            self._board[py, px] = self._color
        self._pos = [r, c]
        self._board[r, c] = self._color

        if r == GRID - 1 and c == GRID - 1:
            self._levels += 1
            self._state = (
                GameState.WIN
                if self._levels >= self._win_levels
                else GameState.NOT_FINISHED
            )
        return self._snapshot(action_id)
