"""A Gymnasium environment wrapping the ARC-AGI-3 game API.

One env, usable both standalone and directly with stable-worldmodel (swm).

Importing this module registers two plain Gymnasium ids::

    gym.make("arcagi3/ArcAgi3-v0", game_id="ls20")   # real API (needs ARC_API_KEY)
    gym.make("arcagi3/ArcAgi3Mock-v0")               # offline mock, no key/network

and :func:`register_swm` registers the same env with swm under ``swm/`` ids.

Mapping of ARC-AGI-3 concepts onto the Gymnasium API
----------------------------------------------------
* ``reset()``   -> POST /api/cmd/RESET  (also re-used to restart after GAME_OVER)
* ``step()``    -> POST /api/cmd/ACTIONn
* observation   -> the current 64x64 frame (top layer), ``Box(0..15)``
* action        -> a flat ``Discrete(5 + 64*64) = Discrete(4101)``:

      index 0..4              -> ACTION1..ACTION5    (simple actions)
      index 5 + (y*64 + x)    -> ACTION6 click at (x, y)

  A ``Discrete`` space (rather than the more natural ``Dict{id, x, y}``) is the
  canonical action because swm requires it: its ``EverythingToInfoWrapper``
  rejects ``dict`` actions and ``CategoricalCEMSolver`` asserts ``Discrete``.
  For convenience ``step`` still also accepts a ``{"id", "x", "y"}`` dict (with
  an optional ``"reasoning"``) or, equivalently, a flat int.
* reward        -> increase in ``levels_completed`` since the previous step,
                   with a +1 bonus when the game reaches ``WIN``.
* terminated    -> state is ``WIN`` or ``GAME_OVER``.
* truncated     -> handled by the ``max_episode_steps`` TimeLimit wrapper.

Everything else the API returns (game state, available actions, level counters,
guid, the full multi-layer frame) is exposed through the ``info`` dict, following
the ``_get_obs`` / ``_get_info`` pattern from the Gymnasium custom-env tutorial.

swm compatibility, baked in
---------------------------
* flat ``Discrete`` action space (above);
* ``render()`` returns an ``rgb_array`` (swm adds a resized ``pixels`` to info);
* a ``variation_space`` (``swm.spaces.Dict``) exposing a background-colour knob.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
from gymnasium import spaces

from client import ArcClient, HttpArcClient, MockArcClient
from data_models import (
    GRID,
    NUM_COLORS,
    PALETTE,
    Color,
    FrameData,
    GameState,
)

__all__ = [
    "ArcAgi3Env",
    "make_mock_env",
    "register_swm",
    "encode_action",
    "decode_action",
    "ACTION_NAMES",
    "N_SIMPLE",
    "N_ACTIONS",
    "ArcClient",
    "HttpArcClient",
    "MockArcClient",
    "FrameData",
    "GameState",
    "Color",
    "PALETTE",
]

# Human-readable names for the simple/complex actions, indexed by id (1..6).
ACTION_NAMES = {
    1: "ACTION1",  # e.g. up
    2: "ACTION2",  # e.g. down
    3: "ACTION3",  # e.g. left
    4: "ACTION4",  # e.g. right
    5: "ACTION5",  # interact / select
    6: "ACTION6",  # click at (x, y)
}

N_SIMPLE = 5  # ACTION1..ACTION5 (ACTION6 is the click action, encoded per-cell)
N_ACTIONS = N_SIMPLE + GRID * GRID  # 5 + 64*64 = 4101


def encode_action(action_id: int, x: int = 0, y: int = 0) -> int:
    """Map a (game action id, x, y) triple to a flat Discrete index."""
    if 1 <= action_id <= N_SIMPLE:
        return action_id - 1
    if action_id == 6:
        return N_SIMPLE + (int(y) * GRID + int(x))
    raise ValueError(f"Cannot encode action id {action_id}")


def decode_action(index: int) -> dict[str, int]:
    """Inverse of :func:`encode_action`; returns a ``{"id", "x", "y"}`` action."""
    index = int(index)
    if index < N_SIMPLE:
        return {"id": index + 1, "x": 0, "y": 0}
    pos = index - N_SIMPLE
    return {"id": 6, "x": pos % GRID, "y": pos // GRID}


class ArcAgi3Env(gym.Env):
    """Play a single ARC-AGI-3 game through the official REST API."""

    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 10}

    def __init__(
        self,
        game_id: Optional[str] = None,
        client: Optional[ArcClient] = None,
        root_url: str = "https://three.arcprize.org",
        api_key: Optional[str] = None,
        tags: Optional[list[str]] = None,
        render_mode: Optional[str] = None,
        win_bonus: float = 1.0,
        game_over_penalty: float = 0.0,
        manage_scorecard: bool = True,
        game: Optional[str] = None,
        max_frame_history: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Dependency-injectable client. When none is given we build an HTTP
        # client for the real API; pass MockArcClient() for offline use.
        self.client: ArcClient = client or HttpArcClient(root_url=root_url, api_key=api_key)
        # `game` is accepted as an alias for `game_id`.
        self._game_id = game_id if game_id is not None else game
        # Keep a rolling history of FrameData snapshots (None = unbounded).
        self.max_frame_history = max_frame_history
        self.frames: list[FrameData] = []
        self.tags = tags or []
        self.render_mode = render_mode
        self.win_bonus = win_bonus
        self.game_over_penalty = game_over_penalty
        self.manage_scorecard = manage_scorecard

        # --- spaces -------------------------------------------------------
        # Observation: the current frame's top layer as a 64x64 grid of
        # colour ids in [0, 15].
        self.observation_space = spaces.Box(
            low=0, high=NUM_COLORS - 1, shape=(GRID, GRID), dtype=np.uint8
        )
        # Action: a flat Discrete index (see module docstring / encode_action).
        self.action_space = spaces.Discrete(N_ACTIONS)

        # --- episode state ------------------------------------------------
        self.card_id: str = ""
        self.guid: str = ""
        self._frame: Optional[FrameData] = None
        self._prev_levels: int = 0

        # --- stable-worldmodel compatibility ------------------------------
        # Background colour that empty (Color.BLACK / id 0) cells render as,
        # driven by the variation space.
        self._bg_color = np.array(PALETTE[0], dtype=np.uint8)
        self.variation_space = self._build_variation_space()

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        self._reset_variations(seed, options)

        if options.get("game_id"):
            self._game_id = options["game_id"]
        if self._game_id is None:
            games = self.client.list_games()
            if not games:
                raise RuntimeError("No ARC-AGI-3 games available for this API key.")
            self._game_id = games[0]
        elif "-" not in self._game_id:
            # A bare prefix like "ls20": the suffix after the dash is only a
            # version, so resolve it against the games this key can actually see
            # (e.g. "ls20" -> "ls20-9607627b"). Cached: the resolved id contains
            # a dash, so this runs at most once per env.
            games = self.client.list_games()
            matches = [g for g in games if g.split("-", 1)[0] == self._game_id]
            if not matches:
                raise RuntimeError(
                    f"Game {self._game_id!r} matched none of the available games: {games}"
                )
            self._game_id = matches[0]

        if self.manage_scorecard and not self.card_id:
            self.card_id = self.client.open_scorecard(tags=self.tags)

        # RESET starts (or restarts) the game session and returns the first frame.
        self.frames = []
        frame = self.client.reset(self._game_id, self.card_id, guid=self.guid)
        self._ingest(frame)
        self._prev_levels = frame.levels_completed
        return self._get_obs(), self._get_info()

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_id, x, y = self._parse_action(action)
        reasoning = action["reasoning"] if isinstance(action, dict) and "reasoning" in action else None

        frame = self.client.action(
            action_id,
            self._game_id,
            self.guid,
            x=x,
            y=y,
            reasoning=reasoning,
        )
        self._ingest(frame)

        # Reward: progress in levels completed, plus terminal bonuses.
        reward = float(frame.levels_completed - self._prev_levels)
        self._prev_levels = frame.levels_completed

        terminated = frame.state.is_terminal
        if frame.state is GameState.WIN:
            reward += self.win_bonus
        elif frame.state is GameState.GAME_OVER:
            reward -= self.game_over_penalty

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self) -> Any:
        mode = self.render_mode
        if mode == "ansi":
            return self._render_ansi()
        if mode == "human":
            print(self._render_ansi())
            return None
        if mode == "rgb_array":
            return self._render_rgb()
        return None

    def close(self) -> None:
        if self.manage_scorecard and self.card_id:
            try:
                self.client.close_scorecard(self.card_id)
            except Exception:
                pass
            self.card_id = ""
        self.client.close()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _ingest(self, frame: FrameData) -> None:
        self._frame = frame
        if frame.guid:
            self.guid = frame.guid
        self.frames.append(frame)
        if self.max_frame_history is not None and len(self.frames) > self.max_frame_history:
            del self.frames[: -self.max_frame_history]

    def _grid(self) -> np.ndarray:
        """Top frame layer normalised to a (GRID, GRID) uint8 array."""
        if not self._frame or not self._frame.frame:
            return np.zeros((GRID, GRID), dtype=np.uint8)
        top = np.asarray(self._frame.frame[-1], dtype=np.uint8)
        out = np.zeros((GRID, GRID), dtype=np.uint8)
        h, w = top.shape[:2]
        out[:h, :w] = top[:GRID, :GRID]
        return np.clip(out, 0, NUM_COLORS - 1)

    def _get_obs(self) -> np.ndarray:
        return self._grid()

    def _get_info(self) -> dict[str, Any]:
        f = self._frame or FrameData()
        return {
            "state": f.state,
            "available_actions": list(f.available_actions),
            "levels_completed": f.levels_completed,
            "win_levels": f.win_levels,
            "guid": self.guid,
            "game_id": self._game_id,
            "card_id": self.card_id,
            "action_input": f.action_input,
            "frame_stack": f.frame,  # full (possibly multi-layer) frame
        }

    def _parse_action(self, action: Any) -> tuple[int, int, int]:
        """Normalise any accepted action form to (action_id, x, y).

        Accepts a flat Discrete index (int / numpy int) or a convenience
        ``{"id", "x", "y"}`` dict.
        """
        if isinstance(action, dict):
            a = action
        else:
            a = decode_action(int(action))
        action_id = int(a.get("id", 1))
        x = int(a.get("x", 0))
        y = int(a.get("y", 0))
        if action_id not in ACTION_NAMES:
            raise ValueError(f"Invalid action id {action_id}; expected 1..6.")
        return action_id, x, y

    def _render_ansi(self) -> str:
        grid = self._grid()
        # Show only the non-empty bounding box to keep output readable.
        nz = np.argwhere(grid != 0)
        if nz.size == 0:
            return "<empty frame>"
        (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
        chars = "0123456789ABCDEF"
        lines = [
            "".join(chars[v] for v in row) for row in grid[r0:r1, c0:c1]
        ]
        return "\n".join(lines)

    def _render_rgb(self) -> np.ndarray:
        # Map each colour id to its RGB via the fixed ARC palette, with the
        # empty-cell colour overridable by the background variation.
        palette = np.array(PALETTE, dtype=np.uint8).copy()
        palette[Color.BLACK] = self._bg_color
        return palette[self._grid()]

    # -- swm variation space ------------------------------------------- #
    def _build_variation_space(self) -> Any:
        """A minimal swm variation space.

        ARC-AGI-3 variation is controlled by the game server, so the only knob
        we expose locally is the colour empty cells render as.
        """
        return swm.spaces.Dict(
            {
                "background": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(
                            init_value=np.array(PALETTE[0], dtype=np.uint8)
                        ),
                    }
                ),
            }
        )

    def _reset_variations(self, seed: Optional[int], options: dict[str, Any]) -> None:
        """Follow the swm convention for resampling the variation space."""
        from collections.abc import Sequence

        self.variation_space.seed(seed)
        self.variation_space.reset()
        variations = options.get("variation", ())
        if not isinstance(variations, Sequence):
            raise ValueError("'variation' option must be a Sequence of names")
        self.variation_space.update(variations)
        self._bg_color = np.asarray(
            self.variation_space["background"]["color"].value, dtype=np.uint8
        )


def make_mock_env(render_mode: Optional[str] = None, **kwargs: Any) -> ArcAgi3Env:
    """Convenience constructor for an offline env backed by MockArcClient."""
    win_levels = kwargs.pop("win_levels", 1)
    return ArcAgi3Env(
        game_id="mock-grid-v0",
        client=MockArcClient(win_levels=win_levels),
        render_mode=render_mode,
        **kwargs,
    )


# --------------------------------------------------------------------- #
# Gymnasium registration (runs on import). Uses callable entry points so the
# flat-module layout needs no importable package path.
# --------------------------------------------------------------------- #
gym.register(
    id="arcagi3/ArcAgi3-v0",
    entry_point=ArcAgi3Env,
    max_episode_steps=80,  # mirrors the reference agent's MAX_ACTIONS
)
gym.register(
    id="arcagi3/ArcAgi3Mock-v0",
    entry_point=make_mock_env,
    max_episode_steps=300,
)


_SWM_REGISTERED = False


def register_swm() -> None:
    """Register this same env with stable-worldmodel under ``swm/`` ids.

    Idempotent. swm needs ``discrete=True`` (the action space is Discrete) so
    the env lands in ``stable_worldmodel.envs.DISCRETE_WORLDS``.
    """
    global _SWM_REGISTERED
    if _SWM_REGISTERED:
        return
    import stable_worldmodel.envs as swm_envs

    swm_envs.register(
        id="swm/ArcAgi3-v0",
        entry_point=lambda render_mode="rgb_array", **kw: ArcAgi3Env(
            render_mode=render_mode, **kw
        ),
        discrete=True,
        max_episode_steps=80,
    )
    swm_envs.register(
        id="swm/ArcAgi3Mock-v0",
        entry_point=lambda render_mode="rgb_array", **kw: make_mock_env(
            render_mode=render_mode, **kw
        ),
        discrete=True,
        max_episode_steps=300,
    )
    _SWM_REGISTERED = True
