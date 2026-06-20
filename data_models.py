"""Data models and constants for the ARC-AGI-3 game API.

Holds the wire types shared by the client and the environment: the colour and
game-state enums, the colour palette, and the :class:`FrameData` snapshot
(a Pydantic model).
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any, Optional

from pydantic import BaseModel, Field

GRID = 64  # frames are GRID x GRID
NUM_COLORS = 16  # colour ids 0..15


class Color(IntEnum):
    """The 16 fixed ARC-AGI colour ids. Every frame cell holds one of these.

    Subclasses ``int`` so a member is usable directly as a colour id / array
    index (``frame[y][x] == Color.RED``) while naming the otherwise-magic 0..15.
    RGB values are in :data:`PALETTE`, indexed by the colour id.
    """

    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    FUCHSIA = 6
    ORANGE = 7
    AZURE = 8
    MAROON = 9
    WHITE = 10
    DARK_GREY = 11
    PINK = 12
    TEAL = 13
    PURPLE = 14
    NEAR_BLACK = 15

    @property
    def rgb(self) -> tuple[int, int, int]:
        """The (R, G, B) triple this colour renders to."""
        return PALETTE[int(self)]


# RGB palette indexed by colour id (Color). Used by the env renderers.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),        # 0  BLACK
    (0, 116, 217),    # 1  BLUE
    (255, 65, 54),    # 2  RED
    (46, 204, 64),    # 3  GREEN
    (255, 220, 0),    # 4  YELLOW
    (170, 170, 170),  # 5  GREY
    (240, 18, 190),   # 6  FUCHSIA
    (255, 133, 27),   # 7  ORANGE
    (127, 219, 255),  # 8  AZURE
    (135, 12, 37),    # 9  MAROON
    (255, 255, 255),  # 10 WHITE
    (99, 99, 99),     # 11 DARK_GREY
    (255, 167, 167),  # 12 PINK
    (0, 128, 128),    # 13 TEAL
    (128, 0, 128),    # 14 PURPLE
    (60, 60, 60),     # 15 NEAR_BLACK
)


class GameState(str, Enum):
    """Lifecycle state of an ARC-AGI-3 game.

    Subclasses ``str`` so members compare equal to their wire value
    (``GameState.WIN == "WIN"``) and serialise transparently to JSON/Lance.
    """

    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"

    @classmethod
    def _missing_(cls, value: Any) -> "Optional[GameState]":
        # Tolerate casing and the ``NOT_STARTED`` alias seen in some API docs.
        if isinstance(value, str):
            v = value.upper()
            if v == "NOT_STARTED":
                return cls.NOT_PLAYED
            for member in cls:
                if member.value == v:
                    return member
        return None

    @property
    def is_terminal(self) -> bool:
        """True once the episode is over (win or loss)."""
        return self in (GameState.WIN, GameState.GAME_OVER)


class FrameData(BaseModel):
    """Normalised snapshot returned by RESET / ACTION* commands.

    A Pydantic model: validates/coerces incoming JSON (including the
    :class:`GameState` enum) and round-trips via ``model_dump_json``.

    Wire shape::

        {
          "game_id": "...",
          "guid": "...",                 # session id, echoed on every call
          "frame": [[[0..15, ...]]],      # list of 64x64 grids of colour ids
          "state": "NOT_PLAYED" | "NOT_FINISHED" | "WIN" | "GAME_OVER",
          "levels_completed": int,        # (formerly "score")
          "win_levels": int,              # (formerly "win_score")
          "action_input": {"id": int, ...},
          "available_actions": [1, 2, ...]
        }
    """

    game_id: str = ""
    guid: str = ""
    frame: list[list[list[int]]] = Field(default_factory=list)  # list of 2D grids
    state: GameState = GameState.NOT_PLAYED
    levels_completed: int = 0
    win_levels: int = 0
    action_input: dict[str, Any] = Field(default_factory=dict)
    available_actions: list[int] = Field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "FrameData":
        return cls(
            game_id=d.get("game_id", ""),
            guid=d.get("guid", ""),
            frame=d.get("frame", []) or [],
            # GameState(...) resolves casing / the NOT_STARTED alias up front.
            state=GameState(d.get("state", "NOT_PLAYED")),
            # accept both new and legacy field names
            levels_completed=d.get("levels_completed", d.get("score", 0)) or 0,
            win_levels=d.get("win_levels", d.get("win_score", 0)) or 0,
            action_input=d.get("action_input", {}) or {},
            available_actions=d.get("available_actions", []) or [],
        )
