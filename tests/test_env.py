import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import env as env_mod  # noqa: F401  (registers the gym ids)
from client import MockArcClient
from data_models import PALETTE, Color, FrameData, GameState
from env import ArcAgi3Env, decode_action, encode_action


def make():
    return ArcAgi3Env(game_id="mock-grid-v0", client=MockArcClient(win_levels=1))


def test_gamestate_enum():
    assert GameState.WIN == "WIN"  # str-enum compares to wire value
    assert GameState("GAME_OVER") is GameState.GAME_OVER
    assert GameState("NOT_STARTED") is GameState.NOT_PLAYED  # alias
    assert GameState.WIN.is_terminal and GameState.GAME_OVER.is_terminal
    assert not GameState.NOT_FINISHED.is_terminal


def test_color_enum_and_palette():
    assert len(PALETTE) == 16 == len(Color)
    assert Color.BLACK == 0 and Color.RED == 2
    assert Color.RED.rgb == (255, 65, 54)
    assert all(len(rgb) == 3 for rgb in PALETTE)


def test_info_state_is_enum():
    env = make()
    _, info = env.reset(seed=0)
    assert info["state"] is GameState.NOT_FINISHED


def test_framedata_is_pydantic_model():
    import pydantic

    f = FrameData.from_json(
        {"game_id": "g", "state": "WIN", "score": 3, "win_score": 5}
    )
    assert isinstance(f, pydantic.BaseModel)
    assert f.state is GameState.WIN
    assert (f.levels_completed, f.win_levels) == (3, 5)  # legacy field remap
    # round-trips through JSON with the enum serialised to its wire value
    assert '"state":"WIN"' in f.model_dump_json()


def test_check_env():
    # check_env unwraps to the bare env and asserts the Gymnasium contract.
    check_env(make().unwrapped, skip_render_check=True)


def test_reset_returns_valid_obs():
    env = make()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["state"] == "NOT_FINISHED"
    assert info["available_actions"] == [1, 2, 3, 4, 5, 6]


def test_step_tuple_and_spaces():
    env = make()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step({"id": 2, "x": 0, "y": 0})
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_flat_int_action_accepted():
    env = make()
    env.reset(seed=0)
    obs, *_ = env.step(0)  # flat Discrete index 0 == ACTION1
    assert env.observation_space.contains(obs)


def test_action_codec_roundtrip():
    assert decode_action(0) == {"id": 1, "x": 0, "y": 0}
    assert decode_action(4) == {"id": 5, "x": 0, "y": 0}
    assert decode_action(encode_action(6, 10, 5)) == {"id": 6, "x": 10, "y": 5}
    assert encode_action(2) == 1  # ACTION2 -> index 1


def test_win_gives_reward_and_terminates():
    # 63 "down" then 63 "right" walks the cursor to the bottom-right -> WIN.
    # Use {"id": ...} dict actions so we address game actions, not flat indices.
    env = make()
    env.reset(seed=0)
    terminated = False
    total = 0.0
    for action_id in [2] * 63 + [4] * 63:
        _, reward, terminated, _, info = env.step({"id": action_id})
        total += reward
        if terminated:
            break
    assert terminated
    assert info["state"] == "WIN"
    assert total >= 1.0  # level-up delta (1) + win bonus (1)


def test_action6_paints_coordinate():
    env = make()
    env.reset(seed=0)
    obs, *_ = env.step({"id": 6, "x": 10, "y": 5})
    assert obs[5, 10] != 0  # grid is [row=y, col=x]


def test_registered_mock_env():
    env = gym.make("arcagi3/ArcAgi3Mock-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (64, 64)
    env.close()


def test_invalid_action_id():
    env = make()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step({"id": 9, "x": 0, "y": 0})


def test_rgb_render_shape():
    env = ArcAgi3Env(client=MockArcClient(), game_id="mock-grid-v0", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8
