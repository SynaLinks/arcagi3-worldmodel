"""Tests for the stable-worldmodel integration.

The action-codec tests run unconditionally; the World-level tests are skipped
when stable-worldmodel (and its heavy deps) are not installed.
"""

import numpy as np
import pytest

from data_models import GRID
from env import (
    N_ACTIONS,
    N_SIMPLE,
    decode_action,
    encode_action,
    register_swm,
)

swm = pytest.importorskip("stable_worldmodel")


def test_action_count():
    assert N_ACTIONS == N_SIMPLE + GRID * GRID == 4101


@pytest.mark.parametrize("idx", [0, 1, 4])
def test_decode_simple_actions(idx):
    assert decode_action(idx) == {"id": idx + 1, "x": 0, "y": 0}


@pytest.mark.parametrize("x,y", [(0, 0), (10, 5), (63, 63), (31, 0)])
def test_codec_roundtrip_clicks(x, y):
    idx = encode_action(6, x, y)
    assert N_SIMPLE <= idx < N_ACTIONS
    assert decode_action(idx) == {"id": 6, "x": x, "y": y}


def test_encode_rejects_bad_id():
    with pytest.raises(ValueError):
        encode_action(7, 0, 0)


def test_registration_and_discrete_world():
    register_swm()
    assert "swm/ArcAgi3Mock-v0" in swm.envs.DISCRETE_WORLDS
    assert "swm/ArcAgi3-v0" in swm.envs.WORLDS


def test_world_reset_and_step():
    register_swm()
    world = swm.World(
        "swm/ArcAgi3Mock-v0", num_envs=2, image_shape=(32, 32), max_episode_steps=20
    )
    world.set_policy(swm.policy.RandomPolicy(seed=0))
    world.reset(seed=0)

    # swm renders each env and lifts pixels into the stacked info dict.
    assert "pixels" in world.infos
    assert np.asarray(world.infos["pixels"]).shape == (2, 1, 32, 32, 3)

    actions = world._get_actions()
    assert np.asarray(actions).shape == (2,)
    _, rewards, terminated, truncated, _ = world.envs.step(actions)
    assert np.asarray(rewards).shape == (2,)
    world.close()


def test_world_collect_roundtrip(tmp_path):
    register_swm()
    world = swm.World(
        "swm/ArcAgi3Mock-v0", num_envs=2, image_shape=(32, 32), max_episode_steps=20
    )
    world.set_policy(swm.policy.RandomPolicy(seed=0))
    path = str(tmp_path / "arc.lance")
    world.collect(path, episodes=4, seed=0)
    world.close()

    ds = swm.data.load_dataset(path, num_steps=4)
    assert len(ds) > 0
    sample = ds[0]
    assert "pixels" in sample and "action" in sample
