# arcagi3-worldmodel

A [Gymnasium](https://gymnasium.farama.org/) environment that wraps the
[ARC-AGI-3](https://github.com/arcprize/ARC-AGI-3-Agents) game API, plus a
DreamerV3-style world model (`train.py`) that learns and plans on it. You can
drive ARC-AGI-3 games with the standard `reset()` / `step()` RL loop and any
Gymnasium-compatible tooling (wrappers, vectorisation, RL libraries).

It is built following the official
[*Create a Custom Environment*](https://gymnasium.farama.org/introduction/create_custom_env/)
guide and the ARC-AGI-3 [REST API](https://docs.arcprize.org).

## Install

```bash
pip install -e .             # core deps include stable-worldmodel (torch, lancedb, ...)
pip install -e ".[http]"     # + requests, for the real ARC-AGI-3 API
pip install -e ".[dev]"      # + pytest, requests
```

> stable-worldmodel is a first-class dependency: the env is swm-ready by design
> (see [below](#use-with-stable-worldmodel)), so `import env` always works with swm.

The whole project is a handful of flat modules — `data_models.py` (wire types),
`client.py` (HTTP + mock clients), and `env.py` (the env). `env.py` works both
standalone and directly with [stable-worldmodel](#use-with-stable-worldmodel) —
there is a single `ArcAgi3Env`, no second adapter class.

## Quick start (offline, no API key)

```python
import gymnasium as gym
import env  # noqa: F401  (importing registers the gym ids)
from env import encode_action

e = gym.make("arcagi3/ArcAgi3Mock-v0", render_mode="ansi")
obs, info = e.reset(seed=0)

obs, reward, terminated, truncated, info = e.step(encode_action(6, x=12, y=30))  # ACTION6
print(info["state"], info["available_actions"])
print(e.render())
e.close()
```

`ArcAgi3Mock-v0` runs a self-contained toy game (no network, no key) — useful
for tests, `check_env`, and wiring up an agent before you go online.

## Quick start (online, real ARC-AGI-3 API)

```bash
cp .env.template .env      # then set ARC_API_KEY (the app loads it via load_dotenv)
```

```python
import gymnasium as gym
import env  # noqa: F401

e = gym.make("arcagi3/ArcAgi3-v0", game_id="ls20")
obs, info = e.reset(seed=0)                  # POST /api/cmd/RESET
obs, reward, term, trunc, info = e.step({"id": 6, "x": 12, "y": 30})  # dict form also ok
e.close()                                    # closes the scorecard
```

## Spaces

| | Gymnasium space | Meaning |
|---|---|---|
| **Observation** | `Box(0, 15, (64, 64), uint8)` | current frame, top layer; one colour id per cell |
| **Action** | `Discrete(4101)` = `5 + 64*64` | index `0..4` → ACTION1–5; index `5 + (y*64 + x)` → ACTION6 click at `(x, y)` |

A flat `Discrete` action is the canonical form (required by stable-worldmodel —
see below). Use `encode_action(id, x, y)` / `decode_action(index)` to convert.
For convenience `step` also accepts a `{"id", "x", "y"}` dict (plus an optional
`"reasoning"`), so `e.step({"id": 2})` works too.

## ARC-AGI-3 → Gymnasium mapping

| ARC-AGI-3 | Gymnasium |
|---|---|
| `POST /api/cmd/RESET` | `env.reset()` (also restarts after `GAME_OVER`) |
| `POST /api/cmd/ACTION1..6` | `env.step(action)` |
| `frame` (64×64 grid of colours) | observation `Box` |
| `state` ∈ {`NOT_PLAYED`,`NOT_FINISHED`,`WIN`,`GAME_OVER`} | `info["state"]`; `WIN`/`GAME_OVER` ⇒ `terminated=True` |
| `levels_completed` increase | `reward` (delta), `+win_bonus` on `WIN` |
| `available_actions`, `win_levels`, `guid`, full frame stack | `info[...]` |
| scorecard open/close | done automatically in `reset()`/`close()` |

Truncation is delegated to the standard `TimeLimit` wrapper via
`max_episode_steps` (80 for the online env, mirroring the reference agent's
`MAX_ACTIONS`).

## Info dict

`info` carries everything that doesn't fit the observation `Box`:
`state`, `available_actions`, `levels_completed`, `win_levels`, `guid`,
`game_id`, `card_id`, `action_input`, and `frame_stack` (the full, possibly
multi-layer, raw frame).

## Architecture

```
data_models.py  # GRID/NUM_COLORS, Color + PALETTE, GameState, FrameData (pydantic)
client.py       # ArcClient interface + HttpArcClient (real) + MockArcClient (offline)
env.py          # ArcAgi3Env, make_mock_env, encode/decode_action, register_swm
train.py        # continual learning: world model + CEM-MPC + collect->train->solve loop
tests/
```

The env talks to an injectable `ArcClient`, so the same `ArcAgi3Env` runs
against the live API (`HttpArcClient`) or fully offline (`MockArcClient`). Pass
your own client to point at a local server or to stub the API in tests:

```python
from client import HttpArcClient
from env import ArcAgi3Env
e = ArcAgi3Env(client=HttpArcClient(root_url="http://localhost:8001"), game_id="ls20")
```

## Use with stable-worldmodel

[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) (`swm`)
drives a pool of Gymnasium envs for world-model data collection, training, and
MPC evaluation. `ArcAgi3Env` already satisfies swm's contract — `register_swm()`
just registers that same env under the `swm/` namespace.

```python
import stable_worldmodel as swm
from env import register_swm

register_swm()              # registers swm/ArcAgi3-v0 and swm/ArcAgi3Mock-v0

world = swm.World("swm/ArcAgi3Mock-v0", num_envs=4, image_shape=(64, 64))
world.set_policy(swm.policy.RandomPolicy(seed=0))
world.collect("data/arc.lance", episodes=8, seed=0)   # -> LanceDB dataset

ds = swm.data.load_dataset("data/arc.lance", num_steps=4)  # pixels, action, reward, ...
```

Why `ArcAgi3Env` is swm-ready out of the box:

| swm requirement | how the env meets it |
|---|---|
| **flat action space** (`EverythingToInfoWrapper` rejects `dict` actions; `CategoricalCEMSolver` asserts `Discrete`) | the canonical action *is* `Discrete(4101)` (`5` simple + `64*64` clicks); see `encode_action`/`decode_action` |
| `render()` → `rgb_array` | 16-colour palette render (swm adds resized `pixels` to `info`) |
| `variation_space` (`swm.spaces.Dict`) | a minimal space exposing the empty-cell background colour |

> `register_swm()` is also used internally by `train.py` for data collection
> (it drives a pool of envs with `world.collect`).

## Continual learning (`train.py`)

`train.py` is one self-contained script that closes the loop
**collect → train → solve → collect → …** (Dreamer / Plan2Explore shape):

```bash
# Offline (mock game, no API key) — a complete working run:
python train.py --rounds 10 --episodes-per-round 16 --image-size 32 --eval-episodes 2

python train.py --resume checkpoints/mock-grid-v0.pt --rounds 5  # resume / keep learning
python train.py --rounds 5 --objective explore         # pure exploration instead
```

For the **real API**, pass a concrete `game_id` (the ids look like `sc25-635fd71a`,
not bare prefixes). List what your key can see, then train against one:

```bash
# set ARC_API_KEY in .env first (see "Quick start (online)" above)
python -c "from dotenv import load_dotenv; load_dotenv('.env'); \
from client import HttpArcClient; print(HttpArcClient().list_games())"

python train.py --online --game sc25-635fd71a --rounds 10 --eval-episodes 2
```

> Online is network-bound — every `step` is one HTTP call — so keep `--num-envs`
> modest and expect it to run much slower than the offline mock.

**Monitoring.** Per-round eval already prints `win_rate` and `mean_levels`. Add
`--record-stats` to wrap the eval env in `gymnasium.wrappers.RecordEpisodeStatistics`
(episode return/length/time; `mean_len` is appended to the round line), and
`--video-dir DIR` to record each eval episode to disk via
`gymnasium.wrappers.RecordVideo` (needs `pip install moviepy`):

```bash
python train.py --rounds 10 --episodes-per-round 16 --image-size 32 \
    --eval-episodes 2 --record-stats --video-dir runs/videos
```

It bundles three things:

- **World model** — a DreamerV3-style RSSM: CNN encoder + `nn.Embedding` action
  encoder (for the `Discrete(4101)` action) + a recurrent latent predictor over
  **categorical latents** (`--latent-dim` / `--latent-classes`) + deconv decoder,
  plus a **two-hot symlog reward head** (`--reward-bins`), a **continue head**
  (episode end), and a **learned-surprise head** (novelty). The loss combines
  image reconstruction + reward + continue + surprise + the dynamics/representation
  **KL** (free-bits, `β_dyn`/`β_rep`).
- **CEM-MPC planner** — rolls the model forward in latent space over a `Discrete`
  candidate set (5 simple actions, plus an ACTION6 click grid via `--click-stride`)
  and picks the best first action. Default objective is task-directed:
  `predicted_reward + β·surprise` (`--explore-beta`); `--objective explore`
  is pure novelty-seeking exploration.
- **The loop** — the first round acts **randomly** (no model yet), fills a growing
  in-memory `ReplayBuffer`, trains a model. Every later round acts with the
  CEM-MPC policy driven by the current model, appends the new experience, and
  **fine-tunes the same model** (warm-start = continual, not from scratch).
  Exploration is uncertainty-driven (the learned-surprise bonus), **not**
  ε-greedy. A few greedy eval episodes per round give a progress signal.

```
start: device=cuda env=swm/ArcAgi3Mock-v0 game=mock-grid-v0 rounds=10 objective=reward candidates=4101 ...
round 1/10 [random] collecting 16 episodes over 8 env(s)...
  collected: buffer=16 eps / 1280 steps
  eval: win_rate=0.00 mean_levels=0.00
round 2/10 [mpc] collecting 16 episodes over 8 env(s)...
...
```

> The loop *runs*, learns dynamics + a reward model, and plans for reward;
> actually solving real ARC-AGI-3 games is open research (sparse reward, a tiny
> CNN world model, short CEM horizon). The scaffolding is complete and correct —
> scaling the model/horizon and reward shaping is where the research lives.
>
> ARC-AGI-3 state lives on the game server and can't be set to an arbitrary
> start/goal, so this is episode-rollout planning (auto-reset), not swm's
> dataset-replay `evaluate(_set_state/_set_goal_state)` path.

## Tests

```bash
pytest -q
```

Includes `gymnasium.utils.env_checker.check_env` to validate the env against
the Gymnasium contract.
