"""Fast, env-free tests for the unified continual-learning train.py:
the world model, the CEM-MPC planner, and the MPC policy."""

import numpy as np
import torch

from env import N_ACTIONS, N_SIMPLE
from train import (
    MPCPolicy,
    PixelWorldModel,
    build_candidates,
    build_model,
    cem_plan,
    fit,
    frame_to_tensor,
)


# --- world model ----------------------------------------------------- #
def test_model_forward_shape():
    model = PixelWorldModel(image_size=32, latent_dim=64)
    frame = torch.rand(4, 3, 32, 32)
    action = torch.randint(0, N_ACTIONS, (4,))
    out = model(frame, action)
    assert out.shape == (4, 3, 32, 32)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0  # sigmoid output


def test_reward_head():
    model = PixelWorldModel(image_size=16, latent_dim=32)
    r = model.predict_reward(torch.rand(5, 32))
    assert r.shape == (5,)  # one scalar reward per sample


def test_one_optimizer_step_reduces_loss_on_fixed_batch():
    torch.manual_seed(0)
    model = PixelWorldModel(image_size=16, latent_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    frame = torch.rand(8, 3, 16, 16)
    action = torch.randint(0, N_ACTIONS, (8,))
    target = torch.rand(8, 3, 16, 16)

    losses = []
    for _ in range(20):
        pred = model(frame, action)
        loss = loss_fn(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


# --- surprise head (learned novelty / confidence) -------------------- #
def test_surprise_head_shapes_and_ranges():
    model = PixelWorldModel(image_size=16, latent_dim=32)
    z = model.encode(torch.rand(5, 3, 16, 16))
    a = torch.randint(0, N_ACTIONS, (5,))
    s = model.predict_surprise(z, a)
    c = model.confidence(z, a)
    assert s.shape == (5,) and float(s.min()) >= 0.0  # surprise is a non-neg KL estimate
    assert c.shape == (5,) and 0.0 < float(c.min()) and float(c.max()) <= 1.0  # (0, 1]
    assert torch.allclose(c, torch.exp(-s), atol=1e-6)  # confidence == exp(-surprise)


def _surprise_loader(S, B):
    """8 batches mixing a LEARNABLE transition (fA,aA -> fixed next) with an
    UNPREDICTABLE one (fB,aB -> random next). Returns the loader + probe states."""
    fA = (torch.rand(1, S, S, 3) * 255).byte()
    fA_next = (torch.rand(1, S, S, 3) * 255).byte()
    fB = (torch.rand(1, S, S, 3) * 255).byte()
    aA, aB = 1, 2

    def make():
        half = B // 2
        pix0 = torch.cat([fA.repeat(half, 1, 1, 1), fB.repeat(half, 1, 1, 1)])
        pix1 = torch.cat([
            fA_next.repeat(half, 1, 1, 1),               # deterministic -> learnable
            (torch.rand(half, S, S, 3) * 255).byte(),    # random -> irreducible surprise
        ])
        pixels = torch.stack([pix0, pix1], dim=1)        # (B, 2, H, W, 3)
        action = torch.tensor([[aA, aA]] * half + [[aB, aB]] * half)
        reward = torch.cat([torch.full((B, 1), float("nan")), torch.zeros(B, 1)], dim=1)
        return {"pixels": pixels, "action": action, "reward": reward}

    return [make() for _ in range(8)], fA, fB, aA, aB


def test_surprise_head_learns_novelty():
    """The head should predict MORE surprise (less confidence) for a transition the
    dynamics can't model than for one it can."""
    torch.manual_seed(0)
    S = 16
    model = PixelWorldModel(image_size=S, latent_dim=128, n_classes=32)
    loader, fA, fB, aA, aB = _surprise_loader(S, B=32)
    fit(model, loader, epochs=40, lr=3e-3, device="cpu", verbose=False)

    model.eval()
    with torch.no_grad():
        zA = model.encode(fA.permute(0, 3, 1, 2).float() / 255.0)
        zB = model.encode(fB.permute(0, 3, 1, 2).float() / 255.0)
        sA = float(model.predict_surprise(zA, torch.tensor([aA])))
        sB = float(model.predict_surprise(zB, torch.tensor([aB])))
        cA = float(model.confidence(zA, torch.tensor([aA])))
        cB = float(model.confidence(zB, torch.tensor([aB])))
    assert sB > sA   # unpredictable transition -> higher predicted surprise
    assert cB < cA   # ... and lower confidence


# --- planner --------------------------------------------------------- #
def test_build_candidates():
    assert build_candidates(0) == list(range(N_SIMPLE))  # simple actions only
    cands = build_candidates(32)  # adds a 2x2 click grid (x,y in {0,32})
    assert len(cands) == N_SIMPLE + 4
    assert len(set(cands)) == len(cands)  # no duplicates


def test_frame_to_tensor_shape():
    rgb = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    t = frame_to_tensor(rgb, image_size=32, device="cpu")
    assert t.shape == (1, 3, 32, 32)
    assert t.dtype == torch.float32 and 0.0 <= float(t.min()) <= float(t.max()) <= 1.0


def test_cem_plan_returns_valid_action():
    torch.manual_seed(0)
    model = PixelWorldModel(image_size=16, latent_dim=32).eval()
    frame = torch.rand(1, 3, 16, 16)
    candidates = torch.tensor(build_candidates(0))
    gen = torch.Generator().manual_seed(0)

    action = cem_plan(
        model, frame, candidates,
        horizon=4, samples=32, iters=2, topk=8,
        objective="reward", gen=gen, explore_beta=0.1,
    )
    assert action in candidates.tolist()


def test_cem_plan_explore_objective_returns_valid_action():
    """Pure surprise-driven planning (no reward term) still picks a valid action."""
    torch.manual_seed(0)
    model = PixelWorldModel(image_size=16, latent_dim=32).eval()
    frame = torch.rand(1, 3, 16, 16)
    candidates = torch.tensor(build_candidates(0))
    gen = torch.Generator().manual_seed(0)

    action = cem_plan(
        model, frame, candidates,
        horizon=4, samples=32, iters=2, topk=8,
        objective="explore", gen=gen, explore_beta=0.0,
    )
    assert action in candidates.tolist()


# --- MPC policy (swm-facing) ----------------------------------------- #
def test_mpc_policy_batched_actions():
    device = "cpu"
    model = build_model({"image_size": 16, "latent_dim": 32}, device).eval()
    candidates = torch.tensor(build_candidates(0))
    gen = torch.Generator().manual_seed(0)
    cem = dict(horizon=3, samples=16, iters=2, topk=4,
               objective="reward", explore_beta=0.1)

    policy = MPCPolicy(model, candidates, image_size=16, device=device,
                       cem=cem, gen=gen)

    num_envs = 4
    # swm hands pixels as (num_envs, time, H, W, 3) uint8.
    infos = {"pixels": (np.random.rand(num_envs, 1, 16, 16, 3) * 255).astype(np.uint8)}
    actions = policy.get_action(infos)

    assert isinstance(actions, np.ndarray) and actions.shape == (num_envs,)
    assert set(actions.tolist()).issubset(set(candidates.tolist()))
