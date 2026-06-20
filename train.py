"""Continual learning for ARC-AGI-3: collect -> train -> solve -> collect -> ...

A single self-contained script (no hydra / lightning / the swm repo) that closes
the loop in the Dreamer / Plan2Explore shape:

    round 0 : no model yet -> act RANDOMLY, fill a growing replay buffer, train
    round k : act with CEM-MPC (uncertainty-driven) using the *current* world model,
              append the new experience, and FINE-TUNE the same model (warm-start)

The world model is a DreamerV3-style world model, but FEED-FORWARD (no RSSM GRU):
we adopt V3's representation + training-stability stack, not its recurrence or
its actor-critic. The paper is vendored at ``bibliography/dreamerv3.md`` (extracted
from arXiv:2301.04104); ``# ref:`` comments below point into it by section.

    encoder (CNN)        frame_t             -> POSTERIOR logits  q(z|x)  (categorical)
    action embedding     action_t (Discrete) -> a_t               <- handles Discrete(4101)
    predictor (MLP)      [z_t, a_t]          -> PRIOR/dynamics logits p(z'|z,a)
    decoder (deconv)     z                   -> frame              (recon loss)
    reward head (MLP)    z                   -> two-hot/symlog reward (cross-entropy)
    continue head (MLP)  z                   -> episode-continue flag (logistic)
    surprise head (MLP)  [z_t, a_t]          -> LEARNED novelty (amortized surprise)

The latent is a vector of ``n_cat`` softmax categoricals (V3 default 32 classes
each), sampled with straight-through gradients + 1% unimix. Training minimises
recon + reward + continue + KL(free-bits-clipped, balanced) + surprise-regression.

Exploration is uncertainty-driven (NOT epsilon-greedy): the surprise head learns
to predict the model's own one-step surprise KL(q(z'|x') || p(z'|z,a)) — high where
the dynamics are unsure, i.e. unexplored. Because it is *predicted* from (z, a),
the planner (CEM-MPC) reads it inside imagination with no real future frame, and
either seeks it (``--objective explore``) or adds it as a bonus to predicted reward
(``--objective reward``). ``confidence = exp(-surprise)`` in (0, 1].

ref: bibliography/dreamerv3.md "World model learning" / "Robust predictions";
     bibliography/icm.md (prediction-error curiosity);
     bibliography/plan2explore.md + bibliography/disagreement.md (ensemble upgrade)

Usage
-----
    pip install -e .
    python train.py --rounds 10 --episodes-per-round 16 --image-size 32
    python train.py --online --rounds 10                         # real API (auto-picks a game)
    python train.py > last_run.log                               # capture the full run log
    python train.py --rounds 5 --objective explore                # pure exploration
    python train.py --resume checkpoints/<game>.pt --rounds 5    # resume / continue
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.template")
load_dotenv(dotenv_path=".env", override=True)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

# swm eagerly tries to register the optional Atari (ALE/*) envs we don't use;
# silence its "ale-py not found" warning before the import that emits it.
import warnings

warnings.filterwarnings("ignore", message="ale-py not found")

import stable_worldmodel as swm

# Module logger. Configured in ``setup_logging`` (called from main); every notable
# event goes through this so ``python train.py > last_run.log`` captures the full run.
logger = logging.getLogger("arc")


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """Send logs to stdout (so ``> last_run.log`` works) and optionally a file.

    A StreamHandler flushes per record, so output survives a SIGINT/timeout — the
    reason prints were getting lost when interrupting a long online run.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )

from env import N_ACTIONS, N_SIMPLE, encode_action, register_swm


# ===================================================================== #
# World model
# ===================================================================== #
# ref: bibliography/dreamerv3.md "Robust predictions" (eq. 9): symlog compresses
# large magnitudes; symexp is its inverse. Used for the two-hot reward support.
def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


class PixelWorldModel(nn.Module):
    """DreamerV3-style world model (feed-forward; no RSSM recurrence).

    Latent ``z`` is a vector of ``n_cat`` categoricals with ``n_classes`` classes
    each, flattened to ``latent_dim = n_cat * n_classes``. The encoder is the
    posterior ``q(z|x)``; the action-conditioned predictor is the prior/dynamics
    ``p(z'|z,a)``. A two-hot/symlog reward head and a continue head complete the
    "model state -> reward + continuation + reconstruction" set.

    ref: bibliography/dreamerv3.md "World model learning"
    """

    def __init__(
        self,
        image_size: int = 64,
        latent_dim: int = 256,
        action_dim: int = 64,
        n_actions: int = N_ACTIONS,
        channels: int = 3,
        n_classes: int = 32,
        reward_bins: int = 41,
        reward_vmax: float = 20.0,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        assert image_size % 8 == 0, "image_size must be divisible by 8"
        assert latent_dim % n_classes == 0, "latent_dim must be divisible by n_classes"
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.n_cat = latent_dim // n_classes  # categoricals in the latent vector
        self.unimix = unimix
        self._feat_hw = image_size // 8  # three stride-2 layers
        feat = 128 * self._feat_hw * self._feat_hw

        # Encoder -> POSTERIOR logits q(z|x), shaped (B, n_cat * n_classes).
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, stride=2, padding=1), nn.ReLU(),  # /2
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),        # /4
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),       # /8
            nn.Flatten(),
            nn.Linear(feat, latent_dim),
        )
        # Discrete action -> dense embedding (the key bit for ARC actions).
        self.action_embed = nn.Embedding(n_actions, action_dim)
        # Predictor -> PRIOR/dynamics logits p(z'|z,a).
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.decoder_in = nn.Linear(latent_dim, feat)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),  # x2
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),   # x4
            nn.ConvTranspose2d(32, channels, 4, stride=2, padding=1),        # x8
            nn.Sigmoid(),  # outputs in [0, 1]
        )
        # Reward head: two-hot logits over a symlog-spaced support, decoded to a
        # scalar via symexp. ref: bibliography/dreamerv3.md "Robust predictions".
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2), nn.ReLU(),
            nn.Linear(latent_dim // 2, reward_bins),
        )
        # Continue head: episode-continuation flag c in {0,1} via logistic head.
        self.continue_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2), nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
        )
        # Surprise head: a LEARNED (amortized) estimate of the model's own one-step
        # surprise KL(q(z'|x') || p(z'|z,a)) for a state-action (z, a). The true
        # surprise is retrospective (needs the real next frame), so we regress this
        # head onto it during training and query it during imagination/planning to
        # get an exploration signal + confidence where no x' is available.
        # ref: bibliography/icm.md ("Curiosity-driven Exploration") — prediction-error
        # curiosity; bibliography/plan2explore.md is the disagreement-ensemble upgrade.
        self.surprise_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim // 2), nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
        )
        # Two-hot support: bins evenly spaced in symlog space over [-vmax, vmax].
        self.register_buffer(
            "reward_support", torch.linspace(-reward_vmax, reward_vmax, reward_bins)
        )

    # --- categorical latent helpers ----------------------------------- #
    # ref: bibliography/dreamerv3.md "World model learning": categoricals are a
    # 1% uniform + 99% network mixture (unimix), sampled with straight-through.
    def _dist(self, logits: torch.Tensor) -> torch.Tensor:
        """logits (B, latent_dim) -> probs (B, n_cat, n_classes) with unimix."""
        probs = torch.softmax(logits.view(-1, self.n_cat, self.n_classes), dim=-1)
        if self.unimix > 0:
            probs = (1 - self.unimix) * probs + self.unimix / self.n_classes
        return probs

    def _probs_flat(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected one-hot latent (deterministic), flattened to (B, latent_dim)."""
        return self._dist(logits).reshape(-1, self.latent_dim)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Straight-through one-hot sample, flattened to (B, latent_dim)."""
        probs = self._dist(logits)
        flat = probs.reshape(-1, self.n_classes)
        idx = torch.multinomial(flat, 1).squeeze(-1)
        onehot = torch.zeros_like(flat).scatter_(1, idx[:, None], 1.0)
        onehot = onehot + flat - flat.detach()  # straight-through gradient
        return onehot.view(-1, self.latent_dim)

    def posterior(self, frame: torch.Tensor) -> torch.Tensor:
        """Posterior logits q(z|x)."""
        return self.encoder(frame)

    def prior(self, z: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        """Prior/dynamics logits p(z'|z,a)."""
        return self.predictor(torch.cat([z, self.action_embed(action_t)], dim=-1))

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """Expected posterior latent (used as the planning start state)."""
        return self._probs_flat(self.posterior(frame))

    def step_latent(self, z: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        """Advance the latent by one action (expected prior latent)."""
        return self._probs_flat(self.prior(z, action_t))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_in(z).view(-1, 128, self._feat_hw, self._feat_hw)
        return self.decoder(x)

    def reward_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Two-hot logits over the symlog reward support (training target)."""
        return self.reward_head(z)

    def predict_reward(self, z: torch.Tensor) -> torch.Tensor:
        """Predicted scalar reward for arriving in latent state ``z``."""
        probs = torch.softmax(self.reward_logits(z), dim=-1)
        return symexp((probs * self.reward_support).sum(dim=-1))

    def continue_logit(self, z: torch.Tensor) -> torch.Tensor:
        return self.continue_head(z).squeeze(-1)

    def predict_continue(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.continue_logit(z))

    def surprise_raw(self, z: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        """Raw (symlog-space) surprise prediction for (z, a) — training output."""
        h = torch.cat([z, self.action_embed(action_t)], dim=-1)
        return self.surprise_head(h).squeeze(-1)

    def predict_surprise(self, z: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        """Predicted one-step surprise (epistemic novelty) for taking ``a`` in ``z``.

        Decoded from symlog space; clamped to be non-negative (it estimates a KL).
        High where the dynamics model is unsure -> unexplored state-actions.
        """
        return symexp(self.surprise_raw(z, action_t)).clamp(min=0.0)

    def confidence(self, z: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        """Confidence in (z, a) as exp(-surprise) in (0, 1]: 1 = well-modelled."""
        return torch.exp(-self.predict_surprise(z, action_t))

    def twohot(self, y: torch.Tensor) -> torch.Tensor:
        """Two-hot encode scalar rewards in symlog space. ref: eq. (12)."""
        support = self.reward_support
        t = symlog(y).clamp(float(support[0]), float(support[-1]))
        below = (support[None, :] <= t[:, None]).sum(dim=-1) - 1
        below = below.clamp(0, support.numel() - 2)
        lower, upper = support[below], support[below + 1]
        w_up = (t - lower) / (upper - lower)
        target = torch.zeros(y.shape[0], support.numel(), device=y.device)
        target.scatter_(1, below[:, None], (1 - w_up)[:, None])
        target.scatter_(1, (below + 1)[:, None], w_up[:, None])
        return target

    def forward(self, frame_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        """Deterministic one-step pixel prediction (expected latents)."""
        z_next = self.step_latent(self.encode(frame_t), action_t)
        return self.decode(z_next)


# ===================================================================== #
# Model lifecycle + training
# ===================================================================== #
def build_model(config: dict, device: str = "cpu") -> PixelWorldModel:
    return PixelWorldModel(**config).to(device)


def save_checkpoint(model: PixelWorldModel, config: dict, path: str) -> None:
    torch.save({"model_state": model.state_dict(), "config": config}, path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[PixelWorldModel, dict]:
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["config"], device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["config"]


def prepare_batch(
    batch: dict, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(frame_t, action_t, frame_{t+1}, reward, continue) from a replay batch.

    ``reward`` is recorded on arriving at frame_{t+1} (may be NaN at an episode
    boundary -> masked in the loss). ``continue`` is 1 unless that arrival is a
    terminal step (derived from the buffer's ``terminated`` column when present).
    Handles both pixel layouts: a LanceDataset gives (B, T, 3, H, W); the
    in-memory ReplayBuffer keeps raw (B, T, H, W, 3).
    """
    pixels = batch["pixels"].to(device)
    if pixels.shape[-1] == 3:  # HWC -> CHW
        pixels = pixels.permute(0, 1, 4, 2, 3)
    pixels = pixels.float() / 255.0
    actions = batch["action"].to(device).long()
    if actions.ndim > 2:
        actions = actions.squeeze(-1)
    reward = batch["reward"].to(device).float()
    if reward.ndim > 2:
        reward = reward.squeeze(-1)

    frame_t, frame_next = pixels[:, 0], pixels[:, 1]
    action_t = actions[:, 0].reshape(-1).clamp(0, N_ACTIONS - 1)
    reward_t = reward[:, 1].reshape(-1)  # reward of the t -> t+1 transition

    # continue = 1 - terminated at t+1 (truncations still "continue"). Fall back
    # to all-ones if the buffer didn't store a terminated column.
    if "terminated" in batch:
        term = batch["terminated"].to(device).float()
        if term.ndim > 2:
            term = term.squeeze(-1)
        continue_t = 1.0 - torch.nan_to_num(term[:, 1].reshape(-1), nan=0.0)
    else:
        continue_t = torch.ones_like(reward_t)
    return frame_t, action_t, frame_next, reward_t, continue_t


def _categorical_kl(model: PixelWorldModel, logits_q, logits_p) -> torch.Tensor:
    """KL(q || p) per sample, summed over the latent's categoricals. (B,)."""
    q = model._dist(logits_q)
    log_p = torch.log(model._dist(logits_p))
    return (q * (torch.log(q) - log_p)).sum(dim=-1).sum(dim=-1)


def fit(
    model: PixelWorldModel,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: str,
    reward_weight: float = 1.0,
    continue_weight: float = 1.0,
    surprise_weight: float = 1.0,
    kl_free: float = 1.0,
    kl_dyn: float = 1.0,
    kl_rep: float = 0.1,
    verbose: bool = True,
) -> PixelWorldModel:
    """Train ``model`` in place with the DreamerV3 (feed-forward) loss.

    Loss = recon + reward (two-hot CE) + continue (BCE) + KL(free-bits, balanced)
    + surprise (regression of the learned novelty head onto the observed one-step
    surprise). Warm-starts naturally — pass an already-trained model to keep learning.
    ref: bibliography/dreamerv3.md "World model learning"; bibliography/icm.md.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pixel_loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        # Per-term running sums so the log shows WHICH loss is moving, not just total.
        agg = {"loss": 0.0, "recon": 0.0, "reward": 0.0, "cont": 0.0,
               "surprise": 0.0, "kl": 0.0}
        n = 0
        for batch in loader:
            frame_t, action_t, frame_next, reward_t, continue_t = prepare_batch(
                batch, device
            )
            post_t = model.posterior(frame_t)
            post_n = model.posterior(frame_next)
            z_t, z_n = model._sample(post_t), model._sample(post_n)
            prior_n = model.prior(z_t, action_t)  # dynamics prediction of z_{t+1}

            # Reconstruction of both frames from their posterior samples.
            recon = pixel_loss_fn(model.decode(z_t), frame_t) + pixel_loss_fn(
                model.decode(z_n), frame_next
            )

            # Reward: two-hot cross-entropy in symlog space (mask boundary NaNs).
            reward_loss = frame_t.new_zeros(())
            mask = torch.isfinite(reward_t)
            if mask.any():
                logp = torch.log_softmax(model.reward_logits(z_n[mask]), dim=-1)
                target = model.twohot(reward_t[mask])
                reward_loss = -(target * logp).sum(dim=-1).mean()

            # Continuation: logistic regression on the continue flag.
            cont_loss = F.binary_cross_entropy_with_logits(
                model.continue_logit(z_n), continue_t
            )

            # KL with free bits + balancing: dynamics pulls prior->posterior,
            # representation pulls posterior->prior, each clipped below 1 nat.
            kl_dyn_t = torch.clamp(
                _categorical_kl(model, post_n.detach(), prior_n), min=kl_free
            ).mean()
            kl_rep_t = torch.clamp(
                _categorical_kl(model, post_n, prior_n.detach()), min=kl_free
            ).mean()
            kl = kl_dyn * kl_dyn_t + kl_rep * kl_rep_t

            # Surprise head: regress onto the OBSERVED one-step surprise of this
            # transition (how far the dynamics prior fell from the realised
            # posterior). Target is detached symlog(KL) so the head learns to
            # *predict* novelty without reshaping the representation; the head is
            # fed detached (z_t, a_t) for the same reason. At plan time this lets
            # us read off surprise/confidence from (z, a) alone.
            surprise_target = symlog(
                _categorical_kl(model, post_n.detach(), prior_n.detach())
            )
            surprise_loss = F.mse_loss(
                model.surprise_raw(z_t.detach(), action_t), surprise_target
            )

            loss = (
                recon
                + reward_weight * reward_loss
                + continue_weight * cont_loss
                + surprise_weight * surprise_loss
                + kl
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = frame_t.size(0)
            n += bs
            agg["loss"] += loss.item() * bs
            agg["recon"] += recon.item() * bs
            agg["reward"] += reward_loss.item() * bs
            agg["cont"] += cont_loss.item() * bs
            agg["surprise"] += surprise_loss.item() * bs
            agg["kl"] += kl.item() * bs
        if verbose:
            d = max(n, 1)
            logger.info(
                "    epoch %3d/%d  loss=%.5f (recon=%.4f reward=%.4f cont=%.4f "
                "surprise=%.4f kl=%.4f)",
                epoch + 1, epochs, agg["loss"] / d, agg["recon"] / d,
                agg["reward"] / d, agg["cont"] / d, agg["surprise"] / d, agg["kl"] / d,
            )
    return model


# ===================================================================== #
# Planning (CEM-MPC)
# ===================================================================== #
def build_candidates(click_stride: int) -> list[int]:
    """Flat action indices the planner may choose from."""
    cands = list(range(N_SIMPLE))  # ACTION1..ACTION5
    if click_stride > 0:
        for y in range(0, 64, click_stride):
            for x in range(0, 64, click_stride):
                cands.append(encode_action(6, x, y))
    return cands


def frame_to_tensor(rgb: np.ndarray, image_size: int, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 render -> (1, 3, image_size, image_size) float tensor."""
    img = Image.fromarray(rgb).resize((image_size, image_size), Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.inference_mode()
def latent_rollout(
    model: PixelWorldModel,
    z0: torch.Tensor,            # (S, D)
    action_seqs: torch.Tensor,   # (S, H) flat action indices
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll the prior forward in expected-latent space.

    Returns (z_T, total_surprise, total_reward), where ``total_surprise`` is the
    cumulative *predicted* one-step surprise (epistemic novelty) along the imagined
    trajectory — read from the learned surprise head, so it estimates "how much
    haven't I explored here" without needing the real future frames.

    Both accumulators are discounted by the running probability that the episode is
    still alive, read from the continue head: reward/novelty past a predicted episode
    end is down-weighted, so the planner won't chase value it can't actually reach.
    ref: bibliography/dreamerv3.md "World model learning" (continue predictor c_t).
    """
    z = z0
    total_surprise = torch.zeros(z0.shape[0], device=z0.device)
    total_reward = torch.zeros(z0.shape[0], device=z0.device)
    discount = torch.ones(z0.shape[0], device=z0.device)  # P(episode still alive at z)
    for t in range(action_seqs.shape[1]):
        a_t = action_seqs[:, t]
        # Novelty of exploring (z, a) is only worth it if we're alive to reach z.
        total_surprise = total_surprise + discount * model.predict_surprise(z, a_t)
        z_next = model.step_latent(z, a_t)
        discount = discount * model.predict_continue(z_next)  # survive into z_next
        total_reward = total_reward + discount * model.predict_reward(z_next)
        z = z_next
    return z, total_surprise, total_reward


@torch.inference_mode()
def cem_plan(
    model: PixelWorldModel,
    frame: torch.Tensor,            # (1, 3, H, W)
    candidates: torch.Tensor,       # (C,) flat action indices
    *,
    horizon: int,
    samples: int,
    iters: int,
    topk: int,
    objective: str,
    gen: torch.Generator,
    explore_beta: float = 0.0,
) -> int:
    """Return the best first flat action index via Categorical CEM.

    Objectives (all minimised):
      * ``reward``  -> -(predicted reward + ``explore_beta`` * predicted surprise)
      * ``explore`` -> -(predicted surprise)  (go where the model is least certain)

    ``explore_beta`` weights the learned-surprise exploration bonus added to reward.
    ref: bibliography/icm.md / bibliography/plan2explore.md.
    """
    device = frame.device
    C = candidates.shape[0]
    z0 = model.encode(frame)  # (1, D)

    # Per-timestep categorical over candidate indices, initialised uniform.
    probs = torch.full((horizon, C), 1.0 / C, device=device)

    for _ in range(iters):
        idx = torch.multinomial(probs, samples, replacement=True, generator=gen).T
        seqs = candidates[idx]  # (samples, horizon) flat action indices

        _, surprise, reward = latent_rollout(model, z0.expand(samples, -1), seqs)

        if objective == "reward":  # task-directed, with an exploration bonus
            cost = -(reward + explore_beta * surprise)
        else:  # explore: maximise predicted surprise -> minimise its negation
            cost = -surprise

        elite = torch.topk(cost, min(topk, samples), largest=False).indices
        elite_idx = idx[elite]  # (E, horizon)

        new_probs = torch.zeros_like(probs)
        for t in range(horizon):
            new_probs[t] = torch.bincount(elite_idx[:, t], minlength=C).float()
        new_probs = new_probs + 1e-3  # smoothing
        probs = new_probs / new_probs.sum(dim=-1, keepdim=True)

    best_candidate = int(probs[0].argmax().item())
    return int(candidates[best_candidate].item())


class MPCPolicy(swm.policy.BasePolicy):
    """swm policy that plans each step with the world model.

    Exploration is driven by the planner's learned-surprise term (the CEM
    ``objective``/``explore_beta``), not epsilon-greedy randomness.
    """

    def __init__(
        self,
        model: PixelWorldModel,
        candidates: torch.Tensor,
        image_size: int,
        device: str,
        cem: dict,
        gen: torch.Generator | None = None,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.type = "mpc"
        self.model = model
        self.candidates = candidates
        self.image_size = image_size
        self.device = device
        self.cem = cem
        self.gen = gen

    def get_action(self, infos: dict, **kw) -> np.ndarray:
        pixels = infos["pixels"]
        if torch.is_tensor(pixels):
            pixels = pixels.detach().cpu().numpy()
        pixels = np.asarray(pixels)

        n = pixels.shape[0]
        actions = np.empty(n, dtype=np.int64)
        for i in range(n):
            img = pixels[i]
            while img.ndim > 3:  # squeeze any leading time dim -> (H, W, 3)
                img = img[-1]
            frame = frame_to_tensor(img.astype(np.uint8), self.image_size, self.device)
            actions[i] = cem_plan(
                self.model, frame, self.candidates, gen=self.gen, **self.cem,
            )
        return actions


class RandomCandidatePolicy(swm.policy.BasePolicy):
    """Round-0 exploration: uniform random over the planner's candidate set.

    swm's ``RandomPolicy`` samples the full ``Discrete(N_ACTIONS)`` space, which
    includes ACTION6 clicks at arbitrary cells — some real games reject those with
    HTTP 400. Restricting to ``candidates`` keeps bootstrap collection inside the
    same valid action set the MPC planner uses (simple actions when click-stride=0).
    """

    def __init__(self, candidates: torch.Tensor, gen: torch.Generator, **kw) -> None:
        super().__init__(**kw)
        self.type = "random"
        self.candidates = candidates
        self.gen = gen

    def get_action(self, infos: dict, **kw) -> np.ndarray:
        pixels = infos["pixels"]
        if torch.is_tensor(pixels):
            pixels = pixels.detach().cpu().numpy()
        n = np.asarray(pixels).shape[0]
        idx = torch.randint(
            0, len(self.candidates), (n,), generator=self.gen, device=self.candidates.device
        )
        return self.candidates[idx].detach().cpu().numpy().astype(np.int64)


# ===================================================================== #
# Rollout / evaluation
# ===================================================================== #
def run_episode(env, policy, max_steps: int, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(max_steps):
        action = policy(env, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    # RecordEpisodeStatistics (if the env is wrapped in it) injects an
    # ``episode`` dict on the terminal step: r=return, l=length, t=wall-time.
    ep = info.get("episode")
    return {
        "levels": int(info["levels_completed"]),
        "reward": total_reward,
        "win": info["state"] == "WIN",
        "length": float(ep["l"]) if ep is not None else None,
        "time": float(ep["t"]) if ep is not None else None,
    }


def greedy_eval(model, candidates, image_size, device, cem, gen, *, env_id,
                make_kwargs, episodes, max_steps,
                monitor=False, video_dir=None, video_prefix="eval") -> list[dict]:
    """A few greedy MPC episodes on a fresh env, for a progress signal.

    ``monitor`` wraps the env in ``RecordEpisodeStatistics`` (return/length/time
    land in ``info["episode"]``); ``video_dir`` additionally records each eval
    episode to disk via ``RecordVideo`` (needs moviepy).
    """
    # Match the env's TimeLimit to max_steps so the episode truncates at the
    # loop boundary -> RecordEpisodeStatistics fires (it only emits on a real
    # episode end), and RecordVideo flushes a complete clip.
    env = gym.make(env_id, render_mode="rgb_array",
                   max_episode_steps=max_steps, **make_kwargs)
    if video_dir:
        env = gym.wrappers.RecordVideo(
            env, video_folder=video_dir, name_prefix=video_prefix,
            episode_trigger=lambda ep: True,  # record every eval episode
        )
    if monitor:
        env = gym.wrappers.RecordEpisodeStatistics(env)

    def policy(e, info):
        # Plan off the raw env render so monitoring wrappers can't alter the frame.
        frame = frame_to_tensor(e.unwrapped.render(), image_size, device)
        return cem_plan(model, frame, candidates, gen=gen, **cem)

    results = [run_episode(env, policy, max_steps, seed=i) for i in range(episodes)]
    env.close()
    return results


def log_scorecard(world: swm.World, *, online: bool) -> None:
    """Log ARC-AGI-3 scorecard totals for the episodes just collected (online only).

    Reads card_id(s) from the World's latest infos and GETs each summary *before* the
    envs (and their scorecards) close. No-op offline — the mock env has no server
    scorecard. These are the per-round run stats worth plotting later.
    """
    if not online:
        return
    infos = getattr(world, "infos", None) or {}
    cards = infos.get("card_id")
    if cards is None:
        return
    cards = {str(c) for c in np.atleast_1d(cards).ravel().tolist() if c}
    if not cards:
        return
    from client import HttpArcClient

    client = HttpArcClient()
    try:
        for card in sorted(cards):
            try:
                sc = client.get_scorecard(card)
            except Exception as e:  # noqa: BLE001 — logging must not crash the run
                # swm may close the episode's scorecard before this read -> 404.
                # Best-effort only; keep it at debug so normal runs aren't noisy.
                logger.debug("  scorecard %s: fetch failed (%s)", card[:8], e)
                continue
            logger.info(
                "  scorecard %s: score=%.3f levels=%d/%d actions=%d envs_done=%d/%d",
                card[:8], sc.get("score", 0.0),
                sc.get("total_levels_completed", 0), sc.get("total_levels", 0),
                sc.get("total_actions", 0),
                sc.get("total_environments_completed", 0),
                sc.get("total_environments", 0),
            )
    finally:
        client.close()


# ===================================================================== #
# Continual learning loop
# ===================================================================== #
def run_continual(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    register_swm()

    if args.video_dir:  # fail fast: RecordVideo writes mp4 via moviepy
        try:
            import moviepy  # noqa: F401
        except ImportError as e:
            raise SystemExit("--video-dir needs moviepy: pip install moviepy") from e

    S = args.image_size
    env_id = "swm/ArcAgi3-v0" if args.online else "swm/ArcAgi3Mock-v0"
    make_kwargs = {"game_id": args.game} if (args.online and args.game) else {}
    cem = dict(horizon=args.cem_horizon, samples=args.cem_samples,
               iters=args.cem_iters, topk=args.cem_topk,
               objective=args.objective, explore_beta=args.explore_beta)

    candidates = torch.tensor(build_candidates(args.click_stride), device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # Per-game checkpoint: <ckpt-dir>/<game>.pt, so each game keeps (and reloads)
    # its own world model. game name falls back to "mock"/"online" when unspecified.
    game_name = args.game or ("mock" if not args.online else "online")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"{game_name}.pt")

    # Warm-start: explicit --resume wins; else auto-reload this game's checkpoint if
    # it exists; else start fresh.
    resume_path = args.resume or (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume_path:
        model, config = load_checkpoint(resume_path, device)
        S = config["image_size"]
        logger.info("warm-started from %s (image_size=%d)", resume_path, S)
    else:
        config = {"image_size": S, "latent_dim": args.latent_dim,
                  "n_actions": N_ACTIONS, "n_classes": args.latent_classes,
                  "reward_bins": args.reward_bins}
        model = None

    # One growing replay buffer shared across rounds (continual data).
    buffer = swm.data.ReplayBuffer(max_steps=args.buffer_size, history_len=2, frameskip=1)

    # rounds <= 0 -> run forever until interrupted (Ctrl-C / SIGINT).
    unlimited = args.rounds <= 0
    rounds_iter = itertools.count() if unlimited else range(args.rounds)
    total = "inf" if unlimited else str(args.rounds)

    logger.info(
        "start: device=%s env=%s game=%s rounds=%s objective=%s candidates=%d "
        "buffer_cap=%d ckpt=%s%s",
        device, env_id, game_name, total, args.objective, len(candidates),
        args.buffer_size, ckpt_path,
        "  (Ctrl-C to stop)" if unlimited else "",
    )

    for r in rounds_iter:
        # --- 1. ACT + COLLECT ------------------------------------------- #
        phase = "random" if model is None else "mpc"
        logger.info("round %d/%s [%s] collecting %d episodes over %d env(s)...",
                    r + 1, total, phase, args.episodes_per_round, args.num_envs)
        world = swm.World(env_id, num_envs=args.num_envs, image_shape=(S, S),
                          max_episode_steps=args.max_episode_steps, **make_kwargs)
        if model is None:
            policy = RandomCandidatePolicy(candidates, gen)
        else:
            model.eval()
            policy = MPCPolicy(model, candidates, S, device, cem, gen=gen)
        world.set_policy(policy)
        with warnings.catch_warnings():
            # swm marks the reset step's (unused) action by filling the int
            # action array with NaN -> a benign "invalid value in cast" warning.
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in cast",
                category=RuntimeWarning,
            )
            world.collect(writer=buffer, episodes=args.episodes_per_round,
                          seed=args.seed + 1000 * (r + 1), progress=False)
        log_scorecard(world, online=args.online)  # before close -> scorecards still open
        world.close()
        logger.info("  collected: buffer=%d eps / %d steps",
                    buffer.num_episodes, buffer.num_steps_stored)

        # --- 2. TRAIN (warm-start: same model object persists) ---------- #
        if model is None:
            model = build_model(config, device)
        # drop_last=False so early rounds (fewer steps than batch_size) still train —
        # with drop_last=True a small buffer yields zero batches and fit is a no-op.
        loader = DataLoader(buffer, batch_size=args.batch_size, shuffle=True,
                            drop_last=len(buffer) >= args.batch_size)
        if len(loader) == 0:
            logger.warning("  buffer too small to train (%d clips); skipping round",
                           len(buffer))
            continue
        logger.info("  training %d epochs (lr=%g) on %d clips...",
                    args.epochs_per_round, args.lr, len(buffer))
        fit(model, loader, epochs=args.epochs_per_round, lr=args.lr, device=device,
            reward_weight=args.reward_weight, continue_weight=args.continue_weight,
            surprise_weight=args.surprise_weight,
            kl_free=args.kl_free_bits, kl_dyn=args.kl_dyn, kl_rep=args.kl_rep,
            verbose=True)
        save_checkpoint(model, config, ckpt_path)
        logger.info("  saved checkpoint -> %s", ckpt_path)

        # --- 3. eval ---------------------------------------------------- #
        if args.eval_episodes > 0:
            model.eval()
            res = greedy_eval(model, candidates, S, device, cem, gen,
                              env_id=env_id, make_kwargs=make_kwargs,
                              episodes=args.eval_episodes, max_steps=args.max_episode_steps,
                              monitor=args.record_stats, video_dir=args.video_dir,
                              video_prefix=f"eval-round{r + 1}")
            wins = sum(x["win"] for x in res)
            levels = float(np.mean([x["levels"] for x in res]))
            extra = ""
            if args.record_stats:  # RecordEpisodeStatistics gives per-episode length
                lens = [x["length"] for x in res if x["length"] is not None]
                if lens:
                    extra = f" mean_len={np.mean(lens):.1f}"
            logger.info("  eval: win_rate=%.2f mean_levels=%.2f%s",
                        wins / len(res), levels, extra)

    logger.info("done: final model at %s", ckpt_path)


def _run_continual_until_stopped(args: argparse.Namespace) -> None:
    """Run the loop, exiting cleanly on Ctrl-C (the latest round is already saved)."""
    try:
        run_continual(args)
    except KeyboardInterrupt:
        logger.info("interrupted — latest checkpoint is in %s/", args.ckpt_dir)
    except Exception:  # noqa: BLE001 — log the failure (incl. HTTP errors) then re-raise
        logger.exception("run failed")
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # environment
    p.add_argument("--online", action="store_true", help="use the real ARC-AGI-3 API")
    p.add_argument("--game", default=None, help="game_id (online only)")
    # loop
    p.add_argument("--rounds", type=int, default=5,
                   help="number of collect->train rounds; <=0 runs until Ctrl-C")
    p.add_argument("--episodes-per-round", type=int, default=16)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--max-episode-steps", type=int, default=80)
    p.add_argument("--buffer-size", type=int, default=200_000, help="replay capacity (steps)")
    # model / training
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=128,
                   help="flattened categorical latent dim (must divide by --latent-classes)")
    p.add_argument("--latent-classes", type=int, default=32,
                   help="classes per categorical; n_cat = latent_dim // latent_classes")
    p.add_argument("--reward-bins", type=int, default=41,
                   help="two-hot bins for the symlog reward head")
    p.add_argument("--epochs-per-round", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--reward-weight", type=float, default=1.0,
                   help="weight of the reward-head loss relative to pixel loss")
    p.add_argument("--continue-weight", type=float, default=1.0,
                   help="weight of the continue-head (episode-end) loss")
    p.add_argument("--surprise-weight", type=float, default=1.0,
                   help="weight of the learned-surprise (novelty) regression loss")
    # DreamerV3 KL: free bits = 1 nat, beta_dyn = 1.0, beta_rep = 0.1
    # ref: bibliography/dreamerv3.md "World model learning"
    p.add_argument("--kl-free-bits", type=float, default=1.0,
                   help="clip dynamics/representation KL below this many nats")
    p.add_argument("--kl-dyn", type=float, default=1.0, help="dynamics KL weight (beta_dyn)")
    p.add_argument("--kl-rep", type=float, default=0.1, help="representation KL weight (beta_rep)")
    p.add_argument("--resume", default=None, help="resume/warm-start from a checkpoint")
    # planning (CEM-MPC) / exploration
    p.add_argument("--cem-horizon", type=int, default=5, help="CEM rollout horizon")
    p.add_argument("--cem-samples", type=int, default=64,
                   help="CEM action sequences sampled per iteration")
    p.add_argument("--cem-iters", type=int, default=2, help="CEM refinement iterations")
    p.add_argument("--cem-topk", type=int, default=8, help="CEM elites kept per iteration")
    p.add_argument("--objective", choices=["reward", "explore"], default="reward",
                   help="planning objective: reward = task-directed (+surprise bonus); "
                        "explore = pure learned-surprise novelty seeking")
    p.add_argument("--explore-beta", type=float, default=0.1,
                   help="learned-surprise exploration bonus weight when --objective reward")
    p.add_argument("--click-stride", type=int, default=1,
                   help="ACTION6 click-grid stride in grid cells (0 = simple actions only); "
                        "1 = full 64x64 grid resolution (every cell clickable) -> "
                        "4096 clicks + 5 simple actions = 4101 candidates")
    p.add_argument("--eval-episodes", type=int, default=2, help="greedy eval per round (0 = skip)")
    # monitoring (eval env)
    p.add_argument("--record-stats", action="store_true",
                   help="wrap the eval env in RecordEpisodeStatistics (logs episode return/length/time)")
    p.add_argument("--video-dir", default=None,
                   help="record eval episodes as video to this dir (RecordVideo; needs moviepy)")
    # misc
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default="checkpoints",
                   help="directory for per-game checkpoints (saved as <ckpt-dir>/<game>.pt, "
                        "auto-reloaded when the same game is run again)")
    p.add_argument("--log-file", default=None,
                   help="also write logs to this file (stdout is always used; "
                        "redirect with `python train.py > last_run.log`)")
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    args = p.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO, args.log_file)
    _run_continual_until_stopped(args)


if __name__ == "__main__":
    main()
