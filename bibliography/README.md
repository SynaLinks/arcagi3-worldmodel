# Bibliography

Papers this project builds on. Each `<slug>.md` is a markdown extraction of the
matching `<slug>*.pdf` (via `pymupdf4llm`). Code in `train.py` cites these by
`# ref: bibliography/<slug>.md` comments.

| Paper | arXiv | Year | Why it's here |
|---|---|---|---|
| [Mastering Diverse Domains through World Models (DreamerV3)](dreamerv3.md) | [2301.04104](https://arxiv.org/abs/2301.04104) | 2023 | Baseline world model: stochastic categorical latent + symlog/two-hot/free-bits/KL-balancing. |
| [Learning Latent Dynamics for Planning from Pixels (PlaNet)](planet.md) | [1811.04551](https://arxiv.org/abs/1811.04551) | 2019 | The latent world-model + CEM-MPC planning shape this repo follows. |
| [Dream to Control: Learning Behaviors by Latent Imagination (Dreamer)](dreamerv1.md) | [1912.01603](https://arxiv.org/abs/1912.01603) | 2020 | Dreamer lineage: actor-critic in latent imagination over the RSSM. |
| [Mastering Atari with Discrete World Models (DreamerV2)](dreamerv2.md) | [2010.02193](https://arxiv.org/abs/2010.02193) | 2021 | Origin of the categorical latent + KL balancing we implement. |
| [Planning to Explore via Self-Supervised World Models (Plan2Explore)](plan2explore.md) | [2005.05960](https://arxiv.org/abs/2005.05960) | 2020 | Ensemble-disagreement exploration: the upgrade path for our surprise head. |
| [Self-Supervised Exploration via Disagreement](disagreement.md) | [1906.04161](https://arxiv.org/abs/1906.04161) | 2019 | Why disagreement beats prediction-error curiosity (noisy-TV / aleatoric). |
| [Curiosity-driven Exploration by Self-supervised Prediction (ICM)](icm.md) | [1705.05363](https://arxiv.org/abs/1705.05363) | 2017 | Prediction-error curiosity — the family our learned-surprise head belongs to. |
| [Exploration by Random Network Distillation (RND)](rnd.md) | [1810.12894](https://arxiv.org/abs/1810.12894) | 2018 | Alternative novelty signal via distillation error; exploration reference. |

## How the code uses them

- **DreamerV3** — the world-model architecture + robustness stack (`PixelWorldModel`, `fit`).
- **DreamerV2** — categorical latent & KL balancing.
- **DreamerV1 / PlaNet** — latent-imagination lineage & the CEM-MPC planning shape.
- **ICM** — prediction-error curiosity, the basis of the learned `surprise_head`.
- **Plan2Explore / Disagreement** — the ensemble-disagreement upgrade for the novelty signal.
