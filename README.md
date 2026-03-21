# Formula Mar1 · F1-style race strategy RL

Multi-car **Formula 1–style** simulation with **tyre compounds**, **pit strategy**, **ERS-style battery modes** (Harvest / Standard / Boost / Overtake), and **reinforcement learning** (behavioral cloning + PPO) for the trainable team **BlueCow**. Opponents use **fixed benchmark** pit strategies.

**Documentation**

- **GitHub Pages site** (after you enable Pages — see [docs/README.md](docs/README.md)): `https://<your-username>.github.io/<repo>/`
- [Documentation folder](docs/README.md) · [Guide](docs/guide.md) · [Algorithm & RL reference](docs/FORMULA_MAR1_ALGORITHM.md)
- Build locally: `pip install -r requirements-docs.txt && mkdocs serve`

---

## Features

- **22 cars, 11 teams** — Random grid; lap-by-lap `total_race_time` standings.
- **Strategic actions** — Per car: pit (Soft/Medium/Hard), stay out with Harvest/Boost/Standard; **starting tyre choice** before lap 1.
- **Regulations** — Two-compound rule per car; shaping + terminal penalties/rewards.
- **Baselines** — 1-stop and 2-stop scripted strategies for non-trained teams.
- **Training** — JAX + Flax + Optax; vectorized rollouts; TensorBoard logs under `runs/`.
- **Visualization** — Terminal timing board (`render_utils.py`), optional GIF from `evaluate.py`.

---

## Requirements

- Python **3.10+** recommended.
- **[JAX](https://github.com/google/jax)** — install the variant that matches your OS/GPU (CPU wheels are fine for small experiments).

Install dependencies:

```bash
cd marl-f1
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Then install JAX for your platform, e.g. CPU-only:
# pip install -U "jax[cpu]"
```

---

## Quick start

| Command | What it does |
|---------|----------------|
| `python train.py` | BC pretrain + PPO; writes `f1_best_weights.pkl`, `f1_trained_weights.pkl`, TensorBoard |
| `python evaluate.py` | Load weights, run one showcase race, optional GIF + `full_race_telemetry.csv` |
| `python analyze_rewards.py` | Episodes + plots → `reward_analysis.png` |
| `python main.py` | Random actions + quick GIF smoke test |

**Notebook:** open `gym.ipynb` in Jupyter for interactive experiments.

---

## Project layout

```
marl-f1/
├── README.md                 # This file (GitHub landing page)
├── requirements.txt
├── requirements-docs.txt     # MkDocs only (GitHub Pages)
├── mkdocs.yml                # Documentation site config
├── docs/
│   ├── index.md              # Doc site home (MkDocs)
│   ├── README.md             # Doc index + Pages setup
│   └── FORMULA_MAR1_ALGORITHM.md
├── .github/workflows/
│   └── docs.yml              # Deploy docs to gh-pages
├── env.py                    # Simulation + rewards
├── vec_env.py                # Batched envs for training
├── networks.py               # Policy / value network
├── ppo.py                    # PPO + GAE + logit masks
├── train.py                  # Training entry point
├── evaluate.py               # Evaluation + telemetry
├── analyze_rewards.py
├── render_utils.py
└── main.py
```

---

## Documentation on GitHub

- **Repo home** shows this **`README.md`** first.
- **`docs/`** holds Markdown sources; the **MkDocs** site is defined by **`mkdocs.yml`** and deployed by **`.github/workflows/docs.yml`** to the **`gh-pages`** branch.
- Enable **Settings → Pages → branch `gh-pages` / root** to publish **`https://<user>.github.io/<repo>/`**.

See **[docs/README.md](docs/README.md)** for the full index and setup steps.

---

## License & credits

See the repository license file if present. Simulation credits: Jan A. Krzywda (see `render_utils.py`).
