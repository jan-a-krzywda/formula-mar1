# Formula Mar1 · F1-style race strategy RL

Multi-car **Formula 1–style** simulation with **tyre compounds**, **pit strategy**, **ERS-style battery modes** (Harvest / Standard / Boost / Overtake), and **reinforcement learning** (behavioral cloning + PPO) for the trainable team **BlueCow**. Opponents use **fixed benchmark** pit strategies.

**Documentation**

- **GitHub Pages** (after you enable Pages — see [docs/README.md](docs/README.md)): **`https://jan-a-krzywda.github.io/formula-mar1/`** — **Sphinx** auto-generated Python API (`sphinx.ext.autodoc`)
- **Sphinx:** [`sphinx/`](sphinx/) · build: `pip install -r sphinx/requirements.txt && sphinx-build -b html sphinx/source sphinx/build/html`
- **Narrative Markdown** (browse on GitHub): [docs/](docs/README.md) — guide + algorithm reference
- [Documentation folder](docs/README.md) · [Guide](docs/guide.md) · [Algorithm & RL reference](docs/FORMULA_MAR1_ALGORITHM.md)
- **Python API (Sphinx):** [Module index](https://jan-a-krzywda.github.io/formula-mar1/py-modindex.html) · [Index](https://jan-a-krzywda.github.io/formula-mar1/genindex.html)

---

## Features

- **22 cars, 11 teams** — Random grid; lap-by-lap `total_race_time` standings.
- **Strategic actions** — Per car: pit (Soft/Medium/Hard), stay out with Harvest/Boost/Standard; **starting tyre choice** before lap 1.
- **Regulations** — Two-compound rule per car; shaping + terminal penalties/rewards.
- **Baselines** — 1-stop and 2-stop scripted strategies for non-trained teams.
- **Training** — JAX + Flax + Optax; vectorized rollouts; TensorBoard logs under `runs/`.
- **Visualization** — Terminal timing board (`formula_mar1/render_utils.py`), optional GIF from `evaluate.py`.

---

## Requirements

- Python **3.10+** recommended.
- **[JAX](https://github.com/google/jax)** — install the variant that matches your OS/GPU (CPU wheels are fine for small experiments).

Install dependencies:

```bash
cd formula-mar1
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
| `python analysis/analyze_rewards.py` | Episodes + plots → `reward_analysis.png` |
| `python main.py` | Random actions + quick GIF smoke test |

**Notebook:** open `gym.ipynb` in Jupyter for interactive experiments.

---

## Project layout

```
formula-mar1/
├── README.md                 # This file (GitHub landing page)
├── requirements.txt
├── formula_mar1/             # Core env, policy, PPO, training, rendering
│   ├── env.py
│   ├── vec_env.py
│   ├── networks.py
│   ├── ppo.py
│   ├── train.py
│   ├── render_utils.py
│   └── checkpoint_utils.py
├── analysis/                 # Plots, learning-curve studies, reward analysis
│   ├── analyze_rewards.py
│   ├── learning_curve_study.py
│   ├── plot_learning_curves.py
│   └── plot_best_agent_performance.py
├── sphinx/                   # Sphinx API docs (deployed to GitHub Pages)
│   ├── source/conf.py
│   └── requirements.txt
├── docs/
│   ├── README.md             # Doc index + Pages setup
│   ├── index.md
│   ├── guide.md
│   └── FORMULA_MAR1_ALGORITHM.md
├── .github/workflows/
│   └── docs.yml              # Sphinx HTML → deploy to gh-pages
├── train.py                  # Thin CLI → `formula_mar1.train`
├── evaluate.py               # Evaluation + telemetry
└── main.py                   # Random-action smoke test + GIF
```

---

## Documentation on GitHub

- **Repo home** shows this **`README.md`** first.
- **`sphinx/`** is the **Sphinx** project; **`.github/workflows/docs.yml`** runs **`sphinx-build`** and pushes HTML to the **`gh-pages`** branch.
- Enable **Settings → Pages → branch `gh-pages` / root** to publish **`https://jan-a-krzywda.github.io/formula-mar1/`**.

See **[docs/README.md](docs/README.md)** for the full index and setup steps.

**Sphinx** API HTML is regenerated when you push **`*.py`** (workflow) or re-run **`sphinx-apidoc`** (see `sphinx/README.md`). Narrative content lives in **`docs/`** only.

---

## License & credits

See the repository license file if present. Simulation credits: Jan A. Krzywda (see `formula_mar1/render_utils.py`).
