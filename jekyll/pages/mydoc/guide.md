---
title: Documentation guide
keywords: repository, github pages, jekyll
sidebar: mydoc_sidebar
permalink: guide.html
folder: mydoc
summary: How the repo is organized, how documentation is published, and how to navigate the project.
---

## Published documentation site (GitHub Pages)

This project uses the **[Documentation Theme for Jekyll](https://github.com/tomjoht/documentation-theme-jekyll)** (v6.0-style sidebar) and a **GitHub Actions** workflow that builds the site and deploys to the **`gh-pages`** branch.

1. Push **`jekyll/`**, **`.github/workflows/docs.yml`**, and related files to **`main`** (or **`master`**).
2. Open **Settings → Pages** in the GitHub repo.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
4. Choose branch **`gh-pages`**, folder **`/ (root)`**, Save.
5. After the workflow runs, the site is available at:

   **`https://jan-a-krzywda.github.io/formula-mar1/`**

**Local preview:** from the repo root:

```bash
cd jekyll
bundle install
bundle exec jekyll serve
```

**Customize:** For a fork, edit **`jekyll/_config.yml`** (`url`, `baseurl`, `github_editme_path`, `repository`) and **`jekyll/_data/topnav.yml`** (GitHub link).

---

Welcome to the **Formula Mar1** project documentation. Markdown sources for the themed site live under **`jekyll/`**; **`docs/`** in the repo may still hold copies or pointers for browsing on GitHub.

| Document | Description |
|----------|-------------|
| [**Algorithm & RL details**](formula_mar1_algorithm.html) | Environment, observations, actions, rewards, PPO/BC pipeline, Mermaid diagrams, blog prompts |
| **This guide** | How to navigate docs and how they are published |

---

## Repository map (quick)

| Path | Role |
|------|------|
| `jekyll/` | Jekyll site (Documentation Theme 6.0), `_config.yml`, `pages/mydoc/` |
| `env.py` | `F1TeamEnv`: race logic, rewards, starting tyres, benchmarks |
| `vec_env.py` | Parallel envs for PPO rollouts (BlueCow + scripted opponents) |
| `networks.py` | Flax `F1AgentNN` (policy + value) |
| `ppo.py` | GAE, PPO update, logit masking |
| `train.py` | BC pretraining + PPO training loop, eval, checkpoints |
| `evaluate.py` | Greedy policy vs baselines, telemetry CSV, optional GIF |
| `analyze_rewards.py` | Per-lap plots → `reward_analysis.png` |
| `main.py` | Minimal random-action smoke test + GIF |
| `render_utils.py` | Terminal timing screen + PIL → image for GIFs |
| `gym.ipynb` | Notebook experiments (if present) |

**Artifacts (not always in git):** `f1_best_weights.pkl`, `f1_trained_weights.pkl`, `runs/` (TensorBoard), `full_race_telemetry.csv`, generated GIFs.

---

## Making documentation accessible on GitHub

GitHub is **documentation-first**: visitors land on the repo home and see **`README.md`** at the **root**. Link clearly to **`jekyll/`** or the published Pages URL.

### Root `README.md`

- Short overview, install, how to run train/eval, and links to docs.

### This site (Jekyll)

- **`index.md`** — home page.
- **`pages/mydoc/`** — guides and long-form reference.
- Sidebar TOC is **`_data/sidebars/mydoc_sidebar.yml`**.

### Optional: Wiki

- Separate editable space; avoid duplicating `jekyll/` content unless you link back.

### Discoverability extras

| Feature | Purpose |
|---------|---------|
| **About → Website** | Link to GitHub Pages |
| **About → Topics** | e.g. `reinforcement-learning`, `jax`, `flax`, `simulation`, `formula-1` |
| **`requirements.txt`** | Python deps for training |

---

## License & attribution

Simulation and project attribution appear in the root `README.md` and in-code credits where applicable.
