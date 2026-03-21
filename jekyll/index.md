---
title: Formula Mar1
keywords: documentation, reinforcement learning, F1 simulation
sidebar: mydoc_sidebar
permalink: index.html
summary: Documentation for the Formula Mar1 F1-style race strategy simulation and RL training stack.
---

Welcome to the **documentation site** for the Formula Mar1 project: a **Formula 1–style** multi-car race simulation with **tyre strategy**, **pit stops**, **ERS-style energy modes**, and **reinforcement learning** (behavioral cloning + PPO) for the trainable team **BlueCow**.

---

## Start here

| I want to… | Go to |
|------------|--------|
| Understand the env, rewards, PPO, and BC | [**Algorithm & RL reference**](formula_mar1_algorithm.html) |
| Repo layout, hosting notes, file map | [**Documentation guide**](guide.html) |
| Clone and run training | Root **`README.md`** in the repository (install, `train.py`, `evaluate.py`) |

---

## At a glance

- **22 cars · 11 teams** — Random grid; scripted 1-stop / 2-stop baselines for non-trained teams.
- **Actions** — Starting compound (S/M/H); each lap: pit compound or stay out (Harvest / Boost / Standard).
- **Stack** — JAX, Flax, Optax; vectorized environments; TensorBoard.

The full technical write-up (observation space, reward terms, Mermaid diagrams, blog prompts) lives in the [**algorithm reference**](formula_mar1_algorithm.html).

---

## Build this site locally

```bash
cd jekyll
bundle install
bundle exec jekyll serve
```

Open `http://127.0.0.1:4000/formula-mar1/` when using the default `baseurl: /formula-mar1` in `_config.yml`, or `http://127.0.0.1:4000` when `baseurl` is empty.

The published URL after GitHub Pages is enabled:

`https://jan-a-krzywda.github.io/formula-mar1/`

---

## License

See the repository **LICENSE** if present. Simulation credits appear in the codebase and original README.
