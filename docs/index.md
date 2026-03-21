# Formula Mar1

Welcome to the **documentation site** for the Formula Mar1 project: a **Formula 1–style** multi-car race simulation with **tyre strategy**, **pit stops**, **ERS-style energy modes**, and **reinforcement learning** (behavioral cloning + PPO) for the trainable team **BlueCow**.

---

## Start here

| I want to… | Go to |
|------------|--------|
| Understand the env, rewards, PPO, and BC | [**Algorithm & RL reference**](FORMULA_MAR1_ALGORITHM.md) |
| Repo layout, hosting notes, file map | [**Documentation guide**](guide.md) |
| Clone and run training | Root **`README.md`** in the repository (install, `train.py`, `evaluate.py`) |

---

## At a glance

- **22 cars · 11 teams** — Random grid; scripted 1-stop / 2-stop baselines for non-trained teams.
- **Actions** — Starting compound (S/M/H); each lap: pit compound or stay out (Harvest / Boost / Standard).
- **Stack** — JAX, Flax, Optax; vectorized environments; TensorBoard.

The full technical write-up (observation space, reward terms, Mermaid diagrams, blog prompts) lives in the [**algorithm reference**](FORMULA_MAR1_ALGORITHM.md).

---

## Build this site locally

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Open `http://127.0.0.1:8000` to preview. The published URL after GitHub Pages is enabled is typically:

`https://<your-username>.github.io/<repo-name>/`

---

## License

See the repository **LICENSE** if present. Simulation credits appear in the codebase and original README.
