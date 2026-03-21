# Documentation guide

## Published documentation site (GitHub Pages)

**Default deployment:** **Sphinx** API docs from **`sphinx/`** (see **`docs/README.md`**). **GitHub Actions** runs **`sphinx-build`** and pushes to **`gh-pages`**.

1. Push **`sphinx/`**, **`*.py`**, **`.github/workflows/docs.yml`** to **`main`**.
2. **Settings → Pages** → **`gh-pages`** / root.
3. **`https://jan-a-krzywda.github.io/formula-mar1/`** — [Module index](https://jan-a-krzywda.github.io/formula-mar1/py-modindex.html).

**Local preview:** `pip install -r sphinx/requirements.txt && sphinx-build -b html sphinx/source sphinx/build/html`

---

Welcome to the **Formula Mar1** project documentation. These pages render as plain Markdown on GitHub; the **Python API** is on GitHub Pages (Sphinx).

| Document | Description |
|----------|-------------|
| [**Algorithm & RL details**](FORMULA_MAR1_ALGORITHM.md) | Environment, observations, actions, rewards, PPO/BC pipeline, Mermaid diagrams, blog prompts |
| **This guide** | How to navigate docs and how to expose them on GitHub |

---

## Repository map (quick)

| Path | Role |
|------|------|
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

GitHub is **documentation-first**: visitors land on the repo home and see whatever is in **`README.md`** at the **root**. Everything else is discoverable if you link to it clearly.

### 1. Root `README.md` (required)

- This is the **only** file many users will read.
- Keep a **short** overview, **install**, **how to run train/eval**, and **links** into `docs/`.
- Use **relative links** so they work on GitHub and locally:

  ```markdown
  [Algorithm details](FORMULA_MAR1_ALGORITHM.md)
  ```

### 2. This `docs/` folder

- **`guide.md`** (this file) — repo map and how to publish docs.
- **`index.md`** — short pointer for the GitHub UI.

### 3. GitHub Pages (Sphinx API)

See the top of this page: **`sphinx/`** builds the site deployed to **`gh-pages`**. Narrative Markdown stays in **`docs/`** on GitHub.

### 4. Optional: Wiki

- GitHub **Wiki** is a separate editable space; good for drafts or community notes.
- **Downside:** duplicates content unless you link back to `docs/` in the repo.

### 5. Discoverability extras

| Feature | Purpose |
|---------|--------|
| **About → Website** | Link to Pages or an external blog |
| **About → Topics** | e.g. `reinforcement-learning`, `jax`, `flax`, `simulation`, `formula-1` |
| **`requirements.txt`** | Makes install copy-pasteable; helps tools dependabot |
| **Issues / Discussions** | Q&A and roadmap without cluttering README |

### 6. Linking to a specific section in another `.md`

GitHub generates anchors from headings. Example:

```markdown
See [Reward structure](FORMULA_MAR1_ALGORITHM.md#7-reward-structure-per-lap-then-terminal).
```

### 7. Accessibility (readability)

- Prefer **clear headings** (`##`, `###`), **tables**, and **short paragraphs**.
- For images, store them in `docs/assets/` (or repo root) and use relative paths: `![Eval](assets/eval.gif)`.
- Avoid **only** PDFs for primary docs—search engines and screen readers handle Markdown on GitHub better.

---

## License & attribution

Simulation and project attribution appear in the root `README.md` and in-code credits where applicable.
