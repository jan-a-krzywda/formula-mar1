# Documentation guide

## Published documentation site (GitHub Pages)

This project includes **MkDocs + Material** and a **GitHub Actions** workflow that deploys the site to the **`gh-pages`** branch.

1. Push these files to **`main`** (or **`master`**): `mkdocs.yml`, `docs/`, `requirements-docs.txt`, `.github/workflows/docs.yml`.
2. Open **Settings → Pages** in the GitHub repo.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
4. Choose branch **`gh-pages`**, folder **`/ (root)`**, Save.
5. After the workflow runs, the site is available at:

   **`https://<your-username>.github.io/<repo-name>/`**

**Local preview:** `pip install -r requirements-docs.txt` then `mkdocs serve`.

**Customize:** Edit `repo_url`, `edit_uri`, and `social` links in **`mkdocs.yml`** (replace `YOUR_GITHUB_USERNAME`).

---

Welcome to the **Formula Mar1** project documentation. These pages also render as plain Markdown on GitHub.

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

- **`guide.md`** (this file) acts as a **table of contents** for the doc site and GitHub.
- **`index.md`** is the **MkDocs** homepage; **`README.md`** in `docs/` is a short pointer for the GitHub UI.

### 3. Optional: GitHub Pages (static site)

The repo uses **MkDocs** deployed to **`gh-pages`** (see top of this page). You can also browse Markdown directly on GitHub without Pages.

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
