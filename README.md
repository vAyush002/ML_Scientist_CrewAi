# 🔬 CrewAI Automated ML Scientist

Upload a dataset and a crew of specialist agents does the work of an ML scientist
end to end: profiles the data, frames the problem, engineers features, trains and
compares models, and writes you a report.

Built on **CrewAI Flows + Crews**, with **Claude, OpenAI or Groq** as the
reasoning engine and a **Streamlit** front end.

---

## How it works

A deterministic **Flow** owns the order and branches by problem type. Each stage
runs a specialist **agent** (a Crew of one) that interprets facts produced by a
**tool**. The numbers are always computed in Python — the agents never invent them.

```
Upload → Profile → Analyze → Frame ─┬─ classification ─┐
                                    └─ regression ──────┴─ Train → Evaluate → Report
```

| # | Agent | Tool / power |
|---|-------|--------------|
| 1 | Data Profiler | shape, dtypes, missing %, uniqueness |
| 2 | Exploratory Data Analyst | correlations, leakage, imbalance, bad columns |
| 3 | Problem Framer | chooses target + classification vs regression |
| 4 | Feature Engineer | impute / scale / one-hot plan |
| 5 | Model Trainer | trains Logistic/Linear, Decision Tree, Random Forest, SVM/SVR, XGBoost w/ CV |
| 6 | Model Evaluator | compares candidates, picks a winner, adds caveats |
| 7 | Report Writer | plain-language Markdown report tailored to your stated goal |

**Why this design:** doing the maths deterministically keeps results reproducible
and auditable; the agents add the judgement (what's the target, which model to
trust, what to do next). That's the point of the Flows layer.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add a key (or paste it in the UI)
```

**Pick a provider** — set one key and a matching `MODEL`. The model prefix
selects the provider:

| Provider | Key env var | Example model |
|----------|-------------|---------------|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4-5` |
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o-mini` |
| Groq (Llama / open) | `GROQ_API_KEY` | `groq/llama-3.3-70b-versatile` |

> Groq is fast and has a free tier (get a key at console.groq.com). For the
> agents' tool calls, prefer `llama-3.3-70b-versatile` — `llama-3.1-8b-instant`
> is faster with higher rate limits but less reliable at tool use.

## Run

**UI (recommended):**
```bash
streamlit run app.py
```
Enter your key(s) in the sidebar, upload a CSV/XLSX, optionally pick the target,
then choose a mode:

- **Single run** — one model works the pipeline; the rail on the left lights up
  stage by stage, and you get the report + trained model to download. After the
  run finishes, a **fine-tuning panel** lets you pick any algorithm (Logistic /
  Linear+Ridge / Decision Tree / Random Forest / SVM-SVR / XGBoost) and search
  its hyperparameters — randomized or exhaustive grid — with the tuned CV
  score shown next to the baseline so you see whether tuning actually helped.
- **Compare models** — pick two or more models (mix Anthropic, OpenAI and Groq
  freely); each runs the full pipeline and the results are shown side by side.

### Telling the agents your goal

Above the mode switch there's an optional **"What are you trying to solve?"**
box. Whatever you write is passed to the Problem Framer (which uses it to pick
the right target/task) and to the Report Writer (which tailors the
Recommendations & Next Steps section to it).

### Fine-tuning details

- Uses `RandomizedSearchCV` (default) or `GridSearchCV` from scikit-learn over
  per-algorithm grids defined in `tools.py`.
- `linear_regression` has no native hyperparameters, so tuning it transparently
  upgrades to **Ridge regression** with an `alpha` search.
- Tuned model is saved to `artifacts/tuned_model.joblib` and downloadable.
- The panel shows the **improvement** over the untuned baseline — a negative
  number means tuning didn't help, which is real and worth knowing.

### What "Compare models" actually compares

The ML metrics are computed **deterministically** by the tools, not the model —
so two models that frame the problem the same way get identical metrics. The
comparison therefore highlights what genuinely differs:

- **Framing** — which target column and task type each model chose (if they
  disagree, that's flagged, because it *does* change the metrics).
- **Reports & recommendations** — shown in side-by-side tabs.
- **Latency** — a bar chart; Groq is usually much faster than a frontier model.

A training cache means models that agree on the framing don't retrain — the
heavy work runs once.

**CLI / the raw CrewAI Flow:**
```bash
python -m ml_scientist.flow data.csv --target price
python -m ml_scientist.flow data.csv --model groq/llama-3.3-70b-versatile
python -m ml_scientist.flow data.csv --dry-run    # deterministic only, no API key
```

## Dry run

Toggle **Dry run** (or `--dry-run`) to compute every profile, chart and metric
without calling any agent — useful for a quick look or when you have no API key.

## Output

Trained model and report are written to `artifacts/`. The UI also offers direct
downloads of the report (`.md`) and the fitted scikit-learn pipeline (`.joblib`).

---

## Project layout

```
app.py                     Streamlit UI (pipeline rail + results)
ml_scientist/
  config.py                env + Claude LLM factory
  tools.py                 deterministic compute + CrewAI tool wrappers
  agents.py                the seven specialist agents
  pipeline.py              shared stage functions + MLState
  flow.py                  CrewAI Flow (orchestration + branching) and CLI
.streamlit/config.toml     theme
```

## Notes & limits

- Tabular data only (CSV/XLSX). One target column.
- Models: LogReg/Linear, RandomForest, and XGBoost if installed.
- Small/imbalanced datasets: CV folds adapt down automatically; trust the
  evaluator's caveats.
- This is a strong baseline auto-ML pipeline, not a replacement for a human on
  high-stakes problems.
