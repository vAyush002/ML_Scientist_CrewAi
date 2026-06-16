"""CrewAI Automated ML Scientist — Streamlit front end.

Modes:
  - Single run: one model works the pipeline; rail lights up stage by stage,
    then you can fine-tune any algorithm.
  - Compare models: several models each run the pipeline; outputs side by side.

Providers: Anthropic, OpenAI, Groq (enter any combination of keys).
Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st

from ml_scientist import STAGE_LABELS, STAGE_ORDER
from ml_scientist import compare as C
from ml_scientist import tools as T
from ml_scientist.pipeline import MLState, STAGE_FUNCS

st.set_page_config(page_title="ML Scientist", page_icon="🔬", layout="wide")

ALGO_LABELS = {
    "logistic_regression": "Logistic Regression", "linear_regression": "Linear / Ridge Regression",
    "decision_tree": "Decision Tree", "random_forest": "Random Forest",
    "svm": "SVM (SVC)", "svr": "SVR", "xgboost": "XGBoost",
}

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=IBM+Plex+Mono:wght@500&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .stApp { background: #FBFBFD; }
      .mono, .stMetric, code { font-family: 'IBM Plex Mono', monospace; }
      .eyebrow { font-family:'IBM Plex Mono',monospace; font-size:.72rem;
                 letter-spacing:.18em; text-transform:uppercase; color:#7A8194; }
      h1 { font-weight:600; letter-spacing:-.02em; }
      .rail-step { display:flex; align-items:flex-start; gap:.7rem; padding:.35rem 0; }
      .rail-num { font-family:'IBM Plex Mono',monospace; font-size:.8rem;
                  width:1.4rem; height:1.4rem; border-radius:50%; flex:none;
                  display:flex; align-items:center; justify-content:center;
                  border:1.5px solid #C7CCDA; color:#9aa1b3; background:#fff; }
      .rail-step.done .rail-num { background:#2D5BFF; border-color:#2D5BFF; color:#fff; }
      .rail-step.active .rail-num { border-color:#2D5BFF; color:#2D5BFF; }
      .rail-label { font-size:.92rem; padding-top:.05rem; color:#1C2230; }
      .rail-step.pending .rail-label { color:#A6ACBC; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_rail(container, current: int):
    html = ['<div class="eyebrow">pipeline</div>']
    for i, name in enumerate(STAGE_ORDER):
        cls = "done" if i < current else "active" if i == current else "pending"
        html.append(f'<div class="rail-step {cls}"><div class="rail-num">{i+1}</div>'
                    f'<div class="rail-label">{STAGE_LABELS[name]}</div></div>')
    container.markdown("".join(html), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar — all three provider keys
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="eyebrow">credentials</div>', unsafe_allow_html=True)
    anthropic_key = st.text_input("Anthropic API key", type="password")
    openai_key = st.text_input("OpenAI API key", type="password")
    groq_key = st.text_input("Groq API key", type="password",
                             help="Free at console.groq.com.")
    st.caption("Enter any combination. Compare mode can mix providers.")
    dry_run = st.toggle("Dry run (no agents)", value=False,
                        help="Deterministic analysis only — no key needed (single-run).")

for env, val in (("ANTHROPIC_API_KEY", anthropic_key),
                 ("OPENAI_API_KEY", openai_key),
                 ("GROQ_API_KEY", groq_key)):
    if val:
        os.environ[env] = val
    else:
        os.environ.pop(env, None)

have = {"anthropic": bool(anthropic_key), "openai": bool(openai_key), "groq": bool(groq_key)}
model_options = C.available_models(have["anthropic"], have["groq"], have["openai"])

# --------------------------------------------------------------------------- #
# Header + upload
# --------------------------------------------------------------------------- #
st.markdown('<div class="eyebrow">crewai · multi-agent</div>', unsafe_allow_html=True)
st.title("Automated ML Scientist")
st.write("Upload a dataset. A crew of specialist agents profiles it, frames the "
         "problem, trains and compares models, and writes you a report.")

upload = st.file_uploader("Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
if upload is None:
    st.info("Upload a CSV or Excel file to begin.")
    st.stop()

# stable temp path per uploaded file (survives reruns; resets results on new file)
sig = f"{upload.name}:{upload.size}"
if st.session_state.get("ds_sig") != sig:
    suffix = os.path.splitext(upload.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getbuffer()); tmp.flush()
    st.session_state["ds_sig"] = sig
    st.session_state["ds_path"] = tmp.name
    st.session_state["ds_suffix"] = suffix
    st.session_state.pop("sr", None)
    st.session_state.pop("tune", None)
dataset_path = st.session_state["ds_path"]
suffix = st.session_state["ds_suffix"]

preview = pd.read_excel(dataset_path) if suffix in {".xlsx", ".xls"} else pd.read_csv(dataset_path)
st.dataframe(preview.head(10), use_container_width=True)
cols = ["(let the agents decide)"] + list(map(str, preview.columns))
choice = st.selectbox("Target column", cols)
target_hint = "" if choice == cols[0] else choice

user_goal = st.text_area(
    "What are you trying to solve, and what's your aim? (optional)",
    placeholder="e.g. Predict which customers will churn so we can target retention offers.",
    height=80,
)

mode = st.radio("Mode", ["Single run", "Compare models"], horizontal=True)


def init_crew(model: str):
    from ml_scientist import config
    from ml_scientist.agents import build_agents
    from ml_scientist.tools import _make_tools
    llm = config.get_llm(model)
    return build_agents(llm, _make_tools()), llm


# =========================================================================== #
# SINGLE RUN
# =========================================================================== #
if mode == "Single run":
    if dry_run:
        model = "(deterministic)"
        st.caption("Dry run: deterministic analysis only, no model calls.")
    elif model_options:
        model = st.selectbox("Model", list(model_options),
                             format_func=lambda m: f"{model_options[m]}  ·  {m}")
    else:
        st.warning("Add a key in the sidebar, or switch on Dry run.")
        st.stop()

    if st.button("Run the crew", type="primary"):
        agents = llm = None
        if not dry_run:
            try:
                agents, llm = init_crew(model)
            except Exception as e:
                st.error(f"Could not initialise the crew: {e}")
                st.stop()

        state = MLState(dataset_path=dataset_path, target_hint=target_hint,
                        user_goal=user_goal, use_llm=not dry_run)
        left, right = st.columns([1, 2.4])
        rail_box = left.empty()
        for i, name in enumerate(STAGE_ORDER):
            render_rail(rail_box, i)
            with right:
                with st.status(f"{i+1}. {STAGE_LABELS[name]}", expanded=False) as status:
                    try:
                        STAGE_FUNCS[name](state, agents, llm)
                    except Exception as e:
                        status.update(label=f"{i+1}. {STAGE_LABELS[name]} — failed", state="error")
                        st.exception(e); st.stop()
                    if state.notes.get(name):
                        st.markdown(state.notes[name])
                    status.update(label=f"{i+1}. {STAGE_LABELS[name]} — done", state="complete")
        render_rail(rail_box, len(STAGE_ORDER))
        st.session_state["sr"] = {
            "dataset_path": dataset_path, "target": state.target, "task_type": state.task_type,
            "modeling": state.modeling, "report": state.report,
        }
        st.session_state.pop("tune", None)

    # ---- results + tuning (rendered from session so they survive reruns) ----
    sr = st.session_state.get("sr")
    if sr and sr["dataset_path"] == dataset_path:
        st.divider()
        st.markdown('<div class="eyebrow">results</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Target", sr["target"]); c2.metric("Task", sr["task_type"])
        c3.metric("Best model", str(sr["modeling"].get("best_model")))

        models = sr["modeling"].get("models", {})
        if models:
            mdf = pd.DataFrame([{"model": k, **v} for k, v in models.items()])
            st.dataframe(mdf, use_container_width=True)
            if "cv_mean" in mdf:
                fig = px.bar(mdf.dropna(subset=["cv_mean"]), x="model", y="cv_mean",
                             error_y="cv_std", title="Cross-validation score by model",
                             color_discrete_sequence=["#2D5BFF"])
                fig.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff")
                st.plotly_chart(fig, use_container_width=True)
        if sr["modeling"].get("holdout_metrics"):
            st.write("**Holdout metrics (best model):**", sr["modeling"]["holdout_metrics"])

        if sr["report"]:
            st.markdown(sr["report"])
            st.download_button("Download report (Markdown)", sr["report"], file_name="ml_report.md")

        # ---------- FINE-TUNING ----------
        st.divider()
        st.markdown('<div class="eyebrow">fine-tuning</div>', unsafe_allow_html=True)
        st.write("Pick an algorithm and search its hyperparameters for a better model.")
        algos = T.available_algorithms(sr["task_type"])
        ta, tb, tc = st.columns([2, 1.4, 1])
        algo = ta.selectbox("Algorithm", algos, format_func=lambda a: ALGO_LABELS.get(a, a))
        search_type = tb.radio("Search", ["random", "grid"], horizontal=True,
                               format_func=lambda s: "Randomized" if s == "random" else "Grid (exhaustive)")
        n_iter = tc.slider("Iters", 5, 50, 20, disabled=(search_type == "grid"),
                           help="Random search only.")

        if st.button(f"Fine-tune {ALGO_LABELS.get(algo, algo)}", type="primary"):
            with st.spinner("Searching hyperparameters…"):
                st.session_state["tune"] = T.tune_model(
                    dataset_path, sr["target"], sr["task_type"], algo,
                    search_type=search_type, n_iter=n_iter)

        tune = st.session_state.get("tune")
        if tune:
            if tune.get("error"):
                st.info(tune["error"])
            else:
                st.markdown(f"**Tuned {ALGO_LABELS.get(tune['algorithm'], tune['algorithm'])}** "
                            f"· {tune['candidates_evaluated']} configs · {tune['cv_metric']}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Tuned CV", tune["tuned_cv_score"])
                m2.metric("Baseline CV", tune.get("baseline_cv_score"),
                          delta=tune.get("improvement"))
                m3.metric("Holdout", str(list(tune["holdout_metrics"].values())[0]))
                st.write("**Best parameters:**", tune["best_params"])
                st.write("**Holdout metrics:**", tune["holdout_metrics"])
                if os.path.exists(tune["saved_model_path"]):
                    with open(tune["saved_model_path"], "rb") as f:
                        st.download_button("Download tuned model (.joblib)", f,
                                           file_name="tuned_model.joblib")

# =========================================================================== #
# COMPARE MODELS
# =========================================================================== #
else:
    st.caption("Each selected model runs the full pipeline. ML metrics are "
               "computed deterministically, so they match unless models pick a "
               "different target/task — the real differences are in the decisions, "
               "the reports, and the speed.")
    if not model_options:
        st.warning("Add at least one provider key in the sidebar to compare models.")
        st.stop()

    default = list(model_options)[: min(2, len(model_options))]
    chosen = st.multiselect("Models to compare", list(model_options), default=default,
                            format_func=lambda m: f"{model_options[m]}  ·  {m}")

    if st.button("Run comparison", type="primary"):
        if len(chosen) < 2:
            st.warning("Pick at least two models to compare."); st.stop()
        cache, results = {}, []
        prog = st.progress(0.0, text="Starting…")
        for idx, model in enumerate(chosen):
            prog.progress(idx / len(chosen), text=f"Running {model_options[model]}…")
            with st.status(f"{model_options[model]}", expanded=False) as status:
                res = C.run_for_model(dataset_path, target_hint, model, cache, user_goal)
                results.append(res)
                if res["ok"]:
                    status.update(label=f"{model_options[model]} — {res['elapsed_s']}s", state="complete")
                else:
                    status.update(label=f"{model_options[model]} — failed", state="error")
                    st.error(res["error"])
        prog.progress(1.0, text="Done")

        ok = [r for r in results if r["ok"]]
        if not ok:
            st.error("No model finished successfully."); st.stop()

        st.divider(); st.markdown('<div class="eyebrow">comparison</div>', unsafe_allow_html=True)
        table = pd.DataFrame([{
            "Model": r["label"], "Target": r["target"], "Task": r["task_type"],
            "Best ML model": r["best_model"], "Score": r["metric"],
            "Metric": r["metric_name"], "Time (s)": r["elapsed_s"]} for r in ok])
        st.dataframe(table, use_container_width=True)

        targets = {r["target"] for r in ok}; tasks = {r["task_type"] for r in ok}
        if len(targets) > 1 or len(tasks) > 1:
            lines = [f"- **{r['label']}** → target `{r['target']}`, {r['task_type']}" for r in ok]
            st.warning("Models **disagreed on how to frame the problem** — this is the "
                       "key difference, and it changes the metrics:\n" + "\n".join(lines))
        else:
            st.success(f"All models agreed: target `{ok[0]['target']}` · {ok[0]['task_type']}. "
                       "Metrics are identical by construction; compare reports and speed below.")

        fig = px.bar(pd.DataFrame([{"Model": r["label"], "Time (s)": r["elapsed_s"]} for r in ok]),
                     x="Model", y="Time (s)", title="Latency by model",
                     color_discrete_sequence=["#2D5BFF"])
        fig.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="eyebrow">reports side by side</div>', unsafe_allow_html=True)
        for tab, r in zip(st.tabs([r["label"] for r in ok]), ok):
            with tab:
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Target", r["target"]); cc2.metric("Task", r["task_type"])
                cc3.metric(r["metric_name"], r["metric"])
                if r.get("framing_reason"):
                    st.caption(f"Framing rationale: {r['framing_reason']}")
                st.markdown(r["report"] or "_No report produced._")
                st.download_button("Download this report", r["report"] or "",
                                   file_name=f"report_{r['label'].replace(' ', '_')}.md",
                                   key=f"dl_{r['model']}")
