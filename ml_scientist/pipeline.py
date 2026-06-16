"""The seven pipeline stages — the single source of truth.

Both the CrewAI Flow (flow.py) and the Streamlit UI (app.py) call these, so the
agents/crews/tools are exercised identically either way.

Each stage:
  1. computes deterministic FACTS with a tool function (never hallucinated), then
  2. if use_llm, runs a one-agent crew that interprets those facts in words.

In dry-run (use_llm=False) the facts are still produced; only the narrative is
skipped. That lets the whole analysis run with no API key.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel

from . import tools as T


class MLState(BaseModel):
    # inputs
    dataset_path: str = ""
    target_hint: str = ""
    user_goal: str = ""   # what the user is trying to solve / their aim
    use_llm: bool = True
    # filled in as the pipeline runs
    profile: dict = {}
    eda: dict = {}
    suggestion: dict = {}
    task_type: str = ""
    target: str = ""
    plan: dict = {}
    modeling: dict = {}
    notes: dict = {}      # stage_name -> agent narrative
    report: str = ""

    def profile_columns(self) -> list[str]:
        return [c["name"] for c in self.profile.get("columns", [])]


STAGE_ORDER = ["profile", "analyze", "frame", "engineer", "train", "evaluate", "report"]
STAGE_LABELS = {
    "profile": "Profile data",
    "analyze": "Analyze data",
    "frame": "Frame problem",
    "engineer": "Engineer features",
    "train": "Train models",
    "evaluate": "Evaluate models",
    "report": "Write report",
}


def _interpret(agent, description: str, expected: str, llm) -> str:
    """Run a single agent on a single task and return its text output."""
    if llm is None:
        return ""
    from crewai import Crew, Process, Task

    task = Task(description=description, expected_output=expected, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(crew.kickoff())


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_profile(state: MLState, agents=None, llm=None) -> MLState:
    state.profile = T.profile_dataset(state.dataset_path)
    if agents:
        state.notes["profile"] = _interpret(
            agents["profiler"],
            f"Here is a profile of the dataset:\n{json.dumps(state.profile)}\n\n"
            "Summarise the dataset's size and quality. Call out missing data, "
            "duplicates and anything that needs attention. Do not invent numbers.",
            "A short plain-language summary of the dataset's structure and quality.",
            llm,
        )
    return state


def stage_analyze(state: MLState, agents=None, llm=None) -> MLState:
    state.eda = T.analyze_dataset(state.dataset_path, state.target_hint or None)
    if agents:
        state.notes["analyze"] = _interpret(
            agents["analyst"],
            f"Exploratory analysis results:\n{json.dumps(state.eda)}\n\n"
            "Explain the most important relationships and any risks: leakage, "
            "imbalance, high-cardinality or constant columns. Be concise.",
            "A concise analysis of relationships and modeling risks.",
            llm,
        )
    return state


def stage_frame(state: MLState, agents=None, llm=None) -> MLState:
    suggestion = T.suggest_target_and_task(state.dataset_path, state.target_hint or None)
    state.suggestion = suggestion
    state.target = suggestion["target"]
    state.task_type = suggestion["task_type"]

    if agents:
        goal_line = (f"The user describes their goal as: \"{state.user_goal}\". "
                     "Take this into account. ") if state.user_goal else ""
        raw = _interpret(
            agents["framer"],
            f"Dataset profile:\n{json.dumps(state.profile)}\n\n"
            f"{goal_line}"
            f"A heuristic suggests target='{suggestion['target']}' and "
            f"task='{suggestion['task_type']}'. Confirm or correct this. "
            'Respond with STRICT JSON only: {"target": "<col>", "task_type": '
            '"classification|regression", "reason": "<one sentence>"}',
            "Strict JSON with target, task_type and a one-sentence reason.",
            llm,
        )
        parsed = _safe_json(raw)
        if parsed and parsed.get("target") in state.profile_columns():
            state.target = parsed["target"]
            if parsed.get("task_type") in {"classification", "regression"}:
                state.task_type = parsed["task_type"]
            state.notes["frame"] = parsed.get("reason", raw)
        else:
            state.notes["frame"] = raw or ""
    return state


def stage_engineer(state: MLState, agents=None, llm=None) -> MLState:
    state.plan = T.engineering_plan(state.dataset_path, state.target)
    if agents:
        state.notes["engineer"] = _interpret(
            agents["engineer"],
            f"Preprocessing plan:\n{json.dumps(state.plan)}\n\n"
            "Explain in plain terms how features will be prepared and why this is "
            "appropriate for the data.",
            "A short rationale for the preprocessing strategy.",
            llm,
        )
    return state


def stage_train(state: MLState, agents=None, llm=None) -> MLState:
    state.modeling = T.run_modeling(state.dataset_path, state.target, state.task_type)
    if agents:
        state.notes["train"] = _interpret(
            agents["trainer"],
            f"Training results:\n{json.dumps(state.modeling)}\n\n"
            "Describe which models were trained and how they were validated "
            "(cross-validation, folds). Do not change any numbers.",
            "A short description of the training and validation setup.",
            llm,
        )
    return state


def stage_evaluate(state: MLState, agents=None, llm=None) -> MLState:
    if agents:
        state.notes["evaluate"] = _interpret(
            agents["evaluator"],
            f"Model results:\n{json.dumps(state.modeling)}\n\n"
            "Compare the candidates, name the winner and explain whether the score "
            "is trustworthy given the dataset size and any imbalance. Add caveats.",
            "An honest comparison, a recommended model, and caveats.",
            llm,
        )
    return state


def _compact_facts(state: MLState) -> dict:
    """A small, token-cheap summary for the reporter (avoids dumping every
    column's samples, which is what blows up the prompt on wide datasets)."""
    p = state.profile
    cols_with_missing = [
        {"name": c["name"], "missing_pct": c["missing_pct"]}
        for c in p.get("columns", [])
        if c.get("missing_pct", 0) > 0
    ][:15]
    return {
        "rows": p.get("n_rows"),
        "cols": p.get("n_cols"),
        "duplicate_rows": p.get("duplicate_rows"),
        "columns_with_missing": cols_with_missing,
        "data_quality_flags": {
            k: state.eda.get(k)
            for k in ("high_missing_cols", "constant_cols", "high_cardinality_cols")
        },
        "top_correlations": state.eda.get("top_correlations", [])[:5],
        "target": state.target,
        "task_type": state.task_type,
        "n_features": state.plan.get("n_features"),
        "models": state.modeling.get("models"),
        "best_model": state.modeling.get("best_model"),
        "best_cv_score": state.modeling.get("best_cv_score"),
        "holdout_metrics": state.modeling.get("holdout_metrics"),
    }


def stage_report(state: MLState, agents=None, llm=None) -> MLState:
    if not agents:
        state.report = _fallback_report(state)
        return state

    try:
        goal_line = (f"The user's stated goal: \"{state.user_goal}\". Tailor the "
                     "Recommendations & Next Steps to it. ") if state.user_goal else ""
        state.report = _interpret(
            agents["reporter"],
            "Write a final Markdown report for a developer from this summary "
            f"(JSON):\n{json.dumps(_compact_facts(state))}\n\n"
            f"{goal_line}"
            "Sections: Overview, Data Quality, Problem Framing, Modeling, "
            "Results, Recommendations & Next Steps. Use the exact numbers given; "
            "do not invent any. Be honest and specific.",
            "A clean Markdown report with the requested sections.",
            llm,
        )
        if not state.report.strip():
            raise ValueError("model returned an empty report")
    except Exception as e:  # never let the last stage crash the whole run
        state.notes["report_error"] = str(e)[:300]
        state.report = (
            "> ⚠️ The report agent could not produce a write-up "
            f"(`{str(e)[:160]}`). Showing the deterministic summary instead.\n\n"
            + _fallback_report(state)
        )
    return state


STAGE_FUNCS = {
    "profile": stage_profile,
    "analyze": stage_analyze,
    "frame": stage_frame,
    "engineer": stage_engineer,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "report": stage_report,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe_json(text: str):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _fallback_report(state: MLState) -> str:
    m = state.modeling
    lines = [
        "# Automated ML Report (deterministic, no agent narration)",
        "",
        f"- **Rows × Columns:** {state.profile.get('n_rows')} × {state.profile.get('n_cols')}",
        f"- **Target:** `{state.target}`  |  **Task:** {state.task_type}",
        f"- **Best model:** {m.get('best_model')}  (CV {m.get('best_cv_score')})",
        f"- **Holdout metrics:** {m.get('holdout_metrics')}",
        f"- **Saved model:** {m.get('saved_model_path')}",
        "",
        "_Run with an API key to get full agent commentary and recommendations._",
    ]
    return "\n".join(lines)
