"""Run the pipeline across several LLMs and compare their outputs.

The ML computation is deterministic (done by tools, not the model), so two
models that frame the problem identically share the exact same metrics. To
avoid retraining in that case, modeling results are cached by (target, task).

What actually differs between models, and what this module surfaces:
  - the framing decision (target column + classification/regression)
  - the recommendation and the written report
  - latency (Groq is typically much faster than a frontier model)
"""
from __future__ import annotations

import time

from . import config
from .agents import build_agents
from .pipeline import (
    MLState,
    stage_analyze,
    stage_engineer,
    stage_evaluate,
    stage_frame,
    stage_profile,
    stage_report,
    stage_train,
)
from .tools import _make_tools

# model string -> (friendly label, provider)
MODEL_CATALOG = {
    "anthropic/claude-sonnet-4-5": ("Claude Sonnet 4.5", "anthropic"),
    "openai/gpt-4o": ("GPT-4o", "openai"),
    "openai/gpt-4o-mini": ("GPT-4o mini", "openai"),
    "groq/llama-3.3-70b-versatile": ("Llama 3.3 70B", "groq"),
    "groq/llama-3.1-8b-instant": ("Llama 3.1 8B Instant", "groq"),
    "groq/openai/gpt-oss-120b": ("GPT-OSS 120B", "groq"),
}


def label_for(model: str) -> str:
    return MODEL_CATALOG.get(model, (model, ""))[0]


def available_models(have_anthropic: bool, have_groq: bool, have_openai: bool = False) -> dict[str, str]:
    """Catalog filtered to models whose provider key is present."""
    present = {"anthropic": have_anthropic, "groq": have_groq, "openai": have_openai}
    return {model: lbl for model, (lbl, prov) in MODEL_CATALOG.items() if present.get(prov)}


def _primary_metric(state: MLState):
    h = state.modeling.get("holdout_metrics") or {}
    if state.task_type == "classification":
        for k in ("f1_weighted", "accuracy"):
            if k in h:
                return k, h[k]
    elif "r2" in h:
        return "r2", h["r2"]
    return "cv_score", state.modeling.get("best_cv_score")


def run_for_model(dataset_path: str, target_hint: str, model: str, cache: dict,
                  user_goal: str = "") -> dict:
    """Run the full pipeline for one model. Never raises — errors are returned."""
    t0 = time.perf_counter()
    try:
        llm = config.get_llm(model)
        agents = build_agents(llm, _make_tools())
        state = MLState(dataset_path=dataset_path, target_hint=target_hint,
                        user_goal=user_goal, use_llm=True)

        stage_profile(state, agents, llm)
        stage_analyze(state, agents, llm)
        stage_frame(state, agents, llm)          # the model's framing decision
        stage_engineer(state, agents, llm)

        key = (state.target, state.task_type)
        if key in cache:                          # another model already trained this
            state.modeling = cache[key]
            state.notes["train"] = (
                f"Reused training for target '{state.target}' "
                f"({state.task_type}) — identical to another model's framing."
            )
        else:
            stage_train(state, agents, llm)
            cache[key] = state.modeling

        stage_evaluate(state, agents, llm)
        stage_report(state, agents, llm)

        metric_name, metric = _primary_metric(state)
        return {
            "model": model,
            "label": label_for(model),
            "ok": True,
            "error": "",
            "target": state.target,
            "task_type": state.task_type,
            "framing_reason": state.notes.get("frame", ""),
            "best_model": state.modeling.get("best_model"),
            "best_cv_score": state.modeling.get("best_cv_score"),
            "metric_name": metric_name,
            "metric": metric,
            "holdout": state.modeling.get("holdout_metrics"),
            "evaluation_note": state.notes.get("evaluate", ""),
            "report": state.report,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
    except Exception as e:
        return {
            "model": model,
            "label": label_for(model),
            "ok": False,
            "error": str(e)[:400],
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }


def run_comparison(dataset_path: str, target_hint: str, models: list[str],
                   user_goal: str = "") -> list[dict]:
    cache: dict = {}
    return [run_for_model(dataset_path, target_hint, m, cache, user_goal) for m in models]
