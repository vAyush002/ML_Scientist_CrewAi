"""CrewAI Flow: the deterministic orchestration layer (Option B).

The Flow owns the order, the shared state, and the branch between classification
and regression. Each step delegates the real work to a pipeline stage function,
which in turn runs the relevant agent/crew + tool. This keeps the high-level
control flow auditable while the agents stay specialised.

Run from the CLI:
    python -m ml_scientist.flow path/to/data.csv --target price
"""
from __future__ import annotations

import argparse
import json
import os

from crewai.flow.flow import Flow, listen, or_, router, start

from . import config
from .agents import build_agents
from .pipeline import MLState, STAGE_FUNCS
from .tools import _make_tools


class MLScientistFlow(Flow[MLState]):
    def __init__(self, use_llm: bool = True):
        super().__init__()
        self._use_llm = use_llm
        if use_llm:
            self._llm = config.get_llm()
            self._agents = build_agents(self._llm, _make_tools())
        else:
            self._llm = None
            self._agents = None

    # ---- linear prefix --------------------------------------------------- #
    @start()
    def profile(self):
        STAGE_FUNCS["profile"](self.state, self._agents, self._llm)
        return "profiled"

    @listen(profile)
    def analyze(self, _):
        STAGE_FUNCS["analyze"](self.state, self._agents, self._llm)
        return "analyzed"

    @listen(analyze)
    def frame(self, _):
        STAGE_FUNCS["frame"](self.state, self._agents, self._llm)
        return self.state.task_type  # routed on below

    # ---- branch ---------------------------------------------------------- #
    @router(frame)
    def route(self):
        return "classification" if self.state.task_type == "classification" else "regression"

    @listen("classification")
    def engineer_classification(self):
        STAGE_FUNCS["engineer"](self.state, self._agents, self._llm)
        STAGE_FUNCS["train"](self.state, self._agents, self._llm)
        return "trained"

    @listen("regression")
    def engineer_regression(self):
        STAGE_FUNCS["engineer"](self.state, self._agents, self._llm)
        STAGE_FUNCS["train"](self.state, self._agents, self._llm)
        return "trained"

    # ---- merge + finish -------------------------------------------------- #
    @listen(or_(engineer_classification, engineer_regression))
    def evaluate(self, _):
        STAGE_FUNCS["evaluate"](self.state, self._agents, self._llm)
        return "evaluated"

    @listen(evaluate)
    def report(self, _):
        STAGE_FUNCS["report"](self.state, self._agents, self._llm)
        return self.state.report


def run(dataset_path: str, target: str = "", use_llm: bool = True) -> MLState:
    flow = MLScientistFlow(use_llm=use_llm)
    flow.kickoff(inputs={"dataset_path": dataset_path, "target_hint": target, "use_llm": use_llm})
    return flow.state


def main():
    ap = argparse.ArgumentParser(description="CrewAI Automated ML Scientist")
    ap.add_argument("dataset", help="Path to a CSV or XLSX file")
    ap.add_argument("--target", default="", help="Target column (optional; inferred if omitted)")
    ap.add_argument("--model", default="", help="Model, e.g. anthropic/claude-sonnet-4-5 or groq/llama-3.3-70b-versatile")
    ap.add_argument("--dry-run", action="store_true", help="Deterministic analysis only, no agents")
    args = ap.parse_args()

    if args.model:
        os.environ["MODEL"] = args.model
        config.DEFAULT_MODEL = args.model

    state = run(args.dataset, args.target, use_llm=not args.dry_run)
    print(json.dumps(state.modeling, indent=2))
    print("\n" + "=" * 60 + "\nREPORT\n" + "=" * 60)
    print(state.report)


if __name__ == "__main__":
    main()
