"""The crew. Each agent is a specialist with a role, goal, backstory and a
specific set of tools (its "powers"). Factories take the LLM so the same
definitions work with any Claude model.
"""
from __future__ import annotations


def build_agents(llm, tools: dict):
    from crewai import Agent

    profiler = Agent(
        role="Data Profiler",
        goal="Understand the raw shape and quality of an unfamiliar dataset.",
        backstory=(
            "A meticulous data engineer who always reads the data before trusting "
            "it. You report structure and quality plainly and never invent numbers."
        ),
        tools=[tools["profile"]],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    analyst = Agent(
        role="Exploratory Data Analyst",
        goal="Surface relationships, risks and data-quality issues that affect modeling.",
        backstory=(
            "A statistician who has seen every way a dataset can mislead: leakage, "
            "imbalance, spurious correlation. You flag these before anyone trains a model."
        ),
        tools=[tools["analyze"]],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    framer = Agent(
        role="Problem Framer",
        goal="Decide the target variable and whether this is classification or regression.",
        backstory=(
            "A pragmatic ML lead who turns a vague 'analyse this' into a precise, "
            "well-posed learning problem and justifies the framing."
        ),
        tools=[],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    engineer = Agent(
        role="Feature Engineer",
        goal="Specify a sound preprocessing strategy for the chosen problem.",
        backstory=(
            "A practitioner who knows that clean inputs beat clever models. You "
            "explain why each feature group is treated the way it is."
        ),
        tools=[tools["engineer"]],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    trainer = Agent(
        role="Model Trainer",
        goal="Train a fair slate of candidate models and report how they were validated.",
        backstory=(
            "An experimenter who trusts cross-validation over hunches and documents "
            "exactly how every score was produced."
        ),
        tools=[tools["model"]],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    evaluator = Agent(
        role="Model Evaluator",
        goal="Compare the candidates honestly and recommend one, with caveats.",
        backstory=(
            "A skeptic who reads metrics in context: a high score on a tiny or "
            "imbalanced dataset earns scrutiny, not celebration."
        ),
        tools=[],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    reporter = Agent(
        role="Report Writer",
        goal="Write a clear, honest report a developer can act on.",
        backstory=(
            "A technical writer who turns an analysis into plain language: what was "
            "found, what was built, how good it is, and what to do next."
        ),
        tools=[],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    return {
        "profiler": profiler,
        "analyst": analyst,
        "framer": framer,
        "engineer": engineer,
        "trainer": trainer,
        "evaluator": evaluator,
        "reporter": reporter,
    }
