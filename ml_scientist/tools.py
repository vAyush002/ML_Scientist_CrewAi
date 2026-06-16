"""Tools = the agents' powers.

Each capability exists twice on purpose:
  1. A plain Python function that does the real computation on the dataset.
  2. A thin CrewAI ``BaseTool`` wrapper around it, so an agent can call it.

The pipeline calls the plain functions directly to guarantee the *numbers*
are real and reproducible, then hands those facts to the agent for judgement.
The agent also holds the tool, so it can re-check or drill in if it wants.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR

warnings.filterwarnings("ignore")

TARGET_HINTS = {"target", "label", "y", "class", "outcome", "result", "churn", "price"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def _json_safe(obj: Any) -> Any:
    """Make numpy/pandas scalars JSON serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(round(float(obj), 4))
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# 1. Profiling
# --------------------------------------------------------------------------- #
def profile_dataset(path: str) -> dict:
    df = load_dataset(path)
    n_rows, n_cols = df.shape
    columns = []
    for col in df.columns:
        s = df[col]
        columns.append(
            {
                "name": str(col),
                "dtype": str(s.dtype),
                "missing_pct": round(float(s.isna().mean() * 100), 2),
                "n_unique": int(s.nunique(dropna=True)),
                "sample": [_json_safe(v) for v in s.dropna().unique()[:3].tolist()],
            }
        )
    numeric = df.select_dtypes("number")
    describe = (
        json.loads(numeric.describe().round(3).to_json()) if not numeric.empty else {}
    )
    return _json_safe(
        {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "duplicate_rows": int(df.duplicated().sum()),
            "columns": columns,
            "numeric_summary": describe,
        }
    )


# --------------------------------------------------------------------------- #
# 2. Exploratory analysis
# --------------------------------------------------------------------------- #
def analyze_dataset(path: str, target: str | None = None) -> dict:
    df = load_dataset(path)
    out: dict = {}

    high_missing = [
        c for c in df.columns if df[c].isna().mean() > 0.4
    ]
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    high_card = [
        c
        for c in df.columns
        if df[c].dtype == object and df[c].nunique(dropna=True) > 0.5 * len(df)
    ]
    out["high_missing_cols"] = high_missing
    out["constant_cols"] = constant
    out["high_cardinality_cols"] = high_card

    numeric = df.select_dtypes("number")
    if numeric.shape[1] >= 2:
        corr = numeric.corr().abs()
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
        pairs.sort(key=lambda t: t[2], reverse=True)
        out["top_correlations"] = [
            {"a": a, "b": b, "abs_corr": round(c, 3)} for a, b, c in pairs[:8]
        ]
        # crude leakage signal: feature almost perfectly correlated with target
        if target and target in numeric.columns:
            tcorr = corr[target].drop(labels=[target], errors="ignore")
            leaks = tcorr[tcorr > 0.98].index.tolist()
            out["possible_leakage"] = leaks

    if target and target in df.columns:
        vc = df[target].value_counts(normalize=True).head(10)
        out["target_distribution"] = {str(k): round(float(v), 3) for k, v in vc.items()}

    return _json_safe(out)


# --------------------------------------------------------------------------- #
# 3. Target / task suggestion (heuristic; the agent makes the final call)
# --------------------------------------------------------------------------- #
def suggest_target_and_task(path: str, target: str | None = None) -> dict:
    df = load_dataset(path)
    if not target:
        named = [c for c in df.columns if str(c).strip().lower() in TARGET_HINTS]
        target = named[0] if named else df.columns[-1]

    s = df[target]
    n_unique = s.nunique(dropna=True)
    if s.dtype == object or s.dtype == bool:
        task = "classification"
    elif n_unique <= max(20, int(0.05 * len(df))) and pd.api.types.is_integer_dtype(s):
        task = "classification"
    else:
        task = "regression"
    return {"target": str(target), "task_type": task, "n_unique_target": int(n_unique)}


# --------------------------------------------------------------------------- #
# 4. Feature-engineering plan (no fitting; just the strategy)
# --------------------------------------------------------------------------- #
def engineering_plan(path: str, target: str) -> dict:
    df = load_dataset(path)
    feats = [c for c in df.columns if c != target]
    numeric = df[feats].select_dtypes("number").columns.tolist()
    categorical = [c for c in feats if c not in numeric]
    return _json_safe(
        {
            "n_features": len(feats),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "numeric_strategy": "median impute + standard scale",
            "categorical_strategy": "most-frequent impute + one-hot (ignore unseen)",
            "dropped_target": target,
        }
    )


# --------------------------------------------------------------------------- #
# 5. Train + evaluate (the heavy lifting, fully deterministic)
# --------------------------------------------------------------------------- #
def _build_preprocessor(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = X.select_dtypes("number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False, max_categories=20))]),
             categorical),
        ],
        remainder="drop",
    )


def available_algorithms(task_type: str) -> list[str]:
    """Algorithm keys the user can train or tune for a given task."""
    if task_type == "classification":
        algos = ["logistic_regression", "decision_tree", "random_forest", "svm"]
    else:
        algos = ["linear_regression", "decision_tree", "random_forest", "svr"]
    try:
        import xgboost  # noqa: F401
        algos.append("xgboost")
    except Exception:
        pass
    return algos


def _make_estimator(algorithm: str, task_type: str, random_state: int = 42):
    """A single estimator for the given algorithm + task."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.svm import SVC, SVR

    clf = task_type == "classification"
    table = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "linear_regression": LinearRegression(),
        "decision_tree": (DecisionTreeClassifier(random_state=random_state) if clf
                          else DecisionTreeRegressor(random_state=random_state)),
        "random_forest": (RandomForestClassifier(n_estimators=200, random_state=random_state) if clf
                          else RandomForestRegressor(n_estimators=200, random_state=random_state)),
        "svm": SVC(probability=False),
        "svr": SVR(),
    }
    if algorithm == "xgboost":
        if clf:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=random_state)
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=200, random_state=random_state)
    return table[algorithm]


def _candidate_algorithms(task_type: str, n_rows: int) -> list[str]:
    """Base slate to compare. SVM is O(n^2)+, so skip it on large data."""
    algos = available_algorithms(task_type)
    if n_rows > 5000:
        algos = [a for a in algos if a not in ("svm", "svr")]
    return algos


def _split(X, y, task_type, random_state=42):
    from sklearn.model_selection import train_test_split

    stratify = y if task_type == "classification" and y.nunique() > 1 else None
    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=stratify)


def _n_splits(y_tr, task_type) -> int:
    if task_type == "classification":
        return max(2, min(5, int(y_tr.value_counts().min())))
    return 5


def _holdout(y_te, pred, task_type) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                                 r2_score, root_mean_squared_error)
    if task_type == "classification":
        return {"accuracy": round(float(accuracy_score(y_te, pred)), 4),
                "f1_weighted": round(float(f1_score(y_te, pred, average="weighted")), 4)}
    return {"r2": round(float(r2_score(y_te, pred)), 4),
            "rmse": round(float(root_mean_squared_error(y_te, pred)), 4),
            "mae": round(float(mean_absolute_error(y_te, pred)), 4)}


def run_modeling(path: str, target: str, task_type: str, random_state: int = 42) -> dict:
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    df = load_dataset(path).dropna(subset=[target])
    X, y = df.drop(columns=[target]), df[target]
    pre = _build_preprocessor(X)
    scoring = "f1_weighted" if task_type == "classification" else "r2"

    X_tr, X_te, y_tr, y_te = _split(X, y, task_type, random_state)
    n_splits = _n_splits(y_tr, task_type)

    results, best_name, best_score, best_pipe = {}, None, -np.inf, None
    for name in _candidate_algorithms(task_type, len(df)):
        pipe = Pipeline([("pre", pre), ("model", _make_estimator(name, task_type, random_state))])
        try:
            cv = cross_val_score(pipe, X_tr, y_tr, cv=n_splits, scoring=scoring)
        except Exception as e:
            results[name] = {"error": str(e)[:160]}
            continue
        results[name] = {"cv_metric": scoring, "cv_mean": round(float(cv.mean()), 4),
                         "cv_std": round(float(cv.std()), 4)}
        if cv.mean() > best_score:
            best_name, best_score, best_pipe = name, float(cv.mean()), pipe

    holdout = {}
    if best_pipe is not None:
        best_pipe.fit(X_tr, y_tr)
        holdout = _holdout(y_te, best_pipe.predict(X_te), task_type)
        joblib.dump(best_pipe, ARTIFACT_DIR / "best_model.joblib")

    return _json_safe({
        "task_type": task_type, "target": target, "n_features": X.shape[1],
        "cv_folds": n_splits, "models": results, "best_model": best_name,
        "best_cv_score": round(float(best_score), 4) if best_pipe is not None else None,
        "holdout_metrics": holdout,
        "saved_model_path": str(ARTIFACT_DIR / "best_model.joblib") if best_pipe is not None else None,
    })


# --------------------------------------------------------------------------- #
# 6. Hyperparameter tuning ("fine-tuning") for a chosen algorithm
# --------------------------------------------------------------------------- #
def _param_grid(algorithm: str, task_type: str) -> dict:
    """Search space, keyed for use inside a Pipeline (model__<param>)."""
    grids = {
        "logistic_regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
        "linear_regression": {},  # no hyperparameters; handled specially below
        "decision_tree": {
            "model__max_depth": [None, 5, 10, 20, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "random_forest": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 10, 20, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2"],
        },
        "svm": {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"]},
        "svr": {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"]},
        "xgboost": {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [3, 6, 9],
            "model__learning_rate": [0.03, 0.1, 0.3],
            "model__subsample": [0.8, 1.0],
        },
    }
    if task_type == "classification" and algorithm == "decision_tree":
        grids["decision_tree"]["model__criterion"] = ["gini", "entropy"]
    return grids.get(algorithm, {})


def tune_model(path: str, target: str, task_type: str, algorithm: str,
               search_type: str = "random", n_iter: int = 20, random_state: int = 42) -> dict:
    """Hyperparameter search for one algorithm. Returns best params + metrics
    and saves the tuned pipeline. Deterministic — no LLM, no API key."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
    from sklearn.pipeline import Pipeline

    df = load_dataset(path).dropna(subset=[target])
    X, y = df.drop(columns=[target]), df[target]
    pre = _build_preprocessor(X)
    scoring = "f1_weighted" if task_type == "classification" else "r2"
    X_tr, X_te, y_tr, y_te = _split(X, y, task_type, random_state)
    cv = _n_splits(y_tr, task_type)

    # Plain linear regression has nothing to tune -> use Ridge(alpha) instead.
    if task_type == "regression" and algorithm == "linear_regression":
        estimator, grid = Ridge(), {"model__alpha": [0.01, 0.1, 1, 10, 100]}
        tuned_label = "ridge_regression"
    else:
        estimator, grid = _make_estimator(algorithm, task_type, random_state), _param_grid(algorithm, task_type)
        tuned_label = algorithm

    pipe = Pipeline([("pre", pre), ("model", estimator)])

    if not grid:
        return {"algorithm": algorithm, "error": "This algorithm has no hyperparameters to tune.",
                "task_type": task_type, "target": target}

    if search_type == "grid":
        search = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1)
    else:
        search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, cv=cv, scoring=scoring,
                                    n_jobs=-1, random_state=random_state)
    search.fit(X_tr, y_tr)

    best = search.best_estimator_
    holdout = _holdout(y_te, best.predict(X_te), task_type)

    # baseline (untuned) for comparison
    base = Pipeline([("pre", pre), ("model", _make_estimator(
        "linear_regression" if tuned_label == "ridge_regression" else algorithm, task_type, random_state))])
    try:
        base_cv = float(cross_val_score(base, X_tr, y_tr, cv=cv, scoring=scoring).mean())
    except Exception:
        base_cv = None

    joblib.dump(best, ARTIFACT_DIR / "tuned_model.joblib")
    clean_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return _json_safe({
        "algorithm": tuned_label, "task_type": task_type, "target": target,
        "search_type": search_type, "cv_metric": scoring, "cv_folds": cv,
        "candidates_evaluated": len(search.cv_results_["params"]),
        "best_params": clean_params,
        "tuned_cv_score": round(float(search.best_score_), 4),
        "baseline_cv_score": round(base_cv, 4) if base_cv is not None else None,
        "improvement": round(float(search.best_score_) - base_cv, 4) if base_cv is not None else None,
        "holdout_metrics": holdout,
        "saved_model_path": str(ARTIFACT_DIR / "tuned_model.joblib"),
    })


# --------------------------------------------------------------------------- #
# CrewAI tool wrappers
# --------------------------------------------------------------------------- #
def _make_tools():
    """Built lazily so the module imports even without crewai installed."""
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class _PathArgs(BaseModel):
        dataset_path: str = Field(..., description="Path to the dataset file")

    class _PathTargetArgs(BaseModel):
        dataset_path: str = Field(..., description="Path to the dataset file")
        target: str = Field(..., description="Name of the target column")

    class _ModelArgs(BaseModel):
        dataset_path: str = Field(..., description="Path to the dataset file")
        target: str = Field(..., description="Target column name")
        task_type: str = Field(..., description="'classification' or 'regression'")

    class ProfileTool(BaseTool):
        name: str = "profile_dataset"
        description: str = "Return shape, column dtypes, missing %, uniqueness and numeric summary."
        args_schema: type[BaseModel] = _PathArgs

        def _run(self, dataset_path: str) -> str:
            return json.dumps(profile_dataset(dataset_path))

    class AnalyzeTool(BaseTool):
        name: str = "analyze_dataset"
        description: str = "Return correlations, high-missing/constant/high-cardinality columns and leakage signals."
        args_schema: type[BaseModel] = _PathArgs

        def _run(self, dataset_path: str) -> str:
            return json.dumps(analyze_dataset(dataset_path))

    class EngineerTool(BaseTool):
        name: str = "engineering_plan"
        description: str = "Return the preprocessing plan: which columns are numeric vs categorical and the strategy for each."
        args_schema: type[BaseModel] = _PathTargetArgs

        def _run(self, dataset_path: str, target: str) -> str:
            return json.dumps(engineering_plan(dataset_path, target))

    class ModelTool(BaseTool):
        name: str = "run_modeling"
        description: str = "Train candidate models with cross-validation and return CV + holdout metrics and the best model."
        args_schema: type[BaseModel] = _ModelArgs

        def _run(self, dataset_path: str, target: str, task_type: str) -> str:
            return json.dumps(run_modeling(dataset_path, target, task_type))

    return {
        "profile": ProfileTool(),
        "analyze": AnalyzeTool(),
        "engineer": EngineerTool(),
        "model": ModelTool(),
    }
