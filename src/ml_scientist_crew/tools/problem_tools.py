import pandas as pd

def detect_problem_type(path: str) -> str:
    df = pd.read_csv(path)
    target = df.columns[-1]
    if df[target].nunique() < 20:
        return "Problem Type: Classification"
    return "Problem Type: Regression"
