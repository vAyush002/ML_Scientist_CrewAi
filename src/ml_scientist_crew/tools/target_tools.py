import pandas as pd

def detect_target_column(path: str) -> str:
    df = pd.read_csv(path)
    if "Survived" in df.columns:
        return "Detected target column: Survived"
    return "Target column not detected"
