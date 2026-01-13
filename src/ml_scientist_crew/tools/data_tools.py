import pandas as pd

def analyze_dataset(path: str) -> str:
    df = pd.read_csv(path)
    return f"""
    Shape: {df.shape}
    Columns: {list(df.columns)}
    Missing (%):
    {(df.isnull().mean()*100).round(2)}
    """
