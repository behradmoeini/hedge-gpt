import numpy as np
import pandas as pd
import ta


def enrich_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df["sma_5"] = ta.trend.sma_indicator(df["adj_close"], window=5).shift(1)
    df["sma_20"] = ta.trend.sma_indicator(df["adj_close"], window=20).shift(1)
    df["rsi"] = ta.momentum.rsi(df["adj_close"], window=14).shift(1)
    df["sentiment"] = np.random.randn(len(df))
    df = df.dropna()
    return df


def save_as_parquet(df: pd.DataFrame, path: str = "data/processed.parquet") -> None:
    df.to_parquet(path)
