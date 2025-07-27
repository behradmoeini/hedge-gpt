import logging
from datetime import datetime

import pandas as pd
from predict.engine import run_prediction_for_date


def backtest(path: str = "data/processed.parquet", horizon: int = 5) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.reset_index()

    logging.basicConfig(level=logging.INFO)
    results = []
    for i in range(len(df) - horizon):
        today = df.loc[i]
        future_price = df.loc[i + horizon, "adj_close"]
        current_price = today["adj_close"]
        actual_return = (future_price - current_price) / current_price

        assert not today[["sma_5", "sma_20", "rsi"]].isna().any(), "Feature leakage detected"

        preds = run_prediction_for_date(today["index"], today)
        for name, direction in preds.items():
            guessed_return = direction * abs(actual_return)
            results.append(
                {
                    "date": today["index"],
                    "agent": name,
                    "predicted_direction": direction,
                    "actual_return": actual_return,
                    "simulated_return": guessed_return,
                }
            )
    result_df = pd.DataFrame(results)
    outdir = "backtest/results"
    import os
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, datetime.now().strftime("%Y%m%d") + ".csv")
    result_df.to_csv(outpath, index=False)
    logging.info("Saved results to %s", outpath)
    return result_df
