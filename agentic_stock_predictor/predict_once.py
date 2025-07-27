from data.fetch import fetch_daily_data
from data.format import enrich_with_indicators
from predict.engine import run_prediction_for_date


def main() -> None:
    df = fetch_daily_data("AAPL")
    enriched = enrich_with_indicators(df)
    last = enriched.iloc[-1]
    preds = run_prediction_for_date(last.name, last)
    print(preds)


if __name__ == "__main__":
    main()
