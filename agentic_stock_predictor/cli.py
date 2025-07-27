import argparse
from data.fetch import fetch_daily_data
from data.format import enrich_with_indicators, save_as_parquet
from predict.engine import run_prediction_for_date
from backtest.simulate import backtest


def cmd_fetch(args: argparse.Namespace) -> None:
    df = fetch_daily_data(args.symbol)
    enriched = enrich_with_indicators(df)
    save_as_parquet(enriched)


def cmd_predict(args: argparse.Namespace) -> None:
    df = fetch_daily_data(args.symbol)
    enriched = enrich_with_indicators(df)
    latest = enriched.iloc[-1]
    preds = run_prediction_for_date(latest.name, latest)
    print(preds)


def cmd_backtest(args: argparse.Namespace) -> None:
    results = backtest(path=args.path, horizon=args.horizon)
    print(results.groupby("agent")["simulated_return"].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Stock Predictor CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_p = sub.add_parser("fetch", help="Fetch and process data")
    fetch_p.add_argument("symbol", default="AAPL", nargs="?")
    fetch_p.set_defaults(func=cmd_fetch)

    pred_p = sub.add_parser("predict", help="Run prediction on latest data")
    pred_p.add_argument("symbol", default="AAPL", nargs="?")
    pred_p.set_defaults(func=cmd_predict)

    bt_p = sub.add_parser("backtest", help="Run backtest")
    bt_p.add_argument("--path", default="data/processed.parquet")
    bt_p.add_argument("--horizon", type=int, default=5)
    bt_p.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
