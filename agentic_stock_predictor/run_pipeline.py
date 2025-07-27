from data.fetch import fetch_daily_data
from data.format import enrich_with_indicators, save_as_parquet


def main() -> None:
    df = fetch_daily_data("AAPL")
    enriched = enrich_with_indicators(df)
    save_as_parquet(enriched)


if __name__ == "__main__":
    main()
