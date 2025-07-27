from backtest.simulate import backtest


def main() -> None:
    results = backtest()
    print(results.groupby("agent")["simulated_return"].mean())


if __name__ == "__main__":
    main()
