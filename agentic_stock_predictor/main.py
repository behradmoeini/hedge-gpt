from run_pipeline import main as pipeline_main
from run_backtest import main as backtest_main


def main() -> None:
    pipeline_main()
    backtest_main()


if __name__ == "__main__":
    main()
