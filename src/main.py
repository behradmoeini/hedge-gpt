import sys
import argparse
from dotenv import load_dotenv
from colorama import Fore, Style, init
import questionary

from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER
from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from src.utils.ollama import ensure_ollama_and_model
from src.hedgefund.core import HedgeFundRunner, default_dates
from src.hedgefund.portfolio import create_portfolio

load_dotenv()
init(autoreset=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial cash position. Defaults to 100000.0")
    parser.add_argument("--margin-requirement", type=float, default=0.0, help="Initial margin requirement. Defaults to 0.0")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of stock ticker symbols")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD). Defaults to 3 months before end date")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today")
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    parser.add_argument("--show-agent-graph", action="store_true", help="Show the agent graph")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference")

    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]

    # Select analysts interactively
    selected_analysts = questionary.checkbox(
        "Select your AI analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction=(
            "\n\nInstructions: \n1. Press Space to select/unselect analysts.\n"
            "2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n"
        ),
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not selected_analysts:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        print(
            f"\nSelected analysts: {', '.join(Fore.GREEN + a.title().replace('_', ' ') + Style.RESET_ALL for a in selected_analysts)}\n"
        )

    # Choose LLM model
    if args.ollama:
        print(f"{Fore.CYAN}Using Ollama for local LLM inference.{Style.RESET_ALL}")
        model_name = questionary.select(
            "Select your Ollama model:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()
        if not model_name:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)
        if model_name == "-":
            model_name = questionary.text("Enter the custom model name:").ask()
            if not model_name:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)
        if not ensure_ollama_and_model(model_name):
            print(f"{Fore.RED}Cannot proceed without Ollama and the selected model.{Style.RESET_ALL}")
            sys.exit(1)
        model_provider = ModelProvider.OLLAMA.value
        print(f"\nSelected {Fore.CYAN}Ollama{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
    else:
        model_choice = questionary.select(
            "Select your LLM model:",
            choices=[questionary.Choice(display, value=(name, provider)) for display, name, provider in LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()
        if not model_choice:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)
        model_name, model_provider = model_choice
        info = get_model_info(model_name, model_provider)
        if info and info.is_custom():
            model_name = questionary.text("Enter the custom model name:").ask()
            if not model_name:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)
        print(
            f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n"
        )

    start_date, end_date = default_dates(args.start_date, args.end_date)
    portfolio = create_portfolio(args.initial_cash, args.margin_requirement, tickers)

    runner = HedgeFundRunner(selected_analysts)
    result = runner.run(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        model_name=model_name,
        model_provider=model_provider,
    )
    print_trading_output(result)


if __name__ == "__main__":
    main()
