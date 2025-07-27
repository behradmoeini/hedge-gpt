# Agentic Stock Predictor

This project demonstrates a simple modular system for stock prediction using multiple reasoning agents. It fetches historical data, generates technical indicators and sentiment signals, and evaluates agents with a backtesting engine.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -r agentic_stock_predictor/requirements.txt
```

Create a `.env` file in `agentic_stock_predictor/`:

```ini
OPENAI_API_KEY=sk-...
ALPHA_VANTAGE_API_KEY=demo
```

## Usage

Run the pipeline and backtest:

```bash
python agentic_stock_predictor/main.py
```

Or use the CLI:

```bash
python -m agentic_stock_predictor.cli fetch AAPL
python -m agentic_stock_predictor.cli predict AAPL
python -m agentic_stock_predictor.cli backtest --horizon 5
```

### Adding a new agent

Create a new class inheriting from `PredictionAgent` and implement `predict(self, features: FeatureSet)`. Register it in `predict/engine.py`.

### Interpreting results

Backtesting saves detailed logs under `backtest/results/` with the average simulated return per agent printed to stdout.
