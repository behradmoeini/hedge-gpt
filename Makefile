install:
python3 -m venv venv && . venv/bin/activate && pip install -e . && pip install -r agentic_stock_predictor/requirements.txt

test:
pytest -q
