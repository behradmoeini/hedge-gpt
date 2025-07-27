from agentic_stock_predictor.predict.features import FeatureSet
from agentic_stock_predictor.agents.technical import TechnicalAgent
from agentic_stock_predictor.agents.sentiment import SentimentAgent


def test_technical_agent():
    fs = FeatureSet(date=None, sma_5=2, sma_20=1, rsi=50, sentiment=0)
    agent = TechnicalAgent()
    assert agent.predict(fs) == 1.0


def test_sentiment_agent():
    fs = FeatureSet(date=None, sma_5=0, sma_20=0, rsi=0, sentiment=-0.1)
    agent = SentimentAgent()
    assert agent.predict(fs) == -1.0
