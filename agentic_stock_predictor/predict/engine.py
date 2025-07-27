from datetime import datetime

from agents.technical import TechnicalAgent
from agents.sentiment import SentimentAgent
from agents.llm import LLMInvestorAgent
from .features import FeatureSet


def run_prediction_for_date(date: datetime, row) -> dict:
    feature_set = FeatureSet(
        date=date,
        sma_5=row.get("sma_5", 0.0),
        sma_20=row.get("sma_20", 0.0),
        rsi=row.get("rsi", 0.0),
        sentiment=row.get("sentiment", 0.0),
    )
    agents = [TechnicalAgent(), SentimentAgent(), LLMInvestorAgent()]
    return {a.__class__.__name__: a.predict(feature_set) for a in agents}
