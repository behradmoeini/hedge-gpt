import pandas as pd
import ta
from predict.features import FeatureSet
from .base import PredictionAgent

class TechnicalAgent(PredictionAgent):
    """Simple technical analysis based on moving averages."""

    def predict(self, features: FeatureSet) -> float:
        sma_fast = features.sma_5
        sma_slow = features.sma_20
        return 1.0 if sma_fast > sma_slow else -1.0
