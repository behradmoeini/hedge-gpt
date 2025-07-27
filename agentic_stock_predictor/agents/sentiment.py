from predict.features import FeatureSet
from .base import PredictionAgent

class SentimentAgent(PredictionAgent):
    """Placeholder sentiment agent using fake data."""

    def predict(self, features: FeatureSet) -> float:
        sentiment = features.sentiment
        return 1.0 if sentiment > 0 else -1.0
