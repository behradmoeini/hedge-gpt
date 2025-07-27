from abc import ABC, abstractmethod

from predict.features import FeatureSet

class PredictionAgent(ABC):
    @abstractmethod
    def predict(self, features: FeatureSet) -> float:
        """Return +1 for bullish, -1 for bearish."""
        pass
