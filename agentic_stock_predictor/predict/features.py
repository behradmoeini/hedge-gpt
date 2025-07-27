from dataclasses import dataclass
from datetime import datetime

@dataclass
class FeatureSet:
    date: datetime
    sma_5: float
    sma_20: float
    rsi: float
    sentiment: float

    def as_dict(self) -> dict:
        return {
            "sma_5": self.sma_5,
            "sma_20": self.sma_20,
            "rsi": self.rsi,
            "sentiment": self.sentiment,
        }
