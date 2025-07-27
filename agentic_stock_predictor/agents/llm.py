import openai
from pathlib import Path

from config import OPENAI_API_KEY
from predict.features import FeatureSet
from .base import PredictionAgent

class LLMInvestorAgent(PredictionAgent):
    """LLM-based investor agent calling OpenAI API."""

    def __init__(self, temperature: float = 0.0, seed: int | None = None, max_tokens: int = 32):
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens

    def _build_prompt(self, features: FeatureSet) -> str:
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "llm_prompt.txt"
        template = prompt_path.read_text()
        return template.format(**features.as_dict())

    def _parse_response(self, text: str) -> float:
        text = text.lower()
        if "up" in text:
            return 1.0
        if "down" in text:
            return -1.0
        return 0.0

    def predict(self, features: FeatureSet) -> float:
        prompt = self._build_prompt(features)
        openai.api_key = OPENAI_API_KEY
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            text = response["choices"][0]["message"]["content"]
        except Exception:
            text = "down"
        return self._parse_response(text)
