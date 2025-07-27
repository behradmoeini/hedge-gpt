from unittest import mock
from agentic_stock_predictor.agents.llm import LLMInvestorAgent
from agentic_stock_predictor.predict.features import FeatureSet


def test_llm_agent_mock():
    fs = FeatureSet(date=None, sma_5=1, sma_20=0, rsi=60, sentiment=0)
    agent = LLMInvestorAgent()
    with mock.patch('openai.ChatCompletion.create') as create:
        create.return_value = {'choices':[{'message':{'content':'up'}}]}
        assert agent.predict(fs) == 1.0
