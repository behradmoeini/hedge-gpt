import pandas as pd
from agentic_stock_predictor.backtest.simulate import backtest


def test_backtest_basic(tmp_path):
    data = pd.DataFrame({
        'adj_close':[1,2,3,4,5,6],
        'sma_5':[None,1,1,2,3,4],
        'sma_20':[None,1,1,1,2,3],
        'rsi':[None,30,40,50,60,70],
        'sentiment':[0,0,0,0,0,0],
    }, index=pd.date_range('2021-01-01', periods=6))
    data = data.dropna()
    path = tmp_path/'data.parquet'
    data.to_parquet(path)
    df = backtest(path=str(path), horizon=1)
    assert not df.empty
