import pandas as pd
from unittest import mock

import agentic_stock_predictor.data.fetch as fetch_mod


def test_fetch_daily_data():
    with mock.patch('agentic_stock_predictor.data.fetch.TimeSeries') as ts_mock:
        ts_instance = ts_mock.return_value
        ts_instance.get_daily_adjusted.return_value = (
            pd.DataFrame({'1. open':[1],'2. high':[1],'3. low':[1],'4. close':[1],'5. adjusted close':[1],'6. volume':[1]}, index=['2021-01-01']),
            None
        )
        df = fetch_mod.fetch_daily_data('TEST')
        assert 'adj_close' in df.columns
        assert df.loc['2021-01-01','adj_close'] == 1
