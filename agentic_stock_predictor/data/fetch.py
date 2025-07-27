from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from config import ALPHA_VANTAGE_API_KEY


def fetch_daily_data(symbol: str = "AAPL") -> pd.DataFrame:
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
    data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize="full")
    data = data.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volume",
        }
    )
    data.index = pd.to_datetime(data.index)
    return data
