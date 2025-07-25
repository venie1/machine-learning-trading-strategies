import pandas as pd
import numpy as np
import datetime as dt
from indicators import sma, bollinger_bands, calculate_macd
from util import get_data

class ManualStrategy:
    def author(self):
        return "pvenieris3" 

    def study_group(self):
        return "pvenieris3"  
    def __init__(self):
        pass

    def author(self):
        return "pvenieris3"  

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        # Fetch price data
        prices = get_data([symbol], pd.date_range(sd, ed))
        prices = prices[symbol]

        # Calculate indicators
        sma_20 = sma(prices, window=20)
        upper_band, lower_band, _, bbp = bollinger_bands(prices, window=20, num_std_dev=2)
        macd, signal = calculate_macd(prices, short_window=12, long_window=26, signal_window=9)

        bbp = bbp.dropna()
        sma_20 = sma_20.dropna()
        macd = macd.dropna()
        signal = signal.dropna()
        aligned_prices = prices.loc[sma_20.index.intersection(bbp.index).intersection(macd.index)]
        trades = pd.Series(0, index=aligned_prices.index)
        COMMAND = 0  # 0: Neutral, 1: Long, -1: Short

        for i in range(len(aligned_prices) - 1):
            bbp_val = bbp.iloc[i]
            macd_val = macd.iloc[i]
            signal_val = signal.iloc[i]
            price_val = aligned_prices.iloc[i]
            sma_val = sma_20.iloc[i]
            if COMMAND == 0:  # No current position
                if bbp_val < 0.2 and macd_val > signal_val and price_val < sma_val:
                    trades.iloc[i] = 1000  # Go long
                    COMMAND = 1
                elif bbp_val > 0.8 and macd_val < signal_val and price_val > sma_val:
                    trades.iloc[i] = -1000  # Go short
                    COMMAND = -1

            elif COMMAND == 1:
                if bbp_val > 0.8 or macd_val < signal_val:
                    trades.iloc[i] = -1000  # Close long
                    COMMAND = 0

            elif COMMAND == -1:
                if bbp_val < 0.2 or macd_val > signal_val:
                    trades.iloc[i] = 1000  # Close short
                    COMMAND = 0

        if COMMAND == 1:
            trades.iloc[-1] = -1000  # Close long position
        elif COMMAND == -1:
            trades.iloc[-1] = 1000  # Close short position

        return trades.to_frame(name=symbol)

if __name__ == "__main__":
    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    print("Final Trades DataFrame:")
    print(df_trades)
