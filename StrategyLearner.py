import datetime as dt
import numpy as np
import pandas as pd
from BagLearner import BagLearner
from RTLearner import RTLearner
from indicators import sma, bollinger_bands, calculate_macd, calculate_momentum, calculate_stochastic_oscillator
import util as ut
class StrategyLearner:
    def author(self):
        return "pvenieris3"  

    def study_group(self):
        return "pvenieris3"  

    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = None

    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices = ut.get_data([symbol], dates)
        prices = prices[symbol]

        sma_20 = sma(prices, window=20)
        upper_band, lower_band, _, bbp = bollinger_bands(prices, window=20, num_std_dev=2)
        macd, signal = calculate_macd(prices, short_window=12, long_window=26, signal_window=9)
        indicators = pd.DataFrame({
            'SMA_20': sma_20,
            'BollingerUpper': upper_band,
            'BollingerLower': lower_band,
            'MACD': macd,
            'MACD_Signal': signal
        })

        indicators.dropna(inplace=True)

        Y_train = self._generate_labels(prices.loc[indicators.index])
        if len(Y_train) != len(indicators):
            indicators = indicators.iloc[:Y_train.shape[0]]
        Y_train = pd.Series(Y_train, index=indicators.index)

        if indicators.shape[0] != Y_train.shape[0]:
            raise ValueError(
                f"Mismatch in lengths after alignment: indicators has {indicators.shape[0]} rows, Y_train has {Y_train.shape[0]} rows.")
        self.learner = BagLearner(
            learner=RTLearner,
            kwargs={"leaf_size": 5},
            bags=20,
            boost=False,
            verbose=self.verbose,
        )
        self.learner.add_evidence(indicators.values, Y_train)

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices = ut.get_data([symbol], dates)
        prices = prices[symbol]
        sma_20 = sma(prices, window=20)
        upper_band, lower_band, _, bbp = bollinger_bands(prices, window=20, num_std_dev=2)
        macd, signal = calculate_macd(prices, short_window=12, long_window=26, signal_window=9)
        indicators = pd.DataFrame({
            'SMA_20': sma_20,
            'BollingerUpper': upper_band,
            'BollingerLower': lower_band,
            'MACD': macd,
            'MACD_Signal': signal
        })

        indicators = indicators.dropna()
        X_test = indicators.values
        predictions = self.learner.query(X_test)
        trades = self._generate_trades(predictions, prices.loc[indicators.index])

        return trades

    def _generate_labels(self, prices):
        lookahead = 5
        price_diff = prices.shift(-lookahead) - prices
        impact_adjusted = self.impact * prices
        price_diff -= impact_adjusted

        labels = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
        return labels[:-lookahead]

    def _generate_trades(self, predictions, prices):
        trades = pd.Series(data=0, index=prices.index)
        holdings = 0
        for i in range(len(predictions)):
            if predictions[i] == 1 and holdings <= 0:
                trades.iloc[i] = 1000 - holdings
                holdings = 1000
            elif predictions[i] == -1 and holdings >= 0:
                trades.iloc[i] = -1000 - holdings
                holdings = -1000

        return trades.to_frame(name=prices.name)
