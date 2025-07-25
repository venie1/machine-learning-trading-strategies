import pandas as pd
from util import get_data

def author():
    return "pvenieris3"

def study_group():
    return "pvenieris3"

def compute_portvals(
    trades,
    start_val=100000,
    commission=0.0,
    impact=0.0
):

    symbols = trades.columns.tolist()
    start_date = trades.index.min()
    end_date = trades.index.max()
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices['Cash'] = 1.0
    holdings = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    cash = pd.Series(start_val, index=prices.index)
    for date in trades.index:
        for symbol in symbols:
            trade = trades.loc[date, symbol]
            if trade != 0:
                price = prices.loc[date, symbol]
                if trade > 0:
                    impacted_price = price * (1 + impact)
                    holdings.loc[date:, symbol] += trade
                    cash.loc[date:] -= (impacted_price * trade) + commission
                elif trade < 0:
                    impacted_price = price * (1 - impact)
                    holdings.loc[date:, symbol] += trade
                    cash.loc[date:] -= (impacted_price * trade) + commission
    portvals = (holdings * prices).sum(axis=1) + cash
    return pd.DataFrame(portvals, columns=['Portfolio'])

def test_code():
    pass

if __name__ == "__main__":
    test_code()
