import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from StrategyLearner import StrategyLearner  
from marketsimcode import compute_portvals  
from util import get_data


def author(self):
    return "pvenieris3"  


def study_group(self):
    return "pvenieris3"  
def experiment2():
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_val = 100000
    commission = 0.0
    impacts = [0.001, 0.005, 0.01,0.05,0.1]  
    cumulative_returns = []
    num_trades = []
    for impact in impacts:
        learner = StrategyLearner(impact=impact, commission=commission)
        learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date, sv=start_val)
        trades = learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_val)
        portvals = compute_portvals(trades, start_val=start_val, commission=commission, impact=impact)
        portvals = portvals / portvals.iloc[0]  # Normalize to 1.0
        cumulative_return = portvals.iloc[-1] - 1  # Total return
        cumulative_returns.append(cumulative_return)
        num_trades.append((trades != 0).sum().sum())  # Count non-zero trades

    plt.figure(figsize=(10, 5))
    plt.plot(impacts, cumulative_returns, marker='o', label="Cumulative Return")
    plt.xlabel("Impact")
    plt.ylabel("Cumulative Return")
    plt.title("Cumulative Return vs Impact")
    plt.grid()
    plt.legend()
    plt.savefig("cumulative_return_vs_impact.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(impacts, num_trades, marker='o', label="Number of Trades", color='orange')
    plt.xlabel("Impact")
    plt.ylabel("Number of Trades")
    plt.title("Number of Trades vs Impact")
    plt.grid()
    plt.legend()
    plt.savefig("num_trades_vs_impact.png")
    plt.show()

if __name__ == "__main__":
    experiment2()
