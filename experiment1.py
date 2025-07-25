import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data
from marketsimcode import compute_portvals
from StrategyLearner import StrategyLearner
from ManualStrategy import ManualStrategy


def author(self):
    return "pvenieris3"  


def study_group(self):
    return "pvenieris3"  
def compute_benchmark_portvals(symbol, start_date, end_date, start_val=100000, commission=9.95, impact=0.005):

    dates = pd.date_range(start_date, end_date)
    prices = get_data([symbol], dates)
    trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
    trades.loc[prices.index[0]] = 1000  # Buy 1000 shares at the start of the period
    trades.loc[prices.index[-1]] = -1000  # Sell 1000 shares at the end of the period
    benchmark_portfolio = compute_portvals(trades, start_val=start_val, commission=commission, impact=impact)
    return benchmark_portfolio


def plot_portfolio_values(manual_portfolio, strategy_portfolio, benchmark_portfolio, title, filename):

    manual_portfolio_norm = manual_portfolio / manual_portfolio.iloc[0]
    strategy_portfolio_norm = strategy_portfolio / strategy_portfolio.iloc[0]
    benchmark_portfolio_norm = benchmark_portfolio / benchmark_portfolio.iloc[0]
    plt.figure(figsize=(10, 6))
    plt.plot(manual_portfolio_norm, label="Manual Strategy", color='black')
    plt.plot(strategy_portfolio_norm, label="Strategy Learner", color='green')
    plt.plot(benchmark_portfolio_norm, label="Benchmark", color='blue')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.savefig(filename)
    plt.show()


def experiment1():
    symbol = "JPM"
    in_sample_start = dt.datetime(2008, 1, 1)
    in_sample_end = dt.datetime(2009, 12, 31)
    out_of_sample_start = dt.datetime(2010, 1, 1)
    out_of_sample_end = dt.datetime(2011, 12, 31)
    start_val = 100000
    commission = 9.95
    impact = 0.005
    manual_strategy = ManualStrategy()
    manual_trades_in = manual_strategy.testPolicy(symbol=symbol, sd=in_sample_start, ed=in_sample_end)
    manual_portfolio_in = compute_portvals(manual_trades_in, start_val=start_val, commission=commission, impact=impact)
    manual_trades_out = manual_strategy.testPolicy(symbol=symbol, sd=out_of_sample_start, ed=out_of_sample_end)
    manual_portfolio_out = compute_portvals(manual_trades_out, start_val=start_val, commission=commission,
                                            impact=impact)
    strategy_learner = StrategyLearner(verbose=False, impact=impact)
    strategy_learner.add_evidence(symbol=symbol, sd=in_sample_start, ed=in_sample_end, sv=start_val)
    strategy_learner_trades_in = strategy_learner.testPolicy(symbol=symbol, sd=in_sample_start, ed=in_sample_end)
    strategy_learner_portfolio_in = compute_portvals(strategy_learner_trades_in, start_val=start_val,
                                                     commission=commission, impact=impact)
    strategy_learner_trades_out = strategy_learner.testPolicy(symbol=symbol, sd=out_of_sample_start,
                                                              ed=out_of_sample_end)
    strategy_learner_portfolio_out = compute_portvals(strategy_learner_trades_out, start_val=start_val,
                                                      commission=commission, impact=impact)
    benchmark_portfolio_in = compute_benchmark_portvals(symbol, in_sample_start, in_sample_end, start_val=start_val,
                                                        commission=commission, impact=impact)
    benchmark_portfolio_out = compute_benchmark_portvals(symbol, out_of_sample_start, out_of_sample_end,
                                                         start_val=start_val, commission=commission, impact=impact)
    plot_portfolio_values(
        manual_portfolio_in,
        strategy_learner_portfolio_in,
        benchmark_portfolio_in,
        title="In-Sample Portfolio Comparison",
        filename="in_sample_comparison.png"
    )
    plot_portfolio_values(
        manual_portfolio_out,
        strategy_learner_portfolio_out,
        benchmark_portfolio_out,
        title="Out-of-Sample Portfolio Comparison",
        filename="out_of_sample_comparison.png"
    )


if __name__ == "__main__":
    experiment1()
