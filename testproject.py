import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from experiment1 import experiment1  # Imported Experiment 1
from experiment2 import experiment2  # Imported Experiment 2
from util import get_data
from marketsimcode import compute_portvals


def author(self):
    return "pvenieris3"  


def study_group(self):
    return "pvenieris3"  
def calculate_metrics(portvals):
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals.squeeze()

    daily_returns = portvals.pct_change().dropna()
    cum_return = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    return float(cum_return), float(std_daily_return), float(avg_daily_return)



def generate_chart(portvals_manual, portvals_benchmark, trades, title, filename):
    plt.figure(figsize=(12, 6))

    plt.plot(portvals_manual, label="Manual Strategy", color="red")
    plt.plot(portvals_benchmark, label="Benchmark", color="purple")
    long_trades = trades[trades > 0].index
    short_trades = trades[trades < 0].index
    for date in long_trades:
        plt.axvline(date, color="blue", linestyle="--", alpha=0.7, linewidth=1)
    for date in short_trades:
        plt.axvline(date, color="black", linestyle="--", alpha=0.7, linewidth=1)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()


def try_manual_strategy(symbol, start_date, end_date, sv, is_insample):
    ms = ManualStrategy()
    trades = ms.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=sv)
    portvals_manual = compute_portvals(trades, start_val=sv, commission=0.00, impact=0.00)
    portvals_manual = portvals_manual / portvals_manual.iloc[0]
    prices = get_data([symbol], pd.date_range(start_date, end_date))[symbol]
    benchmark_trades = pd.Series(0, index=prices.index)
    benchmark_trades.iloc[0] = 1000  # Buy 1000 shares on the first day
    benchmark_trades = pd.DataFrame(benchmark_trades, columns=[symbol])
    portvals_benchmark = compute_portvals(benchmark_trades, start_val=sv, commission=0.00, impact=0.00)
    portvals_benchmark = portvals_benchmark / portvals_benchmark.iloc[0]
    metrics_manual = calculate_metrics(portvals_manual)
    metrics_benchmark = calculate_metrics(portvals_benchmark)

    title = f"Manual Strategy vs Benchmark ({'In-Sample' if is_insample else 'Out-of-Sample'})"
    filename = f"manual_strategy_{'insample' if is_insample else 'outsample'}.png"
    generate_chart(portvals_manual, portvals_benchmark, trades[symbol], title, filename)

    return metrics_manual, metrics_benchmark


def main():

    symbol = "JPM"
    sv = 100000

    in_sample_start = dt.datetime(2008, 1, 1)
    in_sample_end = dt.datetime(2009, 12, 31)
    metrics_manual_in, metrics_benchmark_in = try_manual_strategy(
        symbol, in_sample_start, in_sample_end, sv, is_insample=True
    )

    out_sample_start = dt.datetime(2010, 1, 1)
    out_sample_end = dt.datetime(2011, 12, 31)
    metrics_manual_out, metrics_benchmark_out = try_manual_strategy(
        symbol, out_sample_start, out_sample_end, sv, is_insample=False
    )

    summary = pd.DataFrame(
        {
            "Metric": ["Cumulative Return", "Stdev of Daily Returns", "Mean of Daily Returns"],
            "Benchmark (In-Sample)": metrics_benchmark_in,
            "Manual Strategy (In-Sample)": metrics_manual_in,
            "Benchmark (Out-of-Sample)": metrics_benchmark_out,
            "Manual Strategy (Out-of-Sample)": metrics_manual_out,
        }
    )
    experiment1()
    experiment2()


if __name__ == "__main__":
    main()
