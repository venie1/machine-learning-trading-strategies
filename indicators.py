import pandas as pd
import matplotlib.pyplot as plt
from util import get_data


def author():
    return "pvenieris3"  


def study_group():
    return "pvenieris3"  

def sma(prices, window):
    return prices.rolling(window=window).mean()

def bollinger_bands(prices, window, num_std_dev):
    ssma = sma(prices, window)
    rolling_std = prices.rolling(window=window).std()
    upper = ssma + (rolling_std * num_std_dev)
    lower = ssma - (rolling_std * num_std_dev)
    bbp = (prices - lower) / (upper - lower)  # Bollinger Band Percentage
    return upper, lower, ssma, bbp

def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    short = prices.ewm(span=short_window, adjust=False).mean()
    long = prices.ewm(span=long_window, adjust=False).mean()
    macd = short - long
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_momentum(prices, window):
    return prices / prices.shift(window) - 1

def calculate_stochastic_oscillator(prices, k_window=14, d_window=3):
    low = prices.rolling(window=k_window).min()
    high = prices.rolling(window=k_window).max()
    stoch_k = (prices - low) / (high - low) * 100
    stoch_d = stoch_k.rolling(window=d_window).mean()
    return stoch_k, stoch_d

def plot_bollinger_bands(prices, upper_band, lower_band, sma, bbp):
    plt.figure(figsize=(14, 7))
    plt.plot(prices.index, prices, label='Price', color='blue')
    plt.plot(prices.index, sma, label='SMA', color='orange')
    plt.plot(prices.index, upper_band, label='Upper Band', color='red')
    plt.plot(prices.index, lower_band, label='Lower Band', color='green')
    plt.title('Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.savefig("Bollinger")

    plt.figure(figsize=(14, 7))
    plt.plot(prices.index, bbp, label='Bollinger Band %B', color='purple')
    plt.title('Bollinger Band %B')
    plt.axhline(1, color='red', linestyle='--', label='Overbought')
    plt.axhline(0, color='green', linestyle='--', label='Oversold')
    plt.xlabel('Date')
    plt.ylabel('%B')
    plt.legend()
    plt.grid()
    plt.savefig("BBP")

def plot_sma(prices, sma):
    plt.figure(figsize=(14, 7))
    plt.plot(prices.index, prices, label='Price', color='blue')
    plt.plot(prices.index, sma, label='SMA', color='orange')
    plt.title('Simple Moving Average (SMA)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.savefig("sma")

def plot_macd(prices, macd, signal):
    plt.figure(figsize=(14, 7))
    plt.plot(prices.index, macd, label='MACD', color='blue')
    plt.plot(prices.index, signal, label='Signal Line', color='orange')
    plt.title('Moving Average Convergence Divergence (MACD)')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid()
    plt.savefig("MACD")
    plt.figure(figsize=(14, 7))
    plt.bar(prices.index, macd - signal, label='MACD Histogram', color='gray')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('MACD Histogram')
    plt.xlabel('Date')
    plt.ylabel('MACD Histogram')
    plt.legend()
    plt.grid()
    plt.savefig("histogram")

def plot_momentum(prices, momentum):
    plt.figure(figsize=(14, 7))
    plt.plot(prices.index, momentum, label='Momentum', color='blue')
    plt.title('Momentum Indicator')
    plt.xlabel('Date')
    plt.ylabel('Momentum')
    plt.axhline(0, color='red', linestyle='--')
    plt.legend()
    plt.grid()
    plt.savefig("momentum")

def plot_stochastic_oscillator(stoch_k, stoch_d):
    plt.figure(figsize=(14, 7))
    plt.plot(stoch_k.index, stoch_k, label='%K', color='blue')
    plt.plot(stoch_d.index, stoch_d, label='%D', color='orange')
    plt.axhline(80, color='red', linestyle='--', label='Overbought')
    plt.axhline(20, color='green', linestyle='--', label='Oversold')
    plt.title('Stochastic Oscillator')
    plt.xlabel('Date')
    plt.ylabel('Stochastic Value')
    plt.legend()
    plt.grid()
    plt.savefig("stohastic")

def run(symbol="JPM", start_date="2008-01-01", end_date="2009-12-31"):
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[symbol]  # Get the prices for the specific symbol
    upper_band, lower_band, ssma, bbp = bollinger_bands(prices, window=20, num_std_dev=2)
    plot_bollinger_bands(prices, upper_band, lower_band, ssma, bbp)
    sma_50 = sma(prices, window=50)
    plot_sma(prices, sma_50)
    macd, signal = calculate_macd(prices)
    plot_macd(prices, macd, signal)
    momentum = calculate_momentum(prices, window=10)
    plot_momentum(prices, momentum)
    stoch_k, stoch_d = calculate_stochastic_oscillator(prices)
    plot_stochastic_oscillator(stoch_k, stoch_d)
if __name__ == "__main__":
    run()