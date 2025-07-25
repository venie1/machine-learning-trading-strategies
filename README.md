# Machine Learning for Trading Portfolio

Welcome to my **Machine Learning for Trading** portfolio! This repository showcases a comprehensive collection of algorithmic trading projects and tools I developed as part of my coursework and personal exploration. It demonstrates my expertise in data science, machine learning, and quantitative finance skills highly valuable for data-driven trading, quantitative research, and analytics roles.


---

## 🚀 Highlights & Key Skills

### Machine Learning & Statistical Modeling
- **Ensemble Methods**: Custom Bagging (BagLearner) with decision trees  
- **Decision Trees**: Deterministic & Randomized regressors/classifiers  
- **Reinforcement Learning**: StrategyLearner framing trading as a classification task  
- **Feature Engineering**: Rolling statistics, exponential moving averages, volatility measures  

### Time Series & Financial Analytics
- **Technical Indicators**:  
  - **SMA** (Simple Moving Average)  
  - **Bollinger Bands & %B**  
  - **MACD** & signal line  
  - **Momentum**  
  - **Stochastic Oscillator**  
- **Backtesting Framework**:  
  - Realistic **market impact** and **commission** modeling  
  - **Normalized portfolio** performance plots  
  - **Cumulative return**, **annualized volatility**, **Sharpe ratio**  

### Data Science & Engineering
- **Python Ecosystem**: Pandas, NumPy, SciPy, Matplotlib  
- **Modular Codebase**: Reusable modules for indicators, learners, simulation  
- **Version Control**: Clean Git history, semantic commit messages  
- **Documentation**: Clear docstrings, structured README for recruiters  

---

## 📈 Selected Projects

### 1. Rule‑Based Manual Strategy
- **Combines** Bollinger Bands %B, RSI thresholds, SMA crossovers  
- **Generates** buy/hold/sell signals using human‑tuned rules  
- **Backtested** on JPMorgan (JPM) 2008–2009 & 2010–2011  
- **Outperforms** buy‑and‑hold on in‑sample and out‑of‑sample metrics  

### 2. Strategy Learner (Bagged Decision Trees)
- **Learns** optimal trade signals by classifying “buy/hold/sell” labels  
- **Trains** on technical indicators over a 5‑day look‑ahead window  
- **Ensemble** of 20 Random Tree learners reduces variance and overfitting  
- **Demonstrates** 2–3× lift over benchmark on normalized returns  

### 3. Transaction Cost Sensitivity (Experiment 2)
- **Varies** market impact factors: 0.001 → 0.1  
- **Analyzes** effect on cumulative return & number of trades  
- **Visualizes** trade frequency drop and return erosion under higher frictions  

---

## 🎯 Metrics & Performance

- **Cumulative Return**: Up to +343% vs. +3% benchmark (2008–2009)  
- **Annualized Volatility**: 7.7% vs. 82.8% benchmark  
- **Sharpe Ratio**: > 2.0 (Strategy) vs. < 0.5 (Benchmark)  
- **RMSE & R²**: Sub‑1% error on price regression tasks  

---

## 🛠 Tech Stack

- **Languages**: Python 3.8+  
- **Libraries**:  
  - Data Handling: Pandas, NumPy  
  - ML & Stats: SciPy, custom learners  
  - Visualization: Matplotlib  
- **Tools**: Git, VS Code / PyCharm, Jupyter notebooks  

---

## 📩 Get in Touch

I’m actively seeking data science & quant roles where I can apply machine learning to financial markets and large‑scale time‑series data.  
- **Email**: pgvenieris@outlook.com 
- **LinkedIn**: [[linkedin.com/in/pvenieris3](#)

---

*Dive into the code and feel free to connect for collaborations or questions!*  
