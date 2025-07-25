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
## 📂 File Overview & Business Value

| File                          | Role & Business Value                                                                                                   |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **BagLearner.py**             | Implements a bagging ensemble of decision‑tree learners.  Reduces variance and overfitting key for robust signal generation under noisy market data. |
| **DTLearner.py**              | Correlation‑driven regression tree.  Captures strongest drivers of price movements improves explainability for risk/return trade‑offs.                          |
| **InsaneLearner.py**          | “Extreme” bagging of linear regressors.  Demonstrates scalable ensemble architectures valuable for model benchmarking and capacity planning.                  |
| **ManualStrategy.py**         | Rule‑based “human” strategy combining Bollinger %B, RSI, SMA cross‑over.  Provides a transparent baseline and governance‑friendly logic for compliance.       |
| **QLearner.py**               | Tabular Q‑learning agent framework.  Foundation for reinforcement‑learning–driven execution or order‑sizing policies.                                         |
| **RTLearner.py**              | Randomized regression tree.  Introduces stochasticity enhances resilience across regime shifts.                                                                    |
| **StrategyLearner.py**        | End‑to‑end ML strategy: feature engineering → bagged classification → trade signal generation.  Automates parameter tuning for rapid go‑to‑market.              |
| **experiment1.py**            | In‑sample vs. out‑of‑sample performance comparison.  Validates model generalization critical for live deployment risk assessment.                                |
| **experiment2.py**            | Sensitivity analysis on transaction‑cost (market impact).  Quantifies P&L erosion under varying frictions—essential for cost‑benefit optimization of algos.     |
| **gen_data.py**               | Synthetic data generator (e.g. martingale simulations).  Enables stress‑testing strategies under extreme scenarios.                                              |
| **indicators.py**             | Library of technical indicators: SMA, Bollinger Bands, MACD, Momentum, Stochastic.  Reusable feature‑engineering toolkit for any time‑series use case.         |
| **marketsimcode.py**          | Market simulator with realistic commissions & impact modeling.  Core engine for “what‑if” P&L projection and scenario analysis.                                 |
| **martingale.py**             | Monte Carlo martingale betting simulation.  Demonstrates risk of doubling strategies—parallels in position sizing/risk management.                             |
| **testlearner.py**            | Harness to train & evaluate any Learner (DT, RT, Bag, Insane).  Standardized benchmarking harness for rapid model comparison.                                    |
| **testproject.py**            | End‑to‑end sample workflow: data → strategy → backtest → metrics → plots.  Blueprint for production workflows or prototype demos.                             |
| **metadata.yml**              | Project metadata & configuration.  Useful for CI/CD or documentation generation pipelines.                                                                           |
| **\*_report.pdf**             | Detailed write‑ups for Projects 1,2,3,6,8.  Combines methodology, results, business insights & actionable recommendations—ready for stakeholder presentations.  |

---

## 🎯 Business Impact & Applications

- **Automated Signal Generation**: Ensemble learners & Q‑learning enable adaptive strategies that can be deployed in algorithmic execution platforms.
- **Risk Management**: Transaction‑cost sensitivity and Monte Carlo stress‑tests inform position‑sizing rules and capital allocation.
- **Explainability & Compliance**: Transparent rule‑based and tree‑based models support audit trails and regulatory requirements.
- **Rapid Prototyping**: Modular indicators & standardized test harness accelerate “proof of concept” to production.
- **Portfolio Advisory**: Backtesting framework provides clear performance attribution for pitch decks and client reporting.

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
