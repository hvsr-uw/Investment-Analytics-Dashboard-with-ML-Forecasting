# Investment Analytics Dashboard with ML Forecasting

**A sophisticated financial analytics platform combining time-series forecasting, portfolio optimization, and risk analysis using machine learning.**


---

## Overview

This is a **professional-grade investment analytics dashboard** built with Python, Streamlit, and advanced machine learning models. It provides stock price forecasting using multiple ML algorithms (SVR, Random Forest, Ridge Regression) alongside traditional time-series methods (Holt-Winters), portfolio optimization, risk-adjusted performance metrics, and comprehensive data visualization.

**What makes this different:** Unlike basic stock trackers, this platform implements **production-level ML forecasting** with backtesting, **recursive multi-step prediction**, portfolio optimization algorithms, and **comparative model analysis** - features typically found in institutional trading systems, not student projects.

---

## Key Features

### Machine Learning Price Forecasting
- **Multiple ML algorithms**:
  - **SVR (Support Vector Regression)** with RBF kernel
  - **Random Forest** with feature importance analysis
  - **Ridge Regression** with L2 regularization
  - **Holt-Winters** exponential smoothing (classical time-series)
- **Recursive multi-step ahead prediction** (1-180 day forecasts)
- **Automated hyperparameter tuning** with user-adjustable controls
- **Backtest validation** with MAPE (Mean Absolute Percentage Error)
- **Model comparison framework** to evaluate multiple algorithms simultaneously
- **95% confidence intervals** for forecast uncertainty

### Advanced Feature Engineering
- **Lagged price features** (AR model structure: up to 30 lags)
- **Rolling window statistics** (5-day mean, 5-day std dev)
- **Log transformation** for variance stabilization
- **StandardScaler normalization** for ML model stability
- **Feature importance analysis** (for Random Forest models)

### Portfolio Analysis & Optimization
- **Efficient Frontier** computation using Monte Carlo simulation
- **Sharpe ratio optimization** to find optimal portfolio weights
- **Risk-return tradeoff visualization** with scatter plots
- **Custom portfolio creation** with user-defined weights
- **Rebalancing analysis** and allocation recommendations

### Risk-Adjusted Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns (industry standard)
- **Maximum Drawdown**: Peak-to-trough decline analysis
- **Annualized Returns**: Compound growth calculations (252 trading days)
- **Annualized Volatility**: Standard deviation with sqrt(252) scaling
- **Rolling Sharpe Ratio**: Time-varying risk-adjusted performance

### Interactive Data Visualization
- **Multi-ticker price charts** with Plotly (zoom, pan, hover details)
- **Correlation heatmaps** for asset relationships
- **Rolling Sharpe ratio plots** to visualize changing risk profiles
- **Forecast comparison charts** with confidence bands
- **Efficient frontier plots** for portfolio optimization
- **Feature importance bar charts** for model interpretability

### Data Quality & Management
- **Automated data ingestion** from Yahoo Finance (yfinance)
- **SQLite database** with normalized schema
- **Data freshness monitoring** with staleness alerts
- **Missing data detection** and handling
- **Date range filtering** for custom analysis periods

---

## Tech Stack

### Core Technologies
- **Python** 3.8+
- **Streamlit** - Interactive web dashboard framework
- **SQLite** - Lightweight relational database
- **Plotly** - Interactive data visualizations

### Machine Learning & Statistics
- **scikit-learn** - ML models (SVR, RandomForest, Ridge)
- **statsmodels** - Time-series forecasting (Holt-Winters)
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and time-series analysis

### Financial Data
- **yfinance** - Yahoo Finance API wrapper (can be replaced with paid APIs)

---

## Architecture

### System Design
```
┌─────────────────┐
│   Streamlit     │  ← User Interface Layer
│   Dashboard     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Analytics      │  ← Business Logic Layer
│  Engine         │     • Portfolio optimization
│  • ML Models    │     • Risk metrics
│  • Forecasting  │     • Feature engineering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   SQLite DB     │  ← Data Layer
│  • assets       │
│  • prices       │
│  • portfolios   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Yahoo Finance  │  ← External Data Source
│     (yfinance)  │
└─────────────────┘
```

### Database Schema

**assets table:**
```sql
CREATE TABLE assets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker TEXT NOT NULL UNIQUE,
  name TEXT
);
```

**prices table:**
```sql
CREATE TABLE prices (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  asset_id INTEGER NOT NULL,
  date TEXT NOT NULL,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  adj_close REAL,
  volume INTEGER,
  UNIQUE(asset_id, date),
  FOREIGN KEY(asset_id) REFERENCES assets(id)
);
```

**Indexing Strategy:**
- Primary keys for fast lookups
- UNIQUE constraint on (asset_id, date) prevents duplicates
- Foreign keys ensure referential integrity

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/investment-analytics.git
cd investment-analytics
```

**2. Create virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Initialize database:**
```bash
python create_db.py
```

**5. Ingest stock data:**
```bash
python ingest_prices.py
```
This downloads 5 years of daily data for AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, and ^GSPC (S&P 500 benchmark).

**6. Run the dashboard:**
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## ML Forecasting Details

### Algorithm Comparison

| Algorithm | Strengths | Best For | Typical MAPE |
|-----------|-----------|----------|--------------|
| **SVR** | Non-linear patterns, kernel trick | Complex price movements | 2-5% |
| **Random Forest** | Feature importance, robust to outliers | Trend identification | 2-6% |
| **Ridge** | Fast training, linear relationships | Simple trends | 3-7% |
| **Holt-Winters** | Classical time-series, interpretable | Smooth trends | 3-8% |

### Recursive Multi-Step Forecasting

The platform uses a **recursive strategy** for multi-step ahead predictions:

```python
def recursive_forecast(model, last_window, horizon):
    """
    Each prediction becomes a feature for the next prediction.
    
    t+1: predict using [t, t-1, t-2, ..., t-19]
    t+2: predict using [t+1, t, t-1, ..., t-18]  <- t+1 is now a feature
    t+3: predict using [t+2, t+1, t, ..., t-17]
    ...
    """
    predictions = []
    current_window = last_window.copy()
    
    for step in range(horizon):
        # Extract features from current window
        features = extract_features(current_window)
        
        # Make prediction
        pred = model.predict(features)
        predictions.append(pred)
        
        # Update window for next step
        current_window = append(current_window, pred)
    
    return predictions
```

### Feature Engineering Process

**Raw Data → Feature Transformation:**
```
Price Series: [100, 102, 101, 103, 105, ...]
       ↓
Log Transform: [4.605, 4.625, 4.615, 4.635, 4.654, ...]
       ↓
Create Lags:
  lag_1:  [4.625, 4.615, 4.635, 4.654, ...]
  lag_2:  [4.615, 4.635, 4.654, ...]
  lag_3:  [4.635, 4.654, ...]
  ...
  lag_20: [...]
       ↓
Rolling Stats:
  5-day mean: [rolling average of last 5 values]
  5-day std:  [rolling std dev of last 5 values]
       ↓
StandardScaler: Scale features to mean=0, std=1
       ↓
Ready for ML model!
```

### Backtest Validation

The platform uses **holdout validation** to assess model accuracy:

```
Full dataset: [━━━━━━━━━━━━━━━━━━━━━━━━━━]
              [━━━━━━━ Train ━━━━━━━][Test]
                                      ↑
                       Hold out last 10% (min 20, max 60 days)

1. Train model on training set
2. Predict on test set
3. Calculate MAPE: mean(|actual - predicted| / actual) × 100
4. Retrain on full dataset for actual forecast
```

---

## Portfolio Optimization Algorithm

### Efficient Frontier Construction

**Monte Carlo Simulation Approach:**

```python
def efficient_frontier(returns, num_portfolios=10000, risk_free_rate=0.02):
    """
    Generate random portfolio weights and compute risk-return metrics.
    
    For each random portfolio:
      1. Generate random weights (sum to 1)
      2. Calculate portfolio return = Σ(weight_i × return_i)
      3. Calculate portfolio volatility = sqrt(w^T Σ w)
      4. Calculate Sharpe ratio = (return - rf) / volatility
    
    Find portfolio with maximum Sharpe ratio = optimal portfolio
    """
    results = []
    
    for _ in range(num_portfolios):
        # Random weights that sum to 1
        weights = np.random.random(len(returns.columns))
        weights /= weights.sum()
        
        # Portfolio metrics
        port_return = np.dot(weights, returns.mean()) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe = (port_return - risk_free_rate) / port_vol
        
        results.append({
            'return': port_return,
            'volatility': port_vol,
            'sharpe': sharpe,
            'weights': weights
        })
    
    # Find optimal portfolio (max Sharpe)
    optimal = max(results, key=lambda x: x['sharpe'])
    
    return results, optimal
```

### Visualization Output:
- **Scatter plot**: 10,000 random portfolios (color = Sharpe ratio)
- **Optimal point**: Highlighted in red (maximum Sharpe)
- **Efficient frontier**: Upper edge of feasible portfolios

---

## Risk Metrics Explained

### Sharpe Ratio
```
Sharpe = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility

Interpretation:
< 1.0: Poor (not worth the risk)
1-2:   Decent
2-3:   Good (outperforming with reasonable risk)
> 3:   Excellent (rare in real markets)
```

**Why it matters:** Tells you if returns justify the volatility. A portfolio with 20% return and 25% volatility (Sharpe = 0.8) is worse than 12% return with 8% volatility (Sharpe = 1.5).

### Maximum Drawdown
```
Max Drawdown = (Trough Value - Peak Value) / Peak Value

Example:
Stock goes: $100 → $150 → $90 → $120
Max Drawdown = ($90 - $150) / $150 = -40%
```

**Why it matters:** Shows worst-case loss from peak. Critical for understanding downside risk.

### Annualized Volatility
```
Annual Vol = Daily StdDev × sqrt(252)

252 = typical trading days per year
sqrt(252) scaling assumes i.i.d. returns
```

**Why it matters:** Standardizes risk measurement across different time periods.
---

##  Future Enhancements

- [ ] LSTM/GRU neural networks for sequence modeling
- [ ] Sentiment analysis integration (news, Twitter)
- [ ] Options pricing and Greeks calculations
- [ ] Backtesting framework with transaction costs
- [ ] Real-time data streaming (WebSocket)
- [ ] Multi-asset class support (bonds, commodities, crypto)
- [ ] Advanced technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Portfolio rebalancing alerts
- [ ] Risk parity optimization
- [ ] Black-Litterman model integration
- [ ] Factor models (Fama-French)

---

##  License

MIT License - see [LICENSE](LICENSE) for details

---

##  Disclaimer

**This project is for educational purposes only.** It uses historical data and machine learning models that should NOT be used for actual investment decisions. Past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.

The forecasts and portfolio recommendations generated by this tool are based on historical patterns and may not account for sudden market changes, black swan events, or other unpredictable factors.

---

##  Acknowledgments

- **scikit-learn** for ML algorithms
- **statsmodels** for time-series forecasting
- **Plotly** for interactive visualizations
- **Streamlit** for rapid dashboard development
- **yfinance** for market data access

---

**Built using Python, scikit-learn, statsmodels, and Streamlit**
