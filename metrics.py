"""
Financial metrics for portfolio analysis.

Standard risk-adjusted return calculations using 252 trading days/year convention.
"""

import numpy as np
import pandas as pd

def to_returns(price_series: pd.Series, use_log: bool = False) -> pd.Series:
    """Convert price series to returns (log or simple)."""
    s = price_series.dropna().astype(float)
    
    if use_log:
        return np.log(s / s.shift(1)).dropna()
    
    return s.pct_change().dropna()

def annualize_return(daily_returns: pd.Series, trading_days: int = 252) -> float:
    """
    Annualize daily returns using compound growth.
    Assumes 252 trading days per year.
    """
    compound_growth = (1 + daily_returns).prod()
    n_days = daily_returns.shape[0]
    
    if n_days == 0:
        return np.nan
    
    return compound_growth ** (trading_days / n_days) - 1

def annualize_vol(daily_returns: pd.Series, trading_days: int = 252) -> float:
    """
    Annualized volatility (standard deviation).
    Uses sqrt(252) scaling assuming i.i.d. returns.
    """
    return daily_returns.std(ddof=0) * np.sqrt(trading_days)

def sharpe(daily_returns: pd.Series, rf_daily: float = 0.0, trading_days: int = 252) -> float:
    """
    Sharpe ratio: risk-adjusted return metric.
    
    Higher is better. Rough interpretation:
    < 1.0: not great | 1-2: decent | > 2: good | > 3: excellent
    """
    excess = daily_returns - rf_daily
    vol = excess.std(ddof=0)
    
    if vol == 0 or np.isnan(vol):
        return np.nan
    
    return np.sqrt(trading_days) * excess.mean() / vol

def max_drawdown(prices: pd.Series) -> float:
    """
    Maximum peak-to-trough decline.
    Returns negative value (or 0 if prices only increased).
    """
    p = prices.dropna().astype(float)
    
    if p.empty:
        return np.nan
    
    cummax = p.cummax()
    drawdown = p / cummax - 1.0
    
    return drawdown.min()

def portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Calculate portfolio returns given asset returns and weights.
    Weights are normalized to sum to 1.
    """
    w = np.array(weights)
    w = w / w.sum()
    
    return (returns_df * w).sum(axis=1)

def rolling_sharpe(daily_returns: pd.Series, window: int = 126, rf_daily: float = 0.0) -> pd.Series:
    """
    Rolling Sharpe ratio over a moving window.
    Default window = 126 days (~6 months).
    """
    excess = daily_returns - rf_daily
    
    mu = excess.rolling(window).mean()
    sigma = excess.rolling(window).std(ddof=0)
    sharpe_ratio = np.sqrt(252) * (mu / sigma)
    
    return sharpe_ratio