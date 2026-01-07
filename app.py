import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from metrics import to_returns, annualize_return, annualize_vol, sharpe, max_drawdown, portfolio_returns, rolling_sharpe

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple, Optional

DB = Path("market.db")

@st.cache_data(show_spinner=False)
def load_assets() -> pd.DataFrame:
    """Grab all tickers we've ingested from the DB"""
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT id, ticker, COALESCE(name, ticker) AS name FROM assets ORDER BY ticker", 
        conn
    )
    conn.close()
    return df

@st.cache_data(show_spinner=False)
def load_prices(asset_ids: List[int], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Pull adjusted close prices for selected assets.
    Returns a nice pivoted DataFrame with dates as index, tickers as columns.
    The date filters are optional - pass None to get everything.
    """
    if not asset_ids:
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB)
    # Build the query with placeholders for safety
    placeholders = ','.join(['?'] * len(asset_ids))
    q = f"""
    SELECT a.ticker, p.date, p.adj_close
    FROM prices p
    JOIN assets a ON a.id = p.asset_id
    WHERE a.id IN ({placeholders})
    """
    params = asset_ids[:]
    
    if start:
        q += " AND p.date >= ?"
        params.append(start)
    if end:
        q += " AND p.date <= ?"
        params.append(end)
    q += " ORDER BY p.date ASC"
    
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    
    if df.empty:
        return df
    
    df["date"] = pd.to_datetime(df["date"])
    # Pivot so each ticker becomes a column - way easier to work with
    pivot = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return pivot

def metrics_table(prices: pd.DataFrame, rf_annual: float = 0.02) -> pd.DataFrame:
    """
    Calculate standard financial metrics for each ticker.
    Returns a sorted DataFrame (by Sharpe, descending).
    """
    if prices.empty:
        return pd.DataFrame()
    
    rets = prices.pct_change().dropna()
    # Convert annual risk-free rate to daily (252 trading days)
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    
    rows = []
    for ticker in rets.columns:
        r = rets[ticker].dropna()
        rows.append({
            "Ticker": ticker,
            "Ann Return": annualize_return(r),
            "Ann Vol": annualize_vol(r),
            "Sharpe": sharpe(r, rf_daily=rf_daily),
            "Max Drawdown": max_drawdown(prices[ticker])
        })
    
    out = pd.DataFrame(rows).set_index("Ticker")
    # Sort by Sharpe - best performers at top
    return out.sort_values("Sharpe", ascending=False)

# Forecasting Models

def make_forecast(prices_series, horizon=90, use_calendar=False):
    """
    Classic exponential smoothing (Holt-Winters) forecast.
    Pretty simple but works decently for trending data.
    
    I'm using a holdout set to estimate MAPE so we can compare models.
    """
    s = prices_series.dropna().astype(float).copy()
    freq = "D" if use_calendar else "B"  # calendar vs business days
    s = s.asfreq(freq).ffill()  # forward fill any gaps

    # Hold out ~10% for backtesting, minimum 10 days
    holdout = min(60, max(10, len(s)//10))
    train, test = s.iloc[:-holdout], s.iloc[-holdout:]

    # Fit on training data first to get MAPE
    model = ExponentialSmoothing(
        train, 
        trend="add",  # additive trend
        damped_trend=True,  # damping helps prevent unrealistic extrapolation
        seasonal=None,  # not modeling seasonality here
        initialization_method="estimated"
    ).fit(optimized=True)

    pred_test = model.predict(start=test.index[0], end=test.index[-1])
    # MAPE = mean absolute percentage error
    mape = float((np.abs((test - pred_test) / np.clip(test, 1e-9, None))).mean() * 100)

    # Now refit on full series for actual forecast
    model_full = ExponentialSmoothing(
        s, trend="add", damped_trend=True, seasonal=None, 
        initialization_method="estimated"
    ).fit(optimized=True)

    future_index = pd.date_range(
        start=s.index[-1] + pd.offsets.Day(1), 
        periods=horizon, 
        freq=freq
    )
    fcst = pd.Series(
        model_full.forecast(horizon).values, 
        index=future_index, 
        name="Forecast"
    )

    # Quick and dirty confidence bands using residual std
    resid = s - model_full.fittedvalues.reindex_like(s)
    sigma = float(resid.std(ddof=1)) if len(resid) > 3 else 0.0
    lower = fcst - 1.96 * sigma  # ~95% confidence
    upper = fcst + 1.96 * sigma

    return {"fcst": fcst, "lower": lower, "upper": upper, "mape": mape}

# ML Helper Functions

def _prepare_series(series, use_log=True, use_calendar=False):
    """
    Clean up the price series and optionally log-transform it.
    Log prices are nice because they stabilize variance and make ML happier.
    """
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    freq = "D" if use_calendar else "B"
    s = s.asfreq(freq).ffill()
    
    if use_log:
        return np.log(s), True, freq
    return s.copy(), False, freq

def _build_xy(series: pd.Series, lags: int = 20) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Create supervised learning dataset from time series.
    Features: lagged values + rolling stats
    Target: next value
    
    This is basically turning AR(p) into a regression problem.
    """
    df = pd.DataFrame({"y": series})
    
    # Add lagged features
    for k in range(1, lags + 1):
        df[f"lag_{k}"] = series.shift(k)
    
    # Some basic rolling window features - helps capture recent momentum
    df["roll_mean_5"] = series.rolling(5).mean()
    df["roll_std_5"]  = series.rolling(5).std()
    
    df = df.dropna()  # kills first ~20 rows but necessary
    
    y = df["y"].values
    X = df.drop(columns=["y"]).values
    idx = df.index
    return X, y, idx

def _backtest_holdout_sizes(n_rows, min_hold=20, max_hold=60):
    """Figure out reasonable holdout size for backtesting"""
    return min(max_hold, max(min_hold, n_rows // 10))

def _recursive_steps(start_series, model, scaler, horizon, lags, freq, model_type="svr"):
    """
    Multi-step ahead forecasting using recursive strategy.
    Each prediction becomes a feature for the next prediction.
    
    This is kinda slow but it's the most straightforward approach.
    Could parallelize or use direct strategy but this works fine for now.
    """
    last = start_series.copy()
    preds, lower, upper = [], [], []
    future_index = pd.date_range(
        start=last.index[-1] + pd.offsets.Day(1), 
        periods=horizon, 
        freq=freq
    )

    for i in range(horizon):
        # Grab recent window for features
        window = last.iloc[-max(lags, 5):]
        
        # Build feature vector: lags + rolling stats
        feats = [last.iloc[-k] for k in range(1, lags + 1)]
        feats += [window.tail(5).mean(), window.tail(5).std()]
        x_next = np.array(feats, dtype=float).reshape(1, -1)
        
        if scaler is not None:
            x_next = scaler.transform(x_next)

        # Predict next step
        if model_type == "rf":
            yhat = float(model.predict(x_next)[0])
            # For RF, use tree predictions to estimate uncertainty
            tree_preds = np.array([est.predict(x_next)[0] for est in model.estimators_])
            lo = float(np.percentile(tree_preds, 10))
            hi = float(np.percentile(tree_preds, 90))
        else:
            yhat = float(model.predict(x_next)[0])
            sigma = getattr(model, "residual_sigma_", 0.0)
            lo = yhat - 1.96 * sigma if sigma and sigma > 0 else np.nan
            hi = yhat + 1.96 * sigma if sigma and sigma > 0 else np.nan

        preds.append(yhat)
        lower.append(lo)
        upper.append(hi)
        
        # Append prediction to series for next iteration
        next_date = future_index[i]
        last = pd.concat([last, pd.Series([yhat], index=[next_date])])

    return future_index, np.array(preds), np.array(lower), np.array(upper)

# ML models

def make_forecast_ridge(prices_series, horizon=90, lags=20, use_log=True, use_calendar=False):
    """
    Ridge regression forecast. Simple linear model with L2 regularization.
    Fast and surprisingly effective as a baseline.
    """
    s_in, used_log, freq = _prepare_series(prices_series, use_log=use_log, use_calendar=use_calendar)
    X, y, idx = _build_xy(s_in, lags=lags)
    
    if len(y) < 80:
        raise ValueError("Not enough history for ML (need 80+ days)")

    holdout = _backtest_holdout_sizes(len(y))
    X_tr, y_tr = X[:-holdout], y[:-holdout]
    X_te, y_te = X[-holdout:], y[-holdout:]
    
    # Standardize features - important for Ridge
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    model = Ridge(alpha=1.0, random_state=0)
    model.fit(X_tr_sc, y_tr)
    y_pred = model.predict(X_te_sc)

    # Calculate MAPE on original scale
    if used_log:
        actual_bt = np.exp(y_te)
        pred_bt = np.exp(y_pred)
    else:
        actual_bt = y_te
        pred_bt = y_pred
    mape = float((np.abs((actual_bt - pred_bt) / np.clip(actual_bt, 1e-9, None))).mean() * 100)

    # Get residual std for confidence bands
    resid = actual_bt - pred_bt
    sigma = float(np.std(resid, ddof=1)) if resid.size > 3 else 0.0

    # Refit on all data
    X_all_sc = scaler.fit_transform(X)
    model.fit(X_all_sc, y)

    # Generate forecast
    fut_idx, preds, lo, hi = _recursive_steps(
        s_in, model, scaler, horizon, lags, freq, model_type="svr"
    )

    # Transform back if we used log
    if used_log:
        fcst = pd.Series(np.exp(preds), index=fut_idx, name="Forecast")
        lower = pd.Series(np.exp(lo), index=fut_idx) if np.isfinite(lo).all() else pd.Series(np.nan, index=fut_idx)
        upper = pd.Series(np.exp(hi), index=fut_idx) if np.isfinite(hi).all() else pd.Series(np.nan, index=fut_idx)
    else:
        fcst = pd.Series(preds, index=fut_idx, name="Forecast")
        lower = pd.Series(lo, index=fut_idx)
        upper = pd.Series(hi, index=fut_idx)

    # Fallback bands if needed
    if lower.isna().any() or upper.isna().any():
        steps = np.arange(1, horizon + 1)
        sigma_scaled = sigma * np.sqrt(steps)  # uncertainty grows with horizon
        lower = fcst - 1.96 * sigma_scaled
        upper = fcst + 1.96 * sigma_scaled

    return {"fcst": fcst, "lower": lower, "upper": upper, "mape": mape}

def make_forecast_svr(prices_series, horizon=90, lags=20, use_log=True, 
                      use_calendar=False, C=5.0, gamma="scale", epsilon=0.1):
    """
    Support Vector Regression with RBF kernel.
    More flexible than Ridge - can capture nonlinear patterns.
    
    Hyperparams (C, gamma, epsilon) matter a lot. Default values work okayish
    but might want to tune these for different stocks.
    """
    s_in, used_log, freq = _prepare_series(prices_series, use_log=use_log, use_calendar=use_calendar)
    X, y, idx = _build_xy(s_in, lags=lags)
    
    if len(y) < 80:
        raise ValueError("Not enough history for ML (need 80+ days)")

    holdout = _backtest_holdout_sizes(len(y))
    X_tr, y_tr = X[:-holdout], y[:-holdout]
    X_te, y_te = X[-holdout:], y[-holdout:]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_tr_sc, y_tr)
    y_pred = model.predict(X_te_sc)

    if used_log:
        actual_bt = np.exp(y_te)
        pred_bt = np.exp(y_pred)
    else:
        actual_bt = y_te
        pred_bt = y_pred
    mape = float((np.abs((actual_bt - pred_bt) / np.clip(actual_bt, 1e-9, None))).mean() * 100)

    # Store residual std for uncertainty estimation
    resid_train = y_tr - model.predict(X_tr_sc)
    sigma = float(np.std(resid_train, ddof=1)) if resid_train.size > 3 else 0.0
    setattr(model, "residual_sigma_", sigma)

    # Refit on everything
    X_all_sc = scaler.fit_transform(X)
    model.fit(X_all_sc, y)

    fut_idx, preds, lo, hi = _recursive_steps(
        s_in, model, scaler, horizon, lags, freq, model_type="svr"
    )

    if used_log:
        fcst = pd.Series(np.exp(preds), index=fut_idx, name="Forecast")
        lower = pd.Series(np.exp(lo), index=fut_idx) if np.isfinite(lo).all() else pd.Series(np.nan, index=fut_idx)
        upper = pd.Series(np.exp(hi), index=fut_idx) if np.isfinite(hi).all() else pd.Series(np.nan, index=fut_idx)
    else:
        fcst = pd.Series(preds, index=fut_idx, name="Forecast")
        lower = pd.Series(lo, index=fut_idx)
        upper = pd.Series(hi, index=fut_idx)

    if lower.isna().any() or upper.isna().any():
        steps = np.arange(1, horizon + 1)
        sigma_scaled = sigma * np.sqrt(steps)
        lower = fcst - 1.96 * sigma_scaled
        upper = fcst + 1.96 * sigma_scaled

    return {"fcst": fcst, "lower": lower, "upper": upper, "mape": mape}

def make_forecast_rf(prices_series, horizon=90, lags=20, use_log=True, use_calendar=False,
                     n_estimators=400, max_depth=None, min_samples_leaf=1, random_state=0):
    """
    Random Forest regression - ensemble of decision trees.
    
    Usually pretty robust and doesn't need scaling. The tree ensemble naturally
    provides uncertainty estimates via variance across trees.
    
    More trees = better but slower. 400 seems like a sweet spot for this data.
    """
    s_in, used_log, freq = _prepare_series(prices_series, use_log=use_log, use_calendar=use_calendar)
    X, y, idx = _build_xy(s_in, lags=lags)
    
    if len(y) < 80:
        raise ValueError("Not enough history for ML (need 80+ days)")

    holdout = _backtest_holdout_sizes(len(y))
    X_tr, y_tr = X[:-holdout], y[:-holdout]
    X_te, y_te = X[-holdout:], y[-holdout:]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,  # use all cores
        random_state=random_state
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    if used_log:
        actual_bt = np.exp(y_te)
        pred_bt = np.exp(y_pred)
    else:
        actual_bt = y_te
        pred_bt = y_pred
    mape = float((np.abs((actual_bt - pred_bt) / np.clip(actual_bt, 1e-9, None))).mean() * 100)

    # Refit on full data
    model.fit(X, y)
    
    fut_idx, preds, lo, hi = _recursive_steps(
        s_in, model, scaler=None, horizon=horizon, lags=lags, freq=freq, model_type="rf"
    )

    if used_log:
        fcst = pd.Series(np.exp(preds), index=fut_idx, name="Forecast")
        lower = pd.Series(np.exp(lo), index=fut_idx)
        upper = pd.Series(np.exp(hi), index=fut_idx)
    else:
        fcst = pd.Series(preds, index=fut_idx, name="Forecast")
        lower = pd.Series(lo, index=fut_idx)
        upper = pd.Series(hi, index=fut_idx)

    return {"fcst": fcst, "lower": lower, "upper": upper, "mape": mape}

# Streamlit UI

def main():
    st.set_page_config(page_title="Investment Analytics", layout="wide")
    st.title("📈 Investment Analytics Dashboard")

    assets_df = load_assets()
    if assets_df.empty:
        st.info("No assets found in DB. Run `python ingest_prices.py` first to pull data.")
        st.stop()

    all_tickers = assets_df["ticker"].tolist()
    # Default to the big tech stocks if they exist
    default_selection = [t for t in ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA"] 
                        if t in all_tickers]
    benchmark = "^GSPC" if "^GSPC" in all_tickers else None

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        tickers = st.multiselect(
            "Select tickers", 
            options=all_tickers, 
            default=default_selection
        )
        date_range = st.date_input("Date range", [])
        show_benchmark = st.checkbox(
            "Compare to S&P 500", 
            value=benchmark is not None
        )
        rf = st.number_input(
            "Risk free rate (annual)", 
            value=0.02, 
            min_value=0.0, 
            max_value=0.15, 
            step=0.005, 
            format="%.3f"
        )
        st.caption("💡 Add more tickers by editing ingest_prices.py and rerunning")
        
        if not tickers:
            st.warning("Pick at least one ticker to continue")
            st.stop()

    # Parse date range
    start = str(date_range[0]) if len(date_range) == 2 else None
    end = str(date_range[1]) if len(date_range) == 2 else None

    # Load price data
    ids = assets_df.loc[assets_df["ticker"].isin(tickers), "id"].tolist()
    prices = load_prices(ids, start, end)
    
    if prices.empty:
        st.warning("No price data available. Try adjusting date filters or running data ingestion.")
        st.stop()

    # Optionally add benchmark
    if show_benchmark and benchmark and benchmark in all_tickers:
        bid = assets_df.loc[assets_df["ticker"] == benchmark, "id"].iloc[0]
        bench_prices = load_prices([bid], start, end)
        if not bench_prices.empty:
            prices = prices.join(bench_prices, how="inner")

    tabs = st.tabs(["📊 Analytics", "🔮 Forecast", "🎯 Data Quality" ])

    # Analytics tab
    with tabs[0]:
        st.subheader("Price History")
        chart_df = prices.copy()
        chart_df.index.name = "Date"
        fig = px.line(
            chart_df, 
            x=chart_df.index, 
            y=chart_df.columns,
            labels={"value": "Adj Close ($)", "variable": "Ticker"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk & Return Metrics")
        table = metrics_table(prices, rf_annual=rf)
        st.dataframe(table.style.format({
            "Ann Return": "{:.2%}",
            "Ann Vol": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Max Drawdown": "{:.2%}"
        }))

        st.subheader("Correlation Heatmap")
        corr = prices.pct_change().dropna().corr()
        heat = px.imshow(
            corr, 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale="RdBu", 
            origin="lower"
        )
        st.plotly_chart(heat, use_container_width=True)

        st.subheader("Portfolio Simulator")
        st.caption("Adjust weights to see how different allocations would have performed")
        
        weights = {}
        cols = st.columns(len(tickers))
        for i, t in enumerate(tickers):
            with cols[i]:
                weights[t] = st.number_input(
                    f"{t} weight", 
                    value=1.0/len(tickers), 
                    min_value=0.0, 
                    max_value=1.0, 
                    step=0.01
                )

        w = np.array([weights[t] for t in tickers])
        if w.sum() == 0:
            st.error("Weights can't all be zero!")
            st.stop()
        w = w / w.sum()  # normalize to sum to 1

        rets = prices[tickers].pct_change().dropna()
        port = portfolio_returns(rets, w)

        bench_col = "^GSPC" if (show_benchmark and "^GSPC" in prices.columns) else None
        rf_daily = (1 + rf) ** (1/252) - 1

        # Growth of $1 chart
        pr = (1 + port).cumprod()
        dfp = pd.DataFrame({"Portfolio": pr})
        if bench_col:
            dfp["Benchmark"] = (1 + rets[bench_col]).cumprod()

        figp = px.line(dfp, labels={"value": "Growth of $1", "index": "Date"})
        st.plotly_chart(figp, use_container_width=True)

        st.write("**Portfolio Metrics**")
        pm = {
            "Ann Return": annualize_return(port),
            "Ann Vol": annualize_vol(port),
            "Sharpe": sharpe(port, rf_daily=rf_daily),
            "Max Drawdown": max_drawdown(dfp["Portfolio"])
        }
        st.json({k: (f"{v:.2%}" if k != "Sharpe" else f"{v:.2f}") for k, v in pm.items()})

        st.subheader("Rolling Sharpe Ratio (6-month)")
        st.caption("Shows how risk-adjusted returns evolved over time")
        roll = rolling_sharpe(port, window=126, rf_daily=rf_daily)
        roll_df = roll.to_frame("Rolling Sharpe")
        figr = px.line(roll_df, labels={"value": "Sharpe Ratio", "index": "Date"})
        st.plotly_chart(figr, use_container_width=True)

    # Forecast tab
    with tabs[1]:
        st.subheader("Price Forecasting with ML")

        base_tickers = [t for t in tickers if t in prices.columns]
        if not base_tickers:
            st.info("Need at least one ticker to forecast. Go back to Analytics and select some!")
            st.stop()

        target = st.selectbox("Ticker to forecast", base_tickers, index=0)

        model_choice = st.radio(
            "Primary forecasting model",
            ["SVM RBF", "Random Forest", "Ridge (ML)", "Holt Winters"],
            horizontal=True,
            help="SVM/RF are nonlinear ML models. Ridge is linear. Holt-Winters is classical time series."
        )
        
        use_log = st.checkbox(
            "Model log(price) instead of raw price", 
            value=True,
            help="Log transform stabilizes variance and often improves ML performance"
        )

        use_calendar = st.checkbox(
            "Use calendar days (vs business days)", 
            value=False,
            help="Business days excludes weekends/holidays - usually better for stocks"
        )
        
        max_h = 365 if use_calendar else 252
        label = "calendar days" if use_calendar else "business days"
        horizon = st.slider(
            f"Forecast horizon ({label})", 
            30, max_h, 90, step=10
        )

        # Advanced settings in expander so they don't clutter the UI
        with st.expander("🔧 Model Hyperparameters"):
            lags = st.slider("Number of lags", 5, 60, 20, 1)
            
            if model_choice == "SVM RBF":
                C = st.number_input("C (regularization)", value=5.0, min_value=0.01, step=0.5)
                epsilon = st.number_input("Epsilon (tube width)", value=0.1, min_value=0.0, step=0.05)
                gamma_opt = st.selectbox("Gamma", ["scale", "auto"], index=0)
            
            elif model_choice == "Random Forest":
                n_estimators = st.slider("Number of trees", 100, 1000, 400, 50)
                max_depth = st.selectbox("Max tree depth", [None, 4, 6, 8, 12, 16], index=0)
                min_samples_leaf = st.slider("Min samples per leaf", 1, 10, 1, 1)

        # Model comparison feature
        st.markdown("### Model Comparison")
        st.caption("Compare multiple forecasting approaches side-by-side")

        ALL_MODELS = ["SVM RBF", "Random Forest", "Ridge (ML)", "Holt Winters"]
        available_for_compare = [m for m in ALL_MODELS if m != model_choice]

        compare_models = st.multiselect(
            "Additional models to overlay on chart",
            available_for_compare,
            default=[],
            help="Adds forecasts from other models to the plot for comparison"
        )

        # Clean up stale selections if user changed primary model
        compare_models = [m for m in compare_models if m in available_for_compare]

        # Get the price series for selected ticker
        series = pd.to_numeric(prices[target], errors="coerce").dropna()
        if series.size < 80:
            st.warning("Not enough historical data to forecast (need 80+ days). "
                      "Try selecting a longer date range or different ticker.")
            st.stop()

        # Helper function to run any model
        def run_model(name):
            """Dispatch to the appropriate forecasting function based on model name"""
            if name == "SVM RBF":
                return make_forecast_svr(
                    series, horizon=horizon, lags=lags, use_log=use_log, 
                    use_calendar=use_calendar,
                    C=C if 'C' in locals() else 5.0,
                    gamma=gamma_opt if 'gamma_opt' in locals() else "scale",
                    epsilon=epsilon if 'epsilon' in locals() else 0.1
                )
            if name == "Random Forest":
                return make_forecast_rf(
                    series, horizon=horizon, lags=lags, use_log=use_log, 
                    use_calendar=use_calendar,
                    n_estimators=n_estimators if 'n_estimators' in locals() else 400,
                    max_depth=None if 'max_depth' not in locals() or max_depth is None else max_depth,
                    min_samples_leaf=min_samples_leaf if 'min_samples_leaf' in locals() else 1
                )
            if name == "Ridge (ML)":
                return make_forecast_ridge(
                    series, horizon=horizon, lags=lags, 
                    use_log=use_log, use_calendar=use_calendar
                )
            if name == "Holt Winters":
                # For Holt-Winters, handle log transform manually if needed
                if use_log:
                    s_in = np.log(series)
                    out = make_forecast(s_in, horizon=horizon, use_calendar=use_calendar)
                    # Transform back to original scale
                    out["fcst"] = np.exp(out["fcst"])
                    out["lower"] = np.exp(out["lower"])
                    out["upper"] = np.exp(out["upper"])
                    return out
                return make_forecast(series, horizon=horizon, use_calendar=use_calendar)
            raise ValueError(f"Unknown model: {name}")

        # Run primary model
        try:
            primary_result = run_model(model_choice)
        except ValueError as e:
            st.error(f"Forecast failed: {str(e)}")
            st.stop()

        st.caption(f"📉 Primary model backtest MAPE: **{primary_result['mape']:.2f}%**")

        # Run comparison models
        comp_results = {}
        for m in compare_models:
            try:
                comp_results[m] = run_model(m)
            except Exception as e:
                st.warning(f"⚠️ Couldn't run {m}: {str(e)}")

        # Build the forecast plot
        fcst = primary_result["fcst"]
        lower = primary_result["lower"] 
        upper = primary_result["upper"]
        
        # Create confidence band (filled area)
        band_x = list(upper.index) + list(lower.index[::-1])
        band_y = list(upper.values) + list(lower.values[::-1])

        figf = go.Figure()
        
        # Confidence band for primary model
        figf.add_trace(go.Scatter(
            x=band_x, y=band_y, 
            fill="toself", 
            mode="lines",
            line=dict(width=0), 
            name=f"{model_choice} confidence (~95%)",
            hoverinfo="skip", 
            showlegend=True, 
            opacity=0.2
        ))
        
        # Historical prices
        figf.add_trace(go.Scatter(
            x=series.index, 
            y=series.values, 
            mode="lines",
            name="Actual Price", 
            line=dict(width=2, color="#1f77b4")
        ))
        
        # Primary forecast line
        figf.add_trace(go.Scatter(
            x=fcst.index, 
            y=fcst.values, 
            mode="lines",
            name=f"{model_choice} forecast", 
            line=dict(width=2, dash="dash")
        ))

        # Overlay comparison forecasts
        dash_styles = ["dot", "dashdot", "longdash"]
        for idx, (name, res) in enumerate(comp_results.items()):
            figf.add_trace(go.Scatter(
                x=res["fcst"].index,
                y=res["fcst"].values,
                mode="lines",
                name=f"{name} forecast",
                line=dict(width=2, dash=dash_styles[idx % len(dash_styles)])
            ))

        figf.update_layout(
            title=f"{target} — {horizon}-day forecast comparison",
            xaxis_title="Date", 
            yaxis_title="Adjusted Close ($)", 
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(figf, use_container_width=True)

        # Show MAPE comparison table if we have multiple models
        if comp_results:
            rows = [{"Model": model_choice, "MAPE": primary_result["mape"]}]
            for name, res in comp_results.items():
                rows.append({"Model": name, "MAPE": res["mape"]})
            
            cmp_df = pd.DataFrame(rows).set_index("Model").sort_values("MAPE")
            st.subheader("Backtest MAPE Comparison")
            st.caption("Lower MAPE = better historical accuracy (but doesn't guarantee future performance!)")
            st.dataframe(cmp_df.style.format({"MAPE": "{:.2f}%"}))

        # Feature importance for Random Forest
        if model_choice == "Random Forest":
            st.markdown("---")
            st.subheader("Feature Importance")
            st.caption(f"Which historical patterns matter most for {target}?")
            
            try:
                # Quick retrain to get importances
                from sklearn.ensemble import RandomForestRegressor
                
                s_prep = series.copy()
                if use_log:
                    s_prep = np.log(s_prep)
                
                freq = "D" if use_calendar else "B"
                s_prep = s_prep.asfreq(freq).ffill()
                
                X, y, _ = _build_xy(s_prep, lags=lags)
                
                temp_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=max_depth if 'max_depth' in locals() else None,
                    min_samples_leaf=min_samples_leaf if 'min_samples_leaf' in locals() else 1,
                    random_state=0,
                    n_jobs=-1
                )
                temp_model.fit(X, y)
                
                feature_names = [f"Lag {i}" for i in range(1, lags+1)] + ["5-day Mean", "5-day Std"]
                
                imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': temp_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Filter to features with >1% importance for cleaner viz
                imp_df_display = imp_df[imp_df['Importance'] > 0.01].head(10)
                
                if len(imp_df_display) < 3:
                    # Show top 10 even if low importance
                    imp_df_display = imp_df.head(10)
                
                fig_imp = px.bar(
                    imp_df_display,
                    x='Importance',
                    y='Feature',
                    orientation='h'
                )
                fig_imp.update_layout(
                    showlegend=False,
                    height=400,
                    yaxis={'categoryorder': 'total ascending'},
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                
                # Smart interpretation
                top = imp_df.iloc[0]
                st.caption(f"💡 **{top['Feature']}** is most predictive ({top['Importance']:.1%} importance)")
                
                if top['Importance'] > 0.6:
                    st.info(f"ℹ️ {top['Feature']} dominates - suggests strong momentum/trend in {target}")
                
            except Exception as e:
                st.warning(f"Couldn't generate feature importance: {str(e)}")

    with tabs[2]:  # New third tab
        st.subheader("Data Quality Check")
        # Data freshness 
        from datetime import datetime

        st.markdown("### Data Freshness")

        most_recent = prices.index.max()
        days_old = (datetime.now() - most_recent).days

        if days_old <= 1:
            st.success(f"✓ Data is current (last updated: {most_recent.strftime('%Y-%m-%d')})")
        elif days_old <= 7:
            st.info(f"ℹ️ Data is {days_old} days old (last updated: {most_recent.strftime('%Y-%m-%d')})")
        else:
            st.warning(f"⚠️ Data is {days_old} days old. Consider running `python ingest_prices.py` to refresh")
    
        # Show missing data
        missing = prices.isnull().sum()
        if missing.sum() > 0:
            st.warning(f"Found {missing.sum()} missing values")
            st.bar_chart(missing)
        else:
            st.success("No missing data! ✓")
    
        # Show date range coverage
        st.write("**Data Coverage:**")
        for ticker in prices.columns:
            start = prices[ticker].first_valid_index()
            end = prices[ticker].last_valid_index()
            st.write(f"{ticker}: {start.date()} to {end.date()}")
    
        # Show basic stats
        st.write("**Price Statistics:**")
        st.dataframe(prices.describe())       

if __name__ == "__main__":
    main()