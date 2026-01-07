"""
Data ingestion script - pulls stock price history from Yahoo Finance and stores in SQLite.

Run this whenever you want to update prices or add new tickers.
Takes a few seconds per ticker depending on how much history you're grabbing.
"""

import sqlite3
from pathlib import Path
import pandas as pd
import yfinance as yf

DB = "market.db"

def upsert_asset(conn, ticker, name=None):
    """
    Add a ticker to the assets table if it doesn't exist yet.
    Returns the asset_id either way.
    """
    cur = conn.cursor()
    # INSERT OR IGNORE is a neat SQLite trick - only inserts if not already there
    cur.execute(
        "INSERT OR IGNORE INTO assets(ticker, name) VALUES(?, ?)", 
        (ticker.upper(), name)
    )
    conn.commit()
    
    # Grab the id for this ticker
    cur.execute("SELECT id FROM assets WHERE ticker = ?", (ticker.upper(),))
    return cur.fetchone()[0]

def upsert_prices(conn, asset_id, df):
    """
    Insert price data into the DB. Uses INSERT OR IGNORE so running this
    multiple times won't create duplicates (thanks to UNIQUE constraint on asset_id + date).
    
    Returns number of rows actually inserted.
    """
    # Sanity check
    if df is None or df.empty:
        print(f"  -> No data returned for asset_id={asset_id}")
        return 0

    # yfinance sometimes returns multi-level columns when downloading multiple tickers
    # For single ticker, we just need the first level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Move date from index to column
    df = df.reset_index()

    # Normalize column names (spaces to underscores, lowercase)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Make sure we have a date column
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")  # SQLite likes text dates

    # Handle adj_close naming variations
    if "adj_close" not in df.columns and "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})

    # Ensure all expected numeric columns exist
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col not in df.columns:
            df[col] = 0  # fill missing with 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Prepare rows for batch insert
    rows = list(
        df[["date", "open", "high", "low", "close", "adj_close", "volume"]]
        .itertuples(index=False, name=None)
    )

    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR IGNORE INTO prices(asset_id, date, open, high, low, close, adj_close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [(asset_id, d, o, h, l, c, ac, int(v)) for (d, o, h, l, c, ac, v) in rows],
    )
    conn.commit()
    
    # rowcount tells us how many were actually inserted (excludes duplicates)
    inserted = cur.rowcount if cur.rowcount is not None else 0
    print(f"  -> Inserted {inserted} new rows for asset_id={asset_id}")
    return inserted

def download_and_store(tickers, period="5y", interval="1d"):
    """
    Main ingestion loop. Downloads data for each ticker and stores it.
    
    Args:
        tickers: list of ticker symbols
        period: how much history to grab (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: bar size (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    """
    conn = sqlite3.connect(DB)
    total_inserted = 0
    
    for ticker in tickers:
        # Try to get a nice name for the ticker
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            name = info.get("shortName") or info.get("longName") or ticker.upper()
        except Exception:
            # If API call fails, just use the ticker as name
            name = ticker.upper()

        # Make sure ticker exists in assets table
        asset_id = upsert_asset(conn, ticker, name)
        
        print(f"\nDownloading {ticker}...")
        
        # Download price data
        # auto_adjust=False gives us raw OHLC + separate adjusted close
        # group_by="column" flattens the columns (useful for multi-ticker downloads)
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,  # suppress the progress bar
            group_by="column",
        )
        
        print(f"  columns: {list(df.columns)}  |  rows: {len(df)}")
        
        # Store it
        rows_added = upsert_prices(conn, asset_id, df)
        total_inserted += rows_added

    conn.close()
    print(f"\n✅ Ingestion complete! Total new rows: {total_inserted}")

if __name__ == "__main__":
    print("=" * 60)
    print("Running:", __file__)
    print("CWD:", Path().resolve())
    print("=" * 60)
    
    # Default tickers - big tech stocks that everyone watches
    default_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
    
    # Also grab S&P 500 as benchmark (ticker is ^GSPC)
    benchmark = ["^GSPC"]
    
    all_tickers = default_tickers + benchmark
    
    print(f"\nTickers to download: {all_tickers}")
    print("Period: 5 years, daily bars\n")
    
    # TODO: Could make this interactive or read from a config file
    # For now just hardcoded
    
    download_and_store(all_tickers, period="5y", interval="1d")