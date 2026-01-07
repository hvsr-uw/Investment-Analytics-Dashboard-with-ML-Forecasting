-- Database schema for the investment analytics app
-- Using SQLite since it's lightweight and perfect for a local dashboard

-- Assets table: stores each ticker we're tracking
CREATE TABLE IF NOT EXISTS assets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker TEXT NOT NULL UNIQUE,  -- ticker symbol (e.g. AAPL, MSFT)
  name TEXT                       -- full company name if available
);

-- Prices table: daily OHLCV data for each asset
CREATE TABLE IF NOT EXISTS prices (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  asset_id INTEGER NOT NULL,
  date TEXT NOT NULL,             -- storing as TEXT in YYYY-MM-DD format
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  adj_close REAL,                 -- adjusted for splits/dividends
  volume INTEGER,
  UNIQUE(asset_id, date),         -- prevent duplicate entries for same asset+date
  FOREIGN KEY(asset_id) REFERENCES assets(id)
);

-- Benchmarks table (optional): track indices separately if needed
-- Right now just storing ^GSPC as a regular asset, but could use this
-- if we wanted more structure
CREATE TABLE IF NOT EXISTS benchmarks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  ticker TEXT NOT NULL
);

-- Portfolio tables: for saving custom portfolio allocations
-- Not using these in the current version, but left them in case I want to
-- add a feature to save/load portfolio configs later

CREATE TABLE IF NOT EXISTS portfolios (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_weights (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  portfolio_id INTEGER NOT NULL,
  asset_id INTEGER NOT NULL,
  weight REAL NOT NULL,           -- weight as decimal (e.g. 0.25 for 25%)
  FOREIGN KEY(portfolio_id) REFERENCES portfolios(id),
  FOREIGN KEY(asset_id) REFERENCES assets(id)
);

-- TODO: Could add an index on prices(date) if queries get slow
-- CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);