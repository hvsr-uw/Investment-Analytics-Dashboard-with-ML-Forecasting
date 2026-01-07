"""
Quick test script to verify yfinance is working and inspect data structure.

I made this when debugging column naming issues - yfinance can return 
different structures depending on how you call it. Keeping it around 
in case I need to troubleshoot data issues later.
"""

import yfinance as yf
import pandas as pd

# Download a month of AAPL data as a test
df = yf.download(
    "AAPL", 
    period="1mo", 
    interval="1d", 
    auto_adjust=False,  # keeps raw OHLC separate from adjusted
    progress=False
)

print("First few rows:")
print(df.head())

print("\nColumn names:", df.columns)
print("Index type:", type(df.index), "| Index name:", df.index.name)

# If this prints correctly, yfinance is working and we know what structure to expect