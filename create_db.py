"""
Simple script to initialize the database.

Just reads schema.sql and creates the tables. Run this once before ingesting data.
"""

import sqlite3
from pathlib import Path

def main():
    db_path = Path("market.db")
    schema_path = Path("schema.sql")

    # Create/connect to DB file
    conn = sqlite3.connect(db_path)

    # Execute the schema file to create tables
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
        conn.executescript(schema_sql)

    conn.commit()
    conn.close()
    
    print("✅ Database initialized successfully!")
    print(f"   Location: {db_path.absolute()}")
    print("   Tables created: assets, prices, benchmarks, portfolios, portfolio_weights")
    print("\nNext step: Run 'python ingest_prices.py' to populate with data")

if __name__ == "__main__":
    main()