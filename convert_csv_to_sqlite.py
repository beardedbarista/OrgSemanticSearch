import pandas as pd
import sqlite3

CSV_PATH = "/home/codebrewx/myenv/FastAPI_TEST/organizations-100.csv"
DB_PATH = "organizations.db"

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

print(f"Converting {len(df)} rows to SQLite database...")
conn = sqlite3.connect(DB_PATH)
df.to_sql("organizations", conn, if_exists="replace", index=False)

# Create indexes for faster search
conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON organizations(Name)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_country ON organizations(Country)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_industry ON organizations(Industry)")

conn.close()
print(f"Success! Database saved as {DB_PATH}")
print("   → Table name: organizations")
print("   → Run: sqlite3 organizations.db \".schema\" to see it")