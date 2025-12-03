import sqlite3

DB_PATH = "organizations.db"
conn = sqlite3.connect(DB_PATH)

conn.execute('''
CREATE TABLE IF NOT EXISTS filtered_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query_params TEXT,
    total_results INTEGER,
    returned_results INTEGER,
    query_time_ms REAL,
    response_json TEXT
)
''')

conn.execute('''
CREATE TABLE IF NOT EXISTS semantic_search_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query TEXT,
    top_k INTEGER,
    results_count INTEGER,
    query_time_ms REAL,
    response_json TEXT
)
''')

conn.commit()
conn.close()
print(f"Tables 'filtered_queries' and 'semantic_search_queries' added to {DB_PATH}")
               
            
               

