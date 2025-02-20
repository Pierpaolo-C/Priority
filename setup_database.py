import sqlite3
import pandas as pd

DB_FILE = "last_visits.db"
DATA_FILE = "data.csv"

# For testing, set TESTING_MODE to True to drop existing tables.
TESTING_MODE = False

# Connect to SQLite database
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

if TESTING_MODE:
    # Drop existing tables to start fresh (for testing only)
    cursor.execute("DROP TABLE IF EXISTS visit_log;")
    cursor.execute("DROP TABLE IF EXISTS visits;")

# Create the visits table (this holds the current state for each store)
cursor.execute("""
CREATE TABLE IF NOT EXISTS visits (
    Site TEXT PRIMARY KEY,
    Last_Visit_Date TEXT,
    Operational_Robots INTEGER
)
""")

# Create the visit_log table (this stores the historical records of visits)
cursor.execute("""
CREATE TABLE IF NOT EXISTS visit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Site TEXT,
    Zone TEXT,
    Visit_Date TEXT,       -- Actual visit date (NULL if not yet visited)\n    Forecast_Date TEXT,    -- Forecasted visit date\n    Trip_Cost REAL,        -- Trip cost (recorded as the maximum cost for that visit)\n    Robots_Fixed INTEGER,  -- Number of robots fixed during that visit\n    Delay_Days INTEGER     -- Calculated delay in days (actual visit date - forecast date)\n)
""")

# Load initial static data from CSV
try:
    data = pd.read_csv(DATA_FILE)
except Exception as e:
    print("Error loading data.csv:", e)
    data = pd.DataFrame(columns=["Site", "Tickets", "Orders", "TripCost", "Zone", "Total_Robots"])

# Insert each site from the CSV into the visits table if it doesn't exist.
for _, row in data.iterrows():
    site = row["Site"]
    total_robots = row["Total_Robots"]  # Total fleet size from CSV
    cursor.execute("SELECT Site FROM visits WHERE Site = ?", (site,))
    existing = cursor.fetchone()
    if not existing:
        cursor.execute("""
            INSERT INTO visits (Site, Last_Visit_Date, Operational_Robots)
            VALUES (?, ?, ?)
        """, (site, None, total_robots))

conn.commit()
conn.close()

print("Database reset and setup complete! ✅")
