import sqlite3
import pandas as pd
from flask import Flask, render_template, request, jsonify
from datetime import datetime

DB_FILE = "last_visits.db"  # SQLite database file
data_file = "data.csv"  # Static data file

app = Flask(__name__)

def init_db():
    """Initialize the database if it doesn't exist."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                site TEXT PRIMARY KEY,
                last_visit_date TEXT,
                operational_robots INTEGER
            )
        """)
        conn.commit()

def load_data():
    """Load data from CSV and merge with database values."""
    df = pd.read_csv(data_file)
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        visits = pd.read_sql("SELECT * FROM visits", conn)
    
    # Merge database values into main dataframe
    df = df.merge(visits, on="site", how="left")
    df["last_visit_date"] = pd.to_datetime(df["last_visit_date"])
    df["days_since_last_visit"] = (datetime.today() - df["last_visit_date"]).dt.days
    return df

def update_visit(site):
    """Update last visit date for a given site."""
    today = datetime.today().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO visits (site, last_visit_date)
            VALUES (?, ?) ON CONFLICT(site) DO UPDATE SET last_visit_date=?
        """, (site, today, today))
        conn.commit()

def update_robots(site, change):
    """Increase or decrease the number of operational robots."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT operational_robots FROM visits WHERE site = ?", (site,))
        result = cursor.fetchone()
        if result:
            new_value = max(0, result[0] + change)
            cursor.execute("UPDATE visits SET operational_robots = ? WHERE site = ?", (new_value, site))
        else:
            new_value = max(0, change)
            cursor.execute("INSERT INTO visits (site, operational_robots) VALUES (?, ?)", (site, new_value))
        conn.commit()
    return new_value

@app.route("/")
def index():
    data = load_data()
    return render_template("index.html", data=data.to_dict(orient="records"))

@app.route("/visit", methods=["POST"])
def record_visit():
    site = request.json["site"]
    update_visit(site)
    return jsonify({"success": True})

@app.route("/update_robots", methods=["POST"])
def modify_robots():
    site = request.json["site"]
    change = int(request.json["change"])
    new_value = update_robots(site, change)
    return jsonify({"success": True, "new_value": new_value})

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
