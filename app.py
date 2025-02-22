import pandas as pd
import sqlite3
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from datetime import datetime, timedelta
import math

app = Flask(__name__)

DB_FILE = "last_visits.db"      # SQLite database file
DATA_FILE = "data.csv"           # CSV file for static store data

# -------------------------------
# Disable Caching so data always refreshes
# -------------------------------
@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response

# -------------------------------
# Data Loading Functions
# -------------------------------

def load_data():
    """Load static store data from CSV."""
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("Warning: data.csv not found!")
        df = pd.DataFrame(columns=["Site", "Tickets", "Orders", "TripCost", "Zone", "Total_Robots"])
    return df

def fetch_last_visits():
    """Fetch last visit data from the visits table in the database."""
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT Site, Last_Visit_Date, Operational_Robots FROM visits"
    last_visits = pd.read_sql_query(query, conn)
    conn.close()
    if not last_visits.empty:
        last_visits["Last_Visit_Date"] = pd.to_datetime(last_visits["Last_Visit_Date"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        last_visits["Days Since Last Visit"] = (today - last_visits["Last_Visit_Date"]).dt.days
        return last_visits[["Site", "Last_Visit_Date", "Days Since Last Visit", "Operational_Robots"]]
    else:
        return pd.DataFrame(columns=["Site", "Last_Visit_Date", "Days Since Last Visit", "Operational_Robots"])

def get_combined_data():
    """Merge static CSV data with dynamic visit data from the database.
       If no visit exists, leave Last_Visit_Date as NaT and set Days Since Last Visit to 0.
    """
    data = load_data()  # static data
    last_visits = fetch_last_visits()  # dynamic data from visits table
    merged_df = data.merge(last_visits, on="Site", how="left")
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Convert Last_Visit_Date to datetime and fill missing with NaT (do not fill with today)
    merged_df["Last_Visit_Date"] = pd.to_datetime(merged_df["Last_Visit_Date"], errors="coerce")
    today_ts = pd.Timestamp.today().normalize()

    # Compute days since last visit: if no visit exists (NaT), we use 0 for calculation.
    merged_df["Days Since Last Visit"] = (today_ts - merged_df["Last_Visit_Date"]).dt.days
    merged_df["Days Since Last Visit"] = pd.to_numeric(merged_df["Days Since Last Visit"], errors="coerce").fillna(0)

    # Ensure Operational_Robots is set properly
    if "Operational_Robots" not in merged_df.columns:
        merged_df["Operational_Robots"] = merged_df["Total_Robots"]
    else:
        merged_df["Operational_Robots"] = pd.to_numeric(merged_df["Operational_Robots"], errors="coerce").fillna(merged_df["Total_Robots"])

    # Create a display column for Last Visit Date:
    merged_df["Last_Visit_Date_str"] = merged_df["Last_Visit_Date"].apply(
        lambda d: d.strftime("%Y.%m.%d") if pd.notnull(d) else "No Visit Yet"
    )
    
    return merged_df

# -------------------------------
# Global Defaults and Helper Functions
# -------------------------------

# Default weights for the four factors: Tickets, Orders, TripCost, Broken Robots Percentage.
weights = {"w1": 0, "w2": 0.35, "w3": 0.3, "w4": 0.35}

def forecast_days(adjusted_priority):
    """Map an adjusted priority score to forecast days (business days)."""
    if adjusted_priority >= 0.50:
        return 3  # Urgent
    elif adjusted_priority >= 0.4:
        return 5  # High
    elif adjusted_priority >= 0.25:
        return 10 # Medium
    else:
        return 20 # Low

def get_priority_level(adjusted_priority):
    """Map an adjusted priority score to a readable priority level."""
    if adjusted_priority >= 0.50:
        return "Urgent"
    elif adjusted_priority >= 0.4:
        return "High"
    elif adjusted_priority >= 0.25:
        return "Medium"
    else:
        return "Low"

def compute_priority(df):
    """
    Compute dynamic columns:
      NTU, ND, NTC, Broken_Perc, Priority, Adjusted Priority,
      Forecast Days, and Next Visit Forecast (using business days).
    """
    # Normalize static factors:
    df["NTU"] = df["Tickets"] / df["Tickets"].max() if df["Tickets"].max() > 0 else 0
    df["ND"] = df["Orders"] / df["Orders"].max() if df["Orders"].max() > 0 else 0
    df["NTC"] = 1 - (df["TripCost"] / df["TripCost"].max()) if df["TripCost"].max() > 0 else 0

    def safe_convert(x):
        try:
            if isinstance(x, bytes):
                x = x.decode("utf-8")
            return int(float(x))
        except (ValueError, TypeError):
            return 0

    df["Operational_Robots"] = df["Operational_Robots"].apply(safe_convert)
    df["Total_Robots"] = df["Total_Robots"].apply(safe_convert)

    # Calculate Broken Robots Percentage:
    df["Broken_Perc"] = 1 - (df["Operational_Robots"] / df["Total_Robots"].replace(0, 1))
    df["Broken_Perc"] = df["Broken_Perc"].round(2)

    # Base Priority:
    df["Priority"] = (weights["w1"] * df["NTU"] +
                      weights["w2"] * df["ND"] +
                      weights["w3"] * df["NTC"] +
                      weights["w4"] * df["Broken_Perc"])
    df["Priority"] = df["Priority"].fillna(0)

    # Adjusted Priority: boost by 1% per day since last visit
    df["Adjusted Priority"] = df["Priority"] * (1 + df["Days Since Last Visit"] / 100)

    # Forecast next visit using business days:
    df["Forecast Days"] = df["Adjusted Priority"].apply(forecast_days)
    df["Next Visit Forecast"] = df["Forecast Days"].apply(lambda x: pd.Timestamp("today") + pd.offsets.BDay(int(x)))
    # Format Next Visit Forecast as string:
    df["Next Visit Forecast"] = df["Next Visit Forecast"].apply(lambda ts: ts.strftime('%Y-%m-%d') if pd.notnull(ts) else "")

    # Compute Priority Level:
    df["Priority Level"] = df["Adjusted Priority"].apply(get_priority_level)
    
    return df

def get_updated_priority_data():
    """
    Recalculate dynamic data and return a list of dictionaries with keys in lower-case camelCase,
    including onTimeStatus.
    """
    df = get_combined_data()
    df = compute_priority(df)
    updated_data = []
    for _, row in df.iterrows():
        updated_data.append({
            "site": row.get("Site", ""),
            "priority": round(row.get("Priority", 0), 2),
            "adjustedPriority": round(row.get("Adjusted Priority", 0), 2),
            "daysSinceLastVisit": int(row.get("Days Since Last Visit", 0)),
            "nextVisitForecast": row.get("Next Visit Forecast", ""),
            "lastVisitDate": row.get("Last_Visit_Date_str", ""),
            "priorityLevel": get_priority_level(row.get("Adjusted Priority", 0))
        })
    # Get on-time performance data and attach it:
    visit_performance = get_visit_data()
    for record in updated_data:
        record["onTimeStatus"] = visit_performance.get(record["site"], "No Data")
    return updated_data

def get_visit_data():
    """
    Retrieve on-time performance data from visit_log.
    Compares Forecast_Date with Visit_Date (if available) for each record.
    Returns a dictionary mapping each site to its latest on-time status.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT Site, Visit_Date, Forecast_Date 
        FROM visit_log
    """)
    visits = cursor.fetchall()
    conn.close()

    visit_dict = {}
    for site, visit_date, forecast_date in visits:
        if visit_date and forecast_date:
            try:
                visit_dt = datetime.strptime(visit_date, "%Y-%m-%d")
                forecast_dt = datetime.strptime(forecast_date, "%Y-%m-%d")
            except Exception:
                status = "⏳ Pending"
            else:
                if visit_dt == forecast_dt:
                    status = "✅ On Time"
                elif visit_dt > forecast_dt:
                    status = "⏳ Late"
                else:
                    status = "⏩ Early"
        else:
            status = "⏳ Pending"
        # For simplicity, we override with the latest status for each site.
        visit_dict[site] = status
    return visit_dict

# -------------------------------
# Flask Routes
# -------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    global weights
    if request.method == "POST":
        try:
            new_w1 = float(request.form.get("w1"))
            new_w2 = float(request.form.get("w2"))
            new_w3 = float(request.form.get("w3"))
            new_w4 = float(request.form.get("w4"))
            if round(new_w1 + new_w2 + new_w3 + new_w4, 2) != 1.00:
                return render_template("index.html", data=compute_priority(get_combined_data()).to_dict(orient="records"), weights=weights, error="Weights must sum to 1.")
            weights["w1"], weights["w2"], weights["w3"], weights["w4"] = new_w1, new_w2, new_w3, new_w4
        except ValueError:
            return render_template("index.html", data=compute_priority(get_combined_data()).to_dict(orient="records"), weights=weights, error="Invalid input! Enter valid numbers.")

    df = get_combined_data()
    df = compute_priority(df)
    # Attach on-time performance data
    visit_performance = get_visit_data()
    df["On-Time Performance"] = df["Site"].map(visit_performance).fillna("No Data")
    return render_template("index.html", data=df.to_dict(orient="records"), weights=weights, error=None)

@app.route("/update_robots", methods=["POST"])
def modify_robots():
    site = request.form.get("site")
    change = request.form.get("change")
    if not site or change is None:
        return jsonify({"success": False, "error": "Missing site or change value"}), 400
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            static_data = load_data()
            total_arr = static_data.loc[static_data["Site"] == site, "Total_Robots"].values
            total_robots = int(total_arr[0]) if len(total_arr) > 0 else 0
            if change == "reset":
                new_value = total_robots
            else:
                change = int(change)
                cursor.execute("SELECT Operational_Robots FROM visits WHERE Site = ?", (site,))
                result = cursor.fetchone()
                current_value = int(result[0]) if result else total_robots
                new_value = min(max(current_value + change, 0), total_robots)
            cursor.execute("UPDATE visits SET Operational_Robots = ? WHERE Site = ?", (new_value, site))
            conn.commit()
        df = get_combined_data()
        df = compute_priority(df)
        updated_row = df[df["Site"] == site].iloc[0]
        return jsonify({
            "success": True,
            "new_value": int(new_value),
            "priority": float(round(updated_row["Priority"], 2)),
            "adjustedPriority": float(round(updated_row["Adjusted Priority"], 2)),
            "daysSinceLastVisit": int(updated_row["Days Since Last Visit"]),
            "nextVisitForecast": updated_row["Next Visit Forecast"],
            "lastVisitDate": updated_row.get("Last_Visit_Date_str", ""),
            "priorityLevel": get_priority_level(updated_row["Adjusted Priority"])
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/refresh_priority', methods=['GET'])
def refresh_priority():
    updated_data = get_updated_priority_data()
    return jsonify(updated_data)

@app.route('/update_forecast', methods=['POST'])
def update_forecast():
    data = request.json
    site = data.get("site")
    if not site:
        return jsonify({"success": False, "error": "Missing site parameter"}), 400

    # Recalculate forecast for this site using current data:
    df = get_combined_data()
    df = compute_priority(df)
    site_row = df[df["Site"] == site]
    if site_row.empty:
        return jsonify({"success": False, "error": "Site not found"}), 404

    # Compute new forecast days from current state:
    new_forecast_days = site_row.iloc[0]["Forecast Days"]
    
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # Look for an open forecast record (one with Visit_Date IS NULL)
        cursor.execute("""
            SELECT id, Forecast_Date FROM visit_log 
            WHERE Site = ? AND Visit_Date IS NULL 
            ORDER BY Forecast_Date ASC LIMIT 1
        """, (site,))
        row = cursor.fetchone()

        today_norm = pd.Timestamp.today().normalize()
        if row:
            forecast_id, base_forecast_str = row
            # Parse the existing forecast date as the base date.
            try:
                base_forecast_date = pd.to_datetime(base_forecast_str)
            except Exception:
                return jsonify({"success": False, "error": "Invalid base forecast date"}), 500

            # Ensure the base date is not in the past relative to today.
            base_date = max(base_forecast_date, today_norm)
            
            # Calculate the new forecast date from the (adjusted) base date using business days.
            new_forecast_date = base_date + pd.offsets.BDay(int(new_forecast_days))
            new_forecast_str = new_forecast_date.strftime('%Y-%m-%d')
            # Update the existing forecast record with the new forecast date.
            cursor.execute("""
                UPDATE visit_log SET Forecast_Date = ?
                WHERE id = ?
            """, (new_forecast_str, forecast_id))
        else:
            # No open forecast exists; use today's date as the base.
            base_date = today_norm
            new_forecast_date = base_date + pd.offsets.BDay(int(new_forecast_days))
            new_forecast_str = new_forecast_date.strftime('%Y-%m-%d')
            static_data = load_data()
            site_row_static = static_data.loc[static_data["Site"] == site]
            if site_row_static.empty:
                return jsonify({"success": False, "error": "Site not found in static data"}), 404
            zone = site_row_static["Zone"].values[0]
            trip_cost = float(site_row_static["TripCost"].values[0])
            cursor.execute("""
                INSERT INTO visit_log (Site, Zone, Visit_Date, Forecast_Date, Trip_Cost, Robots_Fixed)
                VALUES (?, ?, NULL, ?, ?, NULL)
            """, (site, zone, new_forecast_str, trip_cost))
        conn.commit()
    return jsonify({"success": True, "new_forecast": new_forecast_str})

@app.route('/log_forecast_visit', methods=['POST'])
def log_forecast_visit():
    data = request.json
    site = data.get("site")
    forecast_date = data.get("forecast_date")
    if not site or not forecast_date:
        return jsonify({"success": False, "error": "Missing site or forecast date"}), 400
    static_data = load_data()
    site_row = static_data.loc[static_data["Site"] == site]
    if site_row.empty:
        return jsonify({"success": False, "error": "Site not found in static data"}), 404
    zone = site_row["Zone"].values[0]
    trip_cost = float(site_row["TripCost"].values[0])
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO visit_log (Site, Zone, Visit_Date, Forecast_Date, Trip_Cost, Robots_Fixed)
            VALUES (?, ?, NULL, ?, ?, NULL)
        """, (site, zone, forecast_date, trip_cost))
        conn.commit()
    return jsonify({"success": True, "message": "Forecast visit logged successfully", "forecast_date": forecast_date})

@app.route('/log_actual_visit', methods=['POST'])
def log_actual_visit():
    data = request.json
    site = data.get("site")
    if not site:
        return jsonify({"success": False, "error": "Missing site name"}), 400
    today = datetime.today().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()

        # Find the most recent forecast visit with Visit_Date IS NULL
        cursor.execute("""
            SELECT id, Forecast_Date FROM visit_log 
            WHERE Site = ? AND Visit_Date IS NULL 
            ORDER BY Forecast_Date DESC LIMIT 1
        """, (site,))
        row = cursor.fetchone()

        if not row:
            # If no forecast visit exists, create one automatically
            static_data = load_data()
            site_row = static_data.loc[static_data["Site"] == site]
            if site_row.empty:
                return jsonify({"success": False, "error": "Site not found in static data"}), 404

            zone = site_row["Zone"].values[0]
            trip_cost = float(site_row["TripCost"].values[0])
            df = get_combined_data()
            df = compute_priority(df)
            new_forecast = df[df["Site"] == site].iloc[0]["Next Visit Forecast"]
            cursor.execute("""
                INSERT INTO visit_log (Site, Zone, Visit_Date, Forecast_Date, Trip_Cost, Robots_Fixed)
                VALUES (?, ?, NULL, ?, ?, NULL)
            """, (site, zone, new_forecast, trip_cost))
            conn.commit()

            # Re-run the query to get the newly created forecast visit
            cursor.execute("""
                SELECT id, Forecast_Date FROM visit_log 
                WHERE Site = ? AND Visit_Date IS NULL 
                ORDER BY Forecast_Date DESC LIMIT 1
            """, (site,))
            row = cursor.fetchone()

        visit_id, forecast_date = row

        # Get total robots for the site from static data
        static_data = load_data()
        total_arr = static_data.loc[static_data["Site"] == site, "Total_Robots"].values
        total_robots = int(total_arr[0]) if len(total_arr) > 0 else 0

        # Get current operational robots from visits table
        cursor.execute("SELECT Operational_Robots FROM visits WHERE Site = ?", (site,))
        result = cursor.fetchone()
        current_op = int(result[0]) if result else total_robots
        fixed = total_robots - current_op

        delay_days = None
        if forecast_date:
            try:
                forecast_date_obj = datetime.strptime(forecast_date, "%Y-%m-%d")
                actual_date_obj = datetime.strptime(today, "%Y-%m-%d")
                delay_days = (actual_date_obj - forecast_date_obj).days
            except Exception:
                delay_days = None

        # Update the visit_log record with actual visit info
        cursor.execute("""
            UPDATE visit_log
            SET Visit_Date = ?, Delay_Days = ?, Robots_Fixed = ?
            WHERE id = ?
        """, (today, delay_days, fixed, visit_id))
        
        # update the visits table with today's visit 
        cursor.execute("""
            UPDATE visits
            SET Last_Visit_Date = ?
            WHERE Site = ?
        """, (today, site))
        
        conn.commit()
    return jsonify({"success": True, "visit_date": today, "robots_fixed": fixed})

@app.route("/report")
def report():
    # Read the visit_log data from the database.
    conn = sqlite3.connect(DB_FILE)
    df_log = pd.read_sql_query("SELECT * FROM visit_log", conn, parse_dates=["Visit_Date"])
    conn.close()
    
    # If there is no visit log data, create an empty report_data structure.
    if df_log.empty:
        report_data = {
            "Grand Total": {
                "weekly": {},
                "monthly": {},
                "yearly": {},
                "overall": {
                    "Visits": 0,
                    "Total Trip Cost": 0,
                    "Robots Fixed": 0,
                    "Average Trip Cost": 0,
                    "Average Delay": 0
                }
            }
        }
        return render_template("report.html", report_data=report_data, error="No visit data available.")
    
    # Drop rows with no actual visit date (we only want completed visits)
    df_log = df_log.dropna(subset=["Visit_Date"])
    df_log.set_index("Visit_Date", inplace=True)
    
    # Group daily by Zone.
    # For each day and zone:
    # - Trip_Cost: maximum cost of that day,
    # - Robots_Fixed: sum,
    # - Visits: count unique sites,
    # - Delay_Days: we'll compute a weighted delay later.
    daily = df_log.groupby(["Zone", pd.Grouper(freq="D")]).agg({
        "Trip_Cost": "max",
        "Robots_Fixed": "sum",
        "Site": pd.Series.nunique,
        "Delay_Days": "mean"   # initial mean (will be replaced by weighted calc)
    }).rename(columns={"Site": "Visits"})
    daily = daily.reset_index()
    
    # Initialize report_data dict.
    report_data = {}
    zones = daily["Zone"].unique()
    
    for zone in zones:
        zone_daily = daily[daily["Zone"] == zone].copy()
        zone_daily.set_index("Visit_Date", inplace=True)
        
        # Resample weekly, monthly, and yearly for basic sums.
        weekly = zone_daily.resample("W").agg({
            "Trip_Cost": "sum",
            "Robots_Fixed": "sum",
            "Visits": "sum"
        })
        monthly = zone_daily.resample("M").agg({
            "Trip_Cost": "sum",
            "Robots_Fixed": "sum",
            "Visits": "sum"
        })
        yearly = zone_daily.resample("Y").agg({
            "Trip_Cost": "sum",
            "Robots_Fixed": "sum",
            "Visits": "sum"
        })
        
        # Compute weighted average Delay_Days for each period.
        # (Multiply each day's delay by its visit count, sum, then divide by total visits)
        weekly_weighted = (zone_daily["Delay_Days"] * zone_daily["Visits"]).resample("W").sum() / zone_daily["Visits"].resample("W").sum()
        monthly_weighted = (zone_daily["Delay_Days"] * zone_daily["Visits"]).resample("M").sum() / zone_daily["Visits"].resample("M").sum()
        yearly_weighted = (zone_daily["Delay_Days"] * zone_daily["Visits"]).resample("Y").sum() / zone_daily["Visits"].resample("Y").sum()
        
        weekly["Delay_Days"] = weekly_weighted.fillna(0)
        monthly["Delay_Days"] = monthly_weighted.fillna(0)
        yearly["Delay_Days"] = yearly_weighted.fillna(0)
        
        # Compute overall zone values using weighted average for Delay_Days.
        overall_visits = int(zone_daily["Visits"].sum())
        overall_trip_cost = zone_daily["Trip_Cost"].sum()
        overall_robots_fixed = int(zone_daily["Robots_Fixed"].sum())
        overall_avg_trip_cost = overall_trip_cost / overall_visits if overall_visits > 0 else 0
        overall_weighted_delay = (zone_daily["Delay_Days"] * zone_daily["Visits"]).sum() / overall_visits if overall_visits > 0 else 0
        
        report_data[zone] = {
            "weekly": weekly.to_dict(orient="index"),
            "monthly": monthly.to_dict(orient="index"),
            "yearly": yearly.to_dict(orient="index"),
            "overall": {
                "Visits": overall_visits,
                "Total Trip Cost": overall_trip_cost,
                "Robots Fixed": overall_robots_fixed,
                "Average Trip Cost": round(overall_avg_trip_cost, 2),
                "Average Delay": round(overall_weighted_delay, 2)
            }
        }
    
    # Grand Total across zones:
    total_visits = 0
    total_trip_cost = 0
    total_robots_fixed = 0
    total_delay = 0  # Sum of (zone weighted delay * zone visits)
    
    for zone_data in report_data.values():
        total_visits += zone_data["overall"]["Visits"]
        total_trip_cost += zone_data["overall"]["Total Trip Cost"]
        total_robots_fixed += zone_data["overall"]["Robots Fixed"]
        total_delay += zone_data["overall"]["Average Delay"] * zone_data["overall"]["Visits"]
    
    grand_avg_trip_cost = total_trip_cost / total_visits if total_visits > 0 else 0
    grand_avg_delay = total_delay / total_visits if total_visits > 0 else 0
    
    # Consolidate breakdowns across zones.
    if report_data:
        all_weekly = [pd.DataFrame.from_dict(z["weekly"], orient="index") for z in report_data.values()]
        all_monthly = [pd.DataFrame.from_dict(z["monthly"], orient="index") for z in report_data.values()]
        all_yearly = [pd.DataFrame.from_dict(z["yearly"], orient="index") for z in report_data.values()]
        grand_weekly = pd.concat(all_weekly).groupby(level=0).sum()
        grand_monthly = pd.concat(all_monthly).groupby(level=0).sum()
        grand_yearly = pd.concat(all_yearly).groupby(level=0).sum()
        
        # Recompute consolidated weighted delay for each period.
        def compute_consolidated_delay(dfs):
            combined = pd.concat(dfs)
            def weighted_avg(df):
                total = df["Visits"].sum()
                return (df["Delay_Days"] * df["Visits"]).sum() / total if total > 0 else 0
            result = combined.groupby(level=0).apply(weighted_avg)
            return result.to_dict()
        
        consolidated_weekly_delay = compute_consolidated_delay(all_weekly)
        consolidated_monthly_delay = compute_consolidated_delay(all_monthly)
        consolidated_yearly_delay = compute_consolidated_delay(all_yearly)
        
        grand_weekly = grand_weekly.assign(Delay_Days=consolidated_weekly_delay)
        grand_monthly = grand_monthly.assign(Delay_Days=consolidated_monthly_delay)
        grand_yearly = grand_yearly.assign(Delay_Days=consolidated_yearly_delay)
    else:
        grand_weekly = pd.DataFrame()
        grand_monthly = pd.DataFrame()
        grand_yearly = pd.DataFrame()
    
    report_data["Grand Total"] = {
        "weekly": grand_weekly.to_dict(orient="index"),
        "monthly": grand_monthly.to_dict(orient="index"),
        "yearly": grand_yearly.to_dict(orient="index"),
        "overall": {
            "Visits": total_visits,
            "Total Trip Cost": total_trip_cost,
            "Robots Fixed": total_robots_fixed,
            "Average Trip Cost": round(grand_avg_trip_cost, 2),
            "Average Delay": round(grand_avg_delay, 2)
        }
    }
    
    return render_template("report.html", report_data=report_data, error=None)
    
@app.route('/download_visit_log')
def download_visit_log():
    # Query the visit_log table
    conn = sqlite3.connect(DB_FILE)
    df_log = pd.read_sql_query("SELECT * FROM visit_log", conn, parse_dates=["Visit_Date"])
    conn.close()
    
    # Convert DataFrame to CSV
    csv_data = df_log.to_csv(index=False)
    
    # Return the CSV file as a download
    return Response(csv_data, mimetype="text/csv",
                    headers={"Content-disposition": "attachment; filename=visit_log.csv"})

@app.route('/download_report')
def download_report():
    # For simplicity, here we'll just export the raw visit_log as a CSV.
    # Alternatively, later, we can aggregate the data and create a more detailed report CSV.
    conn = sqlite3.connect(DB_FILE)
    df_log = pd.read_sql_query("SELECT * FROM visit_log", conn, parse_dates=["Visit_Date"])
    conn.close()
    
    if df_log.empty:
        return "No report data available"
    
    csv_data = df_log.to_csv(index=False)
    return Response(csv_data, mimetype="text/csv",
                    headers={"Content-disposition": "attachment; filename=report.csv"})

if __name__ == "__main__":
    app.run(debug=True)
