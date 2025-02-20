# Tech Trip Prioritization Web App

This project is a web application designed to manage and prioritize tech visits to stores based on various factors such as past ticket history(to be reviewed),last month orders, trip cost, and % of operational robot fleet.

## Features

- **Prioritization Model:** Calculates priority score using normalized metrics (tickets, orders, trip cost, broken robots percentage).
- **Dynamic Scheduling:** Automatically computes adjusted priority and forecast visit dates based on current data.
- **Visit Logging:** Logs forecast and actual visits, computes on-time performance.
- **Reports:** Generates reports by zone with breakdowns (weekly, monthly, yearly) including average delays.
- **Interactive UI:** Update weights, log visits, and update forecasts through an interactive web interface.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
2. **Set up a Python virtual environment (optional but recommended):**
    python -m venv venv
    source venv/bin/activate   # On Windows use: venv\Scripts\activate
3. **Install the required libraries:**
    pip install -r requirements.txt

## Usage

1. **Initialize the Database (if needed):**
    python setup_database.py
2. **Run the Appication:**
    python app.py
    The app will be available at http://127.0.0.1:5000

## Project structure
· app.py – Main Flask application with routes for updating visits,forecasts, and generating reports.

· setup_database.py – Script to initialize or reset the SQLite database.

· data.csv – CSV file containing static store data.

· templates/ – Contains HTML templates (index.html, report.html).

· README.md – This file.

· requirements.txt – Python package dependencies.
 Save requirements.txt in your project folder, and then run:
 pip install -r requirements.txt


