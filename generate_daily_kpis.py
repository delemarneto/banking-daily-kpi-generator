import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

states = [
    "NSW",
    "VIC",
    "QLD",
    "WA",
    "SA",
    "TAS",
    "ACT",
    "NT"
]


dates = pd.date_range(
    start="2024-01-01",
    periods=730,
    freq="D"
)

base_customers_map = {
    "NSW": 310000,
    "VIC": 250000,
    "QLD": 190000,
    "WA": 100000,
    "SA": 60000,
    "TAS": 22000,
    "ACT": 30000,
    "NT": 10000
}

frames = []

for state in states:

    base_customers = base_customers_map[state]

    growth = rng.normal(
        loc=22,
        scale=5,
        size=len(dates)
    ).cumsum()

    revenue = (
        2_200_000
        + np.linspace(0, 500_000, len(dates))
        + rng.normal(0, 130_000, len(dates))
    )

    profit = revenue * rng.uniform(
        0.20,
        0.34,
        len(dates)
    )

    deposit_book = (
        4_000_000_000
        + np.linspace(0, 850_000_000, len(dates))
        + rng.normal(0, 65_000_000, len(dates))
    )

    loan_book = (
        4_600_000_000
        + np.linspace(0, 950_000_000, len(dates))
        + rng.normal(0, 70_000_000, len(dates))
    )

    npl_ratio = np.clip(
        rng.normal(
            1.10,
            0.16,
            len(dates)
        )
        + np.linspace(0, 0.12, len(dates)),
        0.4,
        3.0
    )

    nim = rng.normal(
        1.92,
        0.10,
        len(dates)
    )

    cost_to_income = rng.normal(
        44.8,
        2.7,
        len(dates)
    )

    customers_total = (
        base_customers + growth
    ).astype(int)

    new_customers = rng.integers(
        20,
        700,
        len(dates)
    )

    churn_customers = rng.integers(
        5,
        350,
        len(dates)
    )

    digital_logins = rng.integers(
        12000,
        90000,
        len(dates)
    )

    branch_visits = rng.integers(
        700,
        6000,
        len(dates)
    )

    frame = pd.DataFrame({
        "snapshot_date": dates,
        "state": state,
        "customers_total": customers_total,
        "new_customers": new_customers,
        "churn_customers": churn_customers,
        "daily_revenue": np.round(revenue, 2),
        "daily_profit": np.round(profit, 2),
        "deposit_book": np.round(deposit_book, 2),
        "loan_book": np.round(loan_book, 2),
        "nim_pct": np.round(nim, 3),
        "cost_to_income_pct": np.round(cost_to_income, 2),
        "npl_ratio_pct": np.round(npl_ratio, 3),
        "digital_logins": digital_logins,
        "branch_visits": branch_visits,

        "loan_to_deposit_ratio_pct": np.round(
            (loan_book / deposit_book) * 100,
            2
        ),

        "profit_margin_pct": np.round(
            (profit / revenue) * 100,
            2
        ),

        "revenue_per_customer": np.round(
            revenue / customers_total,
            2
        ),

        "digital_adoption_pct": np.round(
            (digital_logins / customers_total) * 100,
            2
        )
    })

    frames.append(frame)

daily_kpis = pd.concat(
    frames,
    ignore_index=True
)

daily_kpis.to_csv(
    OUTPUT_DIR / "daily_kpis.csv",
    index=False
)

print("File generated successfully:")
print(OUTPUT_DIR / "daily_kpis.csv")

print("\nRows:", len(daily_kpis))
print("Columns:", len(daily_kpis.columns))

daily_kpis.head()
