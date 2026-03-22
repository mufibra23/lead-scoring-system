"""
Project 4: Lead Scoring System — Exploratory Data Analysis
==========================================================
This script explores the Kaggle Lead Scoring Dataset (~9,240 leads)
to understand patterns, data quality, and feature importance before modeling.

Run: python notebooks/01_eda.py
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("data/Lead_Scoring.csv")
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"Conversion rate: {df['Converted'].mean():.2%}")
print(f"  - Converted (1): {df['Converted'].sum():,}")
print(f"  - Not converted (0): {(df['Converted'] == 0).sum():,}")

# ============================================================
# 2. DATA QUALITY CHECK
# ============================================================
print("\n" + "=" * 60)
print("DATA QUALITY")
print("=" * 60)

# Missing values
print("\nColumns with missing values:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, count in missing.items():
    print(f"  {col}: {count:,} missing ({count/len(df):.1%})")

# Columns with only 1 unique value (zero information)
print("\nUseless columns (only 1 unique value — will DROP):")
useless = [col for col in df.columns if df[col].nunique() <= 1]
for col in useless:
    print(f"  {col}: always '{df[col].unique()[0]}'")

# 'Select' placeholder values (= missing in disguise)
print("\nColumns with 'Select' placeholder values (= hidden missing):")
for col in df.columns:
    if df[col].dtype == "object":
        select_count = (df[col] == "Select").sum()
        if select_count > 0:
            print(f"  {col}: {select_count:,} rows ({select_count/len(df):.1%})")

# ID columns (not predictive)
print("\nID columns (will DROP — not predictive):")
print("  Prospect ID: unique identifier per lead")
print("  Lead Number: sequential number")

# ============================================================
# 3. CONVERSION RATES BY KEY FEATURES
# ============================================================
print("\n" + "=" * 60)
print("CONVERSION RATES BY FEATURE")
print("=" * 60)

key_cats = [
    "Lead Source", "Lead Origin", "Last Activity",
    "What is your current occupation", "Lead Quality",
    "Lead Profile", "City", "Tags"
]

for col in key_cats:
    print(f"\n--- {col} ---")
    grouped = df.groupby(col)["Converted"].agg(["mean", "count"])
    grouped = grouped.sort_values("mean", ascending=False)
    for idx, row in grouped.iterrows():
        if row["count"] >= 10:  # only show groups with enough data
            bar = "█" * int(row["mean"] * 20)
            print(f"  {idx:40s} {row['mean']:6.1%} ({int(row['count']):,} leads) {bar}")

# ============================================================
# 4. NUMERIC FEATURES — CONVERTED VS NOT
# ============================================================
print("\n" + "=" * 60)
print("NUMERIC FEATURES: CONVERTED vs NOT CONVERTED")
print("=" * 60)

numeric_cols = ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit",
                "Asymmetrique Activity Score", "Asymmetrique Profile Score"]

for col in numeric_cols:
    conv = df[df["Converted"] == 1][col]
    not_conv = df[df["Converted"] == 0][col]
    print(f"\n{col}:")
    print(f"  Converted:     mean={conv.mean():.1f}, median={conv.median():.1f}")
    print(f"  Not converted: mean={not_conv.mean():.1f}, median={not_conv.median():.1f}")
    diff = conv.mean() - not_conv.mean()
    print(f"  Difference:    {diff:+.1f} ({'higher for converted' if diff > 0 else 'lower for converted'})")

# ============================================================
# 5. KEY INSIGHTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("KEY INSIGHTS FOR MODELING")
print("=" * 60)
print("""
1. CONVERSION RATE: 38.5% — moderately balanced, no extreme imbalance.
   Stratified split recommended but no need for SMOTE.

2. STRONGEST PREDICTORS (by conversion rate difference):
   - Total Time Spent on Website: Converted leads spend 2.2x more time
   - Lead Source: 'Reference' (91.8%) and 'Welingak Website' (98.6%) are very high
   - Tags: 'Already a student' and 'Closed by Horizzon' have very different rates
   - Last Activity: 'SMS Sent' and 'Email Opened' show different patterns

3. DATA CLEANING NEEDED:
   - 5 useless columns (single value) → DROP
   - 2 ID columns → DROP
   - 'Select' values in Lead Profile (4,146 rows) → treat as 'Unknown'
   - Several columns with >25% missing → fill or drop depending on importance
   - Asymmetrique columns have 45% missing → consider dropping or creating a 'has_score' flag

4. FEATURE ENGINEERING IDEAS:
   - engagement_score = Total Time Spent × Page Views Per Visit
   - is_referred = 1 if Lead Source in ['Reference', 'Welingak Website']
   - is_high_activity = based on Last Activity type
   - source_conversion_rate = historical conversion rate of each Lead Source
""")

print("EDA complete. Next step: src/data_preprocessing.py")
