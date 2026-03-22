"""
Data Preprocessing + Feature Engineering for Lead Scoring
=========================================================
Cleans the raw Kaggle data, handles missing values,
encodes categoricals, and engineers new features.
"""

import pandas as pd
import numpy as np


def load_and_clean(filepath="data/Lead_Scoring.csv"):
    """Load raw CSV and perform all cleaning steps."""
    df = pd.read_csv(filepath)
    
    # --- DROP USELESS COLUMNS ---
    # ID columns (not predictive)
    drop_ids = ["Prospect ID", "Lead Number"]
    
    # Single-value columns (zero information)
    drop_single = [
        "Magazine",
        "Receive More Updates About Our Courses",
        "Update me on Supply Chain Content",
        "Get updates on DM Content",
        "I agree to pay the amount through cheque",
    ]
    
    # Asymmetrique columns (45% missing, unclear value)
    drop_asymmetrique = [
        "Asymmetrique Activity Index",
        "Asymmetrique Profile Index",
        "Asymmetrique Activity Score",
        "Asymmetrique Profile Score",
    ]
    
    # Other high-missing or low-info columns
    drop_other = [
        "How did you hear about X Education",  # 24% missing, noisy
        "What matters most to you in choosing a course",  # 29% missing, low variance
    ]
    
    all_drops = drop_ids + drop_single + drop_asymmetrique + drop_other
    df = df.drop(columns=all_drops, errors="ignore")
    
    # --- HANDLE 'Select' VALUES ---
    # 'Select' is a placeholder = missing data
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace("Select", np.nan)
    
    # --- FILL MISSING VALUES ---
    # Categorical: fill with 'Unknown'
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")
    
    # Numerical: fill with median
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if col != "Converted":  # don't touch the target
            df[col] = df[col].fillna(df[col].median())
    
    return df


def engineer_features(df):
    """Create new features that capture buying signals."""
    
    # 1. Engagement score (time × page views = composite engagement)
    df["engagement_score"] = (
        df["Total Time Spent on Website"] * df["Page Views Per Visit"]
    )
    
    # 2. Is referred (highest converting sources)
    high_conv_sources = ["Reference", "Welingak Website"]
    df["is_referred"] = df["Lead Source"].isin(high_conv_sources).astype(int)
    
    # 3. Is working professional (strong conversion signal from EDA)
    df["is_working_professional"] = (
        df["What is your current occupation"] == "Working Professional"
    ).astype(int)
    
    # 4. High activity flag (activities that correlate with conversion)
    high_activities = ["SMS Sent", "Had a Phone Conversation"]
    df["is_high_activity"] = df["Last Activity"].isin(high_activities).astype(int)
    
    # 5. Website engagement bucket
    df["website_engagement_level"] = pd.cut(
        df["Total Time Spent on Website"],
        bins=[-1, 0, 60, 300, 1000, float("inf")],
        labels=["zero", "bounce", "browser", "engaged", "power_user"]
    )
    
    # 6. Lead quality numeric (ordinal encode)
    quality_map = {
        "Worst": 0,
        "Not Sure": 1,
        "Unknown": 2,
        "Might be": 3,
        "Low in Relevance": 4,
        "High in Relevance": 5,
    }
    df["lead_quality_numeric"] = df["Lead Quality"].map(quality_map).fillna(2)
    
    # 7. Source conversion rate (historical rate per source — like Rokas's enrichment)
    source_rates = df.groupby("Lead Source")["Converted"].transform("mean")
    df["source_historical_conv_rate"] = source_rates
    
    return df


def prepare_for_modeling(df):
    """Convert all features to model-ready format."""
    # Separate target
    y = df["Converted"].copy()
    X = df.drop(columns=["Converted"])
    
    # Convert Yes/No binary columns to 0/1
    binary_cols = ["Do Not Email", "Do Not Call", "Search", "Newspaper Article",
                   "X Education Forums", "Newspaper", "Digital Advertisement",
                   "Through Recommendations", "A free copy of Mastering The Interview"]
    for col in binary_cols:
        if col in X.columns:
            X[col] = (X[col] == "Yes").astype(int)
    
    # Convert remaining categoricals to pandas category dtype
    # (XGBoost can handle these natively with enable_categorical=True)
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    # Also include the engineered bucket
    if "website_engagement_level" in X.columns:
        cat_cols.append("website_engagement_level")
    
    for col in cat_cols:
        X[col] = X[col].astype("category")
    
    return X, y


def run_preprocessing():
    """Full pipeline: load → clean → engineer → prepare."""
    print("Loading and cleaning data...")
    df = load_and_clean()
    print(f"  After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("Engineering features...")
    df = engineer_features(df)
    print(f"  After feature engineering: {df.shape[1]} columns")
    
    print("Preparing for modeling...")
    X, y = prepare_for_modeling(df)
    
    print(f"\nFinal dataset:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Target balance: {y.mean():.2%} converted")
    
    print(f"\nFeature types:")
    print(f"  Numeric: {X.select_dtypes(include=['number']).shape[1]}")
    print(f"  Categorical: {X.select_dtypes(include=['category']).shape[1]}")
    
    print(f"\nAll features:")
    for i, col in enumerate(X.columns):
        print(f"  {i+1:2d}. {col} ({X[col].dtype})")
    
    return X, y


if __name__ == "__main__":
    X, y = run_preprocessing()
