"""
Preprocessing Pipeline
=======================
Transforms the Master Analytical View into a clean, scaled feature matrix
ready for K-Means clustering.

Pipeline steps:
  1. Median imputation for missing vitals / cost
  2. IQR-based outlier removal on Total_Cost
  3. Log-transformation on Total_Cost (right-skewed)
  4. Feature selection for clustering
  5. StandardScaler normalization (μ=0, σ=1)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Features used for clustering
CLUSTER_FEATURES = [
    "Patient_Age",
    "Patient_Severity_Score",
    "Total_Cost_Log",
    "HeartRate",
    "SystolicBP",
    "O2Sat",
]


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 — Median imputation for numeric columns that may have NaN
    (primarily vitals from FactVitals and Total_Cost).
    """
    df = df.copy()
    numeric_cols = [
        "HeartRate", "Temperature", "SystolicBP", "DiastolicBP",
        "RespRate", "O2Sat", "Total_Cost", "BMI", "Patient_Age",
        "Patient_Severity_Score", "Disease_Severity",
    ]
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            df[col].fillna(median_val, inplace=True)
            print(f"  [Impute] {col}: filled {n_missing:,} NaN -> median {median_val:.2f}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, column: str = "Total_Cost") -> pd.DataFrame:
    """
    Step 2 — Remove statistical outliers using the Inter-Quartile Range method.
    Rows outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR] are dropped.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)].copy()
    after = len(df)
    print(f"  [IQR] {column}: removed {before - after:,} outliers "
          f"(bounds: {lower:,.0f} – {upper:,.0f}), {after:,} rows remain")
    return df


def log_transform_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3 — Log-transform Total_Cost to reduce right-skew.
    Uses log1p (log(1+x)) which handles zero values gracefully.
    """
    df = df.copy()
    df["Total_Cost_Log"] = np.log1p(df["Total_Cost"])
    skew_before = df["Total_Cost"].skew()
    skew_after = df["Total_Cost_Log"].skew()
    print(f"  [Log] Total_Cost skew: {skew_before:.3f} -> {skew_after:.3f}")
    return df


def scale_features(df: pd.DataFrame, features: list = None):
    """
    Step 4 & 5 — Select clustering features and apply StandardScaler.

    Returns
    -------
    X_scaled : np.ndarray   — scaled feature matrix  (n_samples × n_features)
    scaler   : StandardScaler — fitted scaler (for inverse-transforming later)
    feature_names : list     — ordered feature names
    """
    if features is None:
        features = CLUSTER_FEATURES

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  [Scale] StandardScaler applied to {len(features)} features")
    return X_scaled, scaler, features


def run_preprocessing(master_df: pd.DataFrame):
    """
    Full preprocessing pipeline.  Returns the cleaned DataFrame,
    the scaled feature matrix, the fitted scaler, and the feature names.
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Imputation
    print("\nStep 1 — Median Imputation")
    df = impute_missing(master_df)

    # Step 2: Outlier removal
    print("\nStep 2 — IQR Outlier Removal (Total_Cost)")
    df = remove_outliers_iqr(df, "Total_Cost")

    # Step 3: Log transform
    print("\nStep 3 — Log Transformation (Total_Cost)")
    df = log_transform_cost(df)

    # Step 4 & 5: Feature selection + Scaling
    print("\nStep 4/5 — Feature Selection & StandardScaler")
    X_scaled, scaler, feature_names = scale_features(df)

    print(f"\n[Preprocessing Complete] Final shape: {X_scaled.shape}")
    return df, X_scaled, scaler, feature_names
