"""
ETL Pipeline — Data Warehousing & Integration
===============================================
Loads fact and dimension tables from the MedSynora DW dataset,
performs multi-table joins, and produces a single Master Analytical View
where each row represents one unique patient encounter.

Tables loaded:
  Facts  — FactEncounter, FactVitals, FactCost
  Dims   — DimPatient, DimDisease, DimInsurance
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "MedSynora DW")


def _path(filename: str) -> str:
    """Return absolute path to a CSV inside the data directory."""
    return os.path.join(DATA_DIR, filename)


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_fact_encounter() -> pd.DataFrame:
    """Load the central fact table."""
    df = pd.read_csv(_path("FactEncounter.csv"))
    # Parse dates
    df["CheckinDate"] = pd.to_datetime(df["CheckinDate"], errors="coerce")
    df["CheckoutDate"] = pd.to_datetime(df["CheckoutDate"], errors="coerce")
    # Calculate Length of Stay in days
    df["LengthOfStay"] = (df["CheckoutDate"] - df["CheckinDate"]).dt.days
    return df


def load_fact_vitals() -> pd.DataFrame:
    """Load vitals and keep only the Admission phase (one row per encounter)."""
    df = pd.read_csv(_path("FactVitals.csv"))
    # Keep admission vitals only — gives one row per Encounter_ID
    admission = df[df["Phase"] == "Admission"].copy()
    admission.drop(columns=["Patient_ID", "Phase"], inplace=True, errors="ignore")
    return admission


def load_fact_cost() -> pd.DataFrame:
    """
    Load cost data.  The raw file stores costs in a *long* format with 17
    cost-type rows per encounter.  We extract only the TotalCost row so each
    encounter has a single cost figure.
    """
    df = pd.read_csv(_path("FactCost.csv"))
    total = df[df["CostType"] == "TotalCost"][["Encounter_ID", "CostAmount"]].copy()
    total.rename(columns={"CostAmount": "Total_Cost"}, inplace=True)
    return total


def load_dim_patient() -> pd.DataFrame:
    """Load patient demographics."""
    df = pd.read_csv(_path("DimPatient.csv"))
    df.rename(columns={
        "First Name": "First_Name",
        "Last Name": "Last_Name",
        "Birth Date": "Birth_Date",
        "Marital Status": "Marital_Status",
        "Blood Type": "Blood_Type",
    }, inplace=True)
    df["Birth_Date"] = pd.to_datetime(df["Birth_Date"], errors="coerce")
    return df


def load_dim_disease() -> pd.DataFrame:
    """Load disease dimension."""
    df = pd.read_csv(_path("DimDisease.csv"))
    df.rename(columns={
        "Admission Diagnosis": "Admission_Diagnosis",
        "Disease Type": "Disease_Type",
        "Disease Severity": "Disease_Severity",
        "Medical Unit": "Medical_Unit",
    }, inplace=True)
    return df


def load_dim_insurance() -> pd.DataFrame:
    """Load insurance dimension."""
    df = pd.read_csv(_path("DimInsurance.csv"))
    df.rename(columns={
        "Insurance Plan Name": "Insurance_Plan",
        "Coverage Limit": "Coverage_Limit",
        "Excluded Treatments": "Excluded_Treatments",
        "Partial Coverage Treatments": "Partial_Coverage_Treatments",
    }, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Master merge
# ---------------------------------------------------------------------------

def build_master_view() -> pd.DataFrame:
    """
    Build the Master Analytical View by joining all tables on their
    foreign keys.  Returns a DataFrame with one row per Encounter_ID.
    """
    print("[ETL] Loading FactEncounter ...")
    enc = load_fact_encounter()

    print("[ETL] Loading FactVitals (Admission phase) ...")
    vitals = load_fact_vitals()

    print("[ETL] Loading FactCost (TotalCost) ...")
    cost = load_fact_cost()

    print("[ETL] Loading DimPatient ...")
    patient = load_dim_patient()

    print("[ETL] Loading DimDisease ...")
    disease = load_dim_disease()

    print("[ETL] Loading DimInsurance ...")
    insurance = load_dim_insurance()

    # --- Merge vitals (1:1 on Encounter_ID) ---
    master = enc.merge(vitals, on="Encounter_ID", how="left")

    # --- Merge cost (1:1 on Encounter_ID) ---
    master = master.merge(cost, on="Encounter_ID", how="left")

    # --- Merge patient demographics ---
    patient_cols = ["Patient_ID", "Gender", "Birth_Date", "Height", "Weight",
                    "Marital_Status", "Blood_Type"]
    master = master.merge(patient[patient_cols], on="Patient_ID", how="left")

    # --- Calculate Patient Age (at time of check-in) ---
    master["Patient_Age"] = (
        (master["CheckinDate"] - master["Birth_Date"]).dt.days / 365.25
    ).round(1)

    # --- Calculate BMI  (Weight in kg, Height in cm) ---
    master["BMI"] = (
        master["Weight"] / ((master["Height"] / 100) ** 2)
    ).round(2)

    # --- Merge disease dimension ---
    disease_cols = ["Disease_ID", "Admission_Diagnosis", "Disease_Type",
                    "Disease_Severity", "Medical_Unit"]
    master = master.merge(disease[disease_cols], on="Disease_ID", how="left")

    # --- Merge insurance dimension ---
    insurance_cols = ["InsuranceKey", "Insurance_Plan", "Coverage_Limit"]
    master = master.merge(insurance[insurance_cols], on="InsuranceKey", how="left")

    # --- Final column selection (keep useful columns, drop raw keys) ---
    keep = [
        "Encounter_ID", "Patient_ID", "Patient_Age", "Gender",
        "Height", "Weight", "BMI", "Marital_Status", "Blood_Type",
        "Patient_Severity_Score", "Disease_Severity",
        "Admission_Diagnosis", "Disease_Type", "Medical_Unit",
        "Insurance_Plan", "Coverage_Limit",
        "HeartRate", "Temperature", "SystolicBP", "DiastolicBP",
        "RespRate", "O2Sat",
        "Total_Cost", "LengthOfStay",
        "CheckinDate", "CheckoutDate",
    ]
    master = master[[c for c in keep if c in master.columns]]

    print(f"[ETL] Master Analytical View: {master.shape[0]:,} rows × {master.shape[1]} columns")
    return master


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = build_master_view()
    print(df.head())
    print(df.dtypes)
    print(df.describe())
