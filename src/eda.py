"""
Exploratory Data Analysis (EDA)
================================
Generates descriptive statistics and professional-quality visualizations
to understand feature distributions, correlations, and class balance
before clustering.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns

# Plot aesthetics
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_eda(df: pd.DataFrame, output_dir: str = "output/plots/eda"):
    """Generate all EDA visualizations and print a statistical summary."""
    ensure_dir(output_dir)

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Descriptive Statistics
    # ---------------------------------------------------------------
    print("\n--- Descriptive Statistics ---")
    desc = df.describe().T
    print(desc.to_string())

    desc.to_csv(os.path.join(output_dir, "descriptive_statistics.csv"))

    # ---------------------------------------------------------------
    # 2. Distribution Histograms — Age, Severity, Cost
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, col, color, title in zip(
        axes,
        ["Patient_Age", "Patient_Severity_Score", "Total_Cost"],
        ["#2196F3", "#FF5722", "#4CAF50"],
        ["Patient Age", "Severity Score", "Total Treatment Cost ($)"],
    ):
        ax.hist(df[col].dropna(), bins=50, color=color, alpha=0.75, edgecolor="white")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("Frequency")
        # Add mean line
        mean_val = df[col].mean()
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2, label=f"Mean: {mean_val:,.1f}")
        ax.legend(fontsize=9)

    fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_distributions.png"))
    plt.close(fig)
    print("  [OK] Distribution histograms saved")

    # ---------------------------------------------------------------
    # 3. Correlation Heatmap
    # ---------------------------------------------------------------
    numeric_cols = [
        "Patient_Age", "Patient_Severity_Score", "Total_Cost",
        "HeartRate", "Temperature", "SystolicBP", "DiastolicBP",
        "RespRate", "O2Sat", "BMI", "LengthOfStay", "Disease_Severity",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.5,
        square=True, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=15)
    fig.savefig(os.path.join(output_dir, "02_correlation_heatmap.png"))
    plt.close(fig)
    print("  [OK] Correlation heatmap saved")

    # ---------------------------------------------------------------
    # 4. Box Plots — Cost by Insurance Plan & Severity by Disease Type
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    if "Insurance_Plan" in df.columns:
        sns.boxplot(data=df, x="Insurance_Plan", y="Total_Cost", ax=axes[0],
                    palette="Set2", showfliers=False)
        axes[0].set_title("Total Cost by Insurance Plan", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Insurance Plan")
        axes[0].set_ylabel("Total Cost ($)")
        axes[0].tick_params(axis="x", rotation=15)

    if "Disease_Type" in df.columns:
        # Show top 8 disease types by frequency
        top_types = df["Disease_Type"].value_counts().head(8).index
        subset = df[df["Disease_Type"].isin(top_types)]
        sns.boxplot(data=subset, x="Disease_Type", y="Patient_Severity_Score",
                    ax=axes[1], palette="Set3", showfliers=False)
        axes[1].set_title("Severity Score by Disease Type (Top 8)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Disease Type")
        axes[1].set_ylabel("Severity Score")
        axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_boxplots.png"))
    plt.close(fig)
    print("  [OK] Box plots saved")

    # ---------------------------------------------------------------
    # 5. Vital Signs Panel
    # ---------------------------------------------------------------
    vitals = ["HeartRate", "Temperature", "SystolicBP", "DiastolicBP", "O2Sat", "RespRate"]
    available_vitals = [v for v in vitals if v in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2", "#00897B", "#F57F17"]
    for ax, col, color in zip(axes.ravel(), available_vitals, colors):
        ax.hist(df[col].dropna(), bins=40, color=color, alpha=0.75, edgecolor="white")
        ax.set_title(col, fontsize=13, fontweight="bold")
        ax.set_ylabel("Frequency")
        mean_v = df[col].mean()
        ax.axvline(mean_v, color="black", linestyle="--", linewidth=1, label=f"μ={mean_v:.1f}")
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(len(available_vitals), len(axes.ravel())):
        axes.ravel()[idx].set_visible(False)

    fig.suptitle("Vital Signs at Admission", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_vital_signs.png"))
    plt.close(fig)
    print("  [OK] Vital signs panel saved")

    # ---------------------------------------------------------------
    # 6. Categorical Count Plots
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if "Gender" in df.columns:
        sns.countplot(data=df, x="Gender", ax=axes[0], palette="pastel",
                      edgecolor="black", linewidth=0.5)
        axes[0].set_title("Gender Distribution", fontsize=14, fontweight="bold")

    if "Disease_Type" in df.columns:
        top_dt = df["Disease_Type"].value_counts().head(8)
        axes[1].barh(top_dt.index, top_dt.values, color=sns.color_palette("coolwarm", 8),
                     edgecolor="black", linewidth=0.5)
        axes[1].set_title("Top 8 Disease Types", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Count")
        axes[1].invert_yaxis()

    if "Insurance_Plan" in df.columns:
        sns.countplot(data=df, x="Insurance_Plan", ax=axes[2], palette="Set2",
                      edgecolor="black", linewidth=0.5)
        axes[2].set_title("Insurance Plan Distribution", fontsize=14, fontweight="bold")
        axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_count_plots.png"))
    plt.close(fig)
    print("  [OK] Count plots saved")

    # ---------------------------------------------------------------
    # 7. Cost vs Severity scatter
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        df["Patient_Severity_Score"], df["Total_Cost"],
        c=df["Patient_Age"], cmap="viridis", alpha=0.3, s=5, edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Patient Age", fontsize=12)
    ax.set_xlabel("Patient Severity Score", fontsize=13)
    ax.set_ylabel("Total Cost ($)", fontsize=13)
    ax.set_title("Severity vs. Cost (colored by Age)", fontsize=15, fontweight="bold")
    fig.savefig(os.path.join(output_dir, "06_severity_vs_cost.png"))
    plt.close(fig)
    print("  [OK] Severity vs. Cost scatter saved")

    print(f"\n[EDA Complete] All plots saved to {output_dir}/")
