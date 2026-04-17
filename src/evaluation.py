"""
Model Evaluation & Diagnostics
================================
1. Elbow Method — WCSS for K = 1..10
2. Silhouette Score — validate cluster separation
3. Cluster Means Heatmap — interpret patient segments
4. Cluster size bar chart
5. Business-readable cluster interpretation
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Elbow Method
# ---------------------------------------------------------------------------

def elbow_method(X_scaled: np.ndarray, k_range: range = range(1, 11),
                 output_dir: str = "output/plots/evaluation",
                 random_state: int = 42) -> int:
    """
    Compute WCSS (inertia) for each K and plot the Elbow curve.
    Returns the automatically detected optimal K (using the "knee" heuristic).
    """
    ensure_dir(output_dir)

    wcss_values = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=random_state)
        km.fit(X_scaled)
        wcss_values.append(km.inertia_)
        print(f"  K={k:2d}  |  WCSS = {km.inertia_:,.1f}")

    # Simple knee-point detection: largest second derivative
    k_list = list(k_range)
    if len(k_list) >= 3:
        diffs = np.diff(wcss_values)
        diffs2 = np.diff(diffs)
        optimal_idx = np.argmax(np.abs(diffs2)) + 2  # +2 to account for double diff offset
        optimal_k = k_list[optimal_idx]
    else:
        optimal_k = 3

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_list, wcss_values, "o-", color="#1565C0", linewidth=2.5, markersize=8)
    ax.axvline(optimal_k, color="#E53935", linestyle="--", linewidth=1.5,
               label=f"Optimal K = {optimal_k}")
    ax.scatter([optimal_k], [wcss_values[optimal_k - k_list[0]]],
               color="#E53935", s=150, zorder=5, edgecolors="white", linewidths=2)

    ax.set_xlabel("Number of Clusters (K)", fontsize=14)
    ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=14)
    ax.set_title("Elbow Method — Optimal K Selection", fontsize=16, fontweight="bold")
    ax.set_xticks(k_list)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(output_dir, "01_elbow_method.png"))
    plt.close(fig)

    print(f"\n  -> Optimal K = {optimal_k}")
    return optimal_k


# ---------------------------------------------------------------------------
# 2. Silhouette Analysis
# ---------------------------------------------------------------------------

def silhouette_analysis(X_scaled: np.ndarray, optimal_k: int,
                        output_dir: str = "output/plots/evaluation",
                        random_state: int = 42,
                        sample_size: int = 10000) -> float:
    """
    Compute and visualize the Silhouette Score for the optimal K.
    Uses random subsampling for the silhouette plot to avoid OOM on large
    datasets (silhouette_samples needs O(n^2) memory).
    """
    ensure_dir(output_dir)
    rng = np.random.RandomState(random_state)

    km = KMeans(n_clusters=optimal_k, n_init=10, max_iter=300, random_state=random_state)
    labels_full = km.fit_predict(X_scaled)

    # Subsample for silhouette computation (full data is too large)
    n = X_scaled.shape[0]
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        X_sample = X_scaled[idx]
        labels_sample = labels_full[idx]
        print(f"  (Subsampled {sample_size:,} / {n:,} points for silhouette computation)")
    else:
        X_sample = X_scaled
        labels_sample = labels_full

    score = silhouette_score(X_sample, labels_sample)
    sample_silhouette_values = silhouette_samples(X_sample, labels_sample)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10
    colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))

    for i in range(optimal_k):
        ith_values = sample_silhouette_values[labels_sample == i]
        ith_values.sort()
        size_cluster_i = ith_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_values,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"C{i}", fontsize=11, fontweight="bold")
        y_lower = y_upper + 10

    ax.axvline(score, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean Silhouette = {score:.4f}")
    ax.set_xlabel("Silhouette Coefficient", fontsize=14)
    ax.set_ylabel("Cluster (sorted samples)", fontsize=14)
    ax.set_title(f"Silhouette Analysis (K={optimal_k})", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3)

    fig.savefig(os.path.join(output_dir, "02_silhouette_analysis.png"))
    plt.close(fig)

    print(f"  Silhouette Score (K={optimal_k}): {score:.4f}")
    return score, labels_full


# ---------------------------------------------------------------------------
# 3. Cluster Means Heatmap & Interpretation
# ---------------------------------------------------------------------------

def cluster_heatmap(df_clean: pd.DataFrame, labels: np.ndarray,
                    feature_names: list,
                    output_dir: str = "output/plots/evaluation"):
    """
    Create a heatmap of cluster centroids (unscaled) and generate
    business-readable cluster labels.
    """
    ensure_dir(output_dir)

    df_work = df_clean.copy()
    df_work["Cluster"] = labels

    # Use original (unscaled) feature values for interpretability
    # Map log-transformed column name back to original
    display_features = []
    for f in feature_names:
        if f == "Total_Cost_Log":
            display_features.append("Total_Cost")
        else:
            display_features.append(f)

    # Compute cluster means
    cluster_means = df_work.groupby("Cluster")[display_features].mean()

    # Heatmap (z-scored for coloring, annotated with raw values)
    z_scored = (cluster_means - cluster_means.mean()) / cluster_means.std()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        z_scored, annot=cluster_means.round(1).values, fmt="",
        cmap="RdYlGn_r", center=0, linewidths=1, linecolor="white",
        ax=ax, cbar_kws={"label": "Z-Score (relative)"},
        xticklabels=[f.replace("_", " ") for f in display_features],
        yticklabels=[f"Cluster {i}" for i in range(len(cluster_means))],
    )
    ax.set_title("Cluster Profile Heatmap (annotated with raw means)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Feature", fontsize=13)
    ax.set_ylabel("Cluster", fontsize=13)

    fig.savefig(os.path.join(output_dir, "03_cluster_heatmap.png"))
    plt.close(fig)
    print("  [OK] Cluster heatmap saved")

    # --- Cluster size bar chart ---
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00",
              "#8E24AA", "#00ACC1", "#D81B60", "#6D4C41"]
    ax.bar(cluster_sizes.index, cluster_sizes.values,
           color=[colors[i % len(colors)] for i in cluster_sizes.index],
           edgecolor="white", linewidth=1.5)
    for i, v in enumerate(cluster_sizes.values):
        ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cluster", fontsize=13)
    ax.set_ylabel("Number of Patients", fontsize=13)
    ax.set_title("Cluster Size Distribution", fontsize=15, fontweight="bold")
    ax.set_xticks(cluster_sizes.index)
    ax.set_xticklabels([f"Cluster {i}" for i in cluster_sizes.index])

    fig.savefig(os.path.join(output_dir, "04_cluster_sizes.png"))
    plt.close(fig)
    print("  [OK] Cluster size chart saved")

    # --- Business interpretation ---
    print("\n" + "=" * 60)
    print("CLUSTER INTERPRETATION")
    print("=" * 60)

    interpretations = []
    for idx, row in cluster_means.iterrows():
        severity = row.get("Patient_Severity_Score", 0)
        cost = row.get("Total_Cost", 0)
        age = row.get("Patient_Age", 0)

        # Determine labels based on relative position
        sev_label = "High-Severity" if severity > cluster_means["Patient_Severity_Score"].median() else "Low-Severity"
        cost_label = "High-Cost" if cost > cluster_means["Total_Cost"].median() else "Low-Cost"
        age_label = "Elderly" if age > cluster_means["Patient_Age"].median() else "Younger"

        label = f"{sev_label} / {cost_label} / {age_label}"
        interpretations.append(label)

        n_patients = cluster_sizes[idx]
        print(f"\n  Cluster {idx} -- \"{label}\"  ({n_patients:,} patients)")
        for feat in display_features:
            print(f"    {feat:30s}  {row[feat]:>10.1f}")

    return cluster_means, interpretations


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_evaluation(X_scaled, df_clean, feature_names,
                   output_dir="output/plots/evaluation"):
    """Run the full evaluation pipeline: Elbow → Silhouette → Heatmap."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION & DIAGNOSTICS")
    print("=" * 60)

    print("\n--- Elbow Method ---")
    optimal_k = elbow_method(X_scaled, output_dir=output_dir)


    print("\n--- Silhouette Analysis ---")
    sil_score, labels_full = silhouette_analysis(X_scaled, optimal_k, output_dir=output_dir)

    return optimal_k, sil_score, labels_full


# ---------------------------------------------------------------------------
# 4. Model Comparison (K-Means vs K-Medoids)
# ---------------------------------------------------------------------------

def compare_clustering_metrics(X_kmeans, labels_kmeans, X_kmedoids, labels_kmedoids,
                               output_dir="output/plots/evaluation"):
    """Calculate DB, CH, and Silhouette indices for both models and plot comparison."""
    ensure_dir(output_dir)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (K-Means vs K-Medoids)")
    print("=" * 60)

    # Subsample if dataset is too large to compute Silhouette quickly
    def get_metrics(X, labels):
        sample_size = 15000
        n_samples = X.shape[0]
        if n_samples > sample_size:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_samples, size=sample_size, replace=False)
            X_eval = X[idx]
            labels_eval = labels[idx]
        else:
            X_eval = X
            labels_eval = labels
        
        sil = silhouette_score(X_eval, labels_eval)
        db = davies_bouldin_score(X_eval, labels_eval)
        ch = calinski_harabasz_score(X_eval, labels_eval)
        return {"Silhouette": sil, "Davies-Bouldin": db, "Calinski-Harabasz": ch}

    print("  Calculating metrics for K-Means...")
    metrics_kmeans = get_metrics(X_kmeans, labels_kmeans)
    
    print("  Calculating metrics for K-Medoids...")
    metrics_kmedoids = get_metrics(X_kmedoids, labels_kmedoids)

    print("\n  [Performance Metrics Comparison]")
    print(f"  {'Metric':<20} | {'K-Means':<15} | {'K-Medoids':<15}")
    print("  " + "-" * 55)
    
    metrics_names = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
    
    for m in metrics_names:
         print(f"  {m:<20} | {metrics_kmeans[m]:<15.4f} | {metrics_kmedoids[m]:<15.4f}")

    # Plot Comparison Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, m in enumerate(metrics_names):
        vals = [metrics_kmeans[m], metrics_kmedoids[m]]
        bars = axes[i].bar(["K-Means", "K-Medoids"], vals, color=["#1E88E5", "#D81B60"])
        axes[i].set_title(m, fontsize=14, fontweight="bold")
        # add labels on top
        for bar in bars:
            yval = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}",
                         ha='center', va='bottom', fontsize=12, fontweight="bold")
    
    fig.suptitle("Performance Measures: K-Means vs K-Medoids\n(Silhouette/CH: Higher is better | DB: Lower is better)", fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "kmeans_vs_kmedoids_comparison.png"))
    plt.close(fig)
    
    print(f"\n  -> Comparison chart saved to {output_dir}/kmeans_vs_kmedoids_comparison.png")

