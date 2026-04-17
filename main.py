# -*- coding: utf-8 -*-
"""
main.py -- Master Orchestrator
===============================
Multi-Dimensional Patient Stratification for Resource Optimization
CMPE 255 Data Mining Class Project

Runs the entire pipeline end-to-end:
  Step 1: ETL -> Master Analytical View
  Step 2: Preprocessing -> Scaled features
  Step 3: EDA -> Exploratory visualizations
  Step 4: Elbow Method -> Optimal K selection
  Step 5: K-Means Iterations -> 10 iteration plots
  Step 6: Evaluation -> Silhouette & Cluster Heatmap
  Step 7: PlantUML -> Star schema diagram
  Step 8: Cluster interpretations

Usage:
    python main.py
"""

import os
import sys
import io
import warnings

# Force UTF-8 stdout/stderr on Windows to avoid cp1252 encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.etl import build_master_view
from src.preprocessing import run_preprocessing
from src.eda import run_eda
from src.kmeans_iterative import run_kmeans_iterative
from src.evaluation import (
    run_evaluation,
    cluster_heatmap,
    compare_clustering_metrics
)
from src.kmedoids_analysis import run_kmedoids
from src.plantuml_schema import generate_plantuml


def main():
    print("+" + "=" * 58 + "+")
    print("|  Multi-Dimensional Patient Stratification               |")
    print("|  Healthcare Data Warehouse - MedSynora DW               |")
    print("|  CMPE 255 Data Mining Class Project                     |")
    print("+" + "=" * 58 + "+")

    # ------------------------------------------------------------------
    # Step 1: ETL -- Build Master Analytical View
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1 -- ETL: DATA WAREHOUSING & INTEGRATION")
    print("=" * 60)
    master_df = build_master_view()

    # Save master view for reference
    os.makedirs("output", exist_ok=True)
    master_df.to_csv("output/master_analytical_view.csv", index=False)
    print("  -> Master view saved to output/master_analytical_view.csv")

    # ------------------------------------------------------------------
    # Step 2: Preprocessing
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 -- PREPROCESSING PIPELINE")
    print("=" * 60)
    df_clean, X_scaled, scaler, feature_names = run_preprocessing(master_df)

    # ------------------------------------------------------------------
    # Step 3: Exploratory Data Analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 -- EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    run_eda(df_clean, output_dir="output/plots/eda")

    # ------------------------------------------------------------------
    # Step 4: Elbow Method -> Optimal K
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 -- ELBOW METHOD & SILHOUETTE ANALYSIS")
    print("=" * 60)
    optimal_k, sil_score, labels_full_kmeans = run_evaluation(
        X_scaled, df_clean, feature_names,
        output_dir="output/plots/evaluation",
    )

    # ------------------------------------------------------------------
    # Step 5: K-Means Iterative Visualization
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 5 -- K-MEANS ITERATIVE VISUALIZATION (K={optimal_k})")
    print("=" * 60)
    labels, centroids, wcss_per_iter = run_kmeans_iterative(
        X_scaled,
        k=optimal_k,
        n_iterations=10,
        output_dir="output/plots/kmeans",
        feature_names=feature_names,
    )

    # ------------------------------------------------------------------
    # Step 6: Cluster Interpretation Heatmap
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 -- CLUSTER PROFILE & INTERPRETATION")
    print("=" * 60)
    cluster_means, interpretations = cluster_heatmap(
        df_clean, labels_full_kmeans, feature_names,
        output_dir="output/plots/evaluation",
    )

    # ------------------------------------------------------------------
    # Step 6.5: K-Medoids Evaluation & Metrics Comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6.5 -- K-MEDOIDS & COMPARISON")
    print("=" * 60)
    kmed_model, X_work_kmedoids, labels_kmedoids = run_kmedoids(
        X_scaled,
        k=optimal_k,
        output_dir="output/plots/kmedoids",
    )
    
    compare_clustering_metrics(
        X_scaled, labels_full_kmeans, 
        X_work_kmedoids, labels_kmedoids,
        output_dir="output/plots/evaluation"
    )

    # ------------------------------------------------------------------
    # Step 7: PlantUML Star Schema
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7 -- PLANTUML STAR SCHEMA DIAGRAM")
    print("=" * 60)
    generate_plantuml(output_dir="output/plantuml")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "+" + "=" * 58 + "+")
    print("|  PIPELINE COMPLETE                                      |")
    print("+" + "=" * 58 + "+")
    print(f"""
  Results Summary:
  ----------------------------------------
  Master View:        {len(master_df):>8,} encounters
  After Preprocessing:{len(df_clean):>8,} encounters
  Features Used:      {len(feature_names)} ({', '.join(feature_names)})
  Optimal K:          {optimal_k}
  Silhouette Score:   {sil_score:.4f}
  ----------------------------------------
  
  Cluster Segments:""")
    for i, interp in enumerate(interpretations):
        n = (labels == i).sum()
        print(f"    Cluster {i}: {interp}  ({n:,} patients)")

    print(f"""
  Output Files:
  ----------------------------------------
  output/master_analytical_view.csv
  output/plots/eda/           (7 EDA plots)
  output/plots/kmeans/        (10 iteration + grid + before-after)
  output/plots/evaluation/    (Elbow + Silhouette + Heatmap + Sizes)
  output/plantuml/            (star_schema.puml)
""")


if __name__ == "__main__":
    main()
