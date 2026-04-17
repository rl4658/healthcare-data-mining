"""
K-Medoids Clustering Analysis
=================================
Executes K-Medoids clustering using pyclustering.
Visualizes the final medoids and compares them visually against the "Before" state.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pyclustering.cluster.kmedoids import kmedoids

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

CLUSTER_COLORS = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00",
    "#8E24AA", "#00ACC1", "#D81B60", "#6D4C41",
    "#3949AB", "#C0CA33",
]

def run_kmedoids(
    X_scaled: np.ndarray,
    k: int = 4,
    output_dir: str = "output/plots/kmedoids",
    random_state: int = 42,
    sample_size: int = 15000
):
    """
    Run K-Medoids clustering. For efficiency on very large datasets,
    clustering may be run on a random subset if it exceeds sample_size.
    Produces a 'before and after' scatter plot similar to K-Means.
    """
    ensure_dir(output_dir)
    print("\n" + "=" * 60)
    print(f"K-MEDOIDS CLUSTERING (K={k})")
    print("=" * 60)

    # Subsample if dataset is extremely large (PAM computation scales poorly)
    n = X_scaled.shape[0]
    rng = np.random.RandomState(random_state)
    if n > sample_size:
        print(f"  Subsampling {sample_size:,} out of {n:,} points for K-Medoids...")
        idx = rng.choice(n, size=sample_size, replace=False)
        X_work = X_scaled[idx]
        n_work = sample_size
    else:
        X_work = X_scaled
        n_work = n
        idx = np.arange(n)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_work)
    explained = pca.explained_variance_ratio_

    # Initialize and fit K-Medoids using pyclustering
    print("  Fitting K-Medoids model (this may take a moment)...")
    initial_medoids = rng.choice(n_work, size=k, replace=False).tolist()
    
    kmed = kmedoids(X_work.tolist(), initial_medoids)
    kmed.process()
    
    clusters = kmed.get_clusters()
    medoids_idx = kmed.get_medoids()
    
    # Convert pyclustering output back into sklearn-style labels and arrays
    labels = np.zeros(n_work, dtype=int)
    for i, cluster in enumerate(clusters):
        for member_idx in cluster:
            labels[member_idx] = i
            
    medoids = X_work[medoids_idx]
    medoids_2d = pca.transform(medoids)

    # Before / After comparison plot
    fig_ba, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(16, 6))

    # Before: just random initial points as "medoids"
    medoids_init_2d = X_2d[initial_medoids]
    
    ax_b.scatter(X_2d[:, 0], X_2d[:, 1], s=3, alpha=0.3, color="gray")
    ax_b.scatter(medoids_init_2d[:, 0], medoids_init_2d[:, 1],
                 c="black", marker="X", s=200, edgecolors="white", linewidths=1.5, zorder=5)
    ax_b.set_title("Before: Random Initial Points", fontsize=14, fontweight="bold")
    ax_b.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax_b.set_ylabel(f"PC2 ({explained[1]:.1%})")

    # After
    for ci in range(k):
        mask = labels == ci
        ax_a.scatter(X_2d[mask, 0], X_2d[mask, 1], s=3, alpha=0.3,
                     color=CLUSTER_COLORS[ci % len(CLUSTER_COLORS)])
    ax_a.scatter(medoids_2d[:, 0], medoids_2d[:, 1],
                 c="black", marker="X", s=200, edgecolors="white", linewidths=1.5, zorder=5)
    ax_a.set_title(f"After: Converged K-Medoids", fontsize=14, fontweight="bold")
    ax_a.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax_a.set_ylabel(f"PC2 ({explained[1]:.1%})")

    fig_ba.suptitle("K-Medoids: Before vs After", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_ba.savefig(os.path.join(output_dir, "before_after_kmedoids.png"))
    plt.close(fig_ba)

    print(f"  [K-Medoids] Saved scatter plot to {output_dir}/before_after_kmedoids.png")
    
    # We will return the model and the working X so evaluation can score exactly what was clustered
    return None, X_work, labels
