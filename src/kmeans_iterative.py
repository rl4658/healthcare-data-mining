"""
K-Means Iterative Visualization
=================================
Implements a custom K-Means training loop that visualizes the
Expectation (assignment) and Maximization (centroid update) steps
for every single iteration (1–10).

Mathematical foundation:
  Objective (WCSS):  J = Σ_i Σ_{x∈C_i} ||x − μ_i||²
  Distance:          d(x,μ) = √( Σ_j (x_j − μ_j)² )
  Centroid update:   μ_i = (1/|C_i|) Σ_{x∈C_i} x
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Core K-Means from scratch
# ---------------------------------------------------------------------------

def euclidean_distance(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance between every sample and every centroid.
    Returns shape (n_samples, k).
    """
    # X: (n, p),  centroids: (k, p)
    return np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """E-step: assign each point to the nearest centroid."""
    distances = euclidean_distance(X, centroids)
    return np.argmin(distances, axis=1)


def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """M-step: recompute centroids as the mean of assigned points."""
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        members = X[labels == i]
        if len(members) > 0:
            new_centroids[i] = members.mean(axis=0)
    return new_centroids


def compute_wcss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Within-Cluster Sum of Squares:  J = Σ_i Σ_{x∈C_i} ||x − μ_i||²"""
    wcss = 0.0
    for i in range(len(centroids)):
        members = X[labels == i]
        if len(members) > 0:
            wcss += ((members - centroids[i]) ** 2).sum()
    return wcss


# ---------------------------------------------------------------------------
# Iteration-by-iteration visualization
# ---------------------------------------------------------------------------

CLUSTER_COLORS = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00",
    "#8E24AA", "#00ACC1", "#D81B60", "#6D4C41",
    "#3949AB", "#C0CA33",
]


def run_kmeans_iterative(
    X_scaled: np.ndarray,
    k: int = 4,
    n_iterations: int = 10,
    output_dir: str = "output/plots/kmeans",
    feature_names: list = None,
    random_state: int = 42,
):
    """
    Run K-Means from scratch for `n_iterations` iterations, producing
    a scatter plot at each iteration showing cluster assignments and
    centroid positions/movement.

    Parameters
    ----------
    X_scaled : scaled feature matrix (n_samples, n_features)
    k : number of clusters
    n_iterations : number of E-M iterations to visualize
    output_dir : where to save plots
    feature_names : names of features (for axis labels)
    random_state : seed for reproducibility

    Returns
    -------
    labels : final cluster assignments
    centroids : final centroid positions (in the original scaled space)
    wcss_per_iter : list of WCSS values at each iteration
    """
    ensure_dir(output_dir)
    rng = np.random.RandomState(random_state)

    print("\n" + "=" * 60)
    print(f"K-MEANS ITERATIVE VISUALIZATION  (K={k}, iters={n_iterations})")
    print("=" * 60)

    # PCA → 2D for visualization
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    print(f"  PCA explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")

    # Initialize centroids: pick k random data points
    indices = rng.choice(X_scaled.shape[0], size=k, replace=False)
    centroids = X_scaled[indices].copy()
    centroids_2d = pca.transform(centroids)

    wcss_per_iter = []

    # ---------- Combined grid figure (2 × 5) ----------
    fig_grid, axes_grid = plt.subplots(2, 5, figsize=(28, 11))
    axes_flat = axes_grid.ravel()

    for it in range(n_iterations):
        # Save previous centroid positions for arrows
        prev_centroids_2d = centroids_2d.copy()

        # E-step: assign clusters
        labels = assign_clusters(X_scaled, centroids)

        # M-step: update centroids
        centroids = update_centroids(X_scaled, labels, k)
        centroids_2d = pca.transform(centroids)

        # WCSS
        wcss = compute_wcss(X_scaled, labels, centroids)
        wcss_per_iter.append(wcss)

        # --- Individual iteration plot ---
        fig_single, ax_s = plt.subplots(figsize=(8, 6))
        _plot_iteration(ax_s, X_2d, labels, centroids_2d, prev_centroids_2d,
                        k, it, wcss, explained)
        fig_single.savefig(os.path.join(output_dir, f"iteration_{it+1:02d}.png"))
        plt.close(fig_single)

        # --- Grid subplot ---
        _plot_iteration(axes_flat[it], X_2d, labels, centroids_2d, prev_centroids_2d,
                        k, it, wcss, explained, compact=True)

        print(f"  Iteration {it+1:2d}  |  WCSS = {wcss:,.1f}")

    fig_grid.suptitle(
        f"K-Means Iterations 1–{n_iterations}  (K={k})",
        fontsize=20, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig_grid.savefig(os.path.join(output_dir, "all_iterations_grid.png"))
    plt.close(fig_grid)

    # ---------- Before / After comparison ----------
    fig_ba, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(16, 6))
    # "Before" = iteration 1 state
    labels_init = assign_clusters(X_scaled, X_scaled[indices])
    centroids_init_2d = pca.transform(X_scaled[indices])
    for ci in range(k):
        mask = labels_init == ci
        ax_b.scatter(X_2d[mask, 0], X_2d[mask, 1], s=3, alpha=0.3,
                     color=CLUSTER_COLORS[ci % len(CLUSTER_COLORS)])
    ax_b.scatter(centroids_init_2d[:, 0], centroids_init_2d[:, 1],
                 c="black", marker="X", s=200, edgecolors="white", linewidths=1.5, zorder=5)
    ax_b.set_title("Before: Initial Random Centroids", fontsize=14, fontweight="bold")
    ax_b.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax_b.set_ylabel(f"PC2 ({explained[1]:.1%})")

    # "After" = final state
    for ci in range(k):
        mask = labels == ci
        ax_a.scatter(X_2d[mask, 0], X_2d[mask, 1], s=3, alpha=0.3,
                     color=CLUSTER_COLORS[ci % len(CLUSTER_COLORS)])
    ax_a.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                 c="black", marker="X", s=200, edgecolors="white", linewidths=1.5, zorder=5)
    ax_a.set_title(f"After: Iteration {n_iterations} (Converged)", fontsize=14, fontweight="bold")
    ax_a.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax_a.set_ylabel(f"PC2 ({explained[1]:.1%})")

    fig_ba.suptitle("K-Means: Before vs After", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_ba.savefig(os.path.join(output_dir, "before_after_comparison.png"))
    plt.close(fig_ba)

    # ---------- WCSS convergence curve ----------
    fig_wc, ax_wc = plt.subplots(figsize=(8, 5))
    ax_wc.plot(range(1, n_iterations + 1), wcss_per_iter, "o-", color="#1565C0",
               linewidth=2, markersize=6)
    ax_wc.set_xlabel("Iteration", fontsize=13)
    ax_wc.set_ylabel("WCSS (Inertia)", fontsize=13)
    ax_wc.set_title("WCSS Convergence Over Iterations", fontsize=15, fontweight="bold")
    ax_wc.set_xticks(range(1, n_iterations + 1))
    ax_wc.grid(True, alpha=0.4)
    fig_wc.savefig(os.path.join(output_dir, "wcss_convergence.png"))
    plt.close(fig_wc)

    print(f"\n[K-Means] All iteration plots saved to {output_dir}/")
    return labels, centroids, wcss_per_iter


def _plot_iteration(ax, X_2d, labels, centroids_2d, prev_centroids_2d,
                    k, iteration, wcss, explained, compact=False):
    """Helper: draw a single iteration's scatter plot on the given axes."""
    for ci in range(k):
        mask = labels == ci
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            s=2 if compact else 4, alpha=0.25,
            color=CLUSTER_COLORS[ci % len(CLUSTER_COLORS)],
            label=f"C{ci}" if not compact else None,
        )

    # Centroid markers
    ax.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        c="black", marker="X",
        s=120 if compact else 200,
        edgecolors="white", linewidths=1.2, zorder=5,
    )

    # Arrows showing centroid movement
    for ci in range(k):
        dx = centroids_2d[ci, 0] - prev_centroids_2d[ci, 0]
        dy = centroids_2d[ci, 1] - prev_centroids_2d[ci, 1]
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            ax.annotate(
                "", xy=(centroids_2d[ci, 0], centroids_2d[ci, 1]),
                xytext=(prev_centroids_2d[ci, 0], prev_centroids_2d[ci, 1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            )

    title = f"Iter {iteration + 1}  (WCSS={wcss:,.0f})"
    fs = 10 if compact else 14
    ax.set_title(title, fontsize=fs, fontweight="bold")

    if not compact:
        ax.set_xlabel(f"PC1 ({explained[0]:.1%})", fontsize=12)
        ax.set_ylabel(f"PC2 ({explained[1]:.1%})", fontsize=12)
        ax.legend(fontsize=8, markerscale=3)
