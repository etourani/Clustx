#!/usr/bin/env python3
# Visualize individual clustering algorithms on one dataset (2D).
# Methods: KMeans, GMM, Agglomerative (Ward), DBSCAN, HDBSCAN (optional), CLIQUE (optional), DE-Grid+OCCA (yours)

import os, math, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score, fowlkes_mallows_score

warnings.filterwarnings("ignore", category=UserWarning)

# --- Optional deps ---
HAVE_HDBSCAN = True
try:
    import hdbscan  # pip install hdbscan
except Exception:
    HAVE_HDBSCAN = False

HAVE_CLIQUE = True
try:
    from pyclustering.cluster.clique import clique  # pip install pyclustering
except Exception:
    HAVE_CLIQUE = False

# --- Your core (assumes clustx is installed/editable) ---
from clustx.core import (
    run_pipeline_2d as run_clustx,
    bin_points_rect, normalize_nonzero, standard_cca, seeded_custom_cca,
    run_diffusion_until_converged, points_to_labels
)

# ------------------ Utilities ------------------

def decode_label_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("b'", "", regex=False).str.replace("'", "", regex=False)
    try:
        return s.astype(int)
    except Exception:
        return s

def external_scores(y_true, y_pred):
    if y_true is None or y_pred is None or len(y_true) != len(y_pred):
        return dict(ARI=np.nan, NMI=np.nan, V=np.nan, FM=np.nan)
    Y = np.where(y_pred < 0, -1, y_pred)
    return dict(
        ARI=float(adjusted_rand_score(y_true, Y)),
        NMI=float(normalized_mutual_info_score(y_true, Y)),
        V=float(v_measure_score(y_true, Y)),
        FM=float(fowlkes_mallows_score(y_true, Y)),
    )

def load_dataset(csv_path, label_col="class"):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        for c in df.columns:
            if c.lower() == label_col.lower():
                label_col = c
                break
    y_true = decode_label_series(df[label_col]) if label_col in df.columns else None
    X = df[["x", "y"]].astype(float).to_numpy()
    return df, X, (y_true.to_numpy() if y_true is not None else None)

def plot_clusters(X, labels, title=""):
    # noise/unassigned as gray
    labels = np.asarray(labels)
    uniq = sorted([int(u) for u in np.unique(labels) if u >= 0])
    cmap = plt.get_cmap("tab20")
    # noise first
    mask_noise = (labels < 0)
    plt.figure(figsize=(6.5, 6))
    if mask_noise.any():
        plt.scatter(X[mask_noise, 0], X[mask_noise, 1], s=12, c="lightgray", edgecolors="none", label="noise")
    # clusters
    for i, cid in enumerate(uniq):
        clr = cmap(i % 20)
        m = (labels == cid)
        plt.scatter(X[m, 0], X[m, 1], s=12, color=clr, edgecolors="none", label=f"c{cid}")
    #plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks([]); plt.yticks([])
    # legend only if #clusters small
    if len(uniq) <= 12:
        plt.legend(markerscale=1.5, fontsize=8, loc="best", frameon=False)
    plt.tight_layout()
    plt.show()

# ------------------ Choose dataset ------------------

HERE = os.path.dirname(__file__) if "__file__" in globals() else "."
CSV = os.path.join(HERE, "s_set1.csv")     # <-- change to "R15.csv" / "s_set1.csv" as you wish
DATASET_NAME = "S-Set1"                     # "R15" / "S-set1"
LABEL_COL = "class"                              # "class" for aggregation/R15; "CLASS" for s_set1 unless you already fixed it

df, X_raw, y_true = load_dataset(CSV, label_col=LABEL_COL)

# For non-grid methods we standardize (helps distance-based clustering).
X_std = StandardScaler().fit_transform(X_raw)

# ------------------ 1) KMeans ------------------
def demo_kmeans(K=15):
    mdl = KMeans(n_clusters=K, n_init="auto", random_state=0)
    labels = mdl.fit_predict(X_std)
    s = external_scores(y_true, labels)
    print("[KMeans]", {"K": K, **{k: round(v,4) for k,v in s.items()}})
    plot_clusters(X_raw, labels, f"{DATASET_NAME} | KMeans (K={K})")

# ------------------ 2) GMM ------------------
def demo_gmm(K=15):
    mdl = GaussianMixture(n_components=K, covariance_type="full", random_state=0)
    labels = mdl.fit_predict(X_std)
    s = external_scores(y_true, labels)
    print("[GMM]", {"K": K, **{k: round(v,4) for k,v in s.items()}})
    plot_clusters(X_raw, labels, f"{DATASET_NAME} | GMM (K={K})")

# ------------------ 3) Agglomerative (Ward) ------------------
def demo_agglomerative(K=15):
    mdl = AgglomerativeClustering(n_clusters=K, linkage="ward")
    labels = mdl.fit_predict(X_std)
    s = external_scores(y_true, labels)
    print("[Agglomerative-Ward]", {"K": K, **{k: round(v,4) for k,v in s.items()}})
    plot_clusters(X_raw, labels, f"{DATASET_NAME} | Agglomerative (Ward, K={K})")

# ------------------ 4) DBSCAN ------------------
def demo_dbscan(eps=0.15, minpts=15):
    mdl = DBSCAN(eps=eps, min_samples=minpts)
    labels = mdl.fit_predict(X_std)
    s = external_scores(y_true, labels)
    print("[DBSCAN]", {"eps": eps, "MinPts": minpts, **{k: round(v,4) for k,v in s.items()}})
    plot_clusters(X_raw, labels, f"{DATASET_NAME} | DBSCAN (eps={eps}, MinPts={minpts})")

# ------------------ 5) HDBSCAN (optional) ------------------
def demo_hdbscan(min_cluster_size=40):
    if not HAVE_HDBSCAN:
        print("[HDBSCAN] Not installed. pip install hdbscan")
        return
    mdl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = mdl.fit_predict(X_std)
    s = external_scores(y_true, labels)
    print("[HDBSCAN]", {"min_cluster_size": min_cluster_size, **{k: round(v,4) for k,v in s.items()}})
    plot_clusters(X_raw, labels, f"{DATASET_NAME} | HDBSCAN (min_cluster_size={min_cluster_size})")

# ------------------ 6) CLIQUE (optional) ------------------
def demo_clique(M=33, R_count=7):
    """
    pyclustering’s CLIQUE expects:
      - M (int): intervals per axis (grid resolution)
      - R_count (int): min points per dense cell (integer threshold)
    We normalize to [0,1] before passing to CLIQUE.
    """
    if not HAVE_CLIQUE:
        print("[CLIQUE] Not installed. pip install pyclustering")
        return
    X01 = (X_std - X_std.min(axis=0)) / (np.ptp(X_std, axis=0) + 1e-12)
    c = clique(X01.tolist(), int(M), int(R_count))
    c.process()
    clusters = c.get_clusters()
    labels = -np.ones(X01.shape[0], dtype=int)
    for cid, idxs in enumerate(clusters):
        if idxs:
            labels[np.asarray(idxs, dtype=int)] = cid
    s = external_scores(y_true, labels)
    print("[CLIQUE]", {"M": M, "R": R_count, **{k: round(v,4) for k,v in s.items()}})
    plot_clusters(X_raw, labels, f"{DATASET_NAME} | CLIQUE (M={M}, R={R_count})")

# ------------------ 7) Your DE-Grid + OCCA ------------------
def demo_degrid_occa():
    """
    Run your tuned pipeline, then rebuild final labels using Stage-A winner and Stage-B (beta*, cthr*).
    We import helpers directly from clustx.core to reconstruct labels for plotting.
    """
    res = run_clustx(
        points_file=CSV,
        out_dir=os.path.join(HERE, f"out_{DATASET_NAME.lower()}_play"),
        TUNING="bo",
        BO_OPT_WEIGHTS=False,
        W_SIL=0.33, W_DBI=0.34, W_COV=0.33,
        BO_N_CALLS=50,
        H_BOUNDS_REL=(0.5, 1.25),
        R_RANGE=(2, 20),
        BETA_CANDIDATES=(0.1, 0.2, 0.25),
        CTHR_VALUES=(0.01, 0.02, 0.05, 0.1),
        MAX_ITERS=5000, MIN_ITERS=100, TOL=1e-6, CHECK_EVERY=10,
        K_MIN=2, K_MAX=50,
        PERIODIC_CCA=False,
        CONNECTIVITY=4,
        MAKE_PLOTS=False, DO_STD_CCA=True,
        LABEL_COL=LABEL_COL,
    )

    # Bounds with tiny padding (mirror pipeline)
    x_min = float(X_raw[:,0].min()); x_max = float(X_raw[:,0].max())
    y_min = float(X_raw[:,1].min()); y_max = float(X_raw[:,1].max())
    pad_x = 1e-9 * (x_max - x_min + 1.0); pad_y = 1e-9 * (y_max - y_min + 1.0)
    x_min -= pad_x; x_max += pad_x; y_min -= pad_y; y_max += pad_y

    bestA = res["stageA_best"]
    nx, ny = int(bestA["nx"]), int(bestA["ny"])

    # Rebuild coarse grid and masks
    counts, dx, dy = bin_points_rect(X_raw, nx, ny, x_min, x_max, y_min, y_max)
    norm, _, _ = normalize_nonzero(counts)
    nonempty = (counts > 0)
    if bestA["mode"] == "quantile":
        dense_thr = float(bestA["dense_thr"])
        dense_before = nonempty & (norm >= dense_thr)
    else:
        R = int(bestA["R"])
        dense_before = (counts >= R) & nonempty

    sparse_before = nonempty & ~dense_before
    empty_mask   = ~nonempty

    beta_eff = float(res.get("best_beta"))
    cthr_eff = float(res["stageB_best"]["cthr"])
    # adaptive weight scale like pipeline
    dense_frac = float(dense_before.sum() / max(1, nonempty.sum()))
    vals = norm[nonempty]
    dense_thr_for_w = float(np.quantile(vals, max(0.0, min(1.0, 1.0 - dense_frac)))) if vals.size else 0.30

    # Diffuse
    c, _ = run_diffusion_until_converged(
        norm, dense_mask=dense_before, update_mask=sparse_before, empty_mask=empty_mask,
        beta=beta_eff, max_iters=50000, min_iters=60, tol=1e-6, check_every=10,
        dense_thr_for_w=dense_thr_for_w, periodic=False
    )

    # Select after
    selected_after = np.zeros_like(dense_before, dtype=bool)
    selected_after[dense_before] = True
    if np.any(sparse_before):
        selected_after[sparse_before] = (c[sparse_before] > cthr_eff)

    labels_after_seeded, _ = seeded_custom_cca(selected_after, dense_before, periodic=False, connectivity=4)
    y_pred = points_to_labels(X_raw, labels_after_seeded, x_min, x_max, y_min, y_max, dx, dy)

    s = external_scores(y_true, y_pred)
    print("[DE-Grid+OCCA]", {"beta*": beta_eff, "cthr*": cthr_eff, **{k: round(v,4) for k,v in s.items()}})
    plot_clusters(X_raw, y_pred, f"{DATASET_NAME} | DE-Grid + OCCA (β*={beta_eff}, cthr*={cthr_eff})")

# ------------------ Run any demos you want ------------------

if __name__ == "__main__":
    # Choose which to run (uncomment)
    demo_kmeans(K=7)
    demo_gmm(K=7)
    demo_agglomerative(K=7)
    demo_dbscan(eps=0.5, minpts=30)
    demo_hdbscan(min_cluster_size=30)     # requires hdbscan
    demo_clique(M=13, R_count=5)          # requires pyclustering
    demo_degrid_occa()                    # your method

