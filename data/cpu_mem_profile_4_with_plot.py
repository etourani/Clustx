#!/usr/bin/env python3
# Profile CPU & memory usage of multiple clusterers on s_set1.csv
# Includes: full DE-Grid+OCCA and Stage-II-only (diffusion + OCCA) timings with ΔRSS.
# Now also saves PNG scatter plots of clustering results for each method.

import os, time, math, json, warnings, threading, gc
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Callable

import matplotlib.pyplot as plt  # <-- added

# --- Stabilize BLAS threading to avoid CPU-time bloat ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    v_measure_score, fowlkes_mallows_score,
    silhouette_score, davies_bouldin_score
)

warnings.filterwarnings("ignore", category=UserWarning)

# Optional deps
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

HAVE_PSUTIL = True
try:
    import psutil
except Exception:
    HAVE_PSUTIL = False

import tracemalloc

# Your method
from clustx.core import run_pipeline_2d as run_clustx
from clustx.core import (
    bin_points_rect, normalize_nonzero, run_diffusion_until_converged,
    seeded_custom_cca, points_to_labels
)


import matplotlib as mpl
import seaborn as sns

# ─── STYLING ────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
mpl.rcParams['font.family'] = 'Nimbus Roman'
mpl.rcParams['font.size'] = 24
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['axes.titlesize'] = 32
mpl.rcParams['xtick.labelsize'] = 26
mpl.rcParams['ytick.labelsize'] = 26
mpl.rcParams['legend.fontsize'] = 22
mpl.rcParams['legend.title_fontsize'] = 28
mpl.rcParams['figure.titlesize'] = 34
mpl.rcParams['grid.alpha'] = 0.01
mpl.rcParams['grid.linewidth'] = 1.0
mpl.rcParams['grid.linestyle'] = '--'



# ------------------------ Utilities ------------------------

def decode_label_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("b'", "", regex=False).str.replace("'", "", regex=False)
    try:
        return s.astype(int)
    except Exception:
        return s

def load_dataset(path: str, label_col: str = "class") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        for c in df.columns:
            if c.lower() == label_col.lower():
                label_col = c
                break
    y_true = decode_label_series(df[label_col]) if label_col in df.columns else None
    X = df[["x", "y"]].astype(float).to_numpy()
    return X, (y_true.to_numpy() if y_true is not None else None)

def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    import collections
    if y_true is None or y_pred is None or len(y_true) != len(y_pred):
        return float("nan")
    y_pred = np.asarray(y_pred)
    valid = (y_pred >= 0)
    if valid.sum() == 0:
        return float("nan")
    total = 0
    for c in np.unique(y_pred[valid]):
        idx = (y_pred == c)
        counts = collections.Counter(y_true[idx])
        total += counts.most_common(1)[0][1]
    return total / float(len(y_true))

def cov_k_sil_dbi(X: np.ndarray, labels: np.ndarray) -> Tuple[float, int, float, float]:
    labels = np.asarray(labels)
    mask = labels >= 0
    k = int(len(np.unique(labels[mask]))) if mask.any() else 0
    coverage = float(mask.mean())
    if mask.sum() >= 2 and k >= 2:
        try:
            sil = float(silhouette_score(X[mask], labels[mask]))
        except Exception:
            sil = float("nan")
        try:
            dbi = float(davies_bouldin_score(X[mask], labels[mask]))
        except Exception:
            dbi = float("nan")
    else:
        sil, dbi = float("nan"), float("nan")
    return coverage, k, sil, dbi

def score_all(X: np.ndarray, y_true: Optional[np.ndarray], labels: np.ndarray) -> Dict:
    coverage, k, sil, dbi = cov_k_sil_dbi(X, labels)
    if y_true is not None and len(y_true) == len(labels):
        Y = np.where(labels < 0, -1, labels)
        ARI = float(adjusted_rand_score(y_true, Y))
        NMI = float(normalized_mutual_info_score(y_true, Y))
        V   = float(v_measure_score(y_true, Y))
        FM  = float(fowlkes_mallows_score(y_true, Y))
        PUR = float(purity_score(y_true, Y))
    else:
        ARI = NMI = V = FM = PUR = float("nan")
    return dict(coverage=coverage, k=k, silhouette=sil, dbi=dbi, ARI=ARI, NMI=NMI, V=V, FM=FM, purity=PUR)

# ------------------------ Plot helper ------------------------

def _slugify(name: str) -> str:
    s = name.lower()
    for ch in ["+", "(", ")", "/", " "]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

def save_cluster_plot(X_raw: np.ndarray, labels: np.ndarray, out_dir: str, method_name: str):
    """
    Save a scatter plot of clustering result:
    - X_raw: original coordinates (N,2)
    - labels: cluster labels (N,)
    - out_dir/method_name.png
    Noise label (-1) plotted in light gray.
    """
    labels = np.asarray(labels)
    fname_png = os.path.join(out_dir, f"s_set1_{_slugify(method_name)}.png")
    fname_pdf = os.path.join(out_dir, f"s_set1_{_slugify(method_name)}.pdf")

    #plt.figure(figsize=(5, 5))
    marker_size = mpl.rcParams['font.size'] / 2
    figsize_scale = 1.2
    fig, ax = plt.subplots(figsize=(10 * figsize_scale, 10 * figsize_scale), sharex=True, sharey=True)
    # Noise
    noise_mask = (labels < 0)
    if noise_mask.any():
        plt.scatter(X_raw[noise_mask, 0], X_raw[noise_mask, 1],
                    s=35, c="lightgray", alpha=0.85, edgecolor='black', label="Noise")
    # Clusters
    uniq = np.unique(labels[labels >= 0])
    if uniq.size > 0:
        # Use a qualitative colormap
        cmap = plt.get_cmap("tab20")
        for i, cid in enumerate(uniq):
            m = labels == cid
            plt.scatter(X_raw[m, 0], X_raw[m, 1],
                        s=35, c=[cmap(i % cmap.N)], alpha=0.9,
                        label=f"C{cid}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(method_name)
    plt.axis("equal")

    from matplotlib.ticker import ScalarFormatter
    # Force plain (non-scientific) tick labels, no 1e6 offset
    fmt = ScalarFormatter(useMathText=False)
    fmt.set_scientific(True)
    fmt.set_useOffset(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


    plt.tight_layout()
    plt.savefig(fname_png, dpi=300, pad_inches=0.05) # bbox_inches="tight",
    plt.savefig(fname_pdf, dpi=300, pad_inches=0.05) # bbox_inches="tight",)
    plt.close()

# ------------------------ Profiling wrapper (ΔRSS-aware) ------------------------

def run_profiled(fn: Callable, *args, **kwargs):
    """
    Measures:
      - wall time (s)
      - CPU time (user+system, s)
      - peak Python heap (MB, via tracemalloc)
      - peak RSS (MB, via psutil sampled ~100Hz; else nan)
      - delta_rss_MB = peak_rss - baseline_rss
    """
    cpu_start = time.process_time()

    baseline_rss = float("nan")
    peak_rss_mb = [float("nan")]
    stop_flag = [False]

    def rss_sampler():
        if not HAVE_PSUTIL:
            return
        proc = psutil.Process()
        nonlocal baseline_rss
        try:
            baseline_rss = proc.memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            baseline_rss = float("nan")
        peak_local = 0
        while not stop_flag[0]:
            try:
                rss = proc.memory_info().rss
                if rss > peak_local:
                    peak_local = rss
            except Exception:
                pass
            time.sleep(0.01)
        peak_rss_mb[0] = peak_local / (1024.0 * 1024.0)

    sampler_thread = None
    if HAVE_PSUTIL:
        sampler_thread = threading.Thread(target=rss_sampler, daemon=True)
        sampler_thread.start()

    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    _, heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stop_flag[0] = True
    if sampler_thread is not None:
        sampler_thread.join(timeout=1.0)

    cpu_time = time.process_time() - cpu_start
    delta_rss = (peak_rss_mb[0] - baseline_rss) if (
        isinstance(peak_rss_mb[0], float) and isinstance(baseline_rss, float)
    ) else float("nan")

    gc.collect()

    return out, {
        "time_sec": wall,
        "cpu_time_sec": cpu_time,
        "heap_peak_MB": heap_peak / (1024.0 * 1024.0),
        "rss_peak_MB": peak_rss_mb[0],
        "delta_rss_MB": delta_rss,
    }

# ------------------------ Clusterer runners ------------------------

def run_kmeans(Xs: np.ndarray, K: int):
    mdl = KMeans(n_clusters=int(K), n_init="auto", random_state=0)
    return mdl.fit_predict(Xs)

def run_gmm(Xs: np.ndarray, K: int):
    mdl = GaussianMixture(n_components=int(K), covariance_type="full", random_state=0)
    return mdl.fit_predict(Xs)

def run_agglomerative(Xs: np.ndarray, K: int):
    mdl = AgglomerativeClustering(n_clusters=int(K), linkage="ward")
    return mdl.fit_predict(Xs)

def run_dbscan(Xs: np.ndarray, eps: float, minpts: int):
    mdl = DBSCAN(eps=float(eps), min_samples=int(minpts))
    return mdl.fit_predict(Xs)

def run_hdbscan(Xs: np.ndarray, min_cluster_size: int):
    if not HAVE_HDBSCAN:
        raise RuntimeError("hdbscan not installed")
    mdl = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples))
    return mdl.fit_predict(Xs)

def run_clique_labels(Xs: np.ndarray, M: int, R_count: int):
    if not HAVE_CLIQUE:
        raise RuntimeError("pyclustering not installed")
    # Normalize to [0,1] for grid partitioning
    X01 = (Xs - Xs.min(axis=0)) / (np.ptp(Xs, axis=0) + 1e-12)
    c = clique(X01.tolist(), int(M), int(R_count))
    c.process()
    clusters = c.get_clusters()
    labels = -np.ones(X01.shape[0], dtype=int)
    for cid, idxs in enumerate(clusters):
        if idxs:
            labels[np.asarray(idxs, dtype=int)] = cid
    return labels

# ------------------------ Stage-II-only from saved JSON ------------------------

def run_stageII_from_json(CSV: str, json_path: str, label_col: str = "class") -> Dict:
    """
    Run ONLY Stage-II (diffusion + OCCA) using params saved by a prior full pipeline run.
    Reads: best_params_summary.json (Stage-I winner + Stage-II beta/cthr).
    """
    with open(json_path, "r") as f:
        summary = json.load(f)

    bestA = summary["stageA_best"]
    stageB = summary["stageB_best"]
    beta_eff = float(stageB["beta"]) if "beta" in stageB else float(summary.get("best_beta"))

    nx, ny = int(bestA["nx"]), int(bestA["ny"])

    modeA = str(bestA.get("mode", "")).lower()
    has_dense_thr = (modeA == "quantile" and "dense_thr" in bestA)
    dense_thr = float(bestA["dense_thr"]) if has_dense_thr else None
    R = int(bestA["R"]) if ("R" in bestA and not has_dense_thr) else None
    cthr_eff = float(stageB["cthr"])

    X_raw, y_true_local = load_dataset(CSV, label_col)

    # bounds (mirror pipeline)
    x_min = float(X_raw[:, 0].min()); x_max = float(X_raw[:, 0].max())
    y_min = float(X_raw[:, 1].min()); y_max = float(X_raw[:, 1].max())
    pad_x = 1e-9 * (x_max - x_min + 1.0); pad_y = 1e-9 * (y_max - y_min + 1.0)
    x_min -= pad_x; x_max += pad_x; y_min -= pad_y; y_max += pad_y

    counts, dx, dy = bin_points_rect(X_raw, nx, ny, x_min, x_max, y_min, y_max)
    norm, _, _ = normalize_nonzero(counts)
    nonempty = (counts > 0)

    if has_dense_thr:
        dense_before = nonempty & (norm >= dense_thr)
    else:
        dense_before = (counts >= int(R)) & nonempty

    sparse_before = nonempty & (~dense_before)
    empty_mask = ~nonempty

    # adaptive diffusion weight threshold (as in pipeline)
    if nonempty.any():
        dense_frac = float(dense_before.sum() / max(1, nonempty.sum()))
        q_equiv = max(0.0, min(1.0, 1.0 - dense_frac))
        vals = norm[nonempty]
        dense_thr_for_w = float(np.quantile(vals, q_equiv)) if vals.size else 0.30
    else:
        dense_thr_for_w = 0.30

    # diffusion
    c, _ = run_diffusion_until_converged(
        norm,
        dense_mask=dense_before,
        update_mask=sparse_before,
        empty_mask=empty_mask,
        beta=beta_eff,
        max_iters=50000,
        min_iters=60,
        tol=1e-6,
        check_every=10,
        dense_thr_for_w=dense_thr_for_w,
        periodic=False,
    )

    # select + seed-CCA
    selected_after = np.zeros_like(dense_before, dtype=bool)
    selected_after[dense_before] = True
    if np.any(sparse_before):
        selected_after[sparse_before] = (c[sparse_before] > cthr_eff)

    labels_after, _ = seeded_custom_cca(selected_after, dense_before, periodic=False, connectivity=4)
    y_pred = points_to_labels(X_raw, labels_after, x_min, x_max, y_min, y_max, dx, dy)

    return {"y_pred": y_pred, "metrics": score_all(X_raw, y_true_local, y_pred)}




def run_stageII_core_only(CSV: str, json_path: str, label_col: str = "class") -> Dict:
    """
    Measure CPU & memory for ONLY the core Stage II step:
      - diffusion
      - O-CCA
      - mapping labels back to points

    Setup (CSV load, grid construction, masks, etc.) and metrics are done
    *outside* the profiled region, so the reported time/memory correspond
    just to diffusion + seeded CCA + points_to_labels.
    """
    # ----- Load best params from full pipeline -----
    with open(json_path, "r") as f:
        summary = json.load(f)

    bestA   = summary["stageA_best"]
    stageB  = summary["stageB_best"]
    beta_eff = float(stageB["beta"]) if "beta" in stageB else float(summary.get("best_beta"))
    cthr_eff = float(stageB["cthr"])

    nx, ny = int(bestA["nx"]), int(bestA["ny"])

    modeA = str(bestA.get("mode", "")).lower()
    has_dense_thr = (modeA == "quantile" and "dense_thr" in bestA)
    dense_thr = float(bestA["dense_thr"]) if has_dense_thr else None
    R = int(bestA["R"]) if ("R" in bestA and not has_dense_thr) else None

    # ----- Load dataset & build grid (unprofiled setup) -----
    X_raw, y_true_local = load_dataset(CSV, label_col)

    x_min = float(X_raw[:, 0].min()); x_max = float(X_raw[:, 0].max())
    y_min = float(X_raw[:, 1].min()); y_max = float(X_raw[:, 1].max())
    pad_x = 1e-9 * (x_max - x_min + 1.0); pad_y = 1e-9 * (y_max - y_min + 1.0)
    x_min -= pad_x; x_max += pad_x; y_min -= pad_y; y_max += pad_y

    counts, dx, dy = bin_points_rect(X_raw, nx, ny, x_min, x_max, y_min, y_max)
    norm, _, _ = normalize_nonzero(counts)
    nonempty = (counts > 0)

    if has_dense_thr:
        dense_before = nonempty & (norm >= dense_thr)
    else:
        dense_before = (counts >= int(R)) & nonempty

    sparse_before = nonempty & (~dense_before)
    empty_mask    = ~nonempty

    # adaptive diffusion weight threshold (same as pipeline)
    if nonempty.any():
        dense_frac = float(dense_before.sum() / max(1, nonempty.sum()))
        q_equiv = max(0.0, min(1.0, 1.0 - dense_frac))
        vals = norm[nonempty]
        dense_thr_for_w = float(np.quantile(vals, q_equiv)) if vals.size else 0.30
    else:
        dense_thr_for_w = 0.30

    # ----- Define the *core* Stage II work to be profiled -----
    def _stageII_core():
        # diffusion
        c, _ = run_diffusion_until_converged(
            norm,
            dense_mask=dense_before,
            update_mask=sparse_before,
            empty_mask=empty_mask,
            beta=beta_eff,
            max_iters=10000,
            min_iters=100,
            tol=1e-6,
            check_every=10,
            dense_thr_for_w=dense_thr_for_w,
            periodic=False,
        )

        # select + O-CCA
        selected_after = np.zeros_like(dense_before, dtype=bool)
        selected_after[dense_before] = True
        if np.any(sparse_before):
            selected_after[sparse_before] = (c[sparse_before] > cthr_eff)

        labels_after, _ = seeded_custom_cca(selected_after, dense_before,
                                            periodic=False, connectivity=4)
        y_pred = points_to_labels(X_raw, labels_after, x_min, x_max, y_min, y_max, dx, dy)
        return y_pred

    # ----- Profile ONLY the core -----
    y_pred, prof = run_profiled(_stageII_core)

    # ----- Compute metrics *after* profiling (not counted in time/mem) -----
    metrics = score_all(X_raw, y_true_local, y_pred)

    return {"y_pred": y_pred, "metrics": metrics, "profile": prof}



# ------------------------ Main: s_set1.csv ------------------------

if __name__ == "__main__":
    HERE = os.path.dirname(__file__) if "__file__" in globals() else "."
    CSV = os.path.join(HERE, "s_set1.csv")  # change if needed
    LABEL_COL = "class"

    # Output dirs for metrics + plots
    OUT = os.path.join(HERE, "out_profile_s_set1_with_plot_newStageIIonly")
    os.makedirs(OUT, exist_ok=True)
    PLOTDIR = os.path.join(OUT, "plots")
    os.makedirs(PLOTDIR, exist_ok=True)

    # Load
    X_raw, y_true = load_dataset(CSV, label_col=LABEL_COL)
    # Standardize for distance-based baselines
    Xs = StandardScaler().fit_transform(X_raw)

    rows = []

    # ---- DE-Grid+OCCA (full pipeline; tuned once) ----
    def _run_de_grid():
        return run_clustx(
            points_file=CSV,
            out_dir=os.path.join(HERE, "out_s_set1_profile"),
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

    res_full, prof_full = run_profiled(_run_de_grid)
    metric_full = (res_full.get("metrics", {}) or {}).get("after_occa", {})
    rows.append({"method": "DE-Grid+OCCA", **prof_full, **metric_full})

    # JSON path from that full run
    JSON_PATH = os.path.join(HERE, "out_s_set1_profile", "best_params_summary.json")

    # ---- Stage II only: diffusion + O-CCA core ----
    res_stage2 = run_stageII_core_only(CSV, JSON_PATH, LABEL_COL)
    prof_stage2 = res_stage2["profile"]
    rows.append({"method": "DE-Grid+OCCA (Stage II only)", **prof_stage2, **res_stage2["metrics"]})

    # Use Stage-II y_pred for plotting DE-Grid+OCCA clusters
    if "y_pred" in res_stage2:
        save_cluster_plot(X_raw, res_stage2["y_pred"], PLOTDIR, "DE-Grid+OCCA")


    # JSON path from that full run
    #JSON_PATH = os.path.join(HERE, "out_s_set1_profile", "best_params_summary.json")

    # ---- Stage II only: load params from JSON, no Stage I rerun ----
    #res_stage2, prof_stage2 = run_profiled(run_stageII_from_json, CSV, JSON_PATH, LABEL_COL)
    #rows.append({"method": "DE-Grid+OCCA (Stage II only)", **prof_stage2, **res_stage2["metrics"]})

    # Use Stage-II y_pred for plotting DE-Grid+OCCA clusters
    #if "y_pred" in res_stage2:
    #    save_cluster_plot(X_raw, res_stage2["y_pred"], PLOTDIR, "DE-Grid+OCCA")

    # ---- Baselines & params (s_set1 defaults) ----
    #K = 7
    #EPS, MINPTS = 0.25, 25
    #MIN_CLUSTER_SIZE, MIN_SAMPLES = 10, 12
    #M, R_count = 10, 8  #12, 7
    K = 15
    EPS, MINPTS = 0.12, 12
    MIN_CLUSTER_SIZE, MIN_SAMPLES = 30, 12
    M, R_count = 33, 7

    # KMeans
    labels, prof = run_profiled(run_kmeans, Xs, K)
    rows.append({"method": "KMeans", **prof, **score_all(Xs, y_true, labels)})
    save_cluster_plot(X_raw, labels, PLOTDIR, "KMeans")

    # GMM
    labels, prof = run_profiled(run_gmm, Xs, K)
    rows.append({"method": "GMM", **prof, **score_all(Xs, y_true, labels)})
    save_cluster_plot(X_raw, labels, PLOTDIR, "GMM")

    # Agglomerative (Ward)
    labels, prof = run_profiled(run_agglomerative, Xs, K)
    rows.append({"method": "Agglomerative", **prof, **score_all(Xs, y_true, labels)})
    save_cluster_plot(X_raw, labels, PLOTDIR, "Agglomerative")

    # DBSCAN
    labels, prof = run_profiled(run_dbscan, Xs, EPS, MINPTS)
    rows.append({"method": "DBSCAN", **prof, **score_all(Xs, y_true, labels)})
    save_cluster_plot(X_raw, labels, PLOTDIR, "DBSCAN")

    # HDBSCAN
    if HAVE_HDBSCAN:
        try:
            labels, prof = run_profiled(run_hdbscan, Xs, MIN_CLUSTER_SIZE, MIN_SAMPLES)
            rows.append({"method": "HDBSCAN", **prof, **score_all(Xs, y_true, labels)})
            save_cluster_plot(X_raw, labels, PLOTDIR, "HDBSCAN")
        except Exception as e:
            rows.append({"method": "HDBSCAN", "time_sec": np.nan, "cpu_time_sec": np.nan,
                         "heap_peak_MB": np.nan, "rss_peak_MB": np.nan, "delta_rss_MB": np.nan,
                         **{k: np.nan for k in ["coverage","k","silhouette","dbi","ARI","NMI","V","FM","purity"]}})
            print("[warn] HDBSCAN failed:", e)
    else:
        rows.append({"method": "HDBSCAN", "time_sec": np.nan, "cpu_time_sec": np.nan,
                     "heap_peak_MB": np.nan, "rss_peak_MB": np.nan, "delta_rss_MB": np.nan,
                     **{k: np.nan for k in ["coverage","k","silhouette","dbi","ARI","NMI","V","FM","purity"]}})

    # CLIQUE
    if HAVE_CLIQUE:
        try:
            labels, prof = run_profiled(run_clique_labels, Xs, M, R_count)
            rows.append({"method": "CLIQUE", **prof, **score_all(Xs, y_true, labels)})
            save_cluster_plot(X_raw, labels, PLOTDIR, "CLIQUE")
        except Exception as e:
            rows.append({"method": "CLIQUE", "time_sec": np.nan, "cpu_time_sec": np.nan,
                         "heap_peak_MB": np.nan, "rss_peak_MB": np.nan, "delta_rss_MB": np.nan,
                         **{k: np.nan for k in ["coverage","k","silhouette","dbi","ARI","NMI","V","FM","purity"]}})
            print("[warn] CLIQUE failed:", e)
    else:
        rows.append({"method": "CLIQUE", "time_sec": np.nan, "cpu_time_sec": np.nan,
                     "heap_peak_MB": np.nan, "rss_peak_MB": np.nan, "delta_rss_MB": np.nan,
                     **{k: np.nan for k in ["coverage","k","silhouette","dbi","ARI","NMI","V","FM","purity"]}})

    df = pd.DataFrame(rows)

    # Formatting preview
    def fmt(v, nd=4):
        return "nan" if (not isinstance(v,(int,float,np.floating)) or not math.isfinite(v)) else f"{v:.{nd}f}"
    preview = df.copy()
    for c, nd in [("time_sec",3), ("cpu_time_sec",3), ("heap_peak_MB",1), ("rss_peak_MB",1), ("delta_rss_MB",1),
                  ("coverage",4), ("silhouette",4), ("dbi",4), ("ARI",4), ("NMI",4), ("V",4), ("FM",4), ("purity",4)]:
        if c in preview.columns:
            preview[c] = preview[c].apply(lambda x: fmt(x, nd))
    print("\n=== s_set1: CPU & Memory Profile (ΔRSS) ===")
    print(preview.to_string(index=False))

    # Save CSV/JSON
    df.to_csv(os.path.join(OUT, "s_set1_cpu_mem.csv"), index=False)
    with open(os.path.join(OUT, "s_set1_cpu_mem.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("\nSaved metrics to:", OUT)
    print("Saved plots to:", PLOTDIR)
