# bench_clusterers.py
# Reproducible benchmark: DE-Grid+OCCA (your method) vs. KMeans, GMM, Agglomerative,
# DBSCAN, HDBSCAN (optional), CLIQUE (pyclustering) with CPU time + peak memory.

import os
import json
import time
import math
import warnings
import tracemalloc
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score,
    davies_bouldin_score,
)

# --- Optional dependencies ---
HAVE_HDBSCAN = True
try:
    import hdbscan  # type: ignore
except Exception:
    HAVE_HDBSCAN = False

HAVE_CLIQUE = True
try:
    from pyclustering.cluster.clique import clique  # type: ignore
except Exception:
    HAVE_CLIQUE = False

# --- Your pipeline (import after venv install / editable install) ---
from clustx.core import run_pipeline_2d as run_clustx

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Utilities
# =============================================================================

def decode_label_series(s: pd.Series) -> pd.Series:
    """
    Convert byte-string style labels like b'3' -> '3' -> int when possible.
    Leaves strings if conversion to int fails.
    """
    s = s.astype(str).str.replace("b'", "", regex=False).str.replace("'", "", regex=False)
    try:
        return s.astype(int)
    except Exception:
        return s

def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Purity = (1/N) * sum_over_pred_clusters max_over_true_labels count(c ∩ t).
    Unassigned points (label < 0) are excluded from cluster counts but still
    present in N (= len(y_true)).
    """
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
    """
    Coverage, number of clusters k, silhouette, and DBI for assigned points (labels >= 0).
    """
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
    """
    Unified metric pack used in all rows.
    """
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

def time_and_peakmem(fn, *args, **kwargs):
    """
    Measure wall-clock time and Python heap peak (MB) using tracemalloc.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, elapsed, peak / (1024.0 * 1024.0)

# =============================================================================
# Baseline algorithm runners
# =============================================================================

def run_kmeans(X: np.ndarray, K: int) -> np.ndarray:
    mdl = KMeans(n_clusters=int(K), n_init="auto", random_state=0)
    mdl.fit(X)
    return mdl.labels_

def run_gmm(X: np.ndarray, K: int) -> np.ndarray:
    mdl = GaussianMixture(n_components=int(K), covariance_type="full", random_state=0)
    return mdl.fit_predict(X)

def run_agglomerative(X: np.ndarray, K: int) -> np.ndarray:
    mdl = AgglomerativeClustering(n_clusters=int(K), linkage="ward")
    return mdl.fit_predict(X)

def run_dbscan(X: np.ndarray, eps: float, minpts: int) -> np.ndarray:
    mdl = DBSCAN(eps=float(eps), min_samples=int(minpts))
    return mdl.fit_predict(X)

def run_hdbscan(X: np.ndarray, min_cluster_size: int) -> np.ndarray:
    if not HAVE_HDBSCAN:
        raise RuntimeError("hdbscan not installed.")
    mdl = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size))
    return mdl.fit_predict(X)

def run_clique_wrapper(X: np.ndarray, M: int, R_count: int) -> np.ndarray:
    """
    CLIQUE via pyclustering:
      - M: number of intervals per dimension (int)
      - R_count: integer threshold on number of points per cell (size_t)
    Data are normalized to [0,1] so M partitions per axis are meaningful.
    """
    if not HAVE_CLIQUE:
        raise RuntimeError("pyclustering not installed.")
    M = int(M)
    R_int = max(1, int(round(R_count)))

    X01 = (X - X.min(axis=0)) / (np.ptp(X, axis=0) + 1e-12)
    c = clique(X01.tolist(), M, R_int)
    c.process()
    clusters = c.get_clusters()

    labels = -np.ones(X01.shape[0], dtype=int)
    for cid, idxs in enumerate(clusters):
        if idxs:
            labels[np.asarray(idxs, dtype=int)] = cid
    return labels

# =============================================================================
# Params from literature table (interpreted)
#   Data1 = Aggregation, Data2 = R15, Data3 = S-set1
# =============================================================================

PARAMS = {
    "Aggregation": {
        "K": 15,
        "DBSCAN": {"eps": 0.15, "MinPts": 15},
        "HDBSCAN": {"min_cluster_size": 40},
        "CLIQUE": {"M": 33, "R": 7},      # R is COUNT per cell
    },
    "R15": {
        "K": 15,
        "DBSCAN": {"eps": 0.10, "MinPts": 19},
        "HDBSCAN": {"min_cluster_size": 45},
        "CLIQUE": {"M": 21, "R": 15},
    },
    "S-set1": {
        "K": 17,
        "DBSCAN": {"eps": 0.09, "MinPts": 18},
        "HDBSCAN": {"min_cluster_size": 45},
        "CLIQUE": {"M": 23, "R": 20},
    },
}

# =============================================================================
# Dataset & DE-Grid runner
# =============================================================================

def load_dataset(path: str, label_col: str = "class") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load CSV with columns ['x','y',label_col]. Decodes byte-like labels to ints if possible.
    """
    df = pd.read_csv(path)
    if label_col not in df.columns:
        # case-insensitive fallback
        for c in df.columns:
            if c.lower() == label_col.lower():
                label_col = c
                break
    y_true = decode_label_series(df[label_col]) if label_col in df.columns else None
    X = df[["x", "y"]].astype(float).to_numpy()
    return X, (y_true.to_numpy() if y_true is not None else None)

def run_all_for_dataset(name: str, csv_path: str, out_dir: str, degrid_kwargs: Dict) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    # Raw data (your method uses raw coords; baselines get standardized)
    X_raw, y_true = load_dataset(csv_path, label_col="class")
    X = StandardScaler().fit_transform(X_raw)

    results = []

    # ---- Your DE-Grid + OCCA (already tunes internally per degrid_kwargs) ----
    def _run_degrid():
        res = run_clustx(points_file=csv_path, out_dir=os.path.join(out_dir, "clustx"), **degrid_kwargs)
        beta = res.get("best_beta", None)
        iters = res.get("best_iters", None)
        M = res.get("metrics", {})
        return {
            "beta": beta,
            "iters": iters,
            "metrics_after_occa": M.get("after_occa", {}),
        }

    degrid_out, t_de, m_de = time_and_peakmem(_run_degrid)
    occa = degrid_out["metrics_after_occa"]
    results.append({
        "method": "DE-Grid+OCCA",
        "params": {"beta*": degrid_out["beta"], "iters*": degrid_out["iters"]},
        "time_sec": t_de, "peak_MB": m_de,
        **occa
    })

    # ---- Baselines ----
    K = int(PARAMS[name]["K"])

    # KMeans
    lbl, t, mem = time_and_peakmem(run_kmeans, X, K)
    results.append({"method": "KMeans", "params": {"K": K}, "time_sec": t, "peak_MB": mem,
                    **score_all(X, y_true, lbl)})

    # GMM
    lbl, t, mem = time_and_peakmem(run_gmm, X, K)
    results.append({"method": "GMM", "params": {"K": K}, "time_sec": t, "peak_MB": mem,
                    **score_all(X, y_true, lbl)})

    # Agglomerative (Ward)
    lbl, t, mem = time_and_peakmem(run_agglomerative, X, K)
    results.append({"method": "Agglomerative", "params": {"K": K, "linkage": "ward"},
                    "time_sec": t, "peak_MB": mem, **score_all(X, y_true, lbl)})

    # DBSCAN
    dbp = PARAMS[name]["DBSCAN"]
    lbl, t, mem = time_and_peakmem(run_dbscan, X, dbp["eps"], dbp["MinPts"])
    results.append({"method": "DBSCAN", "params": dbp, "time_sec": t, "peak_MB": mem,
                    **score_all(X, y_true, lbl)})

    # HDBSCAN (optional)
    if HAVE_HDBSCAN:
        hdp = PARAMS[name]["HDBSCAN"]
        lbl, t, mem = time_and_peakmem(run_hdbscan, X, hdp["min_cluster_size"])
        results.append({"method": "HDBSCAN", "params": hdp, "time_sec": t, "peak_MB": mem,
                        **score_all(X, y_true, lbl)})
    else:
        results.append({"method": "HDBSCAN", "params": "(unavailable)", "time_sec": np.nan, "peak_MB": np.nan,
                        **{k: np.nan for k in ["coverage","k","silhouette","dbi","ARI","NMI","V","FM","purity"]}})

    # CLIQUE (optional; threshold R is COUNT, not density)
    if HAVE_CLIQUE:
        cp = PARAMS[name]["CLIQUE"]
        try:
            M = int(cp["M"])
            R = int(cp["R"])
            lbl, t, mem = time_and_peakmem(run_clique_wrapper, X, M, R)
            results.append({"method": "CLIQUE", "params": {"M": M, "R": R}, "time_sec": t, "peak_MB": mem,
                            **score_all(X, y_true, lbl)})
        except Exception as e:
            results.append({"method": "CLIQUE", "params": {"M": cp["M"], "R": cp["R"]}, "time_sec": np.nan, "peak_MB": np.nan,
                            **{k: np.nan for k in ["coverage","k","silhouette","dbi","ARI","NMI","V","FM","purity"]}})
            print(f"[warn] CLIQUE failed on {name}: {e}")
    else:
        results.append({"method": "CLIQUE", "params": "(unavailable)", "time_sec": np.nan, "peak_MB": np.nan,
                        **{k: np.nan for k in ["coverage","k","silhouette","dbi","ARI","NMI","V","FM","purity"]}})

    # Save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, f"{name}_benchmark.csv"), index=False)
    with open(os.path.join(out_dir, f"{name}_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)
    return df

# =============================================================================
# LaTeX table printer
# =============================================================================

def print_latex_table(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Pretty LaTeX table to paste into Overleaf (accuracy + efficiency).
    """
    disp = df.copy()

    def _fmt_num(v, nd=4):
        if not isinstance(v, (int, float, np.floating)) or not math.isfinite(v):
            return "nan"
        return f"{v:.{nd}f}"

    # Format columns
    for col, nd in [
        ("coverage", 4), ("silhouette", 4), ("dbi", 4),
        ("ARI", 4), ("NMI", 4), ("V", 4), ("FM", 4), ("purity", 4),
    ]:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda x: _fmt_num(x, nd))
    if "time_sec" in disp.columns:
        disp["time_sec"] = disp["time_sec"].apply(lambda x: _fmt_num(x, 3))
    if "peak_MB" in disp.columns:
        disp["peak_MB"] = disp["peak_MB"].apply(lambda x: _fmt_num(x, 1))

    # Ensure all expected columns exist
    cols = ["method","k","coverage","ARI","NMI","V","FM","purity","time_sec","peak_MB"]
    for c in cols:
        if c not in disp.columns:
            disp[c] = ""

    print(r"\begin{table}[ht!]")
    print(r"\centering")
    print(rf"\caption{{Benchmark on {dataset_name}: accuracy vs.\ efficiency. CPU time is wall-clock (s) and memory is peak Python heap (MB).}}")
    print(rf"\label{{tab:{dataset_name.lower().replace(' ','-')}-bench}}")
    print(r"\renewcommand{\arraystretch}{1.15}")
    print(r"\setlength{\tabcolsep}{6pt}")
    print(r"\begin{tabular}{lccccccccc}")
    print(r"\hline")
    print(r"\textbf{Method} & \textbf{$k$} & \textbf{Coverage} & \textbf{ARI} & \textbf{NMI} & \textbf{V} & \textbf{FM} & \textbf{Purity} & \textbf{Time (s)} & \textbf{Peak (MB)} \\")
    print(r"\hline")
    for _, r in disp.iterrows():
        print(f"{r['method']} & {r['k']} & {r['coverage']} & {r['ARI']} & {r['NMI']} & {r['V']} & {r['FM']} & {r['purity']} & {r['time_sec']} & {r['peak_MB']} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    HERE = os.path.dirname(__file__)
    DATA = {
        "Aggregation": os.path.join(HERE, "aggregation.csv"),
        "R15":         os.path.join(HERE, "R15.csv"),
        "S-set1":      os.path.join(HERE, "s_set1.csv"),
    }

    # Use the same tuned settings you already found for your method
    DEGRID_DEFAULTS = {
        "TUNING": "bo",
        "BO_OPT_WEIGHTS": False,
        "W_SIL": 0.33, "W_DBI": 0.34, "W_COV": 0.33,
        "BO_N_CALLS": 50,
        "H_BOUNDS_REL": (0.5, 1.25),
        "R_RANGE": (2, 20),
        "BETA_CANDIDATES": (0.1, 0.2, 0.25),
        "CTHR_VALUES": (0.01, 0.02, 0.05, 0.1),
        "MAX_ITERS": 5000, "MIN_ITERS": 100, "TOL": 1e-6, "CHECK_EVERY": 10,
        "K_MIN": 2, "K_MAX": 50,
        "PERIODIC_CCA": False,
        "CONNECTIVITY": 4,
        "MAKE_PLOTS": False, "DO_STD_CCA": True,
        "LABEL_COL": "class",
    }

    OUT = os.path.join(HERE, "bench_out")
    os.makedirs(OUT, exist_ok=True)

    for name, path in DATA.items():
        df = run_all_for_dataset(name, path, os.path.join(OUT, name), DEGRID_DEFAULTS)
        # LaTeX for each dataset
        print("\n===== LaTeX table:", name, "=====")
        print_latex_table(df, name)
