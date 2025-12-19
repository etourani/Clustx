# core.py
from __future__ import annotations

import os
import math
import json
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ----- Optional skopt (for Bayesian tuning) -----
_HAS_SKOPT = True
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
except Exception:
    _HAS_SKOPT = False


import matplotlib as mpl
import seaborn as sns

# ─── STYLING ────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
mpl.rcParams['font.family'] = 'Nimbus Roman'
mpl.rcParams['font.size'] = 30
mpl.rcParams['axes.labelsize'] = 36
mpl.rcParams['axes.titlesize'] = 38
mpl.rcParams['xtick.labelsize'] = 32
mpl.rcParams['ytick.labelsize'] = 32
mpl.rcParams['legend.fontsize'] = 28
mpl.rcParams['legend.title_fontsize'] = 34
mpl.rcParams['figure.titlesize'] = 36
mpl.rcParams['grid.alpha'] = 0.2
mpl.rcParams['grid.linewidth'] = 1.0
mpl.rcParams['grid.linestyle'] = '--'


# =============================================================================
# 1) GRID SUGGESTION & PREPROCESSING
# =============================================================================

def suggest_grid_size(points: np.ndarray,
                      k: int = 5,
                      alpha: float = 1.0,
                      target_occ: Optional[float] = 4,
                      fd_backup: bool = True,
                      sweep_pct: float = 0.2,
                      max_bins: int = 200) -> Dict:
    """
    Suggest a near-isotropic cell edge h from (kNN / occupancy / Freedman–Diaconis),
    then produce candidate (nx, ny) grids around h (±sweep_pct).
    """
    x = points[:, 0]; y = points[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    Lx, Ly = x_max - x_min, y_max - y_min
    N = points.shape[0]

    # (1) kNN spacing
    try:
        dists, _ = cKDTree(points).query(points, k=k+1)
        s_k = float(np.median(dists[:, -1]))
        h_knn = max(1e-12, alpha * s_k)
    except Exception:
        h_knn = None

    # (2) Occupancy-based h (preserve aspect; aim avg occupancy ~ target_occ)
    h_occ = None
    if target_occ is not None and target_occ > 0:
        total_cells = max(1, int(round(N / float(target_occ))))
        r = Lx / (Ly + 1e-12)
        ny = int(round(np.sqrt(total_cells / (r + 1e-12))))
        nx = int(round(r * ny))
        nx = max(1, min(nx, max_bins))
        ny = max(1, min(ny, max_bins))
        hx = Lx / nx if nx > 0 else Lx
        hy = Ly / ny if ny > 0 else Ly
        h_occ = float(np.sqrt(hx * hy))

    # (3) Freedman–Diaconis per axis -> geometric mean
    h_fd = None
    if fd_backup:
        def fd_h(arr: np.ndarray) -> float:
            q75, q25 = np.percentile(arr, [75, 25])
            iqr = q75 - q25
            return 2.0 * iqr / (N ** (1.0/3.0) + 1e-12)
        bx = max(fd_h(x), 1e-12)
        by = max(fd_h(y), 1e-12)
        h_fd = float(np.sqrt(bx * by))

    seeds = [h for h in [h_knn, h_occ, h_fd] if h is not None and np.isfinite(h)]
    h0 = float(np.mean(seeds)) if seeds else max(1e-12, 0.05 * max(Lx, Ly))
    print(h_knn, h_occ, h_fd, h0)

    nx0 = max(1, min(int(np.ceil(Lx / h0)), max_bins))
    ny0 = max(1, min(int(np.ceil(Ly / h0)), max_bins))
    h_est = float(max(Lx / nx0, Ly / ny0))

    hs = np.linspace((1 - sweep_pct) * h_est, (1 + sweep_pct) * h_est, 7)
    hs = np.clip(hs, 1e-12, max(Lx, Ly))

    candidates: List[Tuple[int, int, float]] = []
    seen = set()
    for h in hs:
        nx = max(1, min(int(np.ceil(Lx / h)), max_bins))
        ny = max(1, min(int(np.ceil(Ly / h)), max_bins))
        key = (nx, ny)
        if key not in seen:
            candidates.append((nx, ny, float(max(Lx / nx, Ly / ny))))
            seen.add(key)

    return {
        "bounds": (x_min, x_max, y_min, y_max),
        "knn_h": h_knn, "occ_h": h_occ, "fd_h": h_fd,
        "initial": (nx0, ny0, h_est),
        "candidates": candidates
    }


def bin_points_rect(points: np.ndarray,
                    nx: int, ny: int,
                    x_min: float, x_max: float,
                    y_min: float, y_max: float) -> Tuple[np.ndarray, float, float]:
    """Count points in (nx, ny) rectilinear bins covering [x_min,x_max]x[y_min,y_max]."""
    counts = np.zeros((nx, ny), dtype=int)
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dx = dx if dx > 0 else 1.0
    dy = dy if dy > 0 else 1.0
    ii = np.floor((points[:, 0] - x_min) / dx).astype(int)
    jj = np.floor((points[:, 1] - y_min) / dy).astype(int)
    ii = np.clip(ii, 0, nx - 1)
    jj = np.clip(jj, 0, ny - 1)
    np.add.at(counts, (ii, jj), 1)
    return counts, dx, dy


def normalize_nonzero(counts: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Min-max normalize only on non-zero cells; zeros remain 0."""
    idx = counts > 0
    if np.any(idx):
        vmin = int(counts[idx].min())
        vmax = int(counts[idx].max())
        denom = (vmax - vmin) if vmax > vmin else 1.0
        norm = np.zeros_like(counts, dtype=float)
        norm[idx] = (counts[idx] - vmin) / denom
        return norm, vmin, vmax
    return np.zeros_like(counts, dtype=float), 0, 0


def preselect_grid(points: np.ndarray,
                   target_occ: int = 4,
                   k: int = 5,
                   alpha: float = 1.0,
                   fd_backup: bool = True,
                   sweep_pct: float = 0.2,
                   max_bins: int = 200) -> Dict:
    """
    High-level preprocessing:
      - suggest h and candidate grids,
      - choose the initial (nx, ny, h),
      - bin points and build masks needed by diffusion/CCA later.
    """
    s = suggest_grid_size(points, k=k, alpha=alpha, target_occ=target_occ,
                          fd_backup=fd_backup, sweep_pct=sweep_pct, max_bins=max_bins)
    x_min, x_max, y_min, y_max = s["bounds"]
    nx0, ny0, h0 = s["initial"]
    dx = (x_max - x_min) / nx0 if nx0 > 0 else (x_max - x_min)
    dy = (y_max - y_min) / ny0 if ny0 > 0 else (y_max - y_min)
    counts, dx, dy = bin_points_rect(points, nx0, ny0, x_min, x_max, y_min, y_max)
    nonempty = counts > 0
    norm, vmin, vmax = normalize_nonzero(counts)

    return {
        "suggest": s,
        "bounds": (x_min, x_max, y_min, y_max),
        "grid": {"nx": nx0, "ny": ny0, "dx": dx, "dy": dy, "h": h0},
        "counts": counts,
        "nonempty_mask": nonempty,
        "norm": norm,
        "norm_minmax": (vmin, vmax),
    }


# =============================================================================
# 2) CCA / O-CCA (periodic + connectivity aligned)
# =============================================================================

def _wrap_idx_2d(i: int, j: int, nx: int, ny: int, periodic: bool):
    if periodic:
        return (i % nx, j % ny)
    if 0 <= i < nx and 0 <= j < ny:
        return (i, j)
    return None

def _idx(i: int, j: int, ny: int) -> int:
    return i * ny + j

def _ufind_find(parent: Dict[int, int], a: int) -> int:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a

def _ufind_union(parent: Dict[int, int], rank: Dict[int, int], a: int, b: int) -> None:
    ra = _ufind_find(parent, a)
    rb = _ufind_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[rb] < rank[ra]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1

_NEIGH4_FWD = [(1, 0), (0, 1)]
_NEIGH4_ALL = [(-1,0), (1,0), (0,-1), (0,1)]
_NEIGH8_ALL = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

def standard_cca_2d(selected: np.ndarray,
                    periodic: bool = False,
                    connectivity: int = 4) -> Tuple[np.ndarray, int]:
    nx, ny = selected.shape
    parent, rank = {}, {}
    for i in range(nx):
        for j in range(ny):
            if selected[i, j]:
                k = _idx(i, j, ny)
                parent[k] = k
                rank[k] = 0

    if connectivity == 4:
        forwards = _NEIGH4_FWD
    elif connectivity == 8:
        forwards = [(1, 0), (0, 1), (1, 1), (1, -1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    for i in range(nx):
        for j in range(ny):
            if not selected[i, j]:
                continue
            a = _idx(i, j, ny)
            for di, dj in forwards:
                nb = _wrap_idx_2d(i + di, j + dj, nx, ny, periodic)
                if nb is None:
                    continue
                ni, nj = nb
                if selected[ni, nj]:
                    _ufind_union(parent, rank, a, _idx(ni, nj, ny))

    labels = -np.ones((nx, ny), dtype=int)
    root_to_id, next_id = {}, 0
    for i in range(nx):
        for j in range(ny):
            if selected[i, j]:
                r = _ufind_find(parent, _idx(i, j, ny))
                if r not in root_to_id:
                    root_to_id[r] = next_id
                    next_id += 1
                labels[i, j] = root_to_id[r]
    return labels, next_id

# Backward-compat alias
standard_cca = standard_cca_2d

def seeded_custom_cca(selected_after: np.ndarray,
                      dense_before: np.ndarray,
                      periodic: bool = False,
                      connectivity: int = 4,
                      max_passes: Optional[int] = None) -> Tuple[np.ndarray, int]:
    if connectivity == 4:
        neighs = _NEIGH4_ALL
    elif connectivity == 8:
        neighs = _NEIGH8_ALL
    else:
        raise ValueError("connectivity must be 4 or 8")

    seeds, _ = standard_cca_2d(dense_before, periodic=periodic, connectivity=connectivity)
    nx, ny = selected_after.shape
    labels = -np.ones((nx, ny), dtype=int)

    seed_mask = (seeds >= 0)
    if not np.any(seed_mask):
        return labels, 0
    labels[seed_mask] = seeds[seed_mask]

    passes = 0
    changed = True
    while changed and (max_passes is None or passes < max_passes):
        changed = False
        passes += 1
        for i in range(nx):
            for j in range(ny):
                if not selected_after[i, j] or labels[i, j] >= 0:
                    continue
                nbrs = set()
                for di, dj in neighs:
                    nb = _wrap_idx_2d(i + di, j + dj, nx, ny, periodic)
                    if nb is None:
                        continue
                    li, lj = nb
                    lab = labels[li, lj]
                    if lab >= 0:
                        nbrs.add(int(lab))
                        if len(nbrs) > 1:
                            break
                if len(nbrs) == 1:
                    labels[i, j] = next(iter(nbrs))
                    changed = True

    uniq = sorted(int(v) for v in np.unique(labels) if v >= 0)
    remap = {old: k for k, old in enumerate(uniq)}
    if remap:
        for i in range(nx):
            for j in range(ny):
                if labels[i, j] >= 0:
                    labels[i, j] = remap[int(labels[i, j])]
    return labels, len(uniq)


# =============================================================================
# 3) DIFFUSION (periodic aligned)
# =============================================================================

def run_diffusion_until_converged(
    norm: np.ndarray,           # [0,1]
    dense_mask: np.ndarray,     # clamp to 1
    update_mask: np.ndarray,    # candidate region (subset of non-dense)
    empty_mask: np.ndarray,     # clamp to 0
    *,
    beta: float = 0.10,
    max_iters: int = 50000,
    min_iters: int = 60,
    tol: float = 1e-6,
    check_every: int = 10,
    dense_thr_for_w: float = 0.30,
    periodic: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Weighted explicit diffusion (5-pt Laplacian). BCs match `periodic`.
    """
    c = np.zeros_like(norm, dtype=float)
    c[dense_mask] = 1.0

    upd = (update_mask & (~dense_mask) & (~empty_mask))
    w = np.zeros_like(norm, dtype=float)
    if np.any(upd):
        eps = 1e-12
        w[upd] = np.clip(norm[upd] / max(dense_thr_for_w, eps), 0.0, 1.0)

    lap = np.array([[0.0, 1.0, 0.0],
                    [1.0,-4.0, 1.0],
                    [0.0, 1.0, 0.0]], dtype=float)

    mode = "wrap" if periodic else "nearest"

    it = 0
    max_delta_last = None
    while it < max_iters:
        it += 1
        lap_c = ndimage.convolve(c, lap, mode=mode)
        dc = np.zeros_like(c)
        if np.any(upd):
            dc_upd = beta * w[upd] * lap_c[upd]
            dc[upd] = dc_upd
            c[upd] = np.clip(c[upd] + dc_upd, 0.0, 1.0)

        c[dense_mask] = 1.0
        c[empty_mask] = 0.0

        if it % check_every == 0:
            max_delta = float(np.max(np.abs(dc[upd]))) if np.any(upd) else 0.0
            max_delta_last = max_delta
            if it >= min_iters and max_delta < tol:
                break

    stops_on = (f"converged (max|Δc|={max_delta_last:.2e})"
                if (it < max_iters and max_delta_last is not None and max_delta_last < tol)
                else f"max_iters ({max_iters})")
    return c, {"iters_used": it, "max_delta_last": max_delta_last, "stops_on": stops_on}


# =============================================================================
# 4) SCORING UTILITIES (no regularization)
# =============================================================================

def points_to_labels(points: np.ndarray,
                     label_grid: np.ndarray,
                     x_min: float, x_max: float,
                     y_min: float, y_max: float,
                     dx: float, dy: float) -> np.ndarray:
    nx, ny = label_grid.shape
    ii = np.floor((points[:, 0] - x_min) / dx).astype(int)
    jj = np.floor((points[:, 1] - y_min) / dy).astype(int)
    ii = np.clip(ii, 0, nx - 1)
    jj = np.clip(jj, 0, ny - 1)
    return label_grid[ii, jj]


def score_partition(points: np.ndarray, y: np.ndarray,
                    k_min: int = 2, k_max: int = 50,
                    w_sil: float = 0.0, w_dbi: float = 0.8,
                    w_cov: float = 0.15) -> Dict:
    """
    Composite score = w_sil * silhouette + w_dbi * (1/(1+DBI)) + w_cov * coverage.
    No regularization terms.
    """
    mask = (y >= 0)
    if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
        return {"sil": 0.0, "dbi": np.inf, "cov": float(mask.mean()), "k": 0, "score": -np.inf}
    X = points[mask]
    labels = y[mask]
    try:
        sil = float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        sil = 0.0
    try:
        dbi = float(davies_bouldin_score(X, labels))
    except Exception:
        dbi = np.inf
    cov = float(mask.mean())
    k = int(len(np.unique(labels)))
    dbi_norm = 1.0 / (1.0 + (dbi if math.isfinite(dbi) else 1e6))

    total = (w_sil * sil) + (w_dbi * dbi_norm) + (w_cov * cov)
    return {"sil": sil, "dbi": dbi, "cov": cov, "k": k, "score": total}


# =============================================================================
# 5) HELPER FOR STAGE-A EVALUATION (h,R + weights) — no regularization arg
# =============================================================================

def _eval_hr_once(points, x_min, x_max, y_min, y_max, Lx, Ly,
                  h, R, W_SIL, W_DBI, W_COV, K_MIN, K_MAX, MAX_BINS,
                  PERIODIC_CCA=False):
    nx = max(1, min(int(np.ceil(Lx / max(h, 1e-12))), MAX_BINS))
    ny = max(1, min(int(np.ceil(Ly / max(h, 1e-12))), MAX_BINS))
    if nx * ny < 2:
        return None, "too-few-cells"

    counts, dx, dy = bin_points_rect(points, nx, ny, x_min, x_max, y_min, y_max)
    nonempty = counts > 0
    if nonempty.sum() < 2:
        return None, "no-occupancy"

    dense = (counts >= int(R)) & nonempty
    if dense.sum() < 1:
        return None, "no-dense"

    labels_grid, _ = standard_cca(dense, periodic=PERIODIC_CCA)
    y_pts = points_to_labels(points, labels_grid, x_min, x_max, y_min, y_max, dx, dy)
    m = score_partition(points, y_pts, K_MIN, K_MAX, W_SIL, W_DBI, W_COV)
    if not np.isfinite(m["score"]):
        return None, "bad-score"

    return {
        "nx": int(nx), "ny": int(ny), "R": int(R),
        "dx": float(dx), "dy": float(dy),
        "sil": float(m["sil"]), "dbi": float(m["dbi"]),
        "cov": float(m["cov"]), "k": int(m["k"]),
        "score": float(m["score"])
    }, "ok"


# =============================================================================
# 6) PLOTTING
# =============================================================================

def plot_cluster_labels_with_points(labels, points, dense_before, title, pdf_path,
                                    x_min, x_max, y_min, y_max, dx, dy):
    nx, ny = labels.shape
    #fig, ax = plt.subplots(figsize=(6, 6))
    marker_size = mpl.rcParams['font.size'] / 2
    figsize_scale = 1.2
    fig, ax = plt.subplots(figsize=(10 * figsize_scale, 10 * figsize_scale), sharex=True, sharey=True)

    cluster_ids = sorted([int(v) for v in np.unique(labels) if v >= 0])
    colors = sns.color_palette("hls", len(cluster_ids)) if len(cluster_ids) > 0 else []
    id_to_color = {cid: colors[i] for i, cid in enumerate(cluster_ids)} if cluster_ids else {}

    for i in range(nx):
        for j in range(ny):
            x0 = x_min + i * dx
            y0 = y_min + j * dy
            lab = labels[i, j]
            rect = plt.Rectangle(
                (x0, y0), dx, dy,
                facecolor="white" if lab < 0 else id_to_color.get(lab, (0.8, 0.8, 0.8)),
                edgecolor="gray", linewidth=0.4, alpha=0.17 if lab < 0 else 0.9
            )
            ax.add_patch(rect)

    for i in range(nx + 1):
        x = x_min + i * dx
        ax.plot([x, x], [y_min, y_max], color="gray", linewidth=0.4, alpha=0.17)
    for j in range(ny + 1):
        y = y_min + j * dy
        ax.plot([x_min, x_max], [y, y], color="gray", linewidth=0.4, alpha=0.17)

    ii = np.floor((points[:, 0] - x_min) / dx).astype(int)
    jj = np.floor((points[:, 1] - y_min) / dy).astype(int)
    ii = np.clip(ii, 0, nx - 1)
    jj = np.clip(jj, 0, ny - 1)
    dense_mask = np.array([dense_before[ii[k], jj[k]] for k in range(points.shape[0])], dtype=bool)
    sp = points[~dense_mask]
    dp = points[dense_mask]
    if sp.size:
        ax.scatter(sp[:, 0], sp[:, 1], s=50, alpha=0.5, color="gray", edgecolor="k")
    if dp.size:
        ax.scatter(dp[:, 0], dp[:, 1], s=50, alpha=0.6, color="indianred", edgecolor="k")

    # Draw bold outer edges
    ax.plot([x_min, x_max], [y_min, y_min], linewidth=1.9, color="dimgray", alpha=0.7)  # bottom
    ax.plot([x_min, x_max], [y_max, y_max], linewidth=1.9, color="dimgray", alpha=0.7)  # top
    ax.plot([x_min, x_min], [y_min, y_max], linewidth=1.9, color="dimgray", alpha=0.7)  # left
    ax.plot([x_max, x_max], [y_min, y_max], linewidth=1.9, color="dimgray", alpha=0.7)  # right

    ax.set_title(title, pad=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=300, pad_inches=0.05) # bbox_inches="tight",
    #plt.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

# =============================================================================
# X) METRICS HELPERS (external + internal + purity)
# =============================================================================
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score as _silhouette_score,
    davies_bouldin_score as _dbi_score,
)

def _find_true_labels_column(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Try common ground-truth column names. Returns None if not found.
    """
    CANDIDATES = ["label", "Label", "labels", "Labels", "class", "Class", "target", "y_true", "gt"]
    for col in CANDIDATES:
        if col in df.columns:
            return df[col].to_numpy()
    return None

def _purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Purity = (1/N) * sum_c max_t | { i : y_pred_i = c and y_true_i = t } |
    Only defined if both y_true and y_pred exist and have same length.
    """
    import collections
    if y_true is None or y_pred is None or len(y_true) != len(y_pred):
        return float("nan")
    N = len(y_true)
    if N == 0:
        return float("nan")
    labels_pred = np.unique(y_pred[y_pred >= 0])
    total = 0
    for c in labels_pred:
        idx = (y_pred == c)
        if not np.any(idx):
            continue
        counts = collections.Counter(y_true[idx])
        total += counts.most_common(1)[0][1]
    return total / float(N)

def _cluster_report(points: np.ndarray, y_pred: np.ndarray, y_true: Optional[np.ndarray]) -> Dict:
    """
    Compute a unified set of metrics.
    - Internal: silhouette, DBI
    - External (if y_true is available): ARI, NMI, V-measure, Fowlkes-Mallows (FM), purity
    - Utility: coverage (fraction of assigned points), k
    """
    m: Dict[str, float] = {}
    mask = (y_pred >= 0)
    k = int(len(np.unique(y_pred[mask]))) if np.any(mask) else 0
    m["coverage"] = float(mask.mean())
    m["k"] = k

    # Internal scores
    if mask.sum() >= 2 and k >= 2:
        try:
            m["silhouette"] = float(_silhouette_score(points[mask], y_pred[mask], metric="euclidean"))
        except Exception:
            m["silhouette"] = float("nan")
        try:
            m["dbi"] = float(_dbi_score(points[mask], y_pred[mask]))
        except Exception:
            m["dbi"] = float("nan")
    else:
        m["silhouette"] = float("nan")
        m["dbi"] = float("nan")

    # External scores (need ground truth)
    if y_true is not None and len(y_true) == len(y_pred):
        try:
            m["ARI"] = float(adjusted_rand_score(y_true, np.where(y_pred < 0, -1, y_pred)))
        except Exception:
            m["ARI"] = float("nan")
        try:
            m["NMI"] = float(normalized_mutual_info_score(y_true, np.where(y_pred < 0, -1, y_pred)))
        except Exception:
            m["NMI"] = float("nan")
        try:
            m["V_measure"] = float(v_measure_score(y_true, np.where(y_pred < 0, -1, y_pred)))
        except Exception:
            m["V_measure"] = float("nan")
        try:
            m["FM"] = float(fowlkes_mallows_score(y_true, np.where(y_pred < 0, -1, y_pred)))
        except Exception:
            m["FM"] = float("nan")
        try:
            m["purity"] = float(_purity(y_true, np.where(y_pred < 0, -1, y_pred)))
        except Exception:
            m["purity"] = float("nan")
    else:
        # No ground truth available
        m["ARI"] = float("nan")
        m["NMI"] = float("nan")
        m["V_measure"] = float("nan")
        m["FM"] = float("nan")
        m["purity"] = float("nan")

    return m

# =============================================================================
# 7) MAIN 2D PIPELINE (with 5D BO; no regularization anywhere)
# =============================================================================

def run_pipeline_2d(
    points_file,
    out_dir="./",
    # ---- FIXED MODE (fast path) ----
    FIXED_GRID=None,            # (nx,ny) or None
    FIXED_DENSE_THR=None,       # [0,1]
    FIXED_BETA=None,
    FIXED_CTHR=None,            # [0,1]
    # ---- TUNED MODE (used if FIXED_GRID is None) ----
    TUNING="grid",              # 'grid' or 'bo'
    H_BOUNDS_REL=(0.5, 1.8),    # for 'bo'
    R_RANGE=(1, 30),            # for 'bo'
    BO_N_CALLS=35,
    BO_OPT_WEIGHTS=True,        # if True, Stage-A BO jointly tunes (h,R,W_sil,W_dbi,W_cov)
    BO_WEIGHT_FLOOR=0.10,       # each weight at least 0.10
    BO_WEIGHT_CEIL=0.90,        # soft ceiling to avoid collapse
    # ---- Grid suggester knobs ----
    K_FOR_KNN=5, ALPHA_FOR_KNN=0.8, TARGET_OCC=4, FD_BACKUP=True,
    SWEEP_PCT=0.4, MAX_BINS=300,
    # ---- Stage-A (quantile mode search) ----
    DENSE_QUANTILES=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
    # ---- Stage-B (diffusion) ----
    BETA_CANDIDATES=(0.02, 0.05, 0.10, 0.20, 0.50),
    CTHR_VALUES=tuple(np.round(np.arange(0.01, 0.21, 0.02), 4)),
    MAX_ITERS=50000, MIN_ITERS=60, TOL=1e-6, CHECK_EVERY=10,
    # ---- scoring weights + k band (CLI defaults; Stage-B may get replaced by BO weights) ----
    W_SIL=0.0, W_DBI=0.80, W_COV=0.15, K_MIN=2, K_MAX=50,
    # ---- runtime ----
    PERIODIC_CCA=True,
    CONNECTIVITY=4,
    MAKE_PLOTS=True,
    DO_STD_CCA=True,
    **_
):
    # --- alias support for callers that use tuning= ---
    if isinstance(_, dict) and "tuning" in _:
        TUNING = str(_.pop("tuning")).lower()
    else:
        TUNING = str(TUNING).lower()
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Load data ----------
    df = pd.read_csv(points_file)
    if not set(["x", "y"]).issubset(df.columns):
        raise ValueError("CSV must contain columns 'x' and 'y'.")
    points = df[["x", "y"]].astype(float).to_numpy()

    # bounds with tiny padding
    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())
    pad_x = 1e-9 * (x_max - x_min + 1.0)
    pad_y = 1e-9 * (y_max - y_min + 1.0)
    x_min -= pad_x; x_max += pad_x; y_min -= pad_y; y_max += pad_y
    Lx, Ly = (x_max - x_min), (y_max - y_min)

    # ===================== FIXED MODE =====================
    if FIXED_GRID is not None:
        nx, ny = map(int, FIXED_GRID)
        if any(v is None for v in (FIXED_DENSE_THR, FIXED_BETA, FIXED_CTHR)):
            raise ValueError("Fixed mode requires FIXED_DENSE_THR, FIXED_BETA, and FIXED_CTHR.")

        counts, dx, dy = bin_points_rect(points, nx, ny, x_min, x_max, y_min, y_max)
        norm, _, _ = normalize_nonzero(counts)
        nonempty = (counts > 0)

        dense_thr = float(FIXED_DENSE_THR)
        beta = float(FIXED_BETA)
        cthr = float(FIXED_CTHR)

        dense_before  = nonempty & (norm >= dense_thr)
        sparse_before = nonempty & ~dense_before
        empty_before  = ~nonempty

        c, dstats = run_diffusion_until_converged(
            norm, dense_mask=dense_before, update_mask=sparse_before, empty_mask=empty_before,
            beta=beta, max_iters=MAX_ITERS, min_iters=MIN_ITERS, tol=TOL, check_every=CHECK_EVERY,
            dense_thr_for_w=dense_thr, periodic=PERIODIC_CCA
        )

        selected_after = np.zeros_like(dense_before, dtype=bool)
        selected_after[dense_before] = True
        if np.any(sparse_before):
            selected_after[sparse_before] = (c[sparse_before] > cthr)

        labels_after_std, _    = standard_cca(selected_after, periodic=PERIODIC_CCA, connectivity=CONNECTIVITY)
        labels_after_seeded, _ = seeded_custom_cca(selected_after, dense_before, periodic=PERIODIC_CCA, connectivity=CONNECTIVITY)

        plots = {}
        if MAKE_PLOTS:
            if DO_STD_CCA:
                p_std = os.path.join(out_dir, f"fig2D_after_std_beta{beta}_cthr{cthr}.pdf")
                plot_cluster_labels_with_points(labels_after_std, points, dense_before,
                                                "After diffusion (standard CCA, 2D)",
                                                p_std, x_min, x_max, y_min, y_max, dx, dy)
                plots["after_std_2d"] = p_std
            p_seed = os.path.join(out_dir, f"fig2D_after_occa_beta{beta}_cthr{cthr}.pdf")
            plot_cluster_labels_with_points(labels_after_seeded, points, dense_before,
                                            "After diffusion (O-CCA, 2D)",
                                            p_seed, x_min, x_max, y_min, y_max, dx, dy)
            plots["after_occa_2d"] = p_seed

        out = {
            "mode": "fixed",
            "fixed_params": {
                "nx": int(nx), "ny": int(ny),
                "dense_thr": float(dense_thr),
                "beta": float(beta),
                "cthr": float(cthr),
                "dx": float(dx), "dy": float(dy),
                "periodic_cca": bool(PERIODIC_CCA),
            },
            "diffusion_stats": {
                "iters_used": int(dstats["iters_used"]),
                "stop": dstats["stops_on"],
            },
            "plots": plots,
        }
        with open(os.path.join(out_dir, "summary2D_fixed.json"), "w") as f:
            json.dump(out, f, indent=2)
        return out

    # ===================== TUNED MODE =====================
    sugg = suggest_grid_size(
        points, k=K_FOR_KNN, alpha=ALPHA_FOR_KNN,
        target_occ=TARGET_OCC, fd_backup=FD_BACKUP,
        sweep_pct=SWEEP_PCT, max_bins=MAX_BINS
    )
    cand_rect = []
    seen = set()
    for nx, ny, _h in sugg["candidates"]:
        if (nx, ny) not in seen:
            cand_rect.append((nx, ny)); seen.add((nx, ny))
    nx0, ny0, h_est = sugg["initial"]
    if (nx0, ny0) not in seen:
        cand_rect.insert(0, (nx0, ny0))

    # ---------- helper: normalize weight triplet (used in objective only) ----------
    def _normalize_weights_tuple(w_sil, w_dbi, w_cov,
                                 lo=BO_WEIGHT_FLOOR, hi=BO_WEIGHT_CEIL):
        w = np.array([w_sil, w_dbi, w_cov], dtype=float)
        w = np.clip(w, lo, hi)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            return (1/3, 1/3, 1/3)
        w /= s
        return (float(w[0]), float(w[1]), float(w[2]))

    # ---------- Stage A search (grid or 5D BO) ----------
    stageA_rows = []
    bestA = None

    if TUNING.lower() == "bo":
        if not _HAS_SKOPT:
            raise RuntimeError("Bayesian tuning selected, but scikit-optimize is not installed.")

        seeds = [sugg.get("knn_h"), sugg.get("occ_h"), sugg.get("fd_h"), h_est]
        seeds = [float(v) for v in seeds if v is not None and np.isfinite(v)]
        h0 = float(np.median(seeds)) if len(seeds) else max(1e-12, 0.05 * max(Lx, Ly))
        hmin = H_BOUNDS_REL[0] * h0
        hmax = H_BOUNDS_REL[1] * h0
        R_lo, R_hi = R_RANGE

        # Prior / fixed weights from args (used if BO_OPT_WEIGHTS=False)
        w_sil0, w_dbi0, w_cov0 = _normalize_weights_tuple(W_SIL, W_DBI, W_COV)

        # Define BO space
        space = [
            Real(np.log(hmin), np.log(hmax), name="log_h"),
            Integer(R_lo, R_hi, name="R"),
        ]
        if BO_OPT_WEIGHTS:
            space.extend([
                Real(BO_WEIGHT_FLOOR, BO_WEIGHT_CEIL, name="w1"),
                Real(BO_WEIGHT_FLOOR, BO_WEIGHT_CEIL, name="w2"),
                Real(BO_WEIGHT_FLOOR, BO_WEIGHT_CEIL, name="w3"),
            ])

        def _obj_vec(x):
            log_h = x[0]
            R_val = int(x[1])
            h = float(np.exp(log_h))
            if BO_OPT_WEIGHTS:
                Wsil, Wdbi, Wcov = _normalize_weights_tuple(x[2], x[3], x[4])
            else:
                Wsil, Wdbi, Wcov = w_sil0, w_dbi0, w_cov0

            out, _ = _eval_hr_once(
                points, x_min, x_max, y_min, y_max, Lx, Ly,
                h, int(R_val), Wsil, Wdbi, Wcov,
                K_MIN, K_MAX, MAX_BINS, PERIODIC_CCA=PERIODIC_CCA
            )
            if out is None:
                return 5.0
            return float(-out["score"])  # maximize score

        # Warm starts (NO normalization here, only clip to box to satisfy skopt)
        from skopt import gp_minimize  # guarded by _HAS_SKOPT
        x0, y0 = [], []
        h0s = [h0, max(hmin, 0.9*h0), min(hmax, 1.1*h0)]
        Rseeds = [max(R_lo, 3), min(R_hi, 5), min(R_hi, 7)]

        def _seed_point(hs, Rseed, w_triplet):
            lgh = float(np.clip(np.log(hs), np.log(hmin), np.log(hmax)))
            Rw  = int(np.clip(Rseed, R_lo, R_hi))
            if BO_OPT_WEIGHTS:
                a, b, c = w_triplet
                a = float(np.clip(a, BO_WEIGHT_FLOOR, BO_WEIGHT_CEIL))
                b = float(np.clip(b, BO_WEIGHT_FLOOR, BO_WEIGHT_CEIL))
                c = float(np.clip(c, BO_WEIGHT_FLOOR, BO_WEIGHT_CEIL))
                x = [lgh, Rw, a, b, c]
            else:
                x = [lgh, Rw]
            val = _obj_vec(x)
            x0.append(x); y0.append(val)

        priors = [
            (w_sil0, w_dbi0, w_cov0),
            (0.50, 0.30, 0.20),
            (0.30, 0.50, 0.20),
        ] if BO_OPT_WEIGHTS else [(w_sil0, w_dbi0, w_cov0)]

        for hs in h0s:
            for Rseed in Rseeds:
                for wtrip in priors:
                    _seed_point(hs, Rseed, wtrip)

        res = gp_minimize(
            _obj_vec, space,
            n_calls=BO_N_CALLS,
            x0=x0 if len(x0) else None,
            y0=y0 if len(y0) else None,
            acq_func="EI",
            noise="gaussian",
            random_state=11
        )

        xs = res.x
        best_logh, best_R = xs[0], int(xs[1])
        h = float(np.exp(best_logh))
        if BO_OPT_WEIGHTS:
            Wsil, Wdbi, Wcov = _normalize_weights_tuple(xs[2], xs[3], xs[4])
        else:
            Wsil, Wdbi, Wcov = w_sil0, w_dbi0, w_cov0

        out, _ = _eval_hr_once(
            points, x_min, x_max, y_min, y_max, Lx, Ly,
            h, best_R, Wsil, Wdbi, Wcov,
            K_MIN, K_MAX, MAX_BINS, PERIODIC_CCA=PERIODIC_CCA
        )
        if out is None:
            raise RuntimeError("BO returned invalid parameters; widen bounds or relax checks.")

        row = {
            "mode": "count",
            "nx": out["nx"], "ny": out["ny"],
            "dense_thr": -1.0, "dense_q": -1.0, "R": out["R"],
            "W_SIL": Wsil, "W_DBI": Wdbi, "W_COV": Wcov,
            "sil": out["sil"], "dbi": out["dbi"], "cov": out["cov"], "k": out["k"], "score": out["score"]
        }
        stageA_rows.append(row)

        bestA = {
            "mode": "count",
            "nx": out["nx"], "ny": out["ny"], "R": out["R"],
            "dx": out["dx"], "dy": out["dy"], "score": out["score"],
            "weights": {"W_SIL": Wsil, "W_DBI": Wdbi, "W_COV": Wcov},
        }

    else:
        # GRID search over quantiles (weights fixed from CLI)
        for (nx, ny) in cand_rect:
            counts, dx, dy = bin_points_rect(points, nx, ny, x_min, x_max, y_min, y_max)
            norm, _, _ = normalize_nonzero(counts)
            nonempty = counts > 0
            vals = norm[nonempty]
            if vals.size == 0:
                continue
            for q in DENSE_QUANTILES:
                thr = float(np.quantile(vals, q))
                dense_before = (counts > 0) & (norm >= thr)
                labels_grid, _ = standard_cca(dense_before, periodic=PERIODIC_CCA)
                y_pts = points_to_labels(points, labels_grid, x_min, x_max, y_min, y_max, dx, dy)
                m = score_partition(points, y_pts, K_MIN, K_MAX, W_SIL, W_DBI, W_COV)
                row = {"mode": "quantile", "nx": nx, "ny": ny, "dense_thr": thr, "dense_q": q, "R": -1,
                       "sil": m["sil"], "dbi": m["dbi"], "cov": m["cov"], "k": m["k"], "score": m["score"]}
                stageA_rows.append(row)
                if (bestA is None) or (m["score"] > bestA["score"]):
                    bestA = {"mode": "quantile", "nx": nx, "ny": ny,
                             "dense_thr": thr, "dense_q": q, "dx": dx, "dy": dy, "score": m["score"]}

    pd.DataFrame(stageA_rows).sort_values("score", ascending=False)\
      .to_csv(os.path.join(out_dir, "stageA_pre_diffusion_candidates.csv"), index=False)

    if bestA is None:
        raise RuntimeError("Stage A failed: no candidates produced a valid score.")

    # ---------- Build baseline masks from Stage-A winner ----------
    nx_c, ny_c = bestA["nx"], bestA["ny"]
    counts_c, dx_c, dy_c = bin_points_rect(points, nx_c, ny_c, x_min, x_max, y_min, y_max)
    norm_c, _, _ = normalize_nonzero(counts_c)
    nonempty_c = (counts_c > 0)

    if bestA["mode"] == "quantile":
        dense_thr_c = bestA["dense_thr"]
        dense_before_c = nonempty_c & (norm_c >= dense_thr_c)
    else:
        R = int(bestA["R"])
        dense_before_c = (counts_c >= R) & nonempty_c
        if np.any(dense_before_c):
            dens_vals_c = norm_c[dense_before_c]
            dense_thr_c = float(np.nanmin(dens_vals_c)) if dens_vals_c.size else 0.30
        else:
            dense_thr_c = 0.30

    sparse_before_c = nonempty_c & ~dense_before_c
    empty_before_c  = ~nonempty_c

    labels_coarse, _ = standard_cca(dense_before_c, periodic=PERIODIC_CCA)
    before_coarse_pdf = None
    if MAKE_PLOTS:
        before_coarse_pdf = os.path.join(out_dir, "fig_before_clusters_overlay_COARSE.pdf")
        plot_cluster_labels_with_points(
            labels_coarse, points, dense_before_c,
            title=f"Number of Clusters: {int(np.max(labels_coarse))+1 if np.max(labels_coarse)>=0 else 0}", #Stage-A baseline (CCA on dense)  |  
            pdf_path=before_coarse_pdf, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, dx=dx_c, dy=dy_c
        )

    dense_frac_c = float(dense_before_c.sum() / max(1, nonempty_c.sum()))

    assert TUNING in ("grid", "bo"), f"Unexpected TUNING={TUNING}"
    print(f"[StageA] mode = {TUNING}  (BO_OPT_WEIGHTS={bool(BO_OPT_WEIGHTS)})")

    # ---------- Stage B (diffusion on same grid) ----------
    nx, ny = nx_c, ny_c
    counts_f, dx_f, dy_f = bin_points_rect(points, nx, ny, x_min, x_max, y_min, y_max)
    norm_f, _, _ = normalize_nonzero(counts_f)
    nonempty_f = (counts_f > 0)

    dense_before_f = dense_before_c.copy()
    update_mask_f  = (sparse_before_c & nonempty_f & (~dense_before_f))
    empty_f        = ~nonempty_f

    # adaptive weight scale from dense fraction (only for diffusion weights, not BO)
    if nonempty_f.any():
        q_equiv = max(0.0, min(1.0, 1.0 - dense_frac_c))
        vals = norm_f[nonempty_f]
        dense_thr_for_w_f = float(np.quantile(vals, q_equiv)) if vals.size else 0.30
    else:
        dense_thr_for_w_f = 0.30

    # ----- Stage-B scoring weights -----
    if TUNING.lower() == "bo" and isinstance(bestA, dict) and ("weights" in bestA):
        W_SIL_B = float(bestA["weights"]["W_SIL"])
        W_DBI_B = float(bestA["weights"]["W_DBI"])
        W_COV_B = float(bestA["weights"]["W_COV"])
    else:
        W_SIL_B, W_DBI_B, W_COV_B = float(W_SIL), float(W_DBI), float(W_COV)

    stageB_rows, bestB = [], None
    for beta in BETA_CANDIDATES:
        c, dstats = run_diffusion_until_converged(
            norm_f, dense_before_f, update_mask_f, empty_f,
            beta=beta, max_iters=MAX_ITERS, min_iters=MIN_ITERS,
            tol=TOL, check_every=CHECK_EVERY, dense_thr_for_w=dense_thr_for_w_f,
            periodic=PERIODIC_CCA
        )
        for cthr in CTHR_VALUES:
            selected_after = np.zeros_like(dense_before_f, dtype=bool)
            selected_after[dense_before_f] = True
            if np.any(update_mask_f):
                selected_after[update_mask_f & (c > cthr)] = True

            labels_seed, _ = seeded_custom_cca(selected_after, dense_before_f,
                                               periodic=PERIODIC_CCA, connectivity=CONNECTIVITY)
            y_pts = points_to_labels(points, labels_seed, x_min, x_max, y_min, y_max, dx_f, dy_f)

            m = score_partition(points, y_pts, K_MIN, K_MAX, W_SIL_B, W_DBI_B, W_COV_B)

            row = {"nx": nx, "ny": ny, "dense_thr": float(dense_thr_c), "beta": float(beta), "cthr": float(cthr),
                   "sil": m["sil"], "dbi": m["dbi"], "cov": m["cov"], "k": m["k"],
                   "iters_used": int(dstats["iters_used"]), "stop": dstats["stops_on"], "score": m["score"],
                   "grid_tag": "COARSE"}
            stageB_rows.append(row)
            if (bestB is None) or (m["score"] > bestB["score"]):
                bestB = row.copy()
                bestB.update({"dx": dx_f, "dy": dy_f})

    pd.DataFrame(stageB_rows).sort_values("score", ascending=False)\
      .to_csv(os.path.join(out_dir, "stageB_post_diffusion_candidates.csv"), index=False)

    # ---------- Final rebuild & plots ----------
    nx = bestB["nx"]; ny = bestB["ny"]; beta_eff = bestB["beta"]; cthr = bestB["cthr"]
    counts_f, dx_f, dy_f = bin_points_rect(points, nx, ny, x_min, x_max, y_min, y_max)
    norm_f, _, _ = normalize_nonzero(counts_f)
    nonempty_f = (counts_f > 0)

    dense_before_f = dense_before_c.copy()
    update_mask_f  = (sparse_before_c & nonempty_f & (~dense_before_f))
    empty_f        = ~nonempty_f

    c, dstats = run_diffusion_until_converged(
        norm_f, dense_before_f, update_mask_f, empty_f,
        beta=beta_eff, max_iters=MAX_ITERS, min_iters=MIN_ITERS,
        tol=TOL, check_every=CHECK_EVERY, dense_thr_for_w=dense_thr_for_w_f,
        periodic=PERIODIC_CCA
    )
    selected_after = np.zeros_like(dense_before_f, dtype=bool)
    selected_after[dense_before_f] = True
    if np.any(update_mask_f):
        selected_after[update_mask_f & (c > cthr)] = True

    labels_after_std, _    = standard_cca(selected_after, periodic=PERIODIC_CCA, connectivity=CONNECTIVITY)
    labels_after_seeded, _ = seeded_custom_cca(selected_after, dense_before_f, periodic=PERIODIC_CCA, connectivity=CONNECTIVITY)

    plots = {"before_coarse": None, "after_std": None, "after_occa": None}
    if MAKE_PLOTS:
        if 'before_coarse_pdf' in locals() and before_coarse_pdf:
            plots["before_coarse"] = before_coarse_pdf

        after_std_pdf    = os.path.join(out_dir, f"fig_after_std_clusters_overlay_COARSE_beta{beta_eff}_cthr{cthr}.pdf")
        after_custom_pdf = os.path.join(out_dir, f"fig_after_custom_clusters_overlay_COARSE_beta{beta_eff}_cthr{cthr}.pdf")

        if DO_STD_CCA:
            plot_cluster_labels_with_points(
                labels_after_std, points, dense_before_f,
                title=f"After diffusion (standard CCA),  #clusters={int(np.max(labels_after_std))+1 if np.max(labels_after_std)>=0 else 0}",
                pdf_path=after_std_pdf, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, dx=dx_f, dy=dy_f
            )
            plots["after_std"] = after_std_pdf

        plot_cluster_labels_with_points(
            labels_after_seeded, points, dense_before_f,
            title=f"After diffusion (O-CCA), #clusters={int(np.max(labels_after_seeded))+1 if np.max(labels_after_seeded)>=0 else 0}",
            pdf_path=after_custom_pdf, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, dx=dx_f, dy=dy_f
        )
        plots["after_occa"] = after_custom_pdf



    # ---------- Metrics (before / after_std / after_occa) ----------
    # Build per-point predictions for all three snapshots
    y_before_pts = points_to_labels(points, labels_coarse, x_min, x_max, y_min, y_max, dx_c, dy_c)
    y_std_pts    = points_to_labels(points, labels_after_std, x_min, x_max, y_min, y_max, dx_f, dy_f)
    y_occa_pts   = points_to_labels(points, labels_after_seeded, x_min, x_max, y_min, y_max, dx_f, dy_f)

    # Try to find ground-truth labels in the CSV (optional)
    y_true = _find_true_labels_column(df)

    metrics = {
        "before":    _cluster_report(points, y_before_pts, y_true),
        "after_std": _cluster_report(points, y_std_pts,    y_true),
        "after_occa":_cluster_report(points, y_occa_pts,   y_true),
    }



    best_summary = {
        "stageA_best": bestA,
        "stageB_best": {k: (float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v)
                        for k, v in bestB.items()},
        "periodic_cca": bool(PERIODIC_CCA),
        "plots": plots
    }

    # expose which weights were actually used in Stage-B scoring
    best_summary["weights_used"] = {
        "stageB_W_SIL": float(W_SIL_B),
        "stageB_W_DBI": float(W_DBI_B),
        "stageB_W_COV": float(W_COV_B),
        "source": "BO" if (TUNING.lower() == "bo" and isinstance(bestA, dict) and ("weights" in bestA)) else "CLI/defaults"
    }


    # Show final diffusion settings and iterations actually used
    best_summary["best_beta"]  = float(beta_eff)
    best_summary["best_iters"] = int(dstats["iters_used"])

    # attach metrics snapshot
    best_summary["metrics"] = metrics



    with open(os.path.join(out_dir, "best_params_summary.json"), "w") as f:
        json.dump(best_summary, f, indent=2)

    return best_summary
