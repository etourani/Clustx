# examples/make_stageI_summaries.py
# Build Stage-I summary table + overlay figs (grid vs bo) for multiple datasets.

import os
import sys
import inspect
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try to import the newer API; fall back to older API if needed.
try:
    from clustx.core import (
        run_pipeline_2d,
        bin_points_rect, normalize_nonzero, standard_cca_2d,
    )
    run_pipeline = run_pipeline_2d
except ImportError:
    # Fallback: older clustx exposes `run_pipeline` only
    from clustx.core import (
        run_pipeline as _run_pipeline,
        bin_points_rect, normalize_nonzero, standard_cca_2d,
    )
    run_pipeline = _run_pipeline

# (Optional) Print which clustx we actually loaded, for sanity:
import clustx
print("Using clustx from:", inspect.getfile(clustx))



HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR  = os.path.join(HERE, "stageI_overlays")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Datasets: filename -> (pretty_name, expected_n_clusters or None)
DATASETS = {
    "aggregation.csv": ("Aggregation", 7),
    "R15.csv":         ("R15",        15),
    "s_set1.csv":      ("s_set1",     15),
}

# -------- Local plotting helper (avoids importing clustx.core’s plotter) --------
def _plot_overlay_dense_cca(labels, points, dense_mask_grid, title, png_path,
                            x_min, x_max, y_min, y_max, dx, dy):
    nx, ny = labels.shape
    fig, ax = plt.subplots(figsize=(6, 6))

    # Color map per cluster id (simple deterministic cycling)
    cluster_ids = sorted([int(v) for v in np.unique(labels) if v >= 0])
    def _color_for(cid):
        # simple, library-light palette
        base = np.array([[0.121,0.466,0.705],
                         [1.000,0.498,0.054],
                         [0.172,0.627,0.172],
                         [0.839,0.152,0.156],
                         [0.580,0.404,0.741],
                         [0.549,0.337,0.294],
                         [0.890,0.467,0.761],
                         [0.498,0.498,0.498],
                         [0.737,0.741,0.133],
                         [0.090,0.745,0.811]])
        return tuple(base[cid % len(base)])

    # draw cells
    for i in range(nx):
        for j in range(ny):
            x0 = x_min + i * dx
            y0 = y_min + j * dy
            lab = int(labels[i, j])
            fc = "white" if lab < 0 else _color_for(lab)
            alpha = 0.15 if lab < 0 else 0.90
            rect = plt.Rectangle((x0, y0), dx, dy, facecolor=fc,
                                 edgecolor="black", linewidth=0.6, alpha=alpha)
            ax.add_patch(rect)

    # light grid
    for i in range(nx + 1):
        x = x_min + i * dx
        ax.plot([x, x], [y_min, y_max], color="black", linewidth=0.6, alpha=0.25)
    for j in range(ny + 1):
        y = y_min + j * dy
        ax.plot([x_min, x_max], [y, y], color="black", linewidth=0.6, alpha=0.25)

    # scatter points: red for dense-cell points, gray otherwise
    ii = np.floor((points[:, 0] - x_min) / dx).astype(int)
    jj = np.floor((points[:, 1] - y_min) / dy).astype(int)
    ii = np.clip(ii, 0, nx - 1)
    jj = np.clip(jj, 0, ny - 1)
    dense_point_mask = dense_mask_grid[ii, jj]
    sp = points[~dense_point_mask]
    dp = points[dense_point_mask]
    if sp.size:
        ax.scatter(sp[:, 0], sp[:, 1], s=33, alpha=0.5, color="gray", edgecolor="k", linewidths=0.3)
    if dp.size:
        ax.scatter(dp[:, 0], dp[:, 1], s=33, alpha=0.6, color="indianred", edgecolor="k", linewidths=0.3)

    ax.set_title(title, pad=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def _stageI_overlay_png(out_root, pretty, strategy):
    os.makedirs(os.path.join(out_root, "figs", "stageI_overlays"), exist_ok=True)
    return os.path.join(out_root, "figs", "stageI_overlays", f"{pretty}_{strategy}.png")

# Common knobs for quick Stage-I (same as before)
COMMON_ARGS = dict(
    MAKE_PLOTS=False,
    DO_STD_CCA=False,
    PERIODIC_CCA=False,
    CONNECTIVITY=4,
    DENSE_QUANTILES=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
    H_BOUNDS_REL=(0.5, 1.25),
    R_RANGE=(2, 10),
    BO_N_CALLS=35,
    W_SIL=0.33, W_DBI=0.33, W_COV=0.34,
    K_FOR_KNN=5, ALPHA_FOR_KNN=0.8, TARGET_OCC=2.5, FD_BACKUP=True,
    SWEEP_PCT=0.2, MAX_BINS=200,
)

rows = []

for fname, (pretty, k_true) in DATASETS.items():
    csv_path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(csv_path)
    if not {"x","y"}.issubset(df.columns):
        raise ValueError(f"{fname} must have columns 'x','y'.")
    points = df[["x","y"]].to_numpy(float)
    n_samples = len(points)

    x_min, x_max = float(points[:,0].min()), float(points[:,0].max())
    y_min, y_max = float(points[:,1].min()), float(points[:,1].max())
    pad_x = 1e-9 * (x_max - x_min + 1.0)
    pad_y = 1e-9 * (y_max - y_min + 1.0)
    x_min -= pad_x; x_max += pad_x; y_min -= pad_y; y_max += pad_y

    for strategy in ["grid", "bo"]:
        out_root = os.path.join(OUT_DIR, f"{pretty}_{strategy}")
        os.makedirs(out_root, exist_ok=True)

        res = run_pipeline(
            points_file=csv_path,
            out_dir=out_root,
            TUNING=strategy,
            **COMMON_ARGS,
        )

        bestA = res["stageA_best"]  # Stage-I winner
        nx, ny = int(bestA["nx"]), int(bestA["ny"])
        counts, dx, dy = bin_points_rect(points, nx, ny, x_min, x_max, y_min, y_max)
        norm, _, _ = normalize_nonzero(counts)
        nonempty = (counts > 0)

        if bestA["mode"] == "quantile":
            dense_thr = float(bestA["dense_thr"])
            dense = nonempty & (norm >= dense_thr)
            params_str = f"q={dense_thr:.2f}"
        else:
            R = int(bestA["R"])
            dense = (counts >= R) & nonempty
            params_str = f"R={R:d}"

        labels_stage1, k_hat = standard_cca_2d(dense, periodic=False, connectivity=4)

        fig_path = _stageI_overlay_png(OUT_DIR, pretty, strategy)
        title = f"{pretty} — Stage I ({strategy})  |  (nx,ny)=({nx},{ny}),  k={k_hat}"
        _plot_overlay_dense_cca(
            labels_stage1, points, dense,
            title, fig_path, x_min, x_max, y_min, y_max, dx, dy
        )

        q_score = float(bestA.get("score", np.nan))
        rows.append({
            "dataset": pretty,
            "n_samples": n_samples,
            "k_true": (k_true if k_true is not None else ""),
            "strategy": strategy,
            "params": params_str,
            "nx": nx, "ny": ny,
            "Q_stageI": q_score,
            "overlay_png": os.path.relpath(fig_path, start=OUT_DIR),
        })

# Write CSV & LaTeX table
summary_csv = os.path.join(OUT_DIR, "stageI_final_used.csv")
pd.DataFrame(rows).to_csv(summary_csv, index=False)

tex_lines = [
    r"\begin{table}",
    r"\centering",
    r"\caption{Stage~I selections (one per strategy per dataset). We report the strategy, parameters, induced grid, and composite score $\mathcal{Q}$.}",
    r"\label{tab:stageI-final-used}",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\setlength{\tabcolsep}{5pt}",
    r"\begin{tabular}{l c c l c c c}",
    r"\toprule",
    r"\textbf{Dataset} & \textbf{\# Samples} & \textbf{\# Clusters} & \textbf{Strategy} & \textbf{Parameters} & \textbf{$(n_x,n_y)$} & $\mathbf{\mathcal{Q}}$ \\",
    r"\midrule",
]
df_rows = pd.DataFrame(rows).sort_values(["dataset","strategy"])
for ds in df_rows["dataset"].unique():
    sub = df_rows[df_rows["dataset"]==ds]
    first = True
    for _, r in sub.iterrows():
        ds_cell   = r["dataset"] if first else ""
        ns_cell   = f"{int(r['n_samples'])}" if first else ""
        k_cell    = f"{int(r['k_true'])}" if (first and str(r['k_true']).strip()!="") else ("" if first else "")
        tex_lines.append(
            f"{ds_cell} & {ns_cell} & {k_cell} & \\texttt{{{r['strategy']}}} & "
            f"{r['params']} & ({int(r['nx'])},{int(r['ny'])}) & {r['Q_stageI']:.3f} \\\\"
        )
        first = False
    tex_lines.append(r"\midrule")
tex_lines[-1] = r"\bottomrule"
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{table}")

table_tex_path = os.path.join(OUT_DIR, "stageI_final_used.tex")
with open(table_tex_path, "w") as f:
    f.write("\n".join(tex_lines))

print("Wrote:")
print("  -", summary_csv)
print("  -", table_tex_path)
print("  - overlay images under:", os.path.join(OUT_DIR, "figs", "stageI_overlays"))
