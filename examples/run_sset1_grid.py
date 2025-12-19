# examples/run_sset1_grid.py
# Grid strategy (Stage I + Stage II) for sset1 dataset

import os
import json
from clustx.core import run_pipeline_2d as run_pipeline

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "s_set1.csv")
OUT  = os.path.join(HERE, "out_sset1_grid_1")
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    res = run_pipeline(
        points_file=DATA,
        out_dir=OUT,

        # ---- Grid suggester knobs ----
        K_FOR_KNN=5, ALPHA_FOR_KNN=0.8, TARGET_OCC=2.5, FD_BACKUP=True,
        SWEEP_PCT=0.2, MAX_BINS=200,

        # ---- Stage-A = GRID (quantile thresholding on normalized counts) ----
        TUNING="grid",
        DENSE_QUANTILES=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50),

        # ---- Stage-B sweeps (diffusion + O-CCA) ----
        BETA_CANDIDATES=(0.10, 0.20, 0.25),
        CTHR_VALUES=(0.01, 0.02, 0.05, 0.10),
        MAX_ITERS=5000, MIN_ITERS=100, TOL=1e-6, CHECK_EVERY=10,

        # ---- Scoring / constraints ----
        # (weights used for the composite Q score)
        W_SIL=0.33, W_DBI=0.34, W_COV=0.33,
        K_MIN=2, K_MAX=50,

        # ---- Runtime / topology ----
        PERIODIC_CCA=False,
        CONNECTIVITY=4,
        MAKE_PLOTS=True,
        DO_STD_CCA=True,
    )

    # --- summarize Stage I (GRID winner) ---
    bestA = res["stageA_best"]
    nx, ny = int(bestA["nx"]), int(bestA["ny"])

    # dx, dy come from the Stage-I winner (coarse grid built there)
    dx_raw = bestA.get("dx")
    dy_raw = bestA.get("dy")
    dx = float(dx_raw[0] if isinstance(dx_raw, (tuple, list)) else dx_raw)
    dy = float(dy_raw[0] if isinstance(dy_raw, (tuple, list)) else dy_raw)
    h  = max(dx, dy)

    # For GRID mode we report q (quantile) and the corresponding dense_thr
    q = float(bestA.get("dense_q"))
    dense_thr = float(bestA.get("dense_thr"))
    Q = float(bestA["score"])

    print("\n[sset1 | Stage-I = GRID]")
    print(f"(nx, ny)  = ({nx}, {ny})")
    print(f"h         = {h:.6g}   (derived as max(dx, dy) with dx={dx:.6g}, dy={dy:.6g})")
    print(f"q         = {q:.3f}")
    print(f"dense_thr = {dense_thr:.6g}   (threshold on normalized cell counts)")
    print(f"Q         = {Q:.6f}")

    # ---- Where are the figures? ----
    plots = res.get("plots", {})
    print("\nSaved figures:")
    for k, v in plots.items():
        if v:
            print(f"  {k}: {v}")

    # Also dump a tiny JSON for the table row
    tiny = {
        "dataset": "sset1",
        "strategy": "grid",
        "nx": nx, "ny": ny,
        "h": h,
        "q": q,
        "dense_thr": dense_thr,
        "Q": Q,
    }
    with open(os.path.join(OUT, "stageI_row_summary.json"), "w") as f:
        json.dump(tiny, f, indent=2)

    print("\nSaved to:", OUT)

