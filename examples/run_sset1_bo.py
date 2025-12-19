# BO on sset1 dataset (stageI + StageII)
import os 
import json
import numpy as np
#from clustx import runpipeline
from clustx.core import run_pipeline_2d as run_pipeline


HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "s_set1.csv")
OUT  = os.path.join(HERE, "out_sset1_bo_12")
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    res = run_pipeline(
            points_file=DATA,
            out_dir=OUT,

            # ---- grid suggester knobs ----
            K_FOR_KNN=5, ALPHA_FOR_KNN=0.8, TARGET_OCC=2.5, FD_BACKUP=True,
            SWEEP_PCT=0.2, MAX_BINS=200,

            # ---- stageA = BO (with(h, R), fixed weights here) ---
            TUNING='bo',
            BO_OPT_WEIGHTS=False, 
            W_SIL=0.33, W_DBI=0.34, W_COV=0.33,
            BO_N_CALLS=50,
            H_BOUNDS_REL=(0.5, 1.25),
            R_RANGE=(2, 20),

            # -- stageB sweeps (diffusion + OCCA) ----
            BETA_CANDIDATES=(0.1, 0.2, 0.25),
            CTHR_VALUES=(0.05, 0.1, 0.2, 0.3),
            MAX_ITERS=5000, MIN_ITERS=100, TOL=1e-6, CHECK_EVERY=10, 

            # ---- scoring / constraints ----
            K_MIN=2, K_MAX=50, 

            # --- Runtime / topology ---
            PERIODIC_CCA=False, 
            CONNECTIVITY=4,
            MAKE_PLOTS=True,
            DO_STD_CCA=True,)


    # --- summarize stageI (BO winner) ---
    bestA = res["stageA_best"]
    nx, ny = int(bestA["nx"]), int(bestA["ny"])

    dx_raw = bestA.get("dx")
    dy_raw = bestA.get("dy")
    dx = float(dx_raw[0] if isinstance(dx_raw, (tuple, list)) else dx_raw)
    dy = float(dy_raw[0] if isinstance(dy_raw, (tuple, list)) else dy_raw)
    h = max(dx, dy)

    R = int(bestA.get("R", -1))
    Q = float(bestA["score"])

    print("\n[sset1 | Stage-I = BO]")
    print(f"(nx, ny) = ({nx}, {ny})")
    print(f"h        = {h:.6g}   (derived as max(dx, dy) with dx={dx:.6g}, dy={dy:.6g})")
    print(f"R        = {R}")
    print(f"Q        = {Q:.6f}")

    bestA = res["stageA_best"]
    nx, ny = int(bestA["nx"]), int(bestA["ny"])

    dx_raw = bestA.get("dx"); dy_raw = bestA.get("dy")
    dx = float(dx_raw[0] if isinstance(dx_raw, (tuple, list)) else dx_raw)
    dy = float(dy_raw[0] if isinstance(dy_raw, (tuple, list)) else dy_raw)

    h_abs = max(dx, dy)
    Lx = dx * nx
    Ly = dy * ny
    h_rel = h_abs / max(Lx, Ly) if max(Lx, Ly) > 0 else float("nan")

    print(f"(nx, ny) = ({nx}, {ny})")
    print(f"h_abs     = {h_abs:.6g}   (data units; dx={dx:.6g}, dy={dy:.6g})")
    print(f"h_rel     = {h_rel:.6g}   (fraction of box; {100*h_rel:.3f}%)")



    # ---- Where are the figures? ----
    plots = res.get("plots", {})
    print("\nSaved figures:")
    for k, v in plots.items():
        if v:
            print(f"  {k}: {v}")

    # Also dump a tiny JSON for the table row
    tiny = {"dataset": "sset1", "strategy": "bo", "nx": nx, "ny": ny, "h": h, "R": R, "Q": Q}
    with open(os.path.join(OUT, "stageI_row_summary.json"), "w") as f:
        json.dump(tiny, f, indent=2)

    print("\nSaved to:", OUT)



    # --- Stage-II winners (beta and iterations) ---
    print("\n[Stage-II (Diffusion) best]")
    print(f"beta*     = {res.get('best_beta')}")
    print(f"iters*    = {res.get('best_iters')}")

    # --- Full metrics (before / after_std / after_occa) ---
    M = res.get("metrics", {})
    def _fmt(x):
        return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)

    rows = []
    for tag in ["before", "after_std", "after_occa"]:
        mt = M.get(tag, {})
        rows.append({
            "phase": tag,
            "k": mt.get("k"),
            "coverage": _fmt(mt.get("coverage")),
            "silhouette": _fmt(mt.get("silhouette")),
            "DBI": _fmt(mt.get("dbi")),
            "ARI": _fmt(mt.get("ARI")),
            "NMI": _fmt(mt.get("NMI")),
            "V": _fmt(mt.get("V_measure")),
            "FM": _fmt(mt.get("FM")),
            "purity": _fmt(mt.get("purity")),
        })

    print("\n[Clustering metrics]")
    print("phase        k   coverage  silhouette     DBI      ARI      NMI       V        FM    purity")
    for r in rows:
        print(f"{r['phase']:<11} {r['k']!s:>2}   {r['coverage']:>8}  {r['silhouette']:>10}  {r['DBI']:>7}  "
              f"{r['ARI']:>7}  {r['NMI']:>7}  {r['V']:>7}  {r['FM']:>7}  {r['purity']:>7}")

    # Also dump to JSON for table-building in the paper
    with open(os.path.join(OUT, "stageB_metrics_summary.json"), "w") as f:
        json.dump(M, f, indent=2)





            
