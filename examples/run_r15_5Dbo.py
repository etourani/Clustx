# examples/run_r15_bo5d.py
# 5D Bayesian Optimization on R15 dataset
import os
from clustx import run_pipeline  # aliased to run_pipeline_2d in __init__.py

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "R15.csv")
OUT  = os.path.join(HERE, "out_r15_cthr")

if __name__ == "__main__":
    res = run_pipeline(
        points_file=DATA,
        out_dir=OUT,


        # ---- Grid suggester knobs ----
        K_FOR_KNN=5, ALPHA_FOR_KNN=0.8, TARGET_OCC=2.5, FD_BACKUP=True,
        SWEEP_PCT=0.2, MAX_BINS=200,

        # ---- Stage-A = BO (with 5D tuning) ----
        tuning="bo",
        #BO_OPT_WEIGHTS=True,         # tune (W_SIL, W_DBI, W_COV) jointly with (h, R)
        #BO_N_CALLS=50,               # BO budget (increase for more thorough search)
        H_BOUNDS_REL=(0.5, 1.25),     # relative bounds around suggested h0
        R_RANGE=(2, 10),             # integer R range
        #BO_WEIGHT_FLOOR=0.1,        # each raw weight sampled in [floor, ceil], then normalized
        #BO_WEIGHT_CEIL=0.40,

        BO_OPT_WEIGHTS=False,   # <- don't tune weights
        W_SIL=0.33,              # whatever ratio you like
        W_DBI=0.33,
        W_COV=0.33,              # <- coverage OFF

        # ---- Stage-B sweeps (diffusion + O-CCA) ----
        BETA_CANDIDATES=(0.05, 0.10, 0.20),
        CTHR_VALUES=(0.31, 0.42, 0.5),

        # ---- Scoring / constraints ----
        K_MIN=2, K_MAX=50,           # acceptable #clusters band

        # ---- Runtime / topology ----
        PERIODIC_CCA=False,          # set True if your domain is periodic
        CONNECTIVITY=4,              # {4, 8}; 8 allows diagonals in 2D CCA
        MAKE_PLOTS=True,             # save before/after overlay figures
        DO_STD_CCA=True,             # also render standard CCA figure
    )

    print("Saved to:", OUT)
    print("Stage-A best:", res.get("stageA_best"))
    print("Stage-B best:", res.get("stageB_best"))
    print("Weights used in Stage-B:", res.get("weights_used"))
