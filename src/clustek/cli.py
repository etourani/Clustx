# cli.py — CLI for 2D pipeline (grid/BO; optional 5D BO with weight tuning)
from __future__ import annotations
import argparse
import json
import sys

from .core2d import run_pipeline_2d


def _parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _ensure_range(vals, lo, hi, name):
    for v in vals:
        if not (lo <= v <= hi):
            raise ValueError(f"{name}: value {v} is out of range [{lo}, {hi}]")


def main():
    ap = argparse.ArgumentParser(
        prog="clustx2d",
        description="Stage-A (grid/BO) + diffusion + O-CCA (2D) with optional 5D Bayesian tuning."
    )

    ap.add_argument("--input", required=True, help="CSV with columns x,y")
    ap.add_argument("--outdir", default="./clustx2d_out", help="Output directory")

    # -------- Fixed mode (fast path) --------
    ap.add_argument("--fixed-grid", type=str, default=None,
                    help="nx,ny (e.g., 120,90). If set, skips tuning and uses fixed params.")
    ap.add_argument("--fixed-dense-thr", type=float, default=None,
                    help="Normalized [0,1] threshold defining pre-imputation dense cells.")
    ap.add_argument("--fixed-beta", type=float, default=None,
                    help="Diffusion beta (recommend ≤ 0.25 for stability).")
    ap.add_argument("--fixed-cthr", type=float, default=None,
                    help="Post-diffusion selection threshold C_sel in [0,1].")

    # -------- Tuned mode (used if fixed-grid is NOT set) --------
    ap.add_argument("--tuning", choices=["grid", "bo"], default="grid",
                    help="Stage-A parameter selection mode.")

    # Grid-mode Stage-A options
    ap.add_argument("--dense-qs", type=str, default="0.20,0.25,0.30,0.35,0.40,0.50",
                    help="Comma-separated quantiles for Stage-A when tuning=grid (each in [0,1]).")

    # BO-mode Stage-A options
    ap.add_argument("--bo-n-calls", type=int, default=35, help="BO evaluation budget.")
    ap.add_argument("--h-bounds-rel", type=str, default="0.5,1.8",
                    help="Relative bounds around h0 for BO, e.g., '0.5,1.8'.")
    ap.add_argument("--R-range", type=str, default="1,30",
                    help="Integer R range for BO, e.g., '1,30'.")

    # 5D BO toggle (if enabled, BO jointly tunes h, R, W_SIL, W_DBI, W_COV)
    try:
        from argparse import BooleanOptionalAction  # Python 3.9+
        bool_action = BooleanOptionalAction
    except Exception:
        bool_action = None

    if bool_action:
        ap.add_argument("--bo-opt-weights", default=True, action=bool_action,
                        help="If true, BO also optimizes (W_SIL, W_DBI, W_COV) with h and R.")
    else:
        ap.add_argument("--bo-opt-weights", action="store_true", default=True,
                        help="Enable BO weight optimization (cannot disable if Python<3.9).")

    # Optional bounds to keep BO weights healthy
    ap.add_argument("--bo-weight-floor", type=float, default=0.10,
                    help="Lower bound for each BO weight component before normalization (default 0.10).")
    ap.add_argument("--bo-weight-ceil", type=float, default=0.90,
                    help="Upper bound for each BO weight component before normalization (default 0.90).")

    # Grid suggester knobs (only used when tuning)
    if bool_action:
        ap.add_argument("--fd-backup", default=True, action=bool_action,
                        help="Enable/disable Freedman–Diaconis backup for h (default: enabled).")
    else:
        ap.add_argument("--fd-backup", action="store_true", default=True,
                        help="Use FD backup (default: enabled). (Cannot disable if Python<3.9)")

    ap.add_argument("--k-for-knn", type=int, default=5)
    ap.add_argument("--alpha-for-knn", type=float, default=0.8)
    ap.add_argument("--target-occ", type=int, default=4)
    ap.add_argument("--sweep-pct", type=float, default=0.4)
    ap.add_argument("--max-bins", type=int, default=300)

    # Stage-B sweeps / iteration controls
    ap.add_argument("--beta-list", type=str, default="0.02,0.05,0.10,0.20,0.50",
                    help="Comma-separated betas for diffusion (tuned mode).")
    ap.add_argument("--cthr-list", type=str,
                    default="0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21",
                    help="Comma-separated thresholds applied after diffusion (tuned mode).")
    ap.add_argument("--max-iters", type=int, default=50000)
    ap.add_argument("--min-iters", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--check-every", type=int, default=10)

    # Scoring weights + k band (CLI defaults; if 5D BO is enabled, BO will override these for Stage-A,
    # and Stage-B will automatically use the BO-found weights)
    ap.add_argument("--w-sil", type=float, default=0.0,
                    help="Default weight for silhouette (used if BO weights are disabled/unavailable).")
    ap.add_argument("--w-dbi", type=float, default=0.80,
                    help="Default weight for 1/(1+DBI) (used if BO weights are disabled/unavailable).")
    ap.add_argument("--w-cov", type=float, default=0.15,
                    help="Default weight for coverage (used if BO weights are disabled/unavailable).")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=50)

    # Run-time options
    ap.add_argument("--no-periodic-cca", action="store_true",
                    help="Disable periodic BCs in CCA/O-CCA & diffusion (default: periodic enabled).")
    ap.add_argument("--no-std-cca-plot", action="store_true",
                    help="Skip plotting the standard CCA figure (only O-CCA).")
    ap.add_argument("--connectivity", type=int, choices=[4, 8], default=4,
                    help="Neighborhood connectivity for CCA/O-CCA (default 4).")

    args = ap.parse_args()

    try:
        # Parse lists / tuples
        BETA_CANDIDATES = tuple(_parse_float_list(args.beta_list))
        CTHR_VALUES     = tuple(_parse_float_list(args.cthr_list))
        DENSE_QS        = tuple(_parse_float_list(args.dense_qs))
        H_BOUNDS_REL    = tuple(_parse_float_list(args.h_bounds_rel))
        R_RANGE_TUP     = tuple(_parse_int_list(args.R_range))

        # Basic validations
        if len(H_BOUNDS_REL) != 2 or not (H_BOUNDS_REL[0] > 0 and H_BOUNDS_REL[1] > 0 and H_BOUNDS_REL[1] >= H_BOUNDS_REL[0]):
            raise ValueError("--h-bounds-rel must be two positive numbers like '0.5,1.8' with hi>=lo.")
        if len(R_RANGE_TUP) != 2 or not (R_RANGE_TUP[0] >= 1 and R_RANGE_TUP[1] >= R_RANGE_TUP[0]):
            raise ValueError("--R-range must be two integers like '1,30' with hi>=lo and lo>=1.")
        _ensure_range(DENSE_QS, 0.0, 1.0, "dense-qs")
        _ensure_range(CTHR_VALUES, 0.0, 1.0, "cthr-list")
        if any(b <= 0 for b in BETA_CANDIDATES):
            raise ValueError("beta-list: all betas must be > 0.")
        if args.sweep_pct <= 0 or args.sweep_pct > 1.0:
            raise ValueError("--sweep-pct should be in (0,1].")
        if args.max_bins < 1:
            raise ValueError("--max-bins must be >= 1.")
        if args.k_min < 1 or args.k_max < args.k_min:
            raise ValueError("--k-min must be >=1 and --k-max >= --k-min.")
        if not (0.0 < args.bo_weight_floor <= args.bo_weight_ceiling := args.bo_weight_ceil <= 1.0):
            pass  # checked more explicitly below
        if not (0.0 <= args.bo_weight_floor <= 1.0):
            raise ValueError("--bo-weight-floor must be in [0,1].")
        if not (0.0 <= args.bo_weight_ceiling := args.bo_weight_ceil <= 1.0):
            raise ValueError("--bo-weight-ceil must be in [0,1].")
        if args.bo_weight_floor > args.bo_weight_ceiling:
            raise ValueError("--bo-weight-floor must be <= --bo-weight-ceil.")

        # Parse fixed-grid
        FIXED_GRID = None
        if args.fixed_grid:
            parts = [p.strip() for p in args.fixed_grid.split(",")]
            if len(parts) != 2:
                raise ValueError("--fixed-grid expects 'nx,ny'")
            FIXED_GRID = (int(parts[0]), int(parts[1]))
            if FIXED_GRID[0] < 1 or FIXED_GRID[1] < 1:
                raise ValueError("--fixed-grid values must be >= 1")

        # Extra sanity for fixed mode thresholds
        if FIXED_GRID is not None:
            for name, val in [("fixed-dense-thr", args.fixed_dense_thr),
                              ("fixed-cthr", args.fixed_cthr)]:
                if val is None:
                    raise ValueError(f"--{name} is required in fixed mode.")
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"--{name} must be in [0,1].")
            if args.fixed_beta is None:
                raise ValueError("--fixed-beta is required in fixed mode.")
            if args.fixed_beta <= 0:
                raise ValueError("--fixed-beta must be > 0.")
            if args.fixed_beta > 0.25:
                print("[warn] fixed-beta > 0.25 may be unstable for explicit 5-point diffusion.", file=sys.stderr)

        # Execute
        result = run_pipeline_2d(
            points_file=args.input,
            out_dir=args.outdir,

            # Fixed fast path
            FIXED_GRID=FIXED_GRID,
            FIXED_DENSE_THR=args.fixed_dense_thr,
            FIXED_BETA=args.fixed_beta,
            FIXED_CTHR=args.fixed_cthr,

            # Tuned path (Stage-A)
            TUNING=args.tuning,
            H_BOUNDS_REL=H_BOUNDS_REL,
            R_RANGE=R_RANGE_TUP,
            BO_N_CALLS=args.bo_n_calls,
            BO_OPT_WEIGHTS=bool(args.bo_opt_weights),
            BO_WEIGHT_FLOOR=args.bo_weight_floor,
            BO_WEIGHT_CEIL=args.bo_weight_ceiling,

            # Grid suggester
            K_FOR_KNN=args.k_for_knn,
            ALPHA_FOR_KNN=args.alpha_for_knn,
            TARGET_OCC=args.target_occ,
            FD_BACKUP=bool(args.fd_backup),
            SWEEP_PCT=args.sweep_pct,
            MAX_BINS=args.max_bins,

            # Stage-A (grid)
            DENSE_QUANTILES=DENSE_QS,

            # Stage-B sweeps + iterations
            BETA_CANDIDATES=BETA_CANDIDATES,
            CTHR_VALUES=CTHR_VALUES,
            MAX_ITERS=args.max_iters,
            MIN_ITERS=args.min_iters,
            TOL=args.tol,
            CHECK_EVERY=args.check_every,

            # scoring defaults (used if BO weights disabled/unavailable)
            W_SIL=args.w_sil,
            W_DBI=args.w_dbi,
            W_COV=args.w_cov,
            K_MIN=args.k_min,
            K_MAX=args.k_max,

            # runtime
            PERIODIC_CCA=not args.no_periodic_cca,
            CONNECTIVITY=args.connectivity,
            MAKE_PLOTS=True,
            DO_STD_CCA=not args.no_std_cca_plot,
        )

        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
