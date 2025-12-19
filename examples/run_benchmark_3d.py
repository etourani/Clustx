#!/usr/bin/env python3
"""
Example: 3D benchmark driver for ClusTEK.

This is intentionally lightweight and meant to be adapted for your paper figures:
- Load one CSV snapshot
- Run atom-based CC baseline once
- Sweep (cell_size, C_thr) and run grid-only / grid+diffusion

Usage:
  python examples/run_benchmark_3d.py path/to/snapshot.csv
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from clustek import ClusTEK3D
from clustek.core3d import DiffusionParams


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    cell_sizes = [(0.8, 0.8, 0.8), (1.0, 1.0, 1.0), (1.2, 1.2, 1.2)]
    c_thrs = [0.3, 0.4, 0.5]
    diffusion = DiffusionParams(beta=0.1, iters=500)

    # atom baseline (once)
    base = ClusTEK3D(df, cell_size=(1.0, 1.0, 1.0), label_thr=0.4, label_col="c_label")
    t0 = time.time()
    atom_labels = base.cluster_atoms_connected_components(cutoff=1.5)
    t_atom = time.time() - t0

    rows = []
    for cs in cell_sizes:
        for thr in c_thrs:
            eng = ClusTEK3D(df, cell_size=cs, label_thr=thr, label_col="c_label")
            eng.particles_to_meshes()

            # grid-only
            t0 = time.time()
            sel0 = eng.compute_filtered_cells(use_diffusion=False)
            cl0 = eng.cluster_cells(sel0)
            t_grid = time.time() - t0

            # grid + diffusion
            t0 = time.time()
            sel1 = eng.compute_filtered_cells(use_diffusion=True, diffusion=diffusion)
            cl1 = eng.cluster_cells(sel1)
            t_diff = time.time() - t0

            rows.append({
                "cell_size": cs,
                "C_thr": thr,
                "atom_time_s": t_atom,
                "grid_time_s": t_grid,
                "diff_time_s": t_diff,
                "n_selected_grid": len(sel0),
                "n_clusters_grid": len(cl0),
                "n_selected_diff": len(sel1),
                "n_clusters_diff": len(cl1),
                "diffusion": asdict(diffusion),
            })

    out = pd.DataFrame(rows)
    out.to_csv("benchmark_summary.csv", index=False)
    print("Wrote benchmark_summary.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1])
