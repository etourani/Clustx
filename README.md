# ClusTEK

**ClusTEK** is a fast, morphology-aware clustering toolkit for molecular simulation data, built around
**grid aggregation + diffusion imputation + connected-component clustering**.

This repository contains:

- **2D pipeline** 
- **3D engine** 

## Key ideas

1. **Grid aggregation:** bin particles into mesh cells and compute a cell-averaged order parameter (e.g., `c_label`).
2. **Diffusion imputation (optional):** stabilize sparse/noisy grids by diffusing the cell field under periodic BCs.
3. **Connected components:** cluster selected cells via fast neighborhood connectivity (KDTree / union-find).

## Install

From the repo root:

```bash
pip install -e ".[dev]"
```

> Requirements are intentionally standard scientific Python. Some 3D “extras” (alpha-shape surface reconstruction, etc.)
> are optional and can be added later as the paper/release matures.

## Quickstart

### 3D (single snapshot)

```python
import pandas as pd
from clustek import ClusTEK3D
from clustek.core3d import DiffusionParams

df = pd.read_csv("data/sample_3d_snapshot.csv")  # or your own snapshot.csv (needs x,y,z,c_label and xlo/xhi/ylo/yhi/zlo/zhi)

engine = ClusTEK3D(df, cell_size=(1.0, 1.0, 1.0), label_thr=0.4, label_col="c_label")
engine.particles_to_meshes()

selected = engine.compute_filtered_cells(use_diffusion=True, diffusion=DiffusionParams(beta=0.1, iters=500))
cell_clusters = engine.cluster_cells(selected)

print(f"Selected cells: {len(selected)}")
print(f"Grid clusters:  {len(cell_clusters)}")
```

### 2D

The original packaged 2D pipeline is available as:

```python
from clustek import run_pipeline_2d
```

(See `examples/2d_quickstart.ipynb` for a fuller run.)

## Reproducible benchmark scripts

- `examples/run_benchmark_3d.py` — template driver for scanning `(cell_size, C_thr)` and recording runtime/memory.

## Development

Run tests:

```bash
pytest -q
```

Lint (optional):

```bash
ruff check .
```

## Citation

If you use ClusTEK in academic work, please cite:

- Tourani, E., Edwards, J. B., Khomami, B. (2025). 
ClusTEK: A grid clustering algorithm augmented with diffusion imputation and origin-constrained connected-component analysis: Application to polymer crystallization,
https://doi.org/10.48550/arXiv.2512.16110


## License

MIT style — see `LICENSE`.
