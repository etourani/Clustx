from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DiffusionParams:
    """Parameters for diffusion-based imputation on the 3D mesh field."""
    beta: float = 0.10
    iters: int = 500


class ClusTEK3D:
    """
    ClusTEK 3D grid clustering engine.

    Core idea (per snapshot):
      1) Bin atoms into a regular 3D grid (cell_size).
      2) Compute a cell-averaged scalar label (e.g., crystallinity index c_label).
      3) Optionally diffuse/impute the grid field (periodic boundary conditions).
      4) Select "active" cells by threshold and cluster them via connectivity (KDTree).
      5) (Optional) Build an atom-based connected-component clustering as a reference.

    Notes
    -----
    - Input dataframe must include columns: x, y, z, c_label (or your label column),
      plus xlo/xhi/ylo/yhi/zlo/zhi for periodic wrapping.
    - This class is snapshot-centric: you pass a dataframe already filtered to a single step.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        cell_size: Tuple[float, float, float],
        label_thr: float = 0.40,
        label_col: str = "c_label",
        bounds_cols: Tuple[str, str, str, str, str, str] = ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi"),
    ) -> None:
        self.data = data.copy()
        self.cell_size = tuple(float(v) for v in cell_size)
        self.cthr = float(label_thr)
        self.label_col = label_col
        self.bounds_cols = bounds_cols

        required = {"x", "y", "z", self.label_col, *self.bounds_cols}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # Box bounds (assumed constant within snapshot)
        xlo, xhi, ylo, yhi, zlo, zhi = (float(self.data[c].iloc[0]) for c in self.bounds_cols)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = xlo, xhi, ylo, yhi, zlo, zhi
        self.Lx, self.Ly, self.Lz = (xhi - xlo), (yhi - ylo), (zhi - zlo)

        # Will be filled after particles_to_meshes()
        self.mesh_df: Optional[pd.DataFrame] = None
        self.grid_shape: Optional[Tuple[int, int, int]] = None

    # ---------------------------------------------------------------------
    # Mesh construction
    # ---------------------------------------------------------------------
    def particles_to_meshes(self) -> pd.DataFrame:
        """Assign each particle to a mesh cell and compute per-cell mean label."""
        dx, dy, dz = self.cell_size
        nx = int(np.ceil(self.Lx / dx))
        ny = int(np.ceil(self.Ly / dy))
        nz = int(np.ceil(self.Lz / dz))
        self.grid_shape = (nx, ny, nz)

        # periodic indexing into [0, n)
        xi = np.floor((self.data["x"].to_numpy() - self.xlo) / dx).astype(int) % nx
        yi = np.floor((self.data["y"].to_numpy() - self.ylo) / dy).astype(int) % ny
        zi = np.floor((self.data["z"].to_numpy() - self.zlo) / dz).astype(int) % nz

        d = self.data.copy()
        d["xi"], d["yi"], d["zi"] = xi, yi, zi

        cell_means = (
            d.groupby(["xi", "yi", "zi"], as_index=False)[self.label_col]
            .mean()
            .rename(columns={self.label_col: "label_mean"})
        )

        # cell centers
        cell_means["x"] = self.xlo + (cell_means["xi"] + 0.5) * dx
        cell_means["y"] = self.ylo + (cell_means["yi"] + 0.5) * dy
        cell_means["z"] = self.zlo + (cell_means["zi"] + 0.5) * dz

        self.mesh_df = cell_means
        return cell_means

    # ---------------------------------------------------------------------
    # Diffusion / imputation
    # ---------------------------------------------------------------------
    def diffuse_grid(self, diffusion: DiffusionParams) -> np.ndarray:
        """
        Build a dense (nx,ny,nz) label field and apply explicit diffusion steps.

        Diffusion update (explicit Euler on a 6-neighbor Laplacian):
            S <- S + beta * Laplacian(S)

        Periodic BCs are enforced via numpy roll.
        """
        if self.mesh_df is None or self.grid_shape is None:
            raise RuntimeError("Call particles_to_meshes() before diffuse_grid().")

        nx, ny, nz = self.grid_shape
        field = np.zeros((nx, ny, nz), dtype=float)

        # Fill known cells
        for row in self.mesh_df.itertuples(index=False):
            field[int(row.xi), int(row.yi), int(row.zi)] = float(row.label_mean)

        beta = float(diffusion.beta)
        iters = int(diffusion.iters)

        for _ in range(iters):
            lap = (
                np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
                + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
                + np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2)
                - 6.0 * field
            )
            field = field + beta * lap

        return field

    def compute_filtered_cells(
        self,
        *,
        use_diffusion: bool = True,
        diffusion: DiffusionParams = DiffusionParams(),
    ) -> pd.DataFrame:
        """
        Return a dataframe of selected cell centers (x,y,z) and indices (xi,yi,zi).

        If use_diffusion=True, threshold uses the diffused field; otherwise uses raw cell means.
        """
        if self.mesh_df is None:
            self.particles_to_meshes()
        assert self.mesh_df is not None
        if self.grid_shape is None:
            raise RuntimeError("Internal error: grid_shape missing after particles_to_meshes().")

        if not use_diffusion:
            sel = self.mesh_df[self.mesh_df["label_mean"] >= self.cthr].copy()
            sel.rename(columns={"label_mean": "label_used"}, inplace=True)
            return sel

        field = self.diffuse_grid(diffusion)
        # read back the diffused values at each occupied cell
        vals = []
        for row in self.mesh_df.itertuples(index=False):
            vals.append(field[int(row.xi), int(row.yi), int(row.zi)])
        m = self.mesh_df.copy()
        m["label_used"] = np.asarray(vals, dtype=float)

        sel = m[m["label_used"] >= self.cthr].copy()
        return sel

    # ---------------------------------------------------------------------
    # Clustering on selected cells
    # ---------------------------------------------------------------------
    def cluster_cells(self, selected_cells: pd.DataFrame, *, radius: Optional[float] = None) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Cluster selected cells via connectivity in index space.

        By default radius=1.01 and clustering is performed in (xi,yi,zi) index space,
        which links face-adjacent and edge-adjacent cells depending on your radius.
        """
        if selected_cells.empty:
            return {}

        if radius is None:
            radius = 1.01

        coords = selected_cells[["xi", "yi", "zi"]].to_numpy(dtype=float)
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=radius)

        # Union-find
        n = len(coords)
        parent = np.arange(n, dtype=int)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in pairs:
            union(i, j)

        clusters: Dict[int, List[Tuple[int, int, int]]] = {}
        xi_yi_zi = selected_cells[["xi", "yi", "zi"]].to_numpy(dtype=int)
        for idx in range(n):
            root = find(idx)
            clusters.setdefault(root, []).append(tuple(map(int, xi_yi_zi[idx])))

        # reindex labels to 1..K for convenience
        remap = {old: new for new, old in enumerate(sorted(clusters.keys()), start=1)}
        return {remap[k]: v for k, v in clusters.items()}

    # ---------------------------------------------------------------------
    # Atom-based reference clustering (optional baseline)
    # ---------------------------------------------------------------------
    def cluster_atoms_connected_components(
        self,
        *,
        cutoff: float = 1.5,
        label_filter_thr: Optional[float] = None,
    ) -> np.ndarray:
        """
        Atom-based connected components on a distance graph.

        Returns an integer array of length N atoms, with 0 meaning 'unlabeled'
        and positive integers indicating cluster ids.

        This is intended as a "ground truth-ish" reference when you filter atoms
        by a crystallinity threshold.
        """
        if label_filter_thr is None:
            label_filter_thr = self.cthr

        pts = self.data[["x", "y", "z"]].to_numpy(dtype=float)
        labels = np.zeros(len(pts), dtype=int)

        mask = self.data[self.label_col].to_numpy(dtype=float) >= float(label_filter_thr)
        if not np.any(mask):
            return labels

        pts_f = pts[mask]
        tree = cKDTree(pts_f)
        pairs = tree.query_pairs(r=float(cutoff))

        n = len(pts_f)
        parent = np.arange(n, dtype=int)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in pairs:
            union(i, j)

        # assign component ids
        comp = np.zeros(n, dtype=int)
        roots = {}
        next_id = 1
        for i in range(n):
            r = find(i)
            if r not in roots:
                roots[r] = next_id
                next_id += 1
            comp[i] = roots[r]

        labels[np.where(mask)[0]] = comp
        return labels

    # ---------------------------------------------------------------------
    # Utility: overlap matching (grid clusters -> atom clusters)
    # ---------------------------------------------------------------------
    @staticmethod
    def match_clusters_by_overlap(
        atom_labels: np.ndarray,
        grid_assignments: np.ndarray,
        *,
        min_jaccard: float = 0.0,
    ) -> Dict[int, int]:
        """
        Match each grid cluster to an atom cluster by maximum Jaccard overlap.

        Parameters
        ----------
        atom_labels:
            Per-atom cluster labels (0 = background).
        grid_assignments:
            Per-atom grid cluster labels (0 = background), e.g. label inherited from the atom's cell.
        min_jaccard:
            If best overlap < min_jaccard, the grid cluster is left unmatched.

        Returns
        -------
        mapping: dict grid_cluster_id -> atom_cluster_id
        """
        mapping: Dict[int, int] = {}
        grid_ids = sorted(set(grid_assignments) - {0})
        atom_ids = sorted(set(atom_labels) - {0})

        if not grid_ids or not atom_ids:
            return mapping

        # precompute masks
        atom_masks = {a: (atom_labels == a) for a in atom_ids}

        for g in grid_ids:
            gmask = (grid_assignments == g)
            best_a = 0
            best_j = -1.0
            for a in atom_ids:
                amask = atom_masks[a]
                inter = np.sum(gmask & amask)
                union = np.sum(gmask | amask)
                j = (inter / union) if union > 0 else 0.0
                if j > best_j:
                    best_j = j
                    best_a = a
            if best_j >= float(min_jaccard):
                mapping[g] = best_a

        return mapping
