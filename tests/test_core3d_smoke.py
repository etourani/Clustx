import numpy as np
import pandas as pd

from clustek import ClusTEK3D


def _toy_snapshot(n=200, seed=0):
    rng = np.random.default_rng(seed)
    # periodic box
    xlo, xhi = 0.0, 10.0
    ylo, yhi = 0.0, 10.0
    zlo, zhi = 0.0, 10.0

    xyz = rng.uniform(0.0, 10.0, size=(n, 3))
    # make a "crystalline" blob near (2,2,2)
    center = np.array([2.0, 2.0, 2.0])
    dist = np.linalg.norm(xyz - center, axis=1)
    c_label = np.exp(-(dist**2) / 1.0)

    df = pd.DataFrame({
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "c_label": c_label,
        "xlo": xlo, "xhi": xhi,
        "ylo": ylo, "yhi": yhi,
        "zlo": zlo, "zhi": zhi,
    })
    return df


def test_core3d_smoke():
    df = _toy_snapshot()
    eng = ClusTEK3D(df, cell_size=(1.0, 1.0, 1.0), label_thr=0.4, label_col="c_label")
    m = eng.particles_to_meshes()
    assert len(m) > 0

    sel = eng.compute_filtered_cells(use_diffusion=False)
    clusters = eng.cluster_cells(sel)
    assert isinstance(clusters, dict)

    atom_labels = eng.cluster_atoms_connected_components(cutoff=1.5)
    assert atom_labels.shape[0] == len(df)
