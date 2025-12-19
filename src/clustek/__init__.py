"""ClusTEK: diffusion-enhanced grid clustering.

Public API:
- run_pipeline_2d: main 2D pipeline (from the original Clustx 2D package)
- ClusTEK3D: 3D grid clustering engine (diffusion-imputation + connected components)
"""

from .core2d import run_pipeline_2d
from .core3d import ClusTEK3D

__all__ = ["run_pipeline_2d", "ClusTEK3D"]
