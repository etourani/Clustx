"""Backward-compatible wrapper.

Historically this project shipped as the `clustx` Python package.
The ClusTEK repo now ships the canonical package name `clustek`.

If you previously used:

    from clustx import run_pipeline

it will continue to work.
"""

from clustek.core2d import run_pipeline_2d as run_pipeline
from clustek.core3d import ClusTEK3D

__all__ = ["run_pipeline", "ClusTEK3D"]
