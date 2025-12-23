"""
Calibration pipeline for microdata reweighting.

Connects administrative targets to microdata, producing reweighted samples
that match official statistics.
"""

from .targets import get_targets, TargetSpec
from .constraints import build_constraint_matrix, Constraint
from .methods import EntropyCalibrator

__all__ = [
    # Targets
    "get_targets",
    "TargetSpec",
    # Constraints
    "build_constraint_matrix",
    "Constraint",
    # Methods
    "EntropyCalibrator",
]
