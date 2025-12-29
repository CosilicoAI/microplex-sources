"""
Calibration pipeline for microdata reweighting.

Connects administrative targets to microdata, producing reweighted samples
that match official statistics.

Supports both flat and hierarchical microdata:
- Flat: Single DataFrame with weights per record
- Hierarchical: Households with linked persons, weights at household level
"""

from .loader import load_microdata
from .targets import get_targets, TargetSpec
from .constraints import (
    build_constraint_matrix,
    build_hierarchical_constraint_matrix,
    Constraint,
)
from .variables import get_entity, infer_target_level
from .methods import EntropyCalibrator

__all__ = [
    # Loader
    "load_microdata",
    # Targets
    "get_targets",
    "TargetSpec",
    # Constraints
    "build_constraint_matrix",
    "build_hierarchical_constraint_matrix",
    "Constraint",
    # Variables
    "get_entity",
    "infer_target_level",
    # Methods
    "EntropyCalibrator",
]
