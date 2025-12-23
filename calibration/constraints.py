"""
Constraint matrix building for calibration.

Maps targets to microdata aggregations, building indicator vectors
for each constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from db.schema import TargetType
from .targets import TargetSpec


@dataclass
class Constraint:
    """
    A single calibration constraint.

    Represents the equation: sum(weights * indicator) = target_value

    Attributes:
        indicator: Vector of indicator values (length = microdata rows)
        target_value: Target aggregate to match
        variable: PolicyEngine variable name
        target_type: COUNT or AMOUNT
        tolerance: Allowed deviation from target (fraction)
        stratum_name: Human-readable description
    """

    indicator: np.ndarray
    target_value: float
    variable: str
    target_type: TargetType
    tolerance: float = 0.01
    stratum_name: Optional[str] = None


def apply_stratum_constraints(
    microdata: pd.DataFrame,
    constraints: list[tuple[str, str, str]],
) -> pd.Series:
    """
    Return boolean mask for records matching stratum.

    Args:
        microdata: DataFrame with microdata
        constraints: List of (variable, operator, value) tuples

    Returns:
        Boolean Series indicating which rows match all constraints
    """
    mask = pd.Series(True, index=microdata.index)

    for variable, operator, value in constraints:
        if variable not in microdata.columns:
            # Skip constraints for variables not in microdata
            continue

        col = microdata[variable]

        # Parse value based on column dtype
        if pd.api.types.is_numeric_dtype(col):
            parsed_value = float(value)
        else:
            parsed_value = value

        if operator == "==":
            mask &= col == parsed_value
        elif operator == "!=":
            mask &= col != parsed_value
        elif operator == ">":
            mask &= col > parsed_value
        elif operator == ">=":
            mask &= col >= parsed_value
        elif operator == "<":
            mask &= col < parsed_value
        elif operator == "<=":
            mask &= col <= parsed_value
        elif operator == "in":
            # Value should be comma-separated list
            values = [v.strip() for v in value.split(",")]
            if pd.api.types.is_numeric_dtype(col):
                values = [float(v) for v in values]
            mask &= col.isin(values)
        else:
            raise ValueError(f"Unknown operator: {operator}")

    return mask


def build_constraint_matrix(
    microdata: pd.DataFrame,
    targets: list[TargetSpec],
    tolerance: float = 0.01,
) -> list[Constraint]:
    """
    Build constraint matrix from targets and microdata.

    For each target, creates an indicator vector:
    - COUNT type: indicator is 1 for matching rows, 0 otherwise
    - AMOUNT type: indicator is variable value for matching rows, 0 otherwise

    Args:
        microdata: DataFrame with microdata records
        targets: List of TargetSpec objects from get_targets()
        tolerance: Default allowed deviation (can be overridden per target)

    Returns:
        List of Constraint objects ready for calibration
    """
    constraints = []

    for target in targets:
        # Build stratum mask
        mask = apply_stratum_constraints(microdata, target.constraints)

        # Build indicator vector based on target type
        if target.target_type == TargetType.COUNT:
            # For counts, indicator is just the mask
            indicator = mask.astype(float).values
        elif target.target_type == TargetType.AMOUNT:
            # For amounts, indicator is variable * mask
            if target.variable in microdata.columns:
                indicator = (microdata[target.variable] * mask).values
            else:
                # If variable not in microdata, use zeros
                indicator = np.zeros(len(microdata))
        else:
            # RATE type - not commonly used for calibration
            indicator = mask.astype(float).values

        constraint = Constraint(
            indicator=indicator,
            target_value=target.value,
            variable=target.variable,
            target_type=target.target_type,
            tolerance=target.tolerance if target.tolerance else tolerance,
            stratum_name=target.stratum_name,
        )
        constraints.append(constraint)

    return constraints
