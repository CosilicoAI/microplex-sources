"""
Target specification and querying.

Provides TargetSpec dataclass and get_targets() function to query
the targets database for calibration constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

from sqlmodel import Session, select

from db.schema import (
    DataSource,
    Stratum,
    StratumConstraint,
    Target,
    TargetType,
    get_engine,
    DEFAULT_DB_PATH,
)


@dataclass
class TargetSpec:
    """
    Specification for a calibration target.

    Combines a target value with its stratum constraints, making it
    ready for constraint matrix building.

    Attributes:
        variable: Variable name. Should use fully qualified format:
            ``{model}:{path}#{var_name}`` (e.g., ``us:statute/26/32#eitc``).
            Legacy unqualified names (e.g., ``tax_unit_count``) are still supported.
        value: Target aggregate value
        target_type: COUNT, AMOUNT, or RATE
        constraints: List of (variable, operator, value) tuples defining stratum
        source: Data source (e.g., IRS_SOI)
        period: Year of the target
        tolerance: Allowed deviation from target (optional)
        stratum_name: Human-readable stratum name (optional)
    """

    variable: str
    value: float
    target_type: TargetType
    constraints: list[tuple[str, str, str]]
    source: DataSource
    period: int
    tolerance: Optional[float] = None
    stratum_name: Optional[str] = None

    @property
    def is_qualified(self) -> bool:
        """Return True if the variable uses the qualified format ``{model}:{path}#{var_name}``."""
        return ":" in self.variable and "#" in self.variable

    @property
    def variable_name(self) -> str:
        """Extract just the variable name from a qualified reference (the part after ``#``).

        For qualified format ``us:statute/26/32#eitc``, returns ``eitc``.
        For unqualified format, returns the full variable string.
        """
        if "#" in self.variable:
            return self.variable.split("#", 1)[1]
        return self.variable

    @property
    def variable_model(self) -> Optional[str]:
        """Extract the model prefix (the part before ``:``).

        For qualified format ``us:statute/26/32#eitc``, returns ``us``.
        For unqualified format, returns None.
        """
        if ":" in self.variable:
            return self.variable.split(":", 1)[0]
        return None

    @property
    def variable_path(self) -> Optional[str]:
        """Extract the path (the part between ``:`` and ``#``).

        For qualified format ``us:statute/26/32#eitc``, returns ``statute/26/32``.
        For unqualified format, returns None.
        """
        if ":" in self.variable and "#" in self.variable:
            after_colon = self.variable.split(":", 1)[1]
            return after_colon.split("#", 1)[0]
        return None


def get_targets(
    db_path: Path | None = None,
    jurisdiction: str = "us",
    year: int | None = None,
    sources: list[str] | None = None,
    variables: list[str] | None = None,
) -> list[TargetSpec]:
    """
    Query targets database and return TargetSpec objects.

    Args:
        db_path: Path to database (default: macro/targets.db)
        jurisdiction: Jurisdiction filter (e.g., "us", "uk")
        year: Filter by target year/period
        sources: List of data source names to include (e.g., ["irs-soi"])
        variables: List of variable names to include

    Returns:
        List of TargetSpec objects ready for constraint building
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    engine = get_engine(db_path)

    with Session(engine) as session:
        # Build query for targets with their strata
        query = select(Target, Stratum).join(Stratum)

        # Filter by year
        if year is not None:
            query = query.where(Target.period == year)

        # Filter by source
        if sources is not None:
            source_enums = [DataSource(s) for s in sources]
            query = query.where(Target.source.in_(source_enums))

        # Filter by variable
        if variables is not None:
            query = query.where(Target.variable.in_(variables))

        # Filter by jurisdiction (match prefix for flexibility)
        # e.g., "us" matches "us", "us-federal", "us-ca", etc.
        if jurisdiction:
            jurisdiction_lower = jurisdiction.lower()
            # For now, simple filter - could be enhanced
            query = query.where(
                Stratum.jurisdiction.in_([
                    j for j in [
                        "us", "us-federal", "us-ca", "us-ny", "us-tx",
                        "uk",
                    ]
                    if j.startswith(jurisdiction_lower)
                ])
            )

        results = session.exec(query).all()

        # Convert to TargetSpec objects
        target_specs = []
        for target, stratum in results:
            # Fetch constraints for this stratum
            constraint_query = select(StratumConstraint).where(
                StratumConstraint.stratum_id == stratum.id
            )
            stratum_constraints = session.exec(constraint_query).all()

            constraints = [
                (c.variable, c.operator, c.value)
                for c in stratum_constraints
            ]

            spec = TargetSpec(
                variable=target.variable,
                value=target.value,
                target_type=target.target_type,
                constraints=constraints,
                source=target.source,
                period=target.period,
                stratum_name=stratum.name,
            )
            target_specs.append(spec)

        return target_specs
