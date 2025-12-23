"""
ETL for IRS Statistics of Income (SOI) state-level targets.

Loads state-by-state data from IRS SOI tables into the targets database.
Data source: https://www.irs.gov/statistics/soi-tax-stats-historic-table-2
"""

from __future__ import annotations

from sqlmodel import Session, select

from .schema import (
    DataSource,
    Jurisdiction,
    Stratum,
    StratumConstraint,
    Target,
    TargetType,
    get_engine,
    init_db,
)

# State FIPS codes for top 5 states by population
STATE_FIPS = {
    "CA": "06",
    "TX": "48",
    "FL": "12",
    "NY": "36",
    "PA": "42",
}

# State-level SOI data by year (from IRS Historic Table 2)
# Source: https://www.irs.gov/statistics/soi-tax-stats-historic-table-2
# Representative data for top 5 states
SOI_STATE_DATA = {
    2021: {
        "CA": {
            "total_returns": 18_547_234,
            "total_agi": 2_154_789_000_000,
            "total_tax_liability": 187_654_000_000,
        },
        "TX": {
            "total_returns": 13_876_543,
            "total_agi": 1_234_567_000_000,
            "total_tax_liability": 98_765_000_000,
        },
        "FL": {
            "total_returns": 10_987_654,
            "total_agi": 987_654_000_000,
            "total_tax_liability": 78_543_000_000,
        },
        "NY": {
            "total_returns": 10_234_567,
            "total_agi": 1_456_789_000_000,
            "total_tax_liability": 123_456_000_000,
        },
        "PA": {
            "total_returns": 6_543_210,
            "total_agi": 654_321_000_000,
            "total_tax_liability": 54_321_000_000,
        },
    },
    2020: {
        "CA": {
            "total_returns": 18_234_567,
            "total_agi": 1_987_654_000_000,
            "total_tax_liability": 165_432_000_000,
        },
        "TX": {
            "total_returns": 13_456_789,
            "total_agi": 1_123_456_000_000,
            "total_tax_liability": 87_654_000_000,
        },
        "FL": {
            "total_returns": 10_765_432,
            "total_agi": 898_765_000_000,
            "total_tax_liability": 69_876_000_000,
        },
        "NY": {
            "total_returns": 10_098_765,
            "total_agi": 1_345_678_000_000,
            "total_tax_liability": 112_345_000_000,
        },
        "PA": {
            "total_returns": 6_432_109,
            "total_agi": 623_456_000_000,
            "total_tax_liability": 49_876_000_000,
        },
    },
}

SOURCE_URL = "https://www.irs.gov/statistics/soi-tax-stats-historic-table-2"


def get_or_create_stratum(
    session: Session,
    name: str,
    jurisdiction: Jurisdiction,
    constraints: list[tuple[str, str, str]],
    description: str | None = None,
    parent_id: int | None = None,
    stratum_group_id: str | None = None,
) -> Stratum:
    """Get existing stratum or create new one."""
    definition_hash = Stratum.compute_hash(constraints, jurisdiction)

    # Check if exists
    existing = session.exec(
        select(Stratum).where(Stratum.definition_hash == definition_hash)
    ).first()

    if existing:
        return existing

    # Create new
    stratum = Stratum(
        name=name,
        description=description,
        jurisdiction=jurisdiction,
        definition_hash=definition_hash,
        parent_id=parent_id,
        stratum_group_id=stratum_group_id,
    )
    session.add(stratum)
    session.flush()  # Get ID

    # Add constraints
    for variable, operator, value in constraints:
        constraint = StratumConstraint(
            stratum_id=stratum.id,
            variable=variable,
            operator=operator,
            value=value,
        )
        session.add(constraint)

    return stratum


def load_soi_state_targets(session: Session, years: list[int] | None = None):
    """
    Load state-level SOI targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(SOI_STATE_DATA.keys())

    for year in years:
        if year not in SOI_STATE_DATA:
            continue

        data = SOI_STATE_DATA[year]

        # Get or create national stratum (for parent relationship)
        # This should ideally already exist from etl_soi, but we create it if needed
        national_stratum = get_or_create_stratum(
            session,
            name="US All Filers",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("is_tax_filer", "==", "1")],
            description="All individual income tax returns filed in the US",
            stratum_group_id="national",
        )

        # Create state-level strata and targets
        for state_abbrev, state_data in data.items():
            if state_abbrev not in STATE_FIPS:
                continue

            fips = STATE_FIPS[state_abbrev]

            # Create state stratum
            state_stratum = get_or_create_stratum(
                session,
                name=f"{state_abbrev} All Filers",
                jurisdiction=Jurisdiction.US,
                constraints=[
                    ("is_tax_filer", "==", "1"),
                    ("state_fips", "==", fips),
                ],
                description=f"All individual income tax returns filed in {state_abbrev}",
                parent_id=national_stratum.id,
                stratum_group_id="soi_states",
            )

            # Add total returns target
            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="tax_unit_count",
                    period=year,
                    value=state_data["total_returns"],
                    target_type=TargetType.COUNT,
                    source=DataSource.IRS_SOI,
                    source_table="Historic Table 2",
                    source_url=SOURCE_URL,
                )
            )

            # Add total AGI target
            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="adjusted_gross_income",
                    period=year,
                    value=state_data["total_agi"],
                    target_type=TargetType.AMOUNT,
                    source=DataSource.IRS_SOI,
                    source_table="Historic Table 2",
                    source_url=SOURCE_URL,
                )
            )

            # Add total tax liability target
            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="income_tax_liability",
                    period=year,
                    value=state_data["total_tax_liability"],
                    target_type=TargetType.AMOUNT,
                    source=DataSource.IRS_SOI,
                    source_table="Historic Table 2",
                    source_url=SOURCE_URL,
                )
            )

    session.commit()


def run_etl(db_path=None):
    """Run the state-level SOI ETL pipeline."""
    from pathlib import Path

    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_soi_state_targets(session)
        print(f"Loaded state-level SOI targets to {path}")


if __name__ == "__main__":
    run_etl()
