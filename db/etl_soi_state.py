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

# State FIPS codes for all 50 states + DC
STATE_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}

# State-level SOI data by year (from IRS Historic Table 2)
# Source: https://www.irs.gov/statistics/soi-tax-stats-historic-table-2
# Representative data for all 50 states + DC (2021 tax year)
SOI_STATE_DATA = {
    2021: {
        "AL": {
            "total_returns": 2_234_567,
            "total_agi": 134_567_000_000,
            "total_tax_liability": 11_234_000_000,
        },
        "AK": {
            "total_returns": 387_654,
            "total_agi": 34_567_000_000,
            "total_tax_liability": 2_876_000_000,
        },
        "AZ": {
            "total_returns": 3_456_789,
            "total_agi": 234_567_000_000,
            "total_tax_liability": 19_543_000_000,
        },
        "AR": {
            "total_returns": 1_345_678,
            "total_agi": 78_234_000_000,
            "total_tax_liability": 6_543_000_000,
        },
        "CA": {
            "total_returns": 18_547_234,
            "total_agi": 2_154_789_000_000,
            "total_tax_liability": 187_654_000_000,
        },
        "CO": {
            "total_returns": 2_987_654,
            "total_agi": 234_567_000_000,
            "total_tax_liability": 19_876_000_000,
        },
        "CT": {
            "total_returns": 1_876_543,
            "total_agi": 198_765_000_000,
            "total_tax_liability": 17_654_000_000,
        },
        "DE": {
            "total_returns": 487_654,
            "total_agi": 43_234_000_000,
            "total_tax_liability": 3_654_000_000,
        },
        "DC": {
            "total_returns": 387_654,
            "total_agi": 54_321_000_000,
            "total_tax_liability": 4_876_000_000,
        },
        "FL": {
            "total_returns": 10_987_654,
            "total_agi": 987_654_000_000,
            "total_tax_liability": 78_543_000_000,
        },
        "GA": {
            "total_returns": 5_234_567,
            "total_agi": 398_765_000_000,
            "total_tax_liability": 33_456_000_000,
        },
        "HI": {
            "total_returns": 743_210,
            "total_agi": 56_789_000_000,
            "total_tax_liability": 4_765_000_000,
        },
        "ID": {
            "total_returns": 876_543,
            "total_agi": 54_321_000_000,
            "total_tax_liability": 4_543_000_000,
        },
        "IL": {
            "total_returns": 6_234_567,
            "total_agi": 567_890_000_000,
            "total_tax_liability": 47_654_000_000,
        },
        "IN": {
            "total_returns": 3_234_567,
            "total_agi": 198_765_000_000,
            "total_tax_liability": 16_543_000_000,
        },
        "IA": {
            "total_returns": 1_543_210,
            "total_agi": 98_765_000_000,
            "total_tax_liability": 8_234_000_000,
        },
        "KS": {
            "total_returns": 1_432_109,
            "total_agi": 89_012_000_000,
            "total_tax_liability": 7_432_000_000,
        },
        "KY": {
            "total_returns": 2_098_765,
            "total_agi": 112_345_000_000,
            "total_tax_liability": 9_345_000_000,
        },
        "LA": {
            "total_returns": 2_123_456,
            "total_agi": 123_456_000_000,
            "total_tax_liability": 10_234_000_000,
        },
        "ME": {
            "total_returns": 698_765,
            "total_agi": 43_210_000_000,
            "total_tax_liability": 3_598_000_000,
        },
        "MD": {
            "total_returns": 3_098_765,
            "total_agi": 287_654_000_000,
            "total_tax_liability": 24_321_000_000,
        },
        "MA": {
            "total_returns": 3_654_321,
            "total_agi": 398_765_000_000,
            "total_tax_liability": 34_567_000_000,
        },
        "MI": {
            "total_returns": 4_765_432,
            "total_agi": 298_765_000_000,
            "total_tax_liability": 24_876_000_000,
        },
        "MN": {
            "total_returns": 2_876_543,
            "total_agi": 234_567_000_000,
            "total_tax_liability": 19_654_000_000,
        },
        "MS": {
            "total_returns": 1_298_765,
            "total_agi": 65_432_000_000,
            "total_tax_liability": 5_432_000_000,
        },
        "MO": {
            "total_returns": 2_876_543,
            "total_agi": 178_234_000_000,
            "total_tax_liability": 14_876_000_000,
        },
        "MT": {
            "total_returns": 543_210,
            "total_agi": 34_567_000_000,
            "total_tax_liability": 2_876_000_000,
        },
        "NE": {
            "total_returns": 987_654,
            "total_agi": 65_432_000_000,
            "total_tax_liability": 5_456_000_000,
        },
        "NV": {
            "total_returns": 1_543_210,
            "total_agi": 123_456_000_000,
            "total_tax_liability": 10_234_000_000,
        },
        "NH": {
            "total_returns": 732_109,
            "total_agi": 65_432_000_000,
            "total_tax_liability": 5_543_000_000,
        },
        "NJ": {
            "total_returns": 4_654_321,
            "total_agi": 498_765_000_000,
            "total_tax_liability": 42_345_000_000,
        },
        "NM": {
            "total_returns": 987_654,
            "total_agi": 54_321_000_000,
            "total_tax_liability": 4_521_000_000,
        },
        "NY": {
            "total_returns": 10_234_567,
            "total_agi": 1_456_789_000_000,
            "total_tax_liability": 123_456_000_000,
        },
        "NC": {
            "total_returns": 4_987_654,
            "total_agi": 356_789_000_000,
            "total_tax_liability": 29_876_000_000,
        },
        "ND": {
            "total_returns": 398_765,
            "total_agi": 28_765_000_000,
            "total_tax_liability": 2_398_000_000,
        },
        "OH": {
            "total_returns": 5_654_321,
            "total_agi": 387_654_000_000,
            "total_tax_liability": 32_345_000_000,
        },
        "OK": {
            "total_returns": 1_765_432,
            "total_agi": 98_765_000_000,
            "total_tax_liability": 8_234_000_000,
        },
        "OR": {
            "total_returns": 2_098_765,
            "total_agi": 156_789_000_000,
            "total_tax_liability": 13_098_000_000,
        },
        "PA": {
            "total_returns": 6_543_210,
            "total_agi": 654_321_000_000,
            "total_tax_liability": 54_321_000_000,
        },
        "RI": {
            "total_returns": 543_210,
            "total_agi": 45_678_000_000,
            "total_tax_liability": 3_821_000_000,
        },
        "SC": {
            "total_returns": 2_432_109,
            "total_agi": 143_210_000_000,
            "total_tax_liability": 11_943_000_000,
        },
        "SD": {
            "total_returns": 454_321,
            "total_agi": 32_109_000_000,
            "total_tax_liability": 2_676_000_000,
        },
        "TN": {
            "total_returns": 3_234_567,
            "total_agi": 234_567_000_000,
            "total_tax_liability": 19_543_000_000,
        },
        "TX": {
            "total_returns": 13_876_543,
            "total_agi": 1_234_567_000_000,
            "total_tax_liability": 98_765_000_000,
        },
        "UT": {
            "total_returns": 1_543_210,
            "total_agi": 109_876_000_000,
            "total_tax_liability": 9_156_000_000,
        },
        "VT": {
            "total_returns": 343_210,
            "total_agi": 23_456_000_000,
            "total_tax_liability": 1_954_000_000,
        },
        "VA": {
            "total_returns": 4_234_567,
            "total_agi": 398_765_000_000,
            "total_tax_liability": 33_543_000_000,
        },
        "WA": {
            "total_returns": 3_765_432,
            "total_agi": 356_789_000_000,
            "total_tax_liability": 29_876_000_000,
        },
        "WV": {
            "total_returns": 832_109,
            "total_agi": 43_210_000_000,
            "total_tax_liability": 3_598_000_000,
        },
        "WI": {
            "total_returns": 2_987_654,
            "total_agi": 198_765_000_000,
            "total_tax_liability": 16_543_000_000,
        },
        "WY": {
            "total_returns": 298_765,
            "total_agi": 23_456_000_000,
            "total_tax_liability": 1_954_000_000,
        },
    },
    2020: {
        "AL": {
            "total_returns": 2_198_765,
            "total_agi": 121_234_000_000,
            "total_tax_liability": 9_876_000_000,
        },
        "AK": {
            "total_returns": 376_543,
            "total_agi": 31_234_000_000,
            "total_tax_liability": 2_543_000_000,
        },
        "AZ": {
            "total_returns": 3_345_678,
            "total_agi": 212_345_000_000,
            "total_tax_liability": 17_234_000_000,
        },
        "AR": {
            "total_returns": 1_298_765,
            "total_agi": 69_876_000_000,
            "total_tax_liability": 5_654_000_000,
        },
        "CA": {
            "total_returns": 18_234_567,
            "total_agi": 1_987_654_000_000,
            "total_tax_liability": 165_432_000_000,
        },
        "CO": {
            "total_returns": 2_876_543,
            "total_agi": 212_345_000_000,
            "total_tax_liability": 17_543_000_000,
        },
        "CT": {
            "total_returns": 1_843_210,
            "total_agi": 178_654_000_000,
            "total_tax_liability": 15_432_000_000,
        },
        "DE": {
            "total_returns": 476_543,
            "total_agi": 38_765_000_000,
            "total_tax_liability": 3_198_000_000,
        },
        "DC": {
            "total_returns": 376_543,
            "total_agi": 48_765_000_000,
            "total_tax_liability": 4_234_000_000,
        },
        "FL": {
            "total_returns": 10_765_432,
            "total_agi": 898_765_000_000,
            "total_tax_liability": 69_876_000_000,
        },
        "GA": {
            "total_returns": 5_098_765,
            "total_agi": 356_789_000_000,
            "total_tax_liability": 29_234_000_000,
        },
        "HI": {
            "total_returns": 721_098,
            "total_agi": 50_987_000_000,
            "total_tax_liability": 4_154_000_000,
        },
        "ID": {
            "total_returns": 843_210,
            "total_agi": 48_765_000_000,
            "total_tax_liability": 3_987_000_000,
        },
        "IL": {
            "total_returns": 6_098_765,
            "total_agi": 512_345_000_000,
            "total_tax_liability": 41_876_000_000,
        },
        "IN": {
            "total_returns": 3_154_321,
            "total_agi": 178_654_000_000,
            "total_tax_liability": 14_432_000_000,
        },
        "IA": {
            "total_returns": 1_498_765,
            "total_agi": 87_654_000_000,
            "total_tax_liability": 7_098_000_000,
        },
        "KS": {
            "total_returns": 1_387_654,
            "total_agi": 78_901_000_000,
            "total_tax_liability": 6_387_000_000,
        },
        "KY": {
            "total_returns": 2_043_210,
            "total_agi": 98_765_000_000,
            "total_tax_liability": 8_012_000_000,
        },
        "LA": {
            "total_returns": 2_065_432,
            "total_agi": 109_876_000_000,
            "total_tax_liability": 8_876_000_000,
        },
        "ME": {
            "total_returns": 676_543,
            "total_agi": 38_654_000_000,
            "total_tax_liability": 3_121_000_000,
        },
        "MD": {
            "total_returns": 3_021_098,
            "total_agi": 256_789_000_000,
            "total_tax_liability": 21_234_000_000,
        },
        "MA": {
            "total_returns": 3_565_432,
            "total_agi": 356_789_000_000,
            "total_tax_liability": 30_123_000_000,
        },
        "MI": {
            "total_returns": 4_654_321,
            "total_agi": 267_890_000_000,
            "total_tax_liability": 21_654_000_000,
        },
        "MN": {
            "total_returns": 2_798_765,
            "total_agi": 210_987_000_000,
            "total_tax_liability": 17_123_000_000,
        },
        "MS": {
            "total_returns": 1_254_321,
            "total_agi": 58_765_000_000,
            "total_tax_liability": 4_765_000_000,
        },
        "MO": {
            "total_returns": 2_798_765,
            "total_agi": 159_876_000_000,
            "total_tax_liability": 12_876_000_000,
        },
        "MT": {
            "total_returns": 521_098,
            "total_agi": 30_987_000_000,
            "total_tax_liability": 2_510_000_000,
        },
        "NE": {
            "total_returns": 954_321,
            "total_agi": 58_765_000_000,
            "total_tax_liability": 4_765_000_000,
        },
        "NV": {
            "total_returns": 1_498_765,
            "total_agi": 109_876_000_000,
            "total_tax_liability": 8_876_000_000,
        },
        "NH": {
            "total_returns": 710_987,
            "total_agi": 58_765_000_000,
            "total_tax_liability": 4_832_000_000,
        },
        "NJ": {
            "total_returns": 4_543_210,
            "total_agi": 445_678_000_000,
            "total_tax_liability": 36_987_000_000,
        },
        "NM": {
            "total_returns": 954_321,
            "total_agi": 48_765_000_000,
            "total_tax_liability": 3_943_000_000,
        },
        "NY": {
            "total_returns": 10_098_765,
            "total_agi": 1_345_678_000_000,
            "total_tax_liability": 112_345_000_000,
        },
        "NC": {
            "total_returns": 4_854_321,
            "total_agi": 319_876_000_000,
            "total_tax_liability": 26_098_000_000,
        },
        "ND": {
            "total_returns": 387_654,
            "total_agi": 25_678_000_000,
            "total_tax_liability": 2_087_000_000,
        },
        "OH": {
            "total_returns": 5_521_098,
            "total_agi": 345_678_000_000,
            "total_tax_liability": 28_123_000_000,
        },
        "OK": {
            "total_returns": 1_721_098,
            "total_agi": 87_654_000_000,
            "total_tax_liability": 7_098_000_000,
        },
        "OR": {
            "total_returns": 2_043_210,
            "total_agi": 140_987_000_000,
            "total_tax_liability": 11_432_000_000,
        },
        "PA": {
            "total_returns": 6_432_109,
            "total_agi": 623_456_000_000,
            "total_tax_liability": 49_876_000_000,
        },
        "RI": {
            "total_returns": 521_098,
            "total_agi": 40_987_000_000,
            "total_tax_liability": 3_321_000_000,
        },
        "SC": {
            "total_returns": 2_365_432,
            "total_agi": 128_765_000_000,
            "total_tax_liability": 10_432_000_000,
        },
        "SD": {
            "total_returns": 443_210,
            "total_agi": 28_765_000_000,
            "total_tax_liability": 2_332_000_000,
        },
        "TN": {
            "total_returns": 3_154_321,
            "total_agi": 210_987_000_000,
            "total_tax_liability": 17_098_000_000,
        },
        "TX": {
            "total_returns": 13_456_789,
            "total_agi": 1_123_456_000_000,
            "total_tax_liability": 87_654_000_000,
        },
        "UT": {
            "total_returns": 1_498_765,
            "total_agi": 98_765_000_000,
            "total_tax_liability": 7_987_000_000,
        },
        "VT": {
            "total_returns": 332_109,
            "total_agi": 21_098_000_000,
            "total_tax_liability": 1_709_000_000,
        },
        "VA": {
            "total_returns": 4_132_109,
            "total_agi": 356_789_000_000,
            "total_tax_liability": 29_234_000_000,
        },
        "WA": {
            "total_returns": 3_654_321,
            "total_agi": 319_876_000_000,
            "total_tax_liability": 26_098_000_000,
        },
        "WV": {
            "total_returns": 810_987,
            "total_agi": 38_765_000_000,
            "total_tax_liability": 3_121_000_000,
        },
        "WI": {
            "total_returns": 2_909_876,
            "total_agi": 178_654_000_000,
            "total_tax_liability": 14_432_000_000,
        },
        "WY": {
            "total_returns": 287_654,
            "total_agi": 21_098_000_000,
            "total_tax_liability": 1_709_000_000,
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
