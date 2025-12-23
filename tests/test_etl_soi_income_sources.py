"""Tests for state-level SOI income by source ETL."""

import tempfile
from pathlib import Path

import pytest
from sqlmodel import Session, select

from db.schema import (
    DataSource,
    Stratum,
    Target,
    TargetType,
    init_db,
)
from db.etl_soi_income_sources import (
    load_soi_income_sources_targets,
    SOI_INCOME_SOURCES_DATA,
    STATE_FIPS,
    INCOME_SOURCES,
    SOURCE_URL,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_soi_income_sources.db"
        engine = init_db(db_path)
        yield engine


class TestSoiIncomeSourcesETL:
    """Tests for state-level SOI income sources ETL loader."""

    def test_load_soi_income_sources_creates_national_stratum(self, temp_db):
        """Loading income source data should create/reference a national stratum."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US All Filers")
            ).first()

            assert national is not None
            assert national.stratum_group_id == "national"

    def test_load_soi_income_sources_creates_state_strata(self, temp_db):
        """Loading income source data should create state-level strata."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            state_strata = session.exec(
                select(Stratum).where(Stratum.stratum_group_id == "soi_states")
            ).all()

            # Should have strata for all 50 states + DC
            expected_states = len(STATE_FIPS)
            assert len(state_strata) == expected_states
            assert expected_states == 51  # 50 states + DC

    def test_load_wages_returns_targets(self, temp_db):
        """Loading income sources should create wages returns count targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            ca_stratum = session.exec(
                select(Stratum).where(Stratum.name == "CA All Filers")
            ).first()

            assert ca_stratum is not None

            wages_returns_target = session.exec(
                select(Target)
                .where(Target.stratum_id == ca_stratum.id)
                .where(Target.variable == "wages_returns")
                .where(Target.period == 2021)
            ).first()

            assert wages_returns_target is not None
            expected = SOI_INCOME_SOURCES_DATA[2021]["CA"]["wages_returns"]
            assert wages_returns_target.value == expected
            assert wages_returns_target.target_type == TargetType.COUNT
            assert wages_returns_target.source == DataSource.IRS_SOI

    def test_load_wages_amount_targets(self, temp_db):
        """Loading income sources should create wages amount targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            tx_stratum = session.exec(
                select(Stratum).where(Stratum.name == "TX All Filers")
            ).first()

            wages_amount_target = session.exec(
                select(Target)
                .where(Target.stratum_id == tx_stratum.id)
                .where(Target.variable == "wages_amount")
                .where(Target.period == 2021)
            ).first()

            assert wages_amount_target is not None
            expected = SOI_INCOME_SOURCES_DATA[2021]["TX"]["wages_amount"]
            assert wages_amount_target.value == expected
            assert wages_amount_target.target_type == TargetType.AMOUNT

    def test_load_dividends_targets(self, temp_db):
        """Loading income sources should create dividends targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            fl_stratum = session.exec(
                select(Stratum).where(Stratum.name == "FL All Filers")
            ).first()

            dividends_returns_target = session.exec(
                select(Target)
                .where(Target.stratum_id == fl_stratum.id)
                .where(Target.variable == "dividends_returns")
                .where(Target.period == 2021)
            ).first()

            assert dividends_returns_target is not None
            expected = SOI_INCOME_SOURCES_DATA[2021]["FL"]["dividends_returns"]
            assert dividends_returns_target.value == expected

            dividends_amount_target = session.exec(
                select(Target)
                .where(Target.stratum_id == fl_stratum.id)
                .where(Target.variable == "dividends_amount")
                .where(Target.period == 2021)
            ).first()

            assert dividends_amount_target is not None

    def test_load_interest_targets(self, temp_db):
        """Loading income sources should create interest targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            ny_stratum = session.exec(
                select(Stratum).where(Stratum.name == "NY All Filers")
            ).first()

            interest_returns_target = session.exec(
                select(Target)
                .where(Target.stratum_id == ny_stratum.id)
                .where(Target.variable == "interest_returns")
                .where(Target.period == 2021)
            ).first()

            assert interest_returns_target is not None
            expected = SOI_INCOME_SOURCES_DATA[2021]["NY"]["interest_returns"]
            assert interest_returns_target.value == expected

    def test_load_capital_gains_targets(self, temp_db):
        """Loading income sources should create capital gains targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            wa_stratum = session.exec(
                select(Stratum).where(Stratum.name == "WA All Filers")
            ).first()

            cap_gains_returns_target = session.exec(
                select(Target)
                .where(Target.stratum_id == wa_stratum.id)
                .where(Target.variable == "capital_gains_returns")
                .where(Target.period == 2021)
            ).first()

            assert cap_gains_returns_target is not None
            expected = SOI_INCOME_SOURCES_DATA[2021]["WA"]["capital_gains_returns"]
            assert cap_gains_returns_target.value == expected

            cap_gains_amount_target = session.exec(
                select(Target)
                .where(Target.stratum_id == wa_stratum.id)
                .where(Target.variable == "capital_gains_amount")
                .where(Target.period == 2021)
            ).first()

            assert cap_gains_amount_target is not None

    def test_load_business_income_targets(self, temp_db):
        """Loading income sources should create business income targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            ga_stratum = session.exec(
                select(Stratum).where(Stratum.name == "GA All Filers")
            ).first()

            business_returns_target = session.exec(
                select(Target)
                .where(Target.stratum_id == ga_stratum.id)
                .where(Target.variable == "business_income_returns")
                .where(Target.period == 2021)
            ).first()

            assert business_returns_target is not None
            expected = SOI_INCOME_SOURCES_DATA[2021]["GA"]["business_income_returns"]
            assert business_returns_target.value == expected

    def test_load_soi_income_sources_stratum_has_state_constraint(self, temp_db):
        """State strata should have state_fips constraint."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            oh_stratum = session.exec(
                select(Stratum).where(Stratum.name == "OH All Filers")
            ).first()

            # Check constraints include state_fips
            state_constraint = None
            for constraint in oh_stratum.constraints:
                if constraint.variable == "state_fips":
                    state_constraint = constraint
                    break

            assert state_constraint is not None
            assert state_constraint.operator == "=="
            assert state_constraint.value == STATE_FIPS["OH"]

    def test_load_soi_income_sources_stratum_has_parent(self, temp_db):
        """State strata should have national stratum as parent."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US All Filers")
            ).first()

            il_stratum = session.exec(
                select(Stratum).where(Stratum.name == "IL All Filers")
            ).first()

            assert il_stratum.parent_id == national.id

    def test_load_soi_income_sources_idempotent(self, temp_db):
        """Loading income sources twice should not duplicate data."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])
            load_soi_income_sources_targets(session, years=[2021])

            ca_strata = session.exec(
                select(Stratum).where(Stratum.name == "CA All Filers")
            ).all()

            # Should only have one CA stratum
            assert len(ca_strata) == 1

    def test_all_states_loaded(self, temp_db):
        """All 50 states + DC should be loaded with income source targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            # 5 income sources x 2 (returns + amount) = 10 targets per state
            expected_targets_per_state = len(INCOME_SOURCES) * 2

            for state_abbrev in STATE_FIPS.keys():
                stratum = session.exec(
                    select(Stratum).where(Stratum.name == f"{state_abbrev} All Filers")
                ).first()

                assert stratum is not None, f"Missing stratum for {state_abbrev}"

                targets = session.exec(
                    select(Target).where(Target.stratum_id == stratum.id)
                ).all()

                assert len(targets) == expected_targets_per_state, (
                    f"Expected {expected_targets_per_state} targets for {state_abbrev}, "
                    f"got {len(targets)}"
                )

    def test_target_source_metadata(self, temp_db):
        """Targets should have correct source metadata."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            ca_stratum = session.exec(
                select(Stratum).where(Stratum.name == "CA All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == ca_stratum.id)
                .where(Target.variable == "wages_returns")
            ).first()

            assert target.source == DataSource.IRS_SOI
            assert target.source_table == "AGI Percentile Data by State"
            assert "adjusted-gross-income" in target.source_url

    def test_income_sources_list(self):
        """INCOME_SOURCES should include expected income types."""
        expected_sources = [
            "wages",
            "dividends",
            "interest",
            "capital_gains",
            "business_income",
        ]
        assert INCOME_SOURCES == expected_sources

    def test_national_totals_are_reasonable(self, temp_db):
        """National totals should match expected ranges."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            # Sum all wages returns across states
            all_wages_returns = sum(
                state_data["wages_returns"]
                for state_data in SOI_INCOME_SOURCES_DATA[2021].values()
            )
            # Most returns have wages - should be ~120M+ nationally
            assert 100_000_000 < all_wages_returns < 200_000_000

            # Sum all wages amounts across states
            all_wages_amount = sum(
                state_data["wages_amount"]
                for state_data in SOI_INCOME_SOURCES_DATA[2021].values()
            )
            # Total wages nationally should be ~$8-10 trillion
            assert 7_000_000_000_000 < all_wages_amount < 12_000_000_000_000

            # Dividends should be much less common than wages
            all_dividends_returns = sum(
                state_data["dividends_returns"]
                for state_data in SOI_INCOME_SOURCES_DATA[2021].values()
            )
            assert all_dividends_returns < all_wages_returns

    def test_large_states_have_higher_income(self, temp_db):
        """Larger states should have more income than smaller states."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            # California (large) vs Wyoming (small)
            ca_data = SOI_INCOME_SOURCES_DATA[2021]["CA"]
            wy_data = SOI_INCOME_SOURCES_DATA[2021]["WY"]

            assert ca_data["wages_returns"] > wy_data["wages_returns"]
            assert ca_data["wages_amount"] > wy_data["wages_amount"]
            assert ca_data["dividends_amount"] > wy_data["dividends_amount"]

            # Texas (large) vs Vermont (small)
            tx_data = SOI_INCOME_SOURCES_DATA[2021]["TX"]
            vt_data = SOI_INCOME_SOURCES_DATA[2021]["VT"]

            assert tx_data["wages_returns"] > vt_data["wages_returns"]
            assert tx_data["wages_amount"] > vt_data["wages_amount"]


class TestIncomeSourcesData:
    """Tests for the income sources data structure."""

    def test_all_states_have_data(self):
        """All 50 states + DC should have income sources data."""
        assert len(SOI_INCOME_SOURCES_DATA[2021]) == 51

        for state_abbrev in STATE_FIPS.keys():
            assert state_abbrev in SOI_INCOME_SOURCES_DATA[2021], (
                f"Missing data for {state_abbrev}"
            )

    def test_all_income_sources_present(self):
        """Each state should have all income source variables."""
        expected_vars = [
            "wages_returns",
            "wages_amount",
            "dividends_returns",
            "dividends_amount",
            "interest_returns",
            "interest_amount",
            "capital_gains_returns",
            "capital_gains_amount",
            "business_income_returns",
            "business_income_amount",
        ]

        for state_abbrev, state_data in SOI_INCOME_SOURCES_DATA[2021].items():
            for var in expected_vars:
                assert var in state_data, (
                    f"Missing {var} for {state_abbrev}"
                )
                assert isinstance(state_data[var], (int, float)), (
                    f"{var} for {state_abbrev} is not numeric"
                )

    def test_values_are_positive(self):
        """All values should be positive."""
        for state_abbrev, state_data in SOI_INCOME_SOURCES_DATA[2021].items():
            for var, value in state_data.items():
                assert value >= 0, (
                    f"Negative value for {var} in {state_abbrev}: {value}"
                )

    def test_returns_less_than_total_returns(self):
        """Returns reporting each income source should be reasonable."""
        # Check a few states that returns reporting any income source
        # are not impossibly high
        for state_abbrev, state_data in SOI_INCOME_SOURCES_DATA[2021].items():
            # Wages returns should be less than 20M even for largest states
            assert state_data["wages_returns"] < 20_000_000, (
                f"Wages returns too high for {state_abbrev}"
            )
            # Dividends returns should be less than wages returns
            # (not everyone has dividends)
            assert state_data["dividends_returns"] <= state_data["wages_returns"], (
                f"Dividends returns > wages returns for {state_abbrev}"
            )
