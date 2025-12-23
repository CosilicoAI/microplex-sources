"""Tests for state-level SOI ETL."""

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
from db.etl_soi_state import load_soi_state_targets, SOI_STATE_DATA, STATE_FIPS


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_soi_state.db"
        engine = init_db(db_path)
        yield engine


class TestSoiStateETL:
    """Tests for state-level SOI ETL loader."""

    def test_load_soi_state_creates_national_stratum(self, temp_db):
        """Loading state SOI data should create/reference a national stratum."""
        with Session(temp_db) as session:
            load_soi_state_targets(session, years=[2021])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US All Filers")
            ).first()

            assert national is not None
            assert national.stratum_group_id == "national"

    def test_load_soi_state_creates_state_strata(self, temp_db):
        """Loading state SOI data should create state-level strata."""
        with Session(temp_db) as session:
            load_soi_state_targets(session, years=[2021])

            state_strata = session.exec(
                select(Stratum).where(Stratum.stratum_group_id == "soi_states")
            ).all()

            # Should have strata for all 5 states in the data
            expected_states = len(STATE_FIPS)
            assert len(state_strata) == expected_states

    def test_load_soi_state_creates_returns_targets(self, temp_db):
        """Loading state SOI should create tax returns count targets."""
        with Session(temp_db) as session:
            load_soi_state_targets(session, years=[2021])

            ca_stratum = session.exec(
                select(Stratum).where(Stratum.name == "CA All Filers")
            ).first()

            assert ca_stratum is not None

            returns_target = session.exec(
                select(Target)
                .where(Target.stratum_id == ca_stratum.id)
                .where(Target.variable == "tax_unit_count")
                .where(Target.period == 2021)
            ).first()

            assert returns_target is not None
            expected = SOI_STATE_DATA[2021]["CA"]["total_returns"]
            assert returns_target.value == expected
            assert returns_target.target_type == TargetType.COUNT
            assert returns_target.source == DataSource.IRS_SOI

    def test_load_soi_state_creates_agi_targets(self, temp_db):
        """Loading state SOI should create AGI amount targets."""
        with Session(temp_db) as session:
            load_soi_state_targets(session, years=[2021])

            tx_stratum = session.exec(
                select(Stratum).where(Stratum.name == "TX All Filers")
            ).first()

            agi_target = session.exec(
                select(Target)
                .where(Target.stratum_id == tx_stratum.id)
                .where(Target.variable == "adjusted_gross_income")
                .where(Target.period == 2021)
            ).first()

            assert agi_target is not None
            expected = SOI_STATE_DATA[2021]["TX"]["total_agi"]
            assert agi_target.value == expected
            assert agi_target.target_type == TargetType.AMOUNT

    def test_load_soi_state_creates_tax_liability_targets(self, temp_db):
        """Loading state SOI should create tax liability targets."""
        with Session(temp_db) as session:
            load_soi_state_targets(session, years=[2021])

            fl_stratum = session.exec(
                select(Stratum).where(Stratum.name == "FL All Filers")
            ).first()

            tax_target = session.exec(
                select(Target)
                .where(Target.stratum_id == fl_stratum.id)
                .where(Target.variable == "income_tax_liability")
                .where(Target.period == 2021)
            ).first()

            assert tax_target is not None
            expected = SOI_STATE_DATA[2021]["FL"]["total_tax_liability"]
            assert tax_target.value == expected
            assert tax_target.target_type == TargetType.AMOUNT

    def test_load_soi_state_stratum_has_state_constraint(self, temp_db):
        """State strata should have state_fips constraint."""
        with Session(temp_db) as session:
            load_soi_state_targets(session, years=[2021])

            ny_stratum = session.exec(
                select(Stratum).where(Stratum.name == "NY All Filers")
            ).first()

            # Check constraints include state_fips
            state_constraint = None
            for constraint in ny_stratum.constraints:
                if constraint.variable == "state_fips":
                    state_constraint = constraint
                    break

            assert state_constraint is not None
            assert state_constraint.operator == "=="
            assert state_constraint.value == STATE_FIPS["NY"]

    def test_load_soi_state_stratum_has_parent(self, temp_db):
        """State strata should have national stratum as parent."""
        with Session(temp_db) as session:
            load_soi_state_targets(session, years=[2021])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US All Filers")
            ).first()

            pa_stratum = session.exec(
                select(Stratum).where(Stratum.name == "PA All Filers")
            ).first()

            assert pa_stratum.parent_id == national.id

    def test_load_multiple_years(self, temp_db):
        """Loading multiple years should create targets for each."""
        with Session(temp_db) as session:
            load_soi_state_targets(session, years=[2020, 2021])

            ca_stratum = session.exec(
                select(Stratum).where(Stratum.name == "CA All Filers")
            ).first()

            targets = session.exec(
                select(Target)
                .where(Target.stratum_id == ca_stratum.id)
                .where(Target.variable == "tax_unit_count")
            ).all()

            years = {t.period for t in targets}
            assert years == {2020, 2021}

    def test_load_soi_state_idempotent(self, temp_db):
        """Loading state SOI twice should not duplicate data."""
        with Session(temp_db) as session:
            load_soi_state_targets(session, years=[2021])
            load_soi_state_targets(session, years=[2021])

            ca_strata = session.exec(
                select(Stratum).where(Stratum.name == "CA All Filers")
            ).all()

            # Should only have one CA stratum
            assert len(ca_strata) == 1

    def test_all_states_loaded(self, temp_db):
        """All 5 states should be loaded."""
        with Session(temp_db) as session:
            load_soi_state_targets(session, years=[2021])

            for state_abbrev in STATE_FIPS.keys():
                stratum = session.exec(
                    select(Stratum).where(Stratum.name == f"{state_abbrev} All Filers")
                ).first()

                assert stratum is not None, f"Missing stratum for {state_abbrev}"

                # Each state should have 3 targets (returns, AGI, tax liability)
                targets = session.exec(
                    select(Target).where(Target.stratum_id == stratum.id)
                ).all()

                assert len(targets) == 3

    def test_target_source_metadata(self, temp_db):
        """Targets should have correct source metadata."""
        with Session(temp_db) as session:
            load_soi_state_targets(session, years=[2021])

            ca_stratum = session.exec(
                select(Stratum).where(Stratum.name == "CA All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == ca_stratum.id)
                .where(Target.variable == "tax_unit_count")
            ).first()

            assert target.source == DataSource.IRS_SOI
            assert target.source_table == "Historic Table 2"
            assert "historic-table-2" in target.source_url
