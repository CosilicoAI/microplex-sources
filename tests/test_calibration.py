"""Tests for calibration pipeline core."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session

from db.schema import (
    DataSource,
    Jurisdiction,
    Stratum,
    StratumConstraint,
    Target,
    TargetType,
    init_db,
)
from db.etl_soi import load_soi_targets


@pytest.fixture
def temp_db():
    """Create a temporary database with SOI targets for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_calibration.db"
        engine = init_db(db_path)
        with Session(engine) as session:
            load_soi_targets(session, years=[2021])
        yield db_path


@pytest.fixture
def sample_microdata():
    """Create sample microdata for testing constraint building."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "weight": np.random.uniform(100, 200, n),
        "is_tax_filer": np.random.choice([0, 1], n, p=[0.2, 0.8]),
        "adjusted_gross_income": np.random.lognormal(10, 1.5, n),
        "age": np.random.randint(18, 85, n),
        "state_fips": np.random.choice(["06", "36", "48"], n),  # CA, NY, TX
        "filing_status": np.random.choice(["1", "2", "3", "4"], n),  # single, mfj, mfs, hoh
    })


class TestTargetSpec:
    """Tests for TargetSpec dataclass."""

    def test_target_spec_creation(self):
        """TargetSpec should store target with constraints."""
        from calibration.targets import TargetSpec

        spec = TargetSpec(
            variable="tax_unit_count",
            value=153_774_296,
            target_type=TargetType.COUNT,
            constraints=[("is_tax_filer", "==", "1")],
            source=DataSource.IRS_SOI,
            period=2021,
        )

        assert spec.variable == "tax_unit_count"
        assert spec.value == 153_774_296
        assert spec.target_type == TargetType.COUNT
        assert len(spec.constraints) == 1

    def test_target_spec_with_multiple_constraints(self):
        """TargetSpec should support multiple constraints."""
        from calibration.targets import TargetSpec

        spec = TargetSpec(
            variable="tax_unit_count",
            value=18_892_456,
            target_type=TargetType.COUNT,
            constraints=[
                ("adjusted_gross_income", ">=", "50000"),
                ("adjusted_gross_income", "<", "75000"),
            ],
            source=DataSource.IRS_SOI,
            period=2021,
        )

        assert len(spec.constraints) == 2


class TestGetTargets:
    """Tests for get_targets() function."""

    def test_get_targets_returns_list(self, temp_db):
        """get_targets should return a list of TargetSpec objects."""
        from calibration.targets import get_targets, TargetSpec

        targets = get_targets(db_path=temp_db, jurisdiction="us", year=2021)

        assert isinstance(targets, list)
        assert len(targets) > 0
        assert all(isinstance(t, TargetSpec) for t in targets)

    def test_get_targets_filters_by_year(self, temp_db):
        """get_targets should filter by year."""
        from calibration.targets import get_targets

        targets = get_targets(db_path=temp_db, jurisdiction="us", year=2021)

        assert all(t.period == 2021 for t in targets)

    def test_get_targets_filters_by_source(self, temp_db):
        """get_targets should filter by source."""
        from calibration.targets import get_targets

        targets = get_targets(
            db_path=temp_db,
            jurisdiction="us",
            year=2021,
            sources=["irs-soi"],
        )

        assert all(t.source == DataSource.IRS_SOI for t in targets)

    def test_get_targets_filters_by_variable(self, temp_db):
        """get_targets should filter by variable name."""
        from calibration.targets import get_targets

        targets = get_targets(
            db_path=temp_db,
            jurisdiction="us",
            year=2021,
            variables=["tax_unit_count"],
        )

        assert all(t.variable == "tax_unit_count" for t in targets)

    def test_get_targets_includes_constraints(self, temp_db):
        """get_targets should include stratum constraints in TargetSpec."""
        from calibration.targets import get_targets

        targets = get_targets(db_path=temp_db, jurisdiction="us", year=2021)

        # Find a bracket target (should have AGI constraints)
        bracket_targets = [
            t for t in targets
            if any("adjusted_gross_income" in c[0] for c in t.constraints)
        ]

        assert len(bracket_targets) > 0

    def test_get_targets_empty_for_nonexistent_year(self, temp_db):
        """get_targets should return empty list for year with no data."""
        from calibration.targets import get_targets

        targets = get_targets(db_path=temp_db, jurisdiction="us", year=1900)

        assert targets == []


class TestBuildConstraintMatrix:
    """Tests for build_constraint_matrix() function."""

    def test_build_constraint_matrix_returns_constraints(self, sample_microdata):
        """build_constraint_matrix should return Constraint objects."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix, Constraint

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=800,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        assert isinstance(constraints, list)
        assert len(constraints) == 1
        assert isinstance(constraints[0], Constraint)

    def test_constraint_has_indicator_vector(self, sample_microdata):
        """Constraint should have indicator vector matching microdata rows."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=800,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        assert len(constraints[0].indicator) == len(sample_microdata)
        assert constraints[0].indicator.dtype == np.float64

    def test_constraint_indicator_matches_stratum(self, sample_microdata):
        """Indicator vector should match stratum definition."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=800,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        # Indicator should be 1 where is_tax_filer == 1
        expected = (sample_microdata["is_tax_filer"] == 1).astype(float).values
        np.testing.assert_array_equal(constraints[0].indicator, expected)

    def test_constraint_with_range(self, sample_microdata):
        """Constraint should handle range conditions (>=, <)."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=100,
                target_type=TargetType.COUNT,
                constraints=[
                    ("adjusted_gross_income", ">=", "50000"),
                    ("adjusted_gross_income", "<", "75000"),
                ],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        # Indicator should be 1 where AGI in range
        agi = sample_microdata["adjusted_gross_income"]
        expected = ((agi >= 50000) & (agi < 75000)).astype(float).values
        np.testing.assert_array_equal(constraints[0].indicator, expected)

    def test_constraint_stores_target_value(self, sample_microdata):
        """Constraint should store the target value."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=800,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        assert constraints[0].target_value == 800

    def test_constraint_amount_type_uses_variable(self, sample_microdata):
        """For AMOUNT type, indicator should be variable * mask."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="adjusted_gross_income",
                value=1_000_000_000,
                target_type=TargetType.AMOUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        # For AMOUNT, indicator should be AGI * is_tax_filer
        mask = sample_microdata["is_tax_filer"] == 1
        expected = (sample_microdata["adjusted_gross_income"] * mask).values
        np.testing.assert_array_almost_equal(constraints[0].indicator, expected)

    def test_constraint_with_tolerance(self, sample_microdata):
        """Constraints should store tolerance."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=800,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(
            sample_microdata, targets, tolerance=0.05
        )

        assert constraints[0].tolerance == 0.05

    def test_multiple_constraints(self, sample_microdata):
        """Should build constraints for multiple targets."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=800,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
            TargetSpec(
                variable="adjusted_gross_income",
                value=1_000_000_000,
                target_type=TargetType.AMOUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        assert len(constraints) == 2


class TestEntropyCalibrator:
    """Tests for EntropyCalibrator (placeholder for future implementation)."""

    def test_entropy_calibrator_exists(self):
        """EntropyCalibrator class should exist."""
        from calibration.methods.entropy import EntropyCalibrator

        calibrator = EntropyCalibrator()
        assert calibrator is not None

    def test_entropy_calibrator_has_calibrate_method(self):
        """EntropyCalibrator should have calibrate method."""
        from calibration.methods.entropy import EntropyCalibrator

        calibrator = EntropyCalibrator()
        assert hasattr(calibrator, "calibrate")
