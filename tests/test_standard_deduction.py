"""TDD tests for standard deduction calculation.

2024 Standard Deduction values:
- Single: $14,600
- MFJ: $29,200
- HOH: $21,900
- Additional for 65+ or blind: $1,550 (MFJ), $1,950 (single/HOH)
"""
import pytest
import numpy as np
import pandas as pd

# Will fail until we implement calculate_standard_deduction
from cosilico_runner import calculate_standard_deduction, PARAMS_2024


class TestStandardDeduction:
    """Test cases for 26 USC § 63 standard deduction."""

    def test_single_filer(self):
        """Single filer under 65 gets $14,600."""
        df = pd.DataFrame({
            "is_joint": [False],
            "is_head_of_household": [False],
            "age_head": [40],
            "age_spouse": [0],
            "is_blind_head": [False],
            "is_blind_spouse": [False],
            "is_dependent": [False],
            "earned_income": [50000],
        })
        result = calculate_standard_deduction(df, PARAMS_2024)
        assert result[0] == 14600

    def test_married_filing_jointly(self):
        """MFJ under 65 gets $29,200."""
        df = pd.DataFrame({
            "is_joint": [True],
            "is_head_of_household": [False],
            "age_head": [40],
            "age_spouse": [38],
            "is_blind_head": [False],
            "is_blind_spouse": [False],
            "is_dependent": [False],
            "earned_income": [100000],
        })
        result = calculate_standard_deduction(df, PARAMS_2024)
        assert result[0] == 29200

    def test_head_of_household(self):
        """HOH under 65 gets $21,900."""
        df = pd.DataFrame({
            "is_joint": [False],
            "is_head_of_household": [True],
            "age_head": [40],
            "age_spouse": [0],
            "is_blind_head": [False],
            "is_blind_spouse": [False],
            "is_dependent": [False],
            "earned_income": [50000],
        })
        result = calculate_standard_deduction(df, PARAMS_2024)
        assert result[0] == 21900

    def test_single_65_plus(self):
        """Single 65+ gets $14,600 + $1,950 = $16,550."""
        df = pd.DataFrame({
            "is_joint": [False],
            "is_head_of_household": [False],
            "age_head": [67],
            "age_spouse": [0],
            "is_blind_head": [False],
            "is_blind_spouse": [False],
            "is_dependent": [False],
            "earned_income": [30000],
        })
        result = calculate_standard_deduction(df, PARAMS_2024)
        assert result[0] == 16550

    def test_mfj_both_65_plus(self):
        """MFJ with both 65+ gets $29,200 + $1,550*2 = $32,300."""
        df = pd.DataFrame({
            "is_joint": [True],
            "is_head_of_household": [False],
            "age_head": [67],
            "age_spouse": [66],
            "is_blind_head": [False],
            "is_blind_spouse": [False],
            "is_dependent": [False],
            "earned_income": [50000],
        })
        result = calculate_standard_deduction(df, PARAMS_2024)
        assert result[0] == 32300

    def test_single_blind(self):
        """Single blind under 65 gets $14,600 + $1,950 = $16,550."""
        df = pd.DataFrame({
            "is_joint": [False],
            "is_head_of_household": [False],
            "age_head": [40],
            "age_spouse": [0],
            "is_blind_head": [True],
            "is_blind_spouse": [False],
            "is_dependent": [False],
            "earned_income": [40000],
        })
        result = calculate_standard_deduction(df, PARAMS_2024)
        assert result[0] == 16550

    def test_mfj_one_65_one_blind(self):
        """MFJ with one 65+ and other blind gets $29,200 + $1,550*2 = $32,300."""
        df = pd.DataFrame({
            "is_joint": [True],
            "is_head_of_household": [False],
            "age_head": [67],
            "age_spouse": [40],
            "is_blind_head": [False],
            "is_blind_spouse": [True],
            "is_dependent": [False],
            "earned_income": [80000],
        })
        result = calculate_standard_deduction(df, PARAMS_2024)
        assert result[0] == 32300

    def test_dependent_limited_to_earned_plus_400(self):
        """Dependent's std ded is min(basic, max($1,300, earned + $450))."""
        # Dependent with $2,000 earned income: max(1300, 2000+450) = 2450
        # Capped at basic $14,600 → gets $2,450
        df = pd.DataFrame({
            "is_joint": [False],
            "is_head_of_household": [False],
            "age_head": [17],
            "age_spouse": [0],
            "is_blind_head": [False],
            "is_blind_spouse": [False],
            "is_dependent": [True],
            "earned_income": [2000],
        })
        result = calculate_standard_deduction(df, PARAMS_2024)
        assert result[0] == 2450

    def test_dependent_minimum(self):
        """Dependent with $0 earned income gets minimum $1,300."""
        df = pd.DataFrame({
            "is_joint": [False],
            "is_head_of_household": [False],
            "age_head": [15],
            "age_spouse": [0],
            "is_blind_head": [False],
            "is_blind_spouse": [False],
            "is_dependent": [True],
            "earned_income": [0],
        })
        result = calculate_standard_deduction(df, PARAMS_2024)
        assert result[0] == 1300

    def test_vectorized(self):
        """Calculation works on multiple rows."""
        df = pd.DataFrame({
            "is_joint": [False, True, False],
            "is_head_of_household": [False, False, True],
            "age_head": [40, 40, 40],
            "age_spouse": [0, 38, 0],
            "is_blind_head": [False, False, False],
            "is_blind_spouse": [False, False, False],
            "is_dependent": [False, False, False],
            "earned_income": [50000, 100000, 50000],
        })
        result = calculate_standard_deduction(df, PARAMS_2024)
        np.testing.assert_array_equal(result, [14600, 29200, 21900])
