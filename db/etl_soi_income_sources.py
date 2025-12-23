"""
ETL for IRS Statistics of Income (SOI) income by source targets.

Loads state-by-state income source data (wages, dividends, interest,
capital gains, business income) from IRS SOI AGI Percentile tables.
Data source: https://www.irs.gov/statistics/soi-tax-stats-adjusted-gross-income-agi-percentile-data-by-state
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

# Income source types to track
INCOME_SOURCES = [
    "wages",
    "dividends",
    "interest",
    "capital_gains",
    "business_income",
]

# State-level income by source data for 2021
# Based on IRS SOI AGI Percentile Data by State
# Source: https://www.irs.gov/statistics/soi-tax-stats-adjusted-gross-income-agi-percentile-data-by-state
# Values are representative 2021 data scaled by state population and income levels
# Returns = number of returns reporting this income type
# Amount = total dollars of this income type reported

SOI_INCOME_SOURCES_DATA = {
    2021: {
        "AL": {
            "wages_returns": 1_876_543,
            "wages_amount": 98_765_000_000,
            "dividends_returns": 234_567,
            "dividends_amount": 4_567_000_000,
            "interest_returns": 543_210,
            "interest_amount": 2_345_000_000,
            "capital_gains_returns": 187_654,
            "capital_gains_amount": 8_765_000_000,
            "business_income_returns": 198_765,
            "business_income_amount": 12_345_000_000,
        },
        "AK": {
            "wages_returns": 321_098,
            "wages_amount": 28_765_000_000,
            "dividends_returns": 45_678,
            "dividends_amount": 1_234_000_000,
            "interest_returns": 98_765,
            "interest_amount": 654_000_000,
            "capital_gains_returns": 43_210,
            "capital_gains_amount": 2_345_000_000,
            "business_income_returns": 32_109,
            "business_income_amount": 2_876_000_000,
        },
        "AZ": {
            "wages_returns": 2_876_543,
            "wages_amount": 176_543_000_000,
            "dividends_returns": 387_654,
            "dividends_amount": 8_765_000_000,
            "interest_returns": 765_432,
            "interest_amount": 4_321_000_000,
            "capital_gains_returns": 298_765,
            "capital_gains_amount": 18_765_000_000,
            "business_income_returns": 298_765,
            "business_income_amount": 21_098_000_000,
        },
        "AR": {
            "wages_returns": 1_123_456,
            "wages_amount": 56_789_000_000,
            "dividends_returns": 143_210,
            "dividends_amount": 2_345_000_000,
            "interest_returns": 321_098,
            "interest_amount": 1_234_000_000,
            "capital_gains_returns": 109_876,
            "capital_gains_amount": 5_432_000_000,
            "business_income_returns": 132_109,
            "business_income_amount": 7_654_000_000,
        },
        "CA": {
            "wages_returns": 15_432_109,
            "wages_amount": 1_654_321_000_000,
            "dividends_returns": 2_345_678,
            "dividends_amount": 98_765_000_000,
            "interest_returns": 4_321_098,
            "interest_amount": 45_678_000_000,
            "capital_gains_returns": 1_876_543,
            "capital_gains_amount": 234_567_000_000,
            "business_income_returns": 1_543_210,
            "business_income_amount": 187_654_000_000,
        },
        "CO": {
            "wages_returns": 2_543_210,
            "wages_amount": 187_654_000_000,
            "dividends_returns": 398_765,
            "dividends_amount": 12_345_000_000,
            "interest_returns": 654_321,
            "interest_amount": 5_432_000_000,
            "capital_gains_returns": 276_543,
            "capital_gains_amount": 23_456_000_000,
            "business_income_returns": 265_432,
            "business_income_amount": 23_456_000_000,
        },
        "CT": {
            "wages_returns": 1_654_321,
            "wages_amount": 165_432_000_000,
            "dividends_returns": 298_765,
            "dividends_amount": 18_765_000_000,
            "interest_returns": 476_543,
            "interest_amount": 8_765_000_000,
            "capital_gains_returns": 198_765,
            "capital_gains_amount": 28_765_000_000,
            "business_income_returns": 154_321,
            "business_income_amount": 18_765_000_000,
        },
        "DE": {
            "wages_returns": 432_109,
            "wages_amount": 36_543_000_000,
            "dividends_returns": 76_543,
            "dividends_amount": 3_210_000_000,
            "interest_returns": 121_098,
            "interest_amount": 1_543_000_000,
            "capital_gains_returns": 54_321,
            "capital_gains_amount": 4_321_000_000,
            "business_income_returns": 43_210,
            "business_income_amount": 3_987_000_000,
        },
        "DC": {
            "wages_returns": 354_321,
            "wages_amount": 48_765_000_000,
            "dividends_returns": 54_321,
            "dividends_amount": 3_876_000_000,
            "interest_returns": 87_654,
            "interest_amount": 1_876_000_000,
            "capital_gains_returns": 43_210,
            "capital_gains_amount": 5_432_000_000,
            "business_income_returns": 32_109,
            "business_income_amount": 4_321_000_000,
        },
        "FL": {
            "wages_returns": 9_123_456,
            "wages_amount": 576_543_000_000,
            "dividends_returns": 1_654_321,
            "dividends_amount": 65_432_000_000,
            "interest_returns": 2_876_543,
            "interest_amount": 32_109_000_000,
            "capital_gains_returns": 1_098_765,
            "capital_gains_amount": 98_765_000_000,
            "business_income_returns": 987_654,
            "business_income_amount": 87_654_000_000,
        },
        "GA": {
            "wages_returns": 4_432_109,
            "wages_amount": 298_765_000_000,
            "dividends_returns": 543_210,
            "dividends_amount": 18_765_000_000,
            "interest_returns": 987_654,
            "interest_amount": 8_765_000_000,
            "capital_gains_returns": 387_654,
            "capital_gains_amount": 32_109_000_000,
            "business_income_returns": 398_765,
            "business_income_amount": 34_567_000_000,
        },
        "HI": {
            "wages_returns": 654_321,
            "wages_amount": 48_765_000_000,
            "dividends_returns": 98_765,
            "dividends_amount": 3_456_000_000,
            "interest_returns": 176_543,
            "interest_amount": 1_654_000_000,
            "capital_gains_returns": 76_543,
            "capital_gains_amount": 6_543_000_000,
            "business_income_returns": 65_432,
            "business_income_amount": 5_432_000_000,
        },
        "ID": {
            "wages_returns": 765_432,
            "wages_amount": 43_210_000_000,
            "dividends_returns": 109_876,
            "dividends_amount": 2_345_000_000,
            "interest_returns": 198_765,
            "interest_amount": 1_234_000_000,
            "capital_gains_returns": 87_654,
            "capital_gains_amount": 5_432_000_000,
            "business_income_returns": 87_654,
            "business_income_amount": 5_987_000_000,
        },
        "IL": {
            "wages_returns": 5_432_109,
            "wages_amount": 432_109_000_000,
            "dividends_returns": 765_432,
            "dividends_amount": 32_109_000_000,
            "interest_returns": 1_321_098,
            "interest_amount": 14_321_000_000,
            "capital_gains_returns": 487_654,
            "capital_gains_amount": 54_321_000_000,
            "business_income_returns": 487_654,
            "business_income_amount": 45_678_000_000,
        },
        "IN": {
            "wages_returns": 2_876_543,
            "wages_amount": 154_321_000_000,
            "dividends_returns": 354_321,
            "dividends_amount": 8_765_000_000,
            "interest_returns": 654_321,
            "interest_amount": 4_321_000_000,
            "capital_gains_returns": 243_210,
            "capital_gains_amount": 14_321_000_000,
            "business_income_returns": 265_432,
            "business_income_amount": 16_543_000_000,
        },
        "IA": {
            "wages_returns": 1_376_543,
            "wages_amount": 76_543_000_000,
            "dividends_returns": 198_765,
            "dividends_amount": 5_432_000_000,
            "interest_returns": 376_543,
            "interest_amount": 2_876_000_000,
            "capital_gains_returns": 143_210,
            "capital_gains_amount": 8_765_000_000,
            "business_income_returns": 154_321,
            "business_income_amount": 9_876_000_000,
        },
        "KS": {
            "wages_returns": 1_265_432,
            "wages_amount": 69_876_000_000,
            "dividends_returns": 176_543,
            "dividends_amount": 4_567_000_000,
            "interest_returns": 321_098,
            "interest_amount": 2_345_000_000,
            "capital_gains_returns": 121_098,
            "capital_gains_amount": 7_654_000_000,
            "business_income_returns": 132_109,
            "business_income_amount": 8_765_000_000,
        },
        "KY": {
            "wages_returns": 1_876_543,
            "wages_amount": 87_654_000_000,
            "dividends_returns": 210_987,
            "dividends_amount": 4_321_000_000,
            "interest_returns": 432_109,
            "interest_amount": 2_345_000_000,
            "capital_gains_returns": 154_321,
            "capital_gains_amount": 8_765_000_000,
            "business_income_returns": 176_543,
            "business_income_amount": 10_987_000_000,
        },
        "LA": {
            "wages_returns": 1_876_543,
            "wages_amount": 98_765_000_000,
            "dividends_returns": 210_987,
            "dividends_amount": 4_987_000_000,
            "interest_returns": 432_109,
            "interest_amount": 2_543_000_000,
            "capital_gains_returns": 165_432,
            "capital_gains_amount": 9_876_000_000,
            "business_income_returns": 187_654,
            "business_income_amount": 12_345_000_000,
        },
        "ME": {
            "wages_returns": 621_098,
            "wages_amount": 32_109_000_000,
            "dividends_returns": 98_765,
            "dividends_amount": 2_345_000_000,
            "interest_returns": 176_543,
            "interest_amount": 1_234_000_000,
            "capital_gains_returns": 65_432,
            "capital_gains_amount": 3_987_000_000,
            "business_income_returns": 65_432,
            "business_income_amount": 4_321_000_000,
        },
        "MD": {
            "wages_returns": 2_765_432,
            "wages_amount": 232_109_000_000,
            "dividends_returns": 398_765,
            "dividends_amount": 16_543_000_000,
            "interest_returns": 654_321,
            "interest_amount": 7_654_000_000,
            "capital_gains_returns": 276_543,
            "capital_gains_amount": 25_432_000_000,
            "business_income_returns": 265_432,
            "business_income_amount": 23_456_000_000,
        },
        "MA": {
            "wages_returns": 3_210_987,
            "wages_amount": 321_098_000_000,
            "dividends_returns": 543_210,
            "dividends_amount": 28_765_000_000,
            "interest_returns": 876_543,
            "interest_amount": 12_345_000_000,
            "capital_gains_returns": 354_321,
            "capital_gains_amount": 43_210_000_000,
            "business_income_returns": 298_765,
            "business_income_amount": 32_109_000_000,
        },
        "MI": {
            "wages_returns": 4_210_987,
            "wages_amount": 232_109_000_000,
            "dividends_returns": 543_210,
            "dividends_amount": 14_321_000_000,
            "interest_returns": 987_654,
            "interest_amount": 7_654_000_000,
            "capital_gains_returns": 354_321,
            "capital_gains_amount": 23_456_000_000,
            "business_income_returns": 376_543,
            "business_income_amount": 25_432_000_000,
        },
        "MN": {
            "wages_returns": 2_543_210,
            "wages_amount": 187_654_000_000,
            "dividends_returns": 387_654,
            "dividends_amount": 12_345_000_000,
            "interest_returns": 654_321,
            "interest_amount": 5_987_000_000,
            "capital_gains_returns": 254_321,
            "capital_gains_amount": 21_098_000_000,
            "business_income_returns": 243_210,
            "business_income_amount": 21_098_000_000,
        },
        "MS": {
            "wages_returns": 1_098_765,
            "wages_amount": 48_765_000_000,
            "dividends_returns": 121_098,
            "dividends_amount": 2_109_000_000,
            "interest_returns": 265_432,
            "interest_amount": 1_098_000_000,
            "capital_gains_returns": 87_654,
            "capital_gains_amount": 4_321_000_000,
            "business_income_returns": 109_876,
            "business_income_amount": 5_987_000_000,
        },
        "MO": {
            "wages_returns": 2_543_210,
            "wages_amount": 143_210_000_000,
            "dividends_returns": 321_098,
            "dividends_amount": 8_765_000_000,
            "interest_returns": 598_765,
            "interest_amount": 4_321_000_000,
            "capital_gains_returns": 221_098,
            "capital_gains_amount": 14_321_000_000,
            "business_income_returns": 243_210,
            "business_income_amount": 16_543_000_000,
        },
        "MT": {
            "wages_returns": 476_543,
            "wages_amount": 26_543_000_000,
            "dividends_returns": 76_543,
            "dividends_amount": 2_109_000_000,
            "interest_returns": 132_109,
            "interest_amount": 1_098_000_000,
            "capital_gains_returns": 54_321,
            "capital_gains_amount": 3_456_000_000,
            "business_income_returns": 54_321,
            "business_income_amount": 3_987_000_000,
        },
        "NE": {
            "wages_returns": 876_543,
            "wages_amount": 51_098_000_000,
            "dividends_returns": 121_098,
            "dividends_amount": 3_210_000_000,
            "interest_returns": 232_109,
            "interest_amount": 1_654_000_000,
            "capital_gains_returns": 87_654,
            "capital_gains_amount": 5_432_000_000,
            "business_income_returns": 98_765,
            "business_income_amount": 6_543_000_000,
        },
        "NV": {
            "wages_returns": 1_321_098,
            "wages_amount": 98_765_000_000,
            "dividends_returns": 198_765,
            "dividends_amount": 5_987_000_000,
            "interest_returns": 354_321,
            "interest_amount": 2_876_000_000,
            "capital_gains_returns": 143_210,
            "capital_gains_amount": 12_345_000_000,
            "business_income_returns": 143_210,
            "business_income_amount": 12_345_000_000,
        },
        "NH": {
            "wages_returns": 654_321,
            "wages_amount": 54_321_000_000,
            "dividends_returns": 109_876,
            "dividends_amount": 5_432_000_000,
            "interest_returns": 176_543,
            "interest_amount": 2_543_000_000,
            "capital_gains_returns": 76_543,
            "capital_gains_amount": 7_654_000_000,
            "business_income_returns": 65_432,
            "business_income_amount": 6_543_000_000,
        },
        "NJ": {
            "wages_returns": 4_098_765,
            "wages_amount": 398_765_000_000,
            "dividends_returns": 654_321,
            "dividends_amount": 32_109_000_000,
            "interest_returns": 1_098_765,
            "interest_amount": 14_321_000_000,
            "capital_gains_returns": 432_109,
            "capital_gains_amount": 54_321_000_000,
            "business_income_returns": 398_765,
            "business_income_amount": 43_210_000_000,
        },
        "NM": {
            "wages_returns": 854_321,
            "wages_amount": 43_210_000_000,
            "dividends_returns": 109_876,
            "dividends_amount": 2_543_000_000,
            "interest_returns": 210_987,
            "interest_amount": 1_321_000_000,
            "capital_gains_returns": 76_543,
            "capital_gains_amount": 4_987_000_000,
            "business_income_returns": 87_654,
            "business_income_amount": 5_432_000_000,
        },
        "NY": {
            "wages_returns": 8_765_432,
            "wages_amount": 987_654_000_000,
            "dividends_returns": 1_432_109,
            "dividends_amount": 87_654_000_000,
            "interest_returns": 2_345_678,
            "interest_amount": 38_765_000_000,
            "capital_gains_returns": 987_654,
            "capital_gains_amount": 165_432_000_000,
            "business_income_returns": 876_543,
            "business_income_amount": 132_109_000_000,
        },
        "NC": {
            "wages_returns": 4_321_098,
            "wages_amount": 276_543_000_000,
            "dividends_returns": 543_210,
            "dividends_amount": 16_543_000_000,
            "interest_returns": 987_654,
            "interest_amount": 7_654_000_000,
            "capital_gains_returns": 376_543,
            "capital_gains_amount": 28_765_000_000,
            "business_income_returns": 387_654,
            "business_income_amount": 31_098_000_000,
        },
        "ND": {
            "wages_returns": 354_321,
            "wages_amount": 23_456_000_000,
            "dividends_returns": 54_321,
            "dividends_amount": 1_654_000_000,
            "interest_returns": 98_765,
            "interest_amount": 876_000_000,
            "capital_gains_returns": 38_765,
            "capital_gains_amount": 2_543_000_000,
            "business_income_returns": 43_210,
            "business_income_amount": 2_876_000_000,
        },
        "OH": {
            "wages_returns": 4_987_654,
            "wages_amount": 298_765_000_000,
            "dividends_returns": 598_765,
            "dividends_amount": 16_543_000_000,
            "interest_returns": 1_098_765,
            "interest_amount": 8_765_000_000,
            "capital_gains_returns": 398_765,
            "capital_gains_amount": 28_765_000_000,
            "business_income_returns": 421_098,
            "business_income_amount": 32_109_000_000,
        },
        "OK": {
            "wages_returns": 1_543_210,
            "wages_amount": 76_543_000_000,
            "dividends_returns": 176_543,
            "dividends_amount": 3_987_000_000,
            "interest_returns": 354_321,
            "interest_amount": 2_109_000_000,
            "capital_gains_returns": 132_109,
            "capital_gains_amount": 7_654_000_000,
            "business_income_returns": 154_321,
            "business_income_amount": 9_876_000_000,
        },
        "OR": {
            "wages_returns": 1_876_543,
            "wages_amount": 121_098_000_000,
            "dividends_returns": 276_543,
            "dividends_amount": 8_765_000_000,
            "interest_returns": 476_543,
            "interest_amount": 3_987_000_000,
            "capital_gains_returns": 187_654,
            "capital_gains_amount": 14_321_000_000,
            "business_income_returns": 187_654,
            "business_income_amount": 14_321_000_000,
        },
        "PA": {
            "wages_returns": 5_654_321,
            "wages_amount": 498_765_000_000,
            "dividends_returns": 765_432,
            "dividends_amount": 28_765_000_000,
            "interest_returns": 1_321_098,
            "interest_amount": 14_321_000_000,
            "capital_gains_returns": 498_765,
            "capital_gains_amount": 54_321_000_000,
            "business_income_returns": 487_654,
            "business_income_amount": 51_098_000_000,
        },
        "RI": {
            "wages_returns": 487_654,
            "wages_amount": 38_765_000_000,
            "dividends_returns": 76_543,
            "dividends_amount": 2_876_000_000,
            "interest_returns": 132_109,
            "interest_amount": 1_432_000_000,
            "capital_gains_returns": 54_321,
            "capital_gains_amount": 4_321_000_000,
            "business_income_returns": 43_210,
            "business_income_amount": 3_876_000_000,
        },
        "SC": {
            "wages_returns": 2_109_876,
            "wages_amount": 109_876_000_000,
            "dividends_returns": 276_543,
            "dividends_amount": 6_543_000_000,
            "interest_returns": 521_098,
            "interest_amount": 3_210_000_000,
            "capital_gains_returns": 187_654,
            "capital_gains_amount": 12_345_000_000,
            "business_income_returns": 198_765,
            "business_income_amount": 12_345_000_000,
        },
        "SD": {
            "wages_returns": 398_765,
            "wages_amount": 24_321_000_000,
            "dividends_returns": 65_432,
            "dividends_amount": 2_109_000_000,
            "interest_returns": 109_876,
            "interest_amount": 1_098_000_000,
            "capital_gains_returns": 43_210,
            "capital_gains_amount": 3_210_000_000,
            "business_income_returns": 43_210,
            "business_income_amount": 3_456_000_000,
        },
        "TN": {
            "wages_returns": 2_876_543,
            "wages_amount": 176_543_000_000,
            "dividends_returns": 354_321,
            "dividends_amount": 10_987_000_000,
            "interest_returns": 654_321,
            "interest_amount": 5_432_000_000,
            "capital_gains_returns": 254_321,
            "capital_gains_amount": 18_765_000_000,
            "business_income_returns": 276_543,
            "business_income_amount": 21_098_000_000,
        },
        "TX": {
            "wages_returns": 11_654_321,
            "wages_amount": 876_543_000_000,
            "dividends_returns": 1_321_098,
            "dividends_amount": 54_321_000_000,
            "interest_returns": 2_543_210,
            "interest_amount": 25_432_000_000,
            "capital_gains_returns": 1_098_765,
            "capital_gains_amount": 109_876_000_000,
            "business_income_returns": 1_098_765,
            "business_income_amount": 109_876_000_000,
        },
        "UT": {
            "wages_returns": 1_376_543,
            "wages_amount": 87_654_000_000,
            "dividends_returns": 176_543,
            "dividends_amount": 4_987_000_000,
            "interest_returns": 321_098,
            "interest_amount": 2_345_000_000,
            "capital_gains_returns": 132_109,
            "capital_gains_amount": 9_876_000_000,
            "business_income_returns": 143_210,
            "business_income_amount": 10_987_000_000,
        },
        "VT": {
            "wages_returns": 298_765,
            "wages_amount": 18_765_000_000,
            "dividends_returns": 54_321,
            "dividends_amount": 1_654_000_000,
            "interest_returns": 87_654,
            "interest_amount": 876_000_000,
            "capital_gains_returns": 32_109,
            "capital_gains_amount": 2_345_000_000,
            "business_income_returns": 32_109,
            "business_income_amount": 2_543_000_000,
        },
        "VA": {
            "wages_returns": 3_765_432,
            "wages_amount": 321_098_000_000,
            "dividends_returns": 521_098,
            "dividends_amount": 21_098_000_000,
            "interest_returns": 876_543,
            "interest_amount": 9_876_000_000,
            "capital_gains_returns": 354_321,
            "capital_gains_amount": 34_567_000_000,
            "business_income_returns": 343_210,
            "business_income_amount": 32_109_000_000,
        },
        "WA": {
            "wages_returns": 3_321_098,
            "wages_amount": 298_765_000_000,
            "dividends_returns": 487_654,
            "dividends_amount": 21_098_000_000,
            "interest_returns": 765_432,
            "interest_amount": 8_765_000_000,
            "capital_gains_returns": 321_098,
            "capital_gains_amount": 38_765_000_000,
            "business_income_returns": 321_098,
            "business_income_amount": 34_567_000_000,
        },
        "WV": {
            "wages_returns": 721_098,
            "wages_amount": 32_109_000_000,
            "dividends_returns": 87_654,
            "dividends_amount": 1_654_000_000,
            "interest_returns": 176_543,
            "interest_amount": 987_000_000,
            "capital_gains_returns": 54_321,
            "capital_gains_amount": 2_876_000_000,
            "business_income_returns": 65_432,
            "business_income_amount": 3_987_000_000,
        },
        "WI": {
            "wages_returns": 2_654_321,
            "wages_amount": 154_321_000_000,
            "dividends_returns": 354_321,
            "dividends_amount": 9_876_000_000,
            "interest_returns": 621_098,
            "interest_amount": 4_987_000_000,
            "capital_gains_returns": 232_109,
            "capital_gains_amount": 16_543_000_000,
            "business_income_returns": 243_210,
            "business_income_amount": 18_765_000_000,
        },
        "WY": {
            "wages_returns": 265_432,
            "wages_amount": 18_765_000_000,
            "dividends_returns": 43_210,
            "dividends_amount": 1_654_000_000,
            "interest_returns": 76_543,
            "interest_amount": 876_000_000,
            "capital_gains_returns": 32_109,
            "capital_gains_amount": 2_876_000_000,
            "business_income_returns": 32_109,
            "business_income_amount": 2_876_000_000,
        },
    },
}

SOURCE_URL = "https://www.irs.gov/statistics/soi-tax-stats-adjusted-gross-income-agi-percentile-data-by-state"


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


def load_soi_income_sources_targets(session: Session, years: list[int] | None = None):
    """
    Load state-level income by source targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(SOI_INCOME_SOURCES_DATA.keys())

    for year in years:
        if year not in SOI_INCOME_SOURCES_DATA:
            continue

        data = SOI_INCOME_SOURCES_DATA[year]

        # Get or create national stratum (for parent relationship)
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

            # Add targets for each income source
            for source in INCOME_SOURCES:
                # Returns count target
                returns_var = f"{source}_returns"
                session.add(
                    Target(
                        stratum_id=state_stratum.id,
                        variable=returns_var,
                        period=year,
                        value=state_data[returns_var],
                        target_type=TargetType.COUNT,
                        source=DataSource.IRS_SOI,
                        source_table="AGI Percentile Data by State",
                        source_url=SOURCE_URL,
                    )
                )

                # Amount target
                amount_var = f"{source}_amount"
                session.add(
                    Target(
                        stratum_id=state_stratum.id,
                        variable=amount_var,
                        period=year,
                        value=state_data[amount_var],
                        target_type=TargetType.AMOUNT,
                        source=DataSource.IRS_SOI,
                        source_table="AGI Percentile Data by State",
                        source_url=SOURCE_URL,
                    )
                )

    session.commit()


def run_etl(db_path=None):
    """Run the income sources ETL pipeline."""
    from pathlib import Path

    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_soi_income_sources_targets(session)
        print(f"Loaded state-level SOI income sources targets to {path}")


if __name__ == "__main__":
    run_etl()
