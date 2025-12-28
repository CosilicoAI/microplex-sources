"""
Microplex Pipeline: Build calibrated microdata from Supabase sources.

Reads raw CPS data and targets from Supabase, runs entropy calibration,
and writes calibrated microplex back to Supabase.

Usage:
    python -m microplex.pipeline --year 2024
    python -m microplex.pipeline --year 2024 --dry-run
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from scipy.optimize import minimize

from db.supabase_client import (
    get_supabase_client,
    query_cps_asec,
    query_targets,
)


@dataclass
class CalibrationResult:
    """Results from entropy calibration."""
    original_weights: np.ndarray
    calibrated_weights: np.ndarray
    adjustment_factors: np.ndarray
    targets_before: Dict[str, Dict]
    targets_after: Dict[str, Dict]
    success: bool
    message: str
    kl_divergence: float


def load_cps_from_supabase(year: int, limit: int = 200000) -> pd.DataFrame:
    """Load raw CPS person data from Supabase."""
    print(f"Loading CPS ASEC {year} from Supabase...")
    df = query_cps_asec(year, table_type="person", limit=limit)
    print(f"  Loaded {len(df):,} person records")
    return df


def load_targets_from_supabase(year: int) -> List[Dict[str, Any]]:
    """Load calibration targets from Supabase."""
    print(f"Loading targets for {year} from Supabase...")
    targets = query_targets(jurisdiction="us", year=year)
    print(f"  Loaded {len(targets)} targets")
    return targets


def build_tax_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build tax units from person-level CPS data.

    Simplified version - groups by household and assigns tax unit based on
    marital status and age. Full version would use relationship variables.
    """
    # Use raw_data if available for additional columns
    if "raw_data" in df.columns:
        # Extract key fields from raw_data
        for col in ["A_MARITL", "A_FAMREL", "TAX_ID"]:
            df[col.lower()] = df["raw_data"].apply(lambda x: x.get(col) if isinstance(x, dict) else None)

    # Calculate total income
    income_cols = ["ptotval", "wsal_val", "semp_val"]
    for col in income_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["total_income"] = df.get("ptotval", 0)
    df["wage_income"] = df.get("wsal_val", 0)
    df["self_employment_income"] = df.get("semp_val", 0)

    # Simple AGI estimate (wages + SE - 1/2 SE tax)
    se_tax = np.maximum(df["self_employment_income"] * 0.0765, 0)
    df["adjusted_gross_income"] = df["wage_income"] + df["self_employment_income"] - se_tax / 2

    # Weight
    if "marsupwt" in df.columns:
        df["weight"] = pd.to_numeric(df["marsupwt"], errors="coerce").fillna(0) / 100
    else:
        df["weight"] = 1.0

    # Filter to likely filers
    filer_mask = (df["total_income"] > 13850) | (df["wage_income"] > 0)
    df = df[filer_mask].copy()
    print(f"  Filtered to {len(df):,} likely filers")

    return df


def assign_agi_bracket(agi: np.ndarray) -> np.ndarray:
    """Assign each record to an AGI bracket matching SOI data."""
    brackets = [
        ("under_1", -np.inf, 1),
        ("1_to_5k", 1, 5000),
        ("5k_to_10k", 5000, 10000),
        ("10k_to_15k", 10000, 15000),
        ("15k_to_20k", 15000, 20000),
        ("20k_to_25k", 20000, 25000),
        ("25k_to_30k", 25000, 30000),
        ("30k_to_40k", 30000, 40000),
        ("40k_to_50k", 40000, 50000),
        ("50k_to_75k", 50000, 75000),
        ("75k_to_100k", 75000, 100000),
        ("100k_to_200k", 100000, 200000),
        ("200k_to_500k", 200000, 500000),
        ("500k_to_1m", 500000, 1000000),
        ("1m_plus", 1000000, np.inf),
    ]

    result = np.empty(len(agi), dtype=object)
    for name, low, high in brackets:
        mask = (agi >= low) & (agi < high)
        result[mask] = name

    return result


def build_constraints_from_targets(
    df: pd.DataFrame,
    targets: List[Dict[str, Any]],
    min_obs: int = 100,
) -> List[Dict]:
    """
    Build calibration constraints from Supabase targets.

    Returns list of constraint dicts with indicator arrays.
    For count targets: indicator = 1 if in stratum, 0 otherwise
    For amount targets (agi_total): skip for now (complex to calibrate)
    """
    constraints = []
    seen_keys = set()  # Deduplicate constraints
    n = len(df)

    df = df.copy()
    df["agi_bracket"] = assign_agi_bracket(df["adjusted_gross_income"].values)

    for target in targets:
        variable = target["variable"]
        value = target["value"]
        target_type = target.get("target_type")

        # Skip rate targets and amount targets (focus on count calibration)
        if target_type == "rate" or target_type == "amount":
            continue

        # Get stratum constraints
        stratum = target.get("strata", {})
        stratum_name = stratum.get("name", "unknown")
        stratum_constraints = stratum.get("stratum_constraints", [])

        # Deduplicate by stratum + variable
        key = (stratum_name, variable)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Build indicator based on stratum
        indicator = np.ones(n, dtype=float)

        for constraint in stratum_constraints:
            var = constraint.get("variable")
            val = constraint.get("value")

            if var == "agi_bracket":
                indicator *= (df["agi_bracket"] == val).astype(float)

        n_obs = indicator.sum()
        if n_obs >= min_obs:
            constraints.append({
                "indicator": indicator,
                "target_value": value,
                "variable": variable,
                "stratum": stratum_name,
                "n_obs": n_obs,
            })

    print(f"  Built {len(constraints)} constraints (min {min_obs} obs each)")
    return constraints


def entropy_calibrate(
    original_weights: np.ndarray,
    constraints: List[Dict],
    bounds: tuple = (0.2, 5.0),
    max_iter: int = 200,
    verbose: bool = True,
) -> tuple:
    """
    Calibrate weights using entropy minimization (gradient descent).
    """
    n = len(original_weights)
    m = len(constraints)

    if verbose:
        print(f"Entropy calibration: {n:,} weights, {m} constraints")

    # Build constraint matrix
    A = np.zeros((m, n))
    targets = np.zeros(m)

    for j, c in enumerate(constraints):
        A[j, :] = c["indicator"]
        targets[j] = c["target_value"]

    def dual_objective(lambdas):
        log_adj = A.T @ lambdas
        log_adj = np.clip(log_adj, -10, 10)
        w = original_weights * np.exp(log_adj)
        return w.sum() - lambdas @ targets

    def dual_gradient(lambdas):
        log_adj = A.T @ lambdas
        log_adj = np.clip(log_adj, -10, 10)
        w = original_weights * np.exp(log_adj)
        return A @ w - targets

    lambda0 = np.zeros(m)
    result = minimize(
        dual_objective,
        lambda0,
        method="L-BFGS-B",
        jac=dual_gradient,
        options={"maxiter": max_iter, "ftol": 1e-8},
    )

    if verbose:
        print(f"Optimization: {result.message}")

    log_adj = A.T @ result.x
    log_adj = np.clip(log_adj, np.log(bounds[0]), np.log(bounds[1]))
    calibrated_weights = original_weights * np.exp(log_adj)

    # KL divergence
    w_safe = np.maximum(calibrated_weights, 1e-10)
    w0_safe = np.maximum(original_weights, 1e-10)
    kl_div = np.sum(w_safe * np.log(w_safe / w0_safe))

    return calibrated_weights, result.success, kl_div


def calibrate_weights(
    df: pd.DataFrame,
    targets: List[Dict[str, Any]],
    verbose: bool = True,
) -> CalibrationResult:
    """Calibrate weights using entropy minimization."""
    original_weights = df["weight"].values.copy()

    if verbose:
        print(f"\nCalibrating {len(df):,} tax units...")
        print(f"Original weighted total: {original_weights.sum():,.0f}")

    constraints = build_constraints_from_targets(df, targets)

    # Pre-scale weights to match total population target
    total_target = None
    for c in constraints:
        if c["variable"] == "tax_unit_count" and c["n_obs"] == len(df):
            total_target = c["target_value"]
            break

    if total_target:
        scale_factor = total_target / original_weights.sum()
        original_weights = original_weights * scale_factor
        if verbose:
            print(f"Pre-scaled weights by {scale_factor:.3f} to match total target {total_target:,.0f}")

    # Pre-calibration values
    targets_before = {}
    for c in constraints:
        current = np.dot(original_weights, c["indicator"])
        targets_before[c["variable"]] = {
            "current": current,
            "target": c["target_value"],
            "error": (current - c["target_value"]) / c["target_value"] if c["target_value"] != 0 else 0,
        }

    # Run calibration
    calibrated_weights, success, kl_div = entropy_calibrate(
        original_weights, constraints, verbose=verbose
    )

    # Post-calibration values
    targets_after = {}
    max_error = 0
    for c in constraints:
        current = np.dot(calibrated_weights, c["indicator"])
        error = (current - c["target_value"]) / c["target_value"] if c["target_value"] != 0 else 0
        targets_after[c["variable"]] = {
            "current": current,
            "target": c["target_value"],
            "error": error,
        }
        max_error = max(max_error, abs(error))

    adjustment_factors = calibrated_weights / original_weights

    if verbose:
        print(f"\nPost-calibration max error: {max_error:.1%}")
        print(f"Weight adjustments: mean={adjustment_factors.mean():.2f}, "
              f"range=[{adjustment_factors.min():.2f}, {adjustment_factors.max():.2f}]")
        print(f"KL divergence: {kl_div:.2f}")

    return CalibrationResult(
        original_weights=original_weights,
        calibrated_weights=calibrated_weights,
        adjustment_factors=adjustment_factors,
        targets_before=targets_before,
        targets_after=targets_after,
        success=success and max_error < 0.05,
        message="Converged" if success else "Did not converge",
        kl_divergence=kl_div,
    )


def write_microplex_to_supabase(
    df: pd.DataFrame,
    year: int,
    chunk_size: int = 200,
) -> int:
    """Write calibrated microplex to Supabase."""
    client = get_supabase_client()
    table_name = f"us_microplex_{year}_person"

    print(f"Writing {len(df):,} records to {table_name}...")

    records = []
    for _, row in df.iterrows():
        record = {
            "source_person_id": int(row.get("id", 0)) if pd.notna(row.get("id")) else None,
            "household_id": int(row.get("ph_seq", 0)) if pd.notna(row.get("ph_seq")) else None,
            "age": int(row.get("a_age", 0)) if pd.notna(row.get("a_age")) else None,
            "state_fips": int(row.get("gestfips", 0)) if pd.notna(row.get("gestfips")) else None,
            "wage_income": float(row.get("wage_income", 0)),
            "self_employment_income": float(row.get("self_employment_income", 0)),
            "total_income": float(row.get("total_income", 0)),
            "adjusted_gross_income": float(row.get("adjusted_gross_income", 0)) if "adjusted_gross_income" in row else None,
            "original_weight": float(row.get("original_weight", 0)),
            "calibrated_weight": float(row.get("weight", 0)),
            "weight_adjustment": float(row.get("weight_adjustment", 1.0)),
            "agi_bracket": row.get("agi_bracket"),
        }
        records.append(record)

    total = 0
    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]
        client.schema("microplex").table(table_name).insert(chunk).execute()
        total += len(chunk)
        if (i + chunk_size) % 5000 == 0:
            print(f"  Inserted {total:,} / {len(records):,}")

    print(f"  Done: {total:,} records")
    return total


def run_pipeline(
    year: int = 2024,
    dry_run: bool = False,
    limit: int = 200000,
) -> pd.DataFrame:
    """Run the full microplex pipeline."""
    print("=" * 60)
    print("MICROPLEX PIPELINE")
    print("=" * 60)

    # Load data
    df = load_cps_from_supabase(year, limit=limit)
    targets = load_targets_from_supabase(year)

    if not targets:
        # Fall back to 2021 targets if year not available
        print(f"  No targets for {year}, trying 2021...")
        targets = load_targets_from_supabase(2021)

    # Build tax units
    df = build_tax_units(df)

    # Calibrate
    result = calibrate_weights(df, targets)

    # Add calibrated weights to dataframe
    df["original_weight"] = result.original_weights
    df["weight"] = result.calibrated_weights
    df["weight_adjustment"] = result.adjustment_factors

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tax units: {len(df):,}")
    print(f"Original weighted: {result.original_weights.sum():,.0f}")
    print(f"Calibrated weighted: {result.calibrated_weights.sum():,.0f}")
    print(f"Calibration success: {result.success}")

    if not dry_run:
        write_microplex_to_supabase(df, year)
    else:
        print("\nDRY RUN - not writing to Supabase")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run microplex pipeline")
    parser.add_argument("--year", type=int, default=2024, help="Data year")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Supabase")
    parser.add_argument("--limit", type=int, default=200000, help="Max records to load")
    args = parser.parse_args()

    run_pipeline(year=args.year, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
