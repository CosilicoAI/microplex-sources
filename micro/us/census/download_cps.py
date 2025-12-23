"""
CPS ASEC Microdata Downloader

Downloads Current Population Survey Annual Social and Economic Supplement
from Census Bureau and processes to parquet format for calibration.

Usage:
    python download_cps.py [--year 2023] [--output micro/us/cps_2023.parquet]
"""

import io
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# Census Bureau CPS ASEC download URLs
# Data for year N is published in March of year N+1
CPS_URL_BY_YEAR = {
    2018: "https://www2.census.gov/programs-surveys/cps/datasets/2019/march/asecpub19csv.zip",
    2019: "https://www2.census.gov/programs-surveys/cps/datasets/2020/march/asecpub20csv.zip",
    2020: "https://www2.census.gov/programs-surveys/cps/datasets/2021/march/asecpub21csv.zip",
    2021: "https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip",
    2022: "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
    2023: "https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub24csv.zip",
    2024: "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asecpub25csv.zip",
}

# Columns to extract from person file (pppub{YY}.csv)
PERSON_COLUMNS = {
    # Identifiers
    "PH_SEQ": "household_id",  # Household sequence number
    "P_SEQ": "person_seq",  # Person sequence within household
    # Demographics
    "A_AGE": "age",
    "A_SEX": "sex",  # 1=Male, 2=Female
    "PRDTRACE": "race",  # Race code
    "A_MARITL": "marital_status",
    # Geographic
    "GESTFIPS": "state_fips",
    # Weight
    "A_FNLWGT": "weight",  # Final weight (person)
    "MARSUPWT": "march_supplement_weight",
    # Employment
    "A_CLSWKR": "class_of_worker",
    "A_WKSTAT": "work_status",
    "A_HRS1": "hours_worked",
    "A_USLHRS": "usual_hours",
    # Income sources
    "WSAL_VAL": "wage_salary_income",
    "SEMP_VAL": "self_employment_income",
    "FRSE_VAL": "farm_self_employment_income",
    "INT_VAL": "interest_income",
    "DIV_VAL": "dividend_income",
    "RNT_VAL": "rental_income",
    "SS_VAL": "social_security_income",
    "SSI_VAL": "ssi_income",
    "PAW_VAL": "public_assistance_income",
    "UC_VAL": "unemployment_compensation",
    "VET_VAL": "veterans_benefits",
    "PENSIONS": "pension_income",
    "OI_OFF": "other_income_type",
    "OI_VAL": "other_income",
    "PTOTVAL": "total_person_income",
    "PEARNVAL": "total_earnings",
    # Tax-related
    "FEDTAX_AC": "federal_tax",
    "STAESSION_AC": "state_tax",
    "FICA": "fica_tax",
    "EIT_CRED": "eitc_received",
    "ACTC_CRD": "actc_received",
    "CTC_CRD": "ctc_received",
    # Family
    "A_FAMREL": "family_relationship",
    "A_PARENT": "parent_present",
    "FOWNU18": "own_children_under_18",
}

# Columns from household file (hhpub{YY}.csv)
HOUSEHOLD_COLUMNS = {
    "H_SEQ": "household_id",
    "H_NUMPER": "household_size",
    "HHINC": "household_income",
    "HHSTATUS": "household_status",
    "HPROP_VAL": "property_value",
    "THINCR": "tenure",  # Owner/renter status
}


def download_cps_zip(year: int, progress: bool = True) -> bytes:
    """Download CPS ASEC ZIP file for given year."""
    if year not in CPS_URL_BY_YEAR:
        available = sorted(CPS_URL_BY_YEAR.keys())
        raise ValueError(f"Year {year} not available. Available: {available}")

    url = CPS_URL_BY_YEAR[year]
    print(f"Downloading CPS ASEC {year} from {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    content = io.BytesIO()

    if progress and total_size:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Download") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                content.write(chunk)
                pbar.update(len(chunk))
    else:
        for chunk in response.iter_content(chunk_size=8192):
            content.write(chunk)

    return content.getvalue()


def extract_person_data(zip_content: bytes, year: int) -> pd.DataFrame:
    """Extract person-level data from CPS ZIP."""
    yy = str(year + 1)[-2:]  # CPS 2023 data is in asecpub24csv.zip

    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        # Find person file
        person_file = None
        for name in zf.namelist():
            if f"pppub{yy}" in name.lower() and name.endswith(".csv"):
                person_file = name
                break

        if not person_file:
            # Try alternate naming
            for name in zf.namelist():
                if "pppub" in name.lower() and name.endswith(".csv"):
                    person_file = name
                    break

        if not person_file:
            raise ValueError(f"Could not find person file in ZIP. Files: {zf.namelist()}")

        print(f"Extracting {person_file}")

        with zf.open(person_file) as f:
            # Read only columns we need
            available_cols = pd.read_csv(f, nrows=0).columns.tolist()
            cols_to_read = [c for c in PERSON_COLUMNS.keys() if c in available_cols]

            f.seek(0)
            df = pd.read_csv(f, usecols=cols_to_read, low_memory=False)

    # Rename columns
    rename_map = {k: v for k, v in PERSON_COLUMNS.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    return df


def process_cps_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw CPS data into calibration-ready format."""
    # Create unique person ID
    df["person_id"] = df["household_id"].astype(str) + "_" + df["person_seq"].astype(str)

    # Derive employment status (1=employed, 0=not employed)
    if "class_of_worker" in df.columns:
        df["employment_status"] = (df["class_of_worker"] > 0).astype(int)
    elif "work_status" in df.columns:
        df["employment_status"] = df["work_status"].isin([1, 2, 3, 4]).astype(int)
    else:
        df["employment_status"] = 0

    # Calculate total income
    income_cols = [
        "wage_salary_income",
        "self_employment_income",
        "farm_self_employment_income",
        "interest_income",
        "dividend_income",
        "rental_income",
    ]
    available_income_cols = [c for c in income_cols if c in df.columns]
    if available_income_cols:
        df["income"] = df[available_income_cols].fillna(0).sum(axis=1)
    elif "total_person_income" in df.columns:
        df["income"] = df["total_person_income"].fillna(0)
    else:
        df["income"] = 0

    # Has children indicator
    if "own_children_under_18" in df.columns:
        df["has_children"] = (df["own_children_under_18"] > 0).astype(int)
    else:
        df["has_children"] = 0

    # Use march supplement weight if available, otherwise final weight
    # CPS weights have 2 implied decimal places, so divide by 100
    if "march_supplement_weight" in df.columns:
        df["weight"] = df["march_supplement_weight"].fillna(0) / 100
    elif "weight" in df.columns:
        df["weight"] = df["weight"].fillna(0) / 100
    else:
        df["weight"] = 1

    # Filter to positive weights only
    df = df[df["weight"] > 0].copy()

    # Select final columns for calibration
    output_cols = [
        "person_id",
        "household_id",
        "weight",
        "age",
        "income",
        "employment_status",
        "has_children",
        "state_fips",
    ]

    # Add optional columns if available
    optional_cols = [
        "sex",
        "race",
        "marital_status",
        "wage_salary_income",
        "self_employment_income",
        "social_security_income",
        "ssi_income",
        "eitc_received",
        "ctc_received",
        "actc_received",
    ]
    for col in optional_cols:
        if col in df.columns:
            output_cols.append(col)

    available_output_cols = [c for c in output_cols if c in df.columns]
    return df[available_output_cols].copy()


def download_and_process_cps(
    year: int,
    output_path: Optional[Path] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Download CPS ASEC data and process to parquet.

    Args:
        year: Tax year (e.g., 2023 for 2023 tax year data)
        output_path: Path to save parquet file (default: micro/us/cps_{year}.parquet)
        progress: Show download progress bar

    Returns:
        Processed DataFrame
    """
    # Download
    zip_content = download_cps_zip(year, progress=progress)

    # Extract person data
    df = extract_person_data(zip_content, year)
    print(f"Extracted {len(df):,} person records")

    # Process
    df = process_cps_data(df)
    print(f"Processed to {len(df):,} records with positive weights")
    print(f"Total weighted population: {df['weight'].sum():,.0f}")

    # Save
    if output_path is None:
        output_path = Path(__file__).parent.parent / f"cps_{year}.parquet"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download CPS ASEC microdata")
    parser.add_argument("--year", type=int, default=2023, help="Tax year")
    parser.add_argument("--output", type=str, help="Output parquet path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    download_and_process_cps(args.year, output_path)


if __name__ == "__main__":
    main()
