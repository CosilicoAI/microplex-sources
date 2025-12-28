"""
Supabase client for microplex-sources.

Provides connection to Cosilico Supabase database for:
- Calibration targets (microplex.targets)
- Microdata (microplex.cps, microplex.puf, microplex.silc)
- Data sources metadata (microplex.sources)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd
from supabase import create_client, Client


@dataclass
class SupabaseConfig:
    """Configuration for Supabase connection."""

    url: str
    secret_key: str

    @classmethod
    def from_env(cls) -> "SupabaseConfig":
        """
        Load configuration from environment variables.

        Required:
            COSILICO_SUPABASE_URL: Supabase project URL
            COSILICO_SUPABASE_SECRET_KEY: Service role key for full access

        Raises:
            ValueError: If required environment variables are missing
        """
        url = os.environ.get("COSILICO_SUPABASE_URL")
        if not url:
            raise ValueError(
                "COSILICO_SUPABASE_URL not set. "
                "Set this to your Supabase project URL."
            )

        secret_key = os.environ.get("COSILICO_SUPABASE_SECRET_KEY")
        if not secret_key:
            raise ValueError(
                "COSILICO_SUPABASE_SECRET_KEY not set. "
                "Set this to your service role key."
            )

        return cls(url=url, secret_key=secret_key)


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Get a Supabase client instance.

    Uses service role key for full database access.
    Client is cached for reuse.

    Returns:
        Supabase client instance

    Raises:
        ValueError: If environment variables are not set
    """
    config = SupabaseConfig.from_env()
    return create_client(config.url, config.secret_key)


def query_sources(
    jurisdiction: Optional[str] = None,
    institution: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query data sources from microplex.sources.

    Args:
        jurisdiction: Filter by jurisdiction (e.g., "us", "uk")
        institution: Filter by institution (e.g., "irs", "census")

    Returns:
        List of source records
    """
    client = get_supabase_client()
    # Use schema() to query from microplex schema
    query = client.schema("microplex").table("sources").select("*")

    if jurisdiction:
        query = query.eq("jurisdiction", jurisdiction)
    if institution:
        query = query.eq("institution", institution)

    result = query.execute()
    return result.data


def query_strata(
    jurisdiction: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query strata with their constraints from microplex.strata.

    Args:
        jurisdiction: Filter by jurisdiction

    Returns:
        List of strata records with nested constraints
    """
    client = get_supabase_client()
    # Join with stratum_constraints
    query = client.schema("microplex").table("strata").select("*, stratum_constraints(*)")

    if jurisdiction:
        query = query.eq("jurisdiction", jurisdiction)

    result = query.execute()
    return result.data


def query_targets(
    jurisdiction: Optional[str] = None,
    year: Optional[int] = None,
    source_id: Optional[str] = None,
    variable: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query calibration targets from microplex.targets.

    Args:
        jurisdiction: Filter by jurisdiction (via stratum join)
        year: Filter by period/year
        source_id: Filter by source UUID
        variable: Filter by variable name

    Returns:
        List of target records with stratum info
    """
    client = get_supabase_client()
    query = client.schema("microplex").table("targets").select("*, strata(*), sources(*)")

    if year:
        query = query.eq("period", year)
    if source_id:
        query = query.eq("source_id", source_id)
    if variable:
        query = query.eq("variable", variable)

    result = query.execute()

    # Filter by jurisdiction if specified (post-query since it's on joined table)
    data = result.data
    if jurisdiction:
        data = [t for t in data if t.get("strata", {}).get("jurisdiction") == jurisdiction]

    return data


def query_cps(
    year: int,
    state_fips: Optional[int] = None,
    limit: int = 100000,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Query CPS microdata from microplex.cps.

    Args:
        year: Data year (required)
        state_fips: Filter by state FIPS code
        limit: Maximum records to return (default 100k)
        columns: Specific columns to select (default all)

    Returns:
        DataFrame with CPS records
    """
    client = get_supabase_client()

    select_cols = ",".join(columns) if columns else "*"
    query = client.schema("microplex").table("cps").select(select_cols)

    query = query.eq("year", year)

    if state_fips:
        query = query.eq("state_fips", state_fips)

    query = query.limit(limit)
    result = query.execute()

    return pd.DataFrame(result.data)


def query_puf(
    year: int,
    limit: int = 100000,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Query PUF microdata from microplex.puf.

    Args:
        year: Data year
        limit: Maximum records
        columns: Specific columns to select

    Returns:
        DataFrame with PUF records
    """
    client = get_supabase_client()

    select_cols = ",".join(columns) if columns else "*"
    query = client.schema("microplex").table("puf").select(select_cols)
    query = query.eq("year", year)
    query = query.limit(limit)

    result = query.execute()
    return pd.DataFrame(result.data)


def query_silc(
    year: int,
    country: Optional[str] = None,
    limit: int = 100000,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Query EU-SILC microdata from microplex.silc.

    Args:
        year: Data year
        country: ISO 2-letter country code (e.g., "DE", "FR")
        limit: Maximum records
        columns: Specific columns to select

    Returns:
        DataFrame with SILC records
    """
    client = get_supabase_client()

    select_cols = ",".join(columns) if columns else "*"
    query = client.schema("microplex").table("silc").select(select_cols)
    query = query.eq("year", year)

    if country:
        query = query.eq("country", country)

    query = query.limit(limit)
    result = query.execute()

    return pd.DataFrame(result.data)


def insert_cps_batch(
    records: List[Dict[str, Any]],
    chunk_size: int = 1000,
) -> int:
    """
    Insert CPS records in batches.

    Args:
        records: List of record dicts
        chunk_size: Records per batch (default 1000)

    Returns:
        Number of records inserted
    """
    client = get_supabase_client()
    total = 0

    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        client.schema("microplex").table("cps").insert(chunk).execute()
        total += len(chunk)

    return total


def insert_targets_batch(
    targets: List[Dict[str, Any]],
    chunk_size: int = 100,
) -> int:
    """
    Insert target records in batches.

    Args:
        targets: List of target dicts
        chunk_size: Records per batch

    Returns:
        Number of records inserted
    """
    client = get_supabase_client()
    total = 0

    for i in range(0, len(targets), chunk_size):
        chunk = targets[i:i + chunk_size]
        client.schema("microplex").table("targets").insert(chunk).execute()
        total += len(chunk)

    return total
