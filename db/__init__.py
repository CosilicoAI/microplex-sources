"""
Database layer for targets and calibration data.

Supports two backends:
- SQLite (local development via SQLModel)
- Supabase (production via supabase-py)

Design follows policyengine-us-data patterns.
"""

from .supabase_client import (
    SupabaseConfig,
    get_supabase_client,
    query_sources,
    query_strata,
    query_targets,
    query_cps,
    query_puf,
    query_silc,
    insert_cps_batch,
    insert_targets_batch,
)

from .schema import (
    DataSource,
    Jurisdiction,
    Stratum,
    StratumConstraint,
    Target,
    TargetType,
    get_engine,
    get_session,
    init_db,
)
from .etl_soi import load_soi_targets
from .etl_soi_state import load_soi_state_targets
from .etl_soi_credits import load_soi_credits_targets
from .etl_soi_income_sources import load_soi_income_sources_targets
from .etl_snap import load_snap_targets
from .etl_hmrc import load_hmrc_targets
from .etl_census import load_census_targets
from .etl_ssa import load_ssa_targets
from .etl_ssi import load_ssi_targets
from .etl_bls import load_bls_targets
from .etl_cps import load_cps_targets
from .etl_cbo import load_cbo_targets
from .etl_obr import load_obr_targets
from .etl_ons import load_ons_targets
from .etl_medicaid import load_medicaid_targets
from .etl_aca_enrollment import load_aca_enrollment_targets

__all__ = [
    # Supabase client
    "SupabaseConfig",
    "get_supabase_client",
    "query_sources",
    "query_strata",
    "query_targets",
    "query_cps",
    "query_puf",
    "query_silc",
    "insert_cps_batch",
    "insert_targets_batch",
    # Schema
    "DataSource",
    "Jurisdiction",
    "Stratum",
    "StratumConstraint",
    "Target",
    "TargetType",
    "get_engine",
    "get_session",
    "init_db",
    # ETL - Historical
    "load_soi_targets",
    "load_soi_state_targets",
    "load_soi_credits_targets",
    "load_soi_income_sources_targets",
    "load_snap_targets",
    "load_hmrc_targets",
    "load_census_targets",
    "load_ssa_targets",
    "load_ssi_targets",
    "load_bls_targets",
    "load_cps_targets",
    "load_medicaid_targets",
    "load_aca_enrollment_targets",
    # ETL - Projections
    "load_cbo_targets",
    "load_obr_targets",
    "load_ons_targets",
]
