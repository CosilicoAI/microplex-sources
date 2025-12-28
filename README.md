# microplex-sources

Data sources for microplex microsimulation: microdata, calibration targets, and country-specific pipelines.

## Purpose

This repository provides:

- **Microdata**: Survey and administrative data (CPS, PUF, FRS)
- **Targets**: Calibration targets from authoritative sources (IRS SOI, Census, SSA)
- **Country Pipelines**: Country-specific microplex builders (US districts, UK regions)

Uses [microplex](https://github.com/CosilicoAI/microplex) for synthesis and calibration algorithms.

## Structure

```
microplex-sources/
├── micro/                       # Country-specific microdata pipelines
│   ├── us/                      # United States
│   │   ├── census/              # CPS download and processing
│   │   ├── district.py          # US district microplex builder
│   │   ├── tax_unit_builder.py  # Tax unit construction
│   │   └── synthesis/           # US-specific synthesis
│   └── uk/                      # United Kingdom (planned)
├── db/                          # Targets database and ETL
│   ├── schema.py                # SQLModel: Target, Stratum, StratumConstraint
│   ├── etl_soi.py               # IRS SOI loader
│   ├── etl_snap.py              # SNAP loader
│   ├── etl_census.py            # Census loader
│   └── etl_*.py                 # All ETL pipelines
├── calibration/                 # Calibration infrastructure
│   ├── targets.py               # TargetSpec, get_targets()
│   └── loader.py                # Constraint matrix builder
├── macro/                       # Aggregate targets
│   └── targets.db               # SQLite (dev); Supabase in prod
└── data/                        # Cached data files
```

## Quick Start

### 1. Install

```bash
pip install microplex-sources
# Or for development:
git clone https://github.com/CosilicoAI/microplex-sources
cd microplex-sources
pip install -e ".[dev]"
```

### 2. Download CPS Data

```bash
python micro/us/census/download_cps.py --year 2024
```

### 3. Build US District Microplex

```python
from micro.us.district import DistrictMicroplex, build_targets_from_db
from calibration.targets import get_targets

# Load targets from database
targets = get_targets(jurisdiction="us", year=2021)

# Build district microplex
dm = DistrictMicroplex(n_per_district=1000, target_sparsity=0.9)
result = dm.build(
    seed_data=cps_data,
    districts=["06", "36", "48"],  # CA, NY, TX
    targets=targets,
)
```

## Targets Database

Three-table schema for calibration targets:

- **strata**: Population subgroups (e.g., "CA filers with AGI $50k-$75k")
- **stratum_constraints**: Rules defining each stratum
- **targets**: Administrative totals linked to strata

```python
from calibration.targets import get_targets

# Query targets
targets = get_targets(
    jurisdiction="us",
    year=2021,
    sources=["irs-soi", "census"],
)
```

## Current Coverage

### Microdata

| Source | Variables | Description |
|--------|-----------|-------------|
| US CPS ASEC | 78 | Census household survey (income, benefits, demographics) |
| US IRS PUF | 33 | Tax return sample (income, deductions, credits) |
| UK FRS | 29 | DWP household survey (income, benefits, housing) |

### Targets (ETL Pipelines)

| Source | Coverage | Description |
|--------|----------|-------------|
| IRS SOI | National + state + AGI brackets | Tax return aggregates |
| Census | Demographics, poverty | Population statistics |
| SSA | OASDI, SSI | Social Security data |
| SNAP | State-level | Food assistance |
| Medicaid | State-level | Health coverage |

## Related Repositories

- **[microplex](https://github.com/CosilicoAI/microplex)** - Core synthesis and calibration algorithms
- **[cosilico-us](https://github.com/CosilicoAI/cosilico-us)** - US statute encodings

## Contributing

1. **Microdata**: Add processing code in `micro/<country>/`
2. **Targets**: Add ETL script in `db/etl_<source>.py`
3. Include official documentation URLs
4. Add tests in `tests/`
