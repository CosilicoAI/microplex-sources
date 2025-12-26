# cosilico-data-sources

Canonical documentation of data sources for tax-benefit microsimulation.

## Purpose

This repository documents both **micro** (individual-level) and **macro** (aggregate) data sources:

- **Micro**: Survey and administrative microdata variable mappings to statutes
- **Macro**: Administrative totals for calibration and validation

## Structure

```
cosilico-data-sources/
├── micro/                       # Individual-level data sources
│   ├── us/                      # United States
│   │   ├── census/cps-asec/     # CPS ASEC variables
│   │   └── irs/puf/             # IRS PUF variables
│   └── uk/                      # United Kingdom
│       └── ons/frs/             # FRS variables
├── macro/                       # Aggregate targets
│   ├── targets.db               # SQLite (dev); Supabase in prod
│   └── [future: YAML configs]
├── db/                          # Database schema and ETL
│   ├── schema.py                # SQLModel tables
│   └── etl_soi.py               # IRS SOI loader
└── README.md
```

## Current Coverage

### Micro (Variable Mappings)

| Source | Variables | Description |
|--------|-----------|-------------|
| US CPS ASEC | 78 | Census household survey (income, benefits, demographics) |
| US IRS PUF | 33 | Tax return sample (income, deductions, credits) |
| UK FRS | 29 | DWP household survey (income, benefits, housing) |
| **Total** | **140** | |

### Macro (Targets)

| Source | Coverage | Description |
|--------|----------|-------------|
| IRS SOI | National + AGI brackets | Tax return aggregates |
| (planned) HMRC | UK national | Tax and benefit totals |

## Variable Schema

Each variable file (YAML) contains:

```yaml
variable: WSAL_VAL              # Source variable name
source: cps-asec                # Source identifier
entity: person                  # person, household, tax_unit, spm_unit
period: year                    # year, month, week, point_in_time
dtype: money                    # money, count, rate, boolean, category

documentation:
  url: "https://www2.census.gov/..."
  section: "Person Income Variables"

concept: wages_and_salaries
definition: "Gross wages, salaries, tips before deductions"

maps_to:
  - jurisdiction: us
    statute: "26 USC § 61(a)(1)"
    variable: wages
    coverage: full

gaps:
  - component: tips_underreporting
    impact: medium
    notes: "Cash tips systematically underreported"
```

## Targets Database

Three-table schema (following policyengine-us-data patterns):

- **strata**: Population subgroups (e.g., "CA filers with AGI $50k-$75k")
- **stratum_constraints**: Rules defining each stratum
- **targets**: Administrative totals linked to strata

Local SQLite for development; Supabase for production.

```python
from db.schema import init_db, get_session
from db.etl_soi import load_soi_targets

# Initialize and load
init_db()
with get_session() as session:
    load_soi_targets(session, years=[2021])
```

## Related Repositories

- **cosilico-microdata** - Builds calibrated datasets using these sources
- **cosilico-us** / **cosilico-uk** - Statute encodings referencing these concepts

## Microsimulation Workflow

### 1. Download CPS Data

```bash
# Download and cache raw CPS ASEC data
python micro/us/census/download_cps.py --year 2024
```

### 2. Convert to Cosilico Format

```bash
# Convert CPS to microsim input format
python scripts/cps_to_cosilico.py --year 2024 --output microsim_2024.parquet

# With entropy calibration to IRS SOI targets
python scripts/cps_to_cosilico.py --year 2024 --calibrate --summary
```

### 3. Run Microsimulation

```bash
# In cosilico-us repository
python -m cosilico_us.microsim --input cosilico_input_2024.parquet
```

See `docs/cps-variable-coverage.md` for detailed variable coverage analysis.

## Contributing

1. **Micro**: Add variable YAML in `micro/<country>/<source>/`
2. **Macro**: Add ETL script in `db/etl_<source>.py`
3. Include official documentation URLs
4. Document coverage and known gaps
