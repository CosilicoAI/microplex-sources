# Microplex Data Architecture

## Overview

Microplex is the final calibrated microdata used for policy simulation. This document describes the full data pipeline from raw sources to calibrated output.

## Storage Layers

### R2 (Object Storage)
Raw files - immutable, versioned.

```
sources/
  irs/soi/2023/table_1_2.xlsx          # IRS SOI individual returns
  census/acs/2023/pums_hh.csv          # ACS PUMS households
  census/cps/2024/asec_raw.zip         # CPS ASEC microdata
  bls/cpi/2024/monthly.csv             # CPI monthly series
  usda/snap/2023/qc_data.xlsx          # SNAP QC data
```

### Supabase Schemas

| Schema | Purpose | Example Tables |
|--------|---------|----------------|
| `arch` | Registry of raw R2 files | sources, files, content, fetch_log |
| `indices` | Processed time series | series, values (CPI, wage growth) |
| `targets` | Calibration targets | strata, constraints, targets |
| `microdata` | Intermediate microdata | cps_asec, acs_pums, synthetic |
| `microplex` | Final calibrated data | households, persons, tax_units |

## Data Flow

```
R2 (raw files)
      │
      ▼
arch.* (registry, points to R2, tracks changes)
      │
      ├──────────────────┬──────────────────┐
      ▼                  ▼                  ▼
indices.*           targets.*          microdata.*
(time series)       (calibration       (intermediate
                     targets)           microdata)
      │                  │                  │
      └──────────────────┴──────────────────┘
                         │
                         ▼
                    microplex.*
                 (final calibrated
                     microdata)
```

## Calibration Pipeline

### 1. Targets (from `targets.*` schema)

Targets define what aggregates the microdata should match:

```sql
-- targets.strata: Population subgroups
INSERT INTO targets.strata (name, jurisdiction, constraints)
VALUES ('CA adults 18-64', 'us', '[{"variable": "age", "operator": ">=", "value": "18"}, ...]');

-- targets.targets: Values to match
INSERT INTO targets.targets (stratum_id, variable, value, period)
VALUES (1, 'us:statute/26/32#eitc_recipients', 2500000, 2023);
```

### 2. Variable Resolution

Target variables use qualified references: `{model}:{path}#{variable}`

```
us:statute/26/32#eitc
│  │            │
│  │            └─ variable name
│  └─ path within model
└─ country model (cosilico-us)
```

The `get_entity()` function resolves this to find the variable's entity (person, tax_unit, household) by parsing the source `.rac` file in cosilico-us.

### 3. Hierarchical Constraint Building

Since all weights are at the household level, person-level targets must be aggregated:

```python
# What we want: count of people aged 18-64 in California
# What we compute: for each household, count matching persons

build_hierarchical_constraint_matrix(
    hh_df=households,      # 18,825 rows
    person_df=persons,     # 48,292 rows
    targets=targets,       # from targets.* schema
)

# Returns: Constraint objects with indicators at household level
# indicator[i] = count of matching persons in household i
```

The key insight: since all persons in a household share the household weight:
```
sum over HH(hh_weight * count_matching_in_hh) = total_matching_persons
```

### 4. IPF Calibration

Iterative Proportional Fitting adjusts household weights to match all targets:

```python
for iteration in range(max_iter):
    for constraint in constraints:
        current = sum(hh_weight * constraint.indicator)
        ratio = constraint.target_value / current
        hh_weight *= clip(ratio, 0.9, 1.1)  # damped

    hh_weight = clip(hh_weight, min_weight, max_weight)
```

### 5. Output (`microplex.*` schema)

```sql
-- microplex.households: Calibrated household weights
SELECT household_id, state_fips, weight, ...
FROM microplex.households;
-- 18,825 rows

-- microplex.persons: Linked to households
SELECT person_id, household_id, age, employment_income, ...
FROM microplex.persons;
-- 48,292 rows
```

## Entity Hierarchy

```
Household (weight lives here)
├── Tax Unit 1
│   ├── Person A (head)
│   └── Person B (spouse)
└── Tax Unit 2
    └── Person C (dependent filing separately)
```

- **Weights are always at household level**
- Person-level targets → aggregate count/sum per household
- Tax-unit-level targets → aggregate count/sum per household
- Household-level targets → direct indicator (0/1)

## Schema Details

### arch.* (Registry)

```sql
arch.sources        -- institution, dataset, url, update_frequency
arch.files          -- r2_key, checksum, source_id, fetched_at
arch.content        -- parsed text/tables for full-text search
arch.fetch_log      -- change detection, version history
```

### indices.* (Time Series)

```sql
indices.series      -- series_id, name, source, frequency
indices.values      -- series_id, date, value
```

Used by `indexing_rule` in .rac files:
```yaml
indexing_rule eitc_inflation:
  series: indices/bls_chained_cpi_u
  base_year: 2015
  rounding: 10
```

### targets.* (Calibration Targets)

```sql
targets.strata           -- population subgroups with constraints
targets.constraints      -- variable, operator, value per stratum
targets.targets          -- stratum_id, variable, value, period, source
```

### microdata.* (Intermediate)

```sql
microdata.cps_asec       -- processed CPS (cleaned, typed)
microdata.acs_pums       -- processed ACS
microdata.synthetic      -- generated synthetic records
```

### microplex.* (Final Output)

```sql
microplex.households     -- calibrated household records with weights
microplex.persons        -- person records linked to households
microplex.tax_units      -- tax unit records (future)
```
