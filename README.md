# NYC Rideshare Wait Time → Cancellation: Causal Analysis

> **What is the causal effect of a 1-minute increase in wait time on rider cancellation probability?**

Naive OLS is confounded — high-demand periods produce both longer waits *and* more cancellations, inflating the association. This project uses **hourly rainfall as an instrumental variable** (real NOAA station data) to isolate exogenous variation in wait time. The result is a textbook confounding story: the naive +3.7pp-per-minute association shrinks to a statistically-zero causal LATE once instrumented, with a strong first stage (F = 83.9) and a Hausman test confirming that OLS and IV differ significantly.

---

## Key Results

| Model | Effect of +1 min wait | Notes |
|---|---|---|
| Naive OLS | +0.0367 (SE 0.0004) | Confounded upward by demand conditions |
| OLS + Controls | +0.0374 (SE 0.0004) | Controls barely move it |
| **IV 2SLS (LATE)** | **+0.0059 (p = 0.68, 95% CI −0.022 to +0.034)** | Causal estimate for rain compliers — indistinguishable from zero |

- **First stage F-stat:** 83.9 (strong instrument — rain robustly shifts wait times)
- **Hausman test:** p = 0.027 → OLS and IV differ significantly, endogeneity confirmed
- **Placebo instrument test:** passed
- **Causal forest (DML) heterogeneity:** mean CATE 0.038–0.041 across boroughs (Bronx highest, Staten Island lowest) and 0.035–0.043 by time of day — a flat gradient. Note these DML estimates do not use the instrument, so they inherit OLS-style confounding and sit near the OLS coefficient, consistent with the IV finding.
- Estimated on a 200,000-row random sample (seed 42) of 54,697,055 cleaned trips, June–August 2023, with real hourly weather from three NOAA stations (JFK, LGA, Central Park).

**Business insight:** the +3.7pp-per-minute correlation overstates the causal effect roughly 6-fold. Investment in wait-time reduction justified by the raw correlation with cancellations would likely be misallocated — the association is driven mostly by demand conditions, not by wait time itself. This is precisely the decision error the IV design exists to catch.

---

## Project Structure

```
nyc-waittime-cancellation/
├── src/
│   ├── config.py               # All constants and thresholds
│   ├── data/
│   │   ├── download.py         # TLC + NOAA download
│   │   ├── clean.py            # Trip cleaning + feature derivation
│   │   └── join.py             # Weather + zone spatial join
│   ├── models/
│   │   ├── ols_baseline.py     # Naive OLS + OLS with controls
│   │   ├── iv_2sls.py          # 2SLS + full diagnostics
│   │   └── causal_forest.py    # HTE via EconML CausalForestDML
│   └── utils/
│       ├── plots.py            # All visualizations
│       └── diagnostics.py     # IV diagnostic utilities
├── app/
│   └── streamlit_app.py        # Interactive dashboard
├── pipeline.py                 # End-to-end runner
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Weather data: fetched automatically from NOAA's token-free NCEI LCD service
# (the legacy CDO v2 API returns 500s on hourly-precipitation queries and is kept
# only as an optional path via NOAA_API_TOKEN; synthetic weather is a last resort
# for offline development and is clearly logged when used)

# 3. Run full pipeline (sample mode — 1 month, fast)
python pipeline.py --sample

# 4. Run on full dataset (3 months)
python pipeline.py

# 5. Launch dashboard
streamlit run app/streamlit_app.py
```

### Run individual steps
```bash
python pipeline.py --step download   # Download raw data only
python pipeline.py --step clean      # Clean trips only
python pipeline.py --step join       # Join weather only
python pipeline.py --step baseline   # OLS baselines only
python pipeline.py --step iv         # IV analysis only
python pipeline.py --step hte        # Causal forest only
python pipeline.py --step plots      # Regenerate figures only
```

---

## Identification Strategy

**Instrument:** Hourly rainfall (mm) at nearest NOAA weather station

**Validity checks:**
- ✓ **Relevance** — Rain increases wait time (F-stat > 10)
- ✓ **Exclusion restriction** — Rain affects cancellation *only* through wait time (argued + placebo tested)
- ✓ **Monotonicity** — Rain increases wait time for all riders, no defiers

**Estimand:** LATE — causal effect for riders whose wait time is affected by rain (outer-borough, non-surge compliers)

---

## Data Sources

| Dataset | Source | Use |
|---|---|---|
| TLC FHVHV Trip Records | nyc.gov/tlc | Trip-level outcomes |
| NOAA Hourly Weather | ncdc.noaa.gov | Instrument (rainfall) |
| TLC Zone Lookup | nyc.gov/tlc | Borough assignment |

---

## Methods & Libraries

- **2SLS:** `linearmodels.IV2SLS` with HC3 robust standard errors
- **Causal Forest:** `econml.dml.CausalForestDML` with cross-fitting
- **Data pipeline:** DuckDB (memory-efficient parquet queries)
- **Visualization:** matplotlib + seaborn
- **Dashboard:** Streamlit + Folium

---

