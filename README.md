# hu-florido-v2

WhatsApp analytics focused on three anonymised couples. The pipeline ingests an exported chat log, enriches it with couple and gender metadata, and renders a gallery of publication-ready plots that surface who talks most, when conversations happen, and how family-related topics evolve over time.

## Overview

- Cleans and normalises WhatsApp exports (CSV or Parquet)
- Maps real names or pseudonyms to couples and genders
- Produces nine figures covering dominance, timing, and topic trends
- Highlights notable behaviours such as “Men rule the day” vs “Women rule the night” messaging peaks

## Project structure

```
.
├── config.toml
├── data/
│   ├── processed/
│   │   ├── couples_reference.json
│   │   ├── output.csv
│   │   └── output.parq
│   └── raw/
├── img/
│   ├── comparing_categories_1.png
│   ├── comparing_categories_2.png
│   ├── distribution_categories_1.png
│   ├── distribution_categories_2.png
│   ├── distribution_categories_4.png
│   ├── distribution_categories_5.png
│   ├── time_series.png
│   ├── time_series_2.png
│   └── time_series_3.png
├── src/
│   ├── myFigures.py
│   └── myPreprocess.py
├── pyproject.toml
└── README.md
```

## Requirements

- macOS (tested) with Python 3.12+
- Dependencies managed via `pyproject.toml`:
  - wa-analyzer
  - pandas, numpy, pyarrow
  - matplotlib, seaborn
  - scipy (spline smoothing)
- Optional: [uv](https://github.com/astral-sh/uv) for fast environment management

## Setup

### Option A: uv (recommended)

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv sync
```

### Option B: venv + pip

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install wa-analyzer pyarrow matplotlib seaborn pandas numpy scipy
```

## Configuration

Point `config.toml` at your preferred inputs:

```toml
processed = "data/processed"
current = "output.parq"
inputpath = "data/raw/whatsapp.txt"
```

## Data preprocessing

1. Create `data/processed/couples_reference.json` with a mapping of message author → couple ID (`couple1`/`couple2`/`couple3`) and gender (`male`/`female`).
2. Run the preprocessing pipeline:

```sh
python src/myPreprocess.py
```

This writes `output.csv` and `output.parq`, which the figure scripts consume.

## Generating the figures

Render the entire gallery in one go:

```sh
python src/myFigures.py
```

Key visuals include:

1. **Messaging dominance gauges** (`img/comparing_categories_1.png`)
   - Semicircular gauges per couple showing who sends more messages.

2. **Weekly cadence comparison** (`img/distribution_categories_1.png`)
   - Side-by-side bars showing each gender’s share of messages per weekday.

3. **Hourly curve – “Who texts when?”** (`img/distribution_categories_2.png`)
   - Smoothed, normalised curves shifted to a 02:00–24:00 window.
   - Annotated with horizontal guides: **Men rule the day** (blue) and **Women rule the night** (pink).

4. **Couple share bar chart** (`img/distribution_categories_4.png`)
   - Weekday vs weekend messaging split by gender.

5. **Weekend flip slope chart** (`img/distribution_categories_5.png`)
   - Two-point slope showing how gender dominance changes from weekdays to weekends.

6. **Quarterly gender timeline** (`img/comparing_categories_2.png`)
   - 100% stacked bars per quarter highlighting shifts inside each couple.

7. **Child-talk focus series** (`img/time_series.png`)
   - Men vs women usage of family-related words with annotated milestones.

8. **With vs without children** (`img/time_series_2.png`)
   - Comparing couples who have children to those who do not.

9. **Combined family-word trend** (`img/time_series_3.png`)
   - Aggregated signal across all couples.

## Notes and conventions

- Media messages (`<Media omitted>`) are removed before analysis.
- Colour palette: men = `#1E3A8A`, women = `#FF69B4`; highlight annotations reuse these tones.
- Hourly curves rely on periodic cubic splines to keep transitions smooth across midnight.
- Family word tracking covers 20+ tokens (baby, child, daycare, stroller, etc.) and flags key life events via vertical markers.

## Troubleshooting

- **Missing processed data** → run `python src/myPreprocess.py` first.
- **Import errors** → confirm `.venv` is activated and dependencies installed.
- **`tomllib` missing** → ensure Python ≥ 3.11 (project targets 3.12).
- **Large files** → PyArrow is used for Parquet; install system dependencies if required.

Chart generation typically finishes in under a minute; time series plots dominate runtime because of smoothing and aggregation.
