hu-florido-v2

Overview

This project processes WhatsApp chat exports and generates visualizations to explore messaging dynamics between three couples. It includes:
- Data preprocessing that anonymizes authors, adds gender, and maps people to couples
- A semicircular gauge chart per couple showing who dominates messaging
- A three-column timeline (per couple) with 100% stacked distributions by gender per quarter
- A stacked timeline of total message counts with a reference threshold

Project structure

```
.
├── config.toml                  # Paths to input and processed data
├── data/
│   ├── processed/
│   │   ├── couples_reference.json  # Real or pseudonym → couple mapping
│   │   ├── output.csv              # Preprocessed messages (generated)
│   │   └── output.parq             # Parquet version (generated)
│   └── raw/
├── img/
│   ├── gauge_charts.png            # Generated gauge charts
│   ├── timeline_couples_bars.png   # Generated stacked timeline
│   └── timeline_couples_columns.png# Generated columns timeline
├── src/
│   ├── gaugeCouples.py             # Gauge charts per couple
│   ├── myPreprocess.py             # Preprocessing pipeline
│   ├── timelineCouples.py          # Stacked timeline (counts)
│   └── timelineCouplesColumns.py   # 3-column 100% distributions by quarter
├── pyproject.toml                  # Python project config (>=3.12)
└── README.md
```

Requirements

- macOS (tested) with Python 3.12+
- Dependencies (managed via pyproject):
	- wa-analyzer
	- pyarrow
- Optional (recommended): uv (fast Python package/deps manager)

Setup

Option A: Using uv (recommended)

```sh
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment in .venv
uv venv .venv
source .venv/bin/activate

# Install dependencies from pyproject
uv sync
```

Option B: Using venv + pip

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install wa-analyzer pyarrow
```

Configuration

Edit `config.toml` to point to your WhatsApp export and processed paths. Example:

```toml
processed = "data/processed"
current = "output.parq"
inputpath = "data/raw/whatsapp.txt"
```

Data preprocessing

1) Prepare couple mapping in `data/processed/couples_reference.json` with either real names or pseudonyms as keys, mapping to couple IDs (couple1/couple2/couple3):

```json
{
	"Florido Meacci": "couple1",
	"Ramunė Meacci": "couple1",
	"Fei": "couple2",
	"Shuyan": "couple2",
	"Alberto Doncato": "couple3",
	"Ludovica": "couple3"
}
```

2) Run preprocessing to generate `data/processed/output.csv` and `output.parq`:

```sh
python src/myPreprocess.py
```

Visualizations

1) Gauge charts (one per couple)

```sh
python src/gaugeCouples.py
# Output: img/gauge_charts.png
```

2) Timeline: 100% distributions per couple (3 columns)

```sh
python src/timelineCouplesColumns.py
# Output: img/timeline_couples_columns.png
```

3) Timeline: Stacked counts per quarter (all couples combined)

```sh
python src/timelineCouples.py
# Output: img/timeline_couples_bars.png
```

Notes and conventions

- Gender is inferred via a small mapping in preprocessing; unknowns default to "unknown".
- The gauge charts color-code by gender per couple palette: dark (male), light (female).
- The columns timeline shows male vs female share per couple per quarter, with y-axis inverted so later quarters appear at the top.
- The stacked timeline sorts individuals by activity per quarter and includes a vertical reference line.

Troubleshooting

- Module import errors: Ensure your virtual environment is active and dependencies are installed.
- Missing output.csv: Run the preprocessing step first.
- tomllib errors: The project requires Python 3.12+, which includes tomllib. Verify your Python version with `python --version`.
