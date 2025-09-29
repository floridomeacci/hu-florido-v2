Florido Jan Meacci

Overview

This project processes WhatsApp chat exports and generates comprehensive visualizations to explore messaging dynamics between three couples. It includes:
- Data preprocessing that anonymizes authors, adds gender, and maps people to couples
- Seven distinct visualization types analyzing different aspects of communication patterns:
  - Semicircular gauge charts showing messaging dominance per couple
  - Character count distribution analysis (Men vs Women)
  - Hour-of-day distribution analysis (Men vs Women) 
  - Three-column timeline with 100% stacked distributions by gender per quarter
  - Time series analysis of child/family-related word usage
  - Comparative time series for couples with vs without children
  - Combined time series showing overall family word trends

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
├── img/                         # Generated visualizations
│   ├── comparing_categories_1.png   # Semicircular gauge charts
│   ├── comparing_categories_2.png   # Three-column timeline
│   ├── distribution_categories.png  # Character count distribution
│   ├── distribution_categories_2.png# Hour-of-day distribution
│   ├── time_series.png             # Men vs Women child word trends
│   ├── time_series_2.png           # With/Without children comparison
│   └── time_series_3.png           # Combined family word trends
├── src/
│   ├── myFigures.py                # Unified plotting system (all charts)
│   └── myPreprocess.py             # Preprocessing pipeline
├── pyproject.toml                  # Python project config (>=3.12)
└── README.md
```

Requirements

- macOS (tested) with Python 3.12+
- Dependencies (managed via pyproject):
	- wa-analyzer
	- pyarrow
	- matplotlib
	- pandas
	- numpy
	- scipy (for smooth interpolation)
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
pip install wa-analyzer pyarrow matplotlib pandas numpy scipy
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


2) Run preprocessing to generate `data/processed/output.csv` and `output.parq`:

```sh
python src/myPreprocess.py
```

Visualizations

Generate all seven visualization types with a single command:

```sh
python src/myFigures.py
```

This creates:

1) **Semicircular gauge charts** (`img/comparing_categories_1.png`)
   - One gauge per couple showing messaging dominance
   - Pink for women, dark blue for men

2) **Character count distribution** (`img/distribution_categories.png`)
   - Horizontal diverging bars showing message length patterns
   - Men (left, blue) vs Women (right, pink)
   - Individual character count bins from 0-149+

3) **Hour-of-day distribution** (`img/distribution_categories_2.png`)
   - Similar diverging bar format but showing messaging by hour (0-23h)
   - Reveals daily communication patterns by gender

4) **Three-column timeline** (`img/comparing_categories_2.png`)
   - 100% stacked distributions per couple by quarter
   - Shows relative gender balance over time

5) **Time series: Men vs Women child words** (`img/time_series.png`)
   - Tracks mentions of child/family-related words over time
   - Smooth interpolated lines with reference dates for baby births/conceptions

6) **Time series: With/Without children** (`img/time_series_2.png`)
   - Compares child word usage between couples with and without children
   - Averaged data showing different communication patterns

7) **Combined time series** (`img/time_series_3.png`)
   - Overall trend of family-related word usage across all couples
   - Purple line showing combined patterns

Notes and conventions

**Data Processing:**
- Gender is inferred via mapping in preprocessing; unknowns default to "unknown"
- Media messages (`<Media omitted>`) are filtered out from character count analysis
- Time series analysis uses quarter-based grouping for temporal patterns

**Visual Design:**
- Consistent color scheme: Pink (#FF69B4) for women/females, Dark blue (#1E3A8A) for men/males
- All charts use "Men vs Women" labeling for consistency
- Distribution charts use diverging horizontal bars with percentage-based widths
- Time series charts include smooth interpolation and reference lines for key family events

**Child/Family Word Analysis:**
- Tracks 22+ family-related terms: baby, child, kids, pregnant, birth, parenting, etc.
- Includes baby-related vocabulary: diaper, stroller, crib, playground, bedtime
- Reference lines mark important dates (baby births and conceptions) when available

**Chart Specifications:**
- Character distribution: Individual bins 0-149+ characters, 1.5% x-axis range
- Hour distribution: 24-hour bins (0-23h), 8% x-axis range for better visibility
- Time series: Quarter-based with smooth spline interpolation

Troubleshooting

**Common Issues:**
- **Module import errors**: Ensure your virtual environment is active and dependencies are installed
- **Missing output.csv**: Run the preprocessing step first with `python src/myPreprocess.py`
- **tomllib errors**: The project requires Python 3.12+, which includes tomllib. Verify with `python --version`
- **scipy import errors**: Install scipy with `pip install scipy` or `uv add scipy`

**Performance Notes:**
- Chart generation typically takes 10-30 seconds depending on data size
- Time series charts with smooth interpolation are most computation-intensive
- Large datasets may require increased memory for character/hour distribution analysis

**Data Quality:**
- Ensure couples_reference.json contains all message authors
- Check that WhatsApp export format matches expected structure
- Verify date formats are properly parsed (preprocessing will show warnings)
