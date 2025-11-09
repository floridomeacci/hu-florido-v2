# HU Florido V2

WhatsApp chat analysis project generating five key visualizations from message data.

## 🎯 Overview

This project analyzes WhatsApp chat data to reveal communication patterns across couples and genders. The analysis is fully automated and configuration-driven using Pydantic models.

## 📊 Generated Figures

1. **`distribution.png`** – Reply speed histogram (minute resolution) split by gender. Stacked bars show raw reply counts, medians are reported in the caption, and the title underlines that women respond quicker on average.

2. **`relationship.png`** – Emoji usage versus response rate. A rigid curve with a solid fill tracks the actual bucketed response rates (zoomed to the 10–30% window) and `n=` counts sit directly beneath each bucket.

3. **`tSNE.png`** – Text clustering per couple. Each couple keeps a single base color for both points and density clouds, with caption lines summarizing within-couple and cross-couple t-SNE distances.

4. **`time_series.png`** – Hourly messaging patterns that reveal who texts when throughout the day via gender-based area fills.

5. **`comparing_categories.png`** – Quarterly dominance view showing which voice leads each period using stacked horizontal bars.

## 🚀 Quick Start

### Installation

```bash
# Clone and navigate to repository
cd hu-florido-v2

# Install dependencies (requires Python 3.12+)
pip install -e .
```

### Configuration

1. Place your WhatsApp chat export in `data/raw/_chat.txt`
2. Configure couple/gender mapping in `data/processed/couples_gender_mapping.json` (auto-created on first run)
3. Adjust settings in `config.toml` if needed

### Run Analysis

```bash
# Generate preprocessing + all five figures
python src/main.py
```

Output images will be saved to `img/` directory.

## 🏗️ Architecture

### Configuration-Driven Design

All settings are centralized in **Pydantic models** (`src/config.py`):
- `DataColumnConfig` - Column name mappings
- `AnalysisConfig` - Analysis parameters (couples, gender mapping, time windows)
- `DistributionPlotConfig` - Distribution plot settings
- `RelationshipPlotConfig` - Emoji analysis settings
- `TSNEPlotConfig` - t-SNE visualization parameters
- `FigureStyleConfig` - Colors, fonts, backgrounds, grid settings
- `FigureOutputConfig` - Output format and filenames
- `MasterConfig` - Combines all configs

### Code Structure

```
src/
├── config.py                    # Pydantic configuration models
└── main.py                     # Main script with analyzers and figure generation
```

**Analyzer Classes** (follow Single Responsibility Principle):
- `ResponseTimeAnalyzer` - Response time calculations
- `EmojiResponseAnalyzer` - Emoji usage analysis
- `TSNEAnalyzer` - Text clustering with t-SNE
- `HourlyPatternAnalyzer` - Time-based messaging patterns
- `QuarterlyDominanceAnalyzer` - Quarterly message statistics
- `FigureStyle` - Consistent styling helper methods (titles, legends, captions)

### Data Pipeline

1. **Preprocessing** - Parse raw chat → clean/anonymize → add gender/couple metadata
2. **Analysis** - Each analyzer class processes data independently
3. **Visualization** - FigureStyle applies consistent styling
4. **Output** - Save to configured locations

## 🔒 Privacy

Sensitive files are excluded from git:
- `data/raw/_chat.txt` - Your private chat data
- `data/processed/couples_gender_mapping.json` - Real name mappings
- All `.csv` and `.parq` files

Configure `.gitignore` as needed for your privacy requirements.

## 📦 Dependencies

Core requirements:
- Python 3.12+
- pandas, numpy, matplotlib, scipy
- scikit-learn (for t-SNE)
- pydantic (for configuration)
- emoji (for emoji detection)

See `pyproject.toml` for complete dependency list.

## 🛠️ Customization

### Change Colors or Styling

Edit `src/config.py` → `FigureStyleConfig`:
```python
gender_colors: Dict[str, str] = {
    'Men': '#1E3A8A',      # Blue
    'Women': '#FF69B4'     # Pink
}
```

### Adjust Analysis Parameters

Edit `src/config.py` → `AnalysisConfig`:
```python
max_response_hours: float = 6.0  # Max time for valid response
couples_list: List[str] = ["couple1", "couple2", "couple3"]
```

### Modify Plot Settings

Each plot type has its own config section in `config.py` with parameters for bins, smoothing, colors, point sizes, etc.

## 📝 Notes

- First run auto-creates `couples_gender_mapping.json` with default values
- Edit the mapping file to match your actual chat participants
- Preprocessing runs automatically before figure generation
- All figures use consistent styling from `FigureStyleConfig`
