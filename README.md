# HU Florido V2

WhatsApp chat analysis project generating 5 key visualizations from message data.

## 🎯 Overview

This project analyzes WhatsApp chat data to reveal communication patterns across couples and genders. The analysis is fully automated and configuration-driven using Pydantic models.

## 📊 Generated Figures

1. **`distribution.png`** - Response time distribution comparing men vs women
   - Shows men reply quicker on average
   - Smoothed probability distributions with median indicators

2. **`relationship.png`** - Emoji usage vs response rate analysis  
   - Proves friends respond more if you use emojis
   - Smooth interpolation with actual data points

3. **`tSNE.png`** - Text clustering visualization by couple
   - Demonstrates each couple has their own language
   - t-SNE dimensionality reduction with KDE contours

4. **`time_series.png`** - Hourly messaging patterns
   - Who texts when throughout the day
   - Smooth time series with gender-based area fills

5. **`comparing_categories.png`** - Quarterly dominance analysis
   - One voice per couple always dominates WhatsApp messages over time
   - Horizontal stacked bars showing message distribution

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
# Generate all 5 figures
python src/generate_5_key_figures.py
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
└── generate_5_key_figures.py   # Main script with analyzer classes
```

**Analyzer Classes** (follow Single Responsibility Principle):
- `ResponseTimeAnalyzer` - Response time calculations
- `EmojiResponseAnalyzer` - Emoji usage analysis
- `TSNEAnalyzer` - Text clustering with t-SNE
- `HourlyPatternAnalyzer` - Time-based messaging patterns
- `QuarterlyDominanceAnalyzer` - Quarterly message statistics
- `FigureStyle` - Consistent styling helper methods

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
