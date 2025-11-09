#!/usr/bin/env python3
"""
Standalone script to run preprocessing and generate 5 key figures.
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Optional, Dict, List, Any
from itertools import combinations
import tomllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import emoji

# Import configuration models
from config import (
    MasterConfig,
    DEFAULT_CONFIG,
    FigureStyleConfig
)


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "config.toml"
MAPPING_FILENAME = "couples_gender_mapping.json"

# Legacy constants - kept for backward compatibility
COUPLES = DEFAULT_CONFIG.analysis.couples_list
TIME_COL = DEFAULT_CONFIG.columns.timestamp

# ============================================================================
# FIGURE STYLING CLASS
# ============================================================================

class FigureStyle:
    """Helper methods for consistent figure styling using configuration."""
    
    def __init__(self, config: FigureStyleConfig = None):
        """Initialize with optional custom config, defaults to DEFAULT_CONFIG.style."""
        if config is None:
            config = DEFAULT_CONFIG.style
        self.config = config
    
    def setup_figure(self, figsize=(16, 8)):
        """Create and configure a figure with consistent styling."""
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(self.config.figure_bg)
        ax.set_facecolor(self.config.axis_bg)
        return fig, ax
    
    def add_title(self, fig, title: str, subtitle: str = None, fontsize: int = None, y_pos: float = 0.95):
        """
        Add a consistent left-aligned title to a figure.
        
        Args:
            fig: matplotlib figure object
            title: Main title text (can include \n for line breaks)
            subtitle: Optional subtitle text (smaller, gray)
            fontsize: Font size for main title (default: from config)
            y_pos: Vertical position of title (default: 0.95)
        """
        if fontsize is None:
            fontsize = self.config.title_fontsize
        
        # Main title
        fig.text(
            0.02 if subtitle else 0.06,
            y_pos,
            title,
            fontsize=fontsize,
            fontweight='bold',
            ha='left',
            va='top',
            transform=fig.transFigure,
        )
        
        # Optional subtitle
        if subtitle:
            subtitle_y = y_pos - 0.055
            fig.text(
                0.02,
                subtitle_y,
                subtitle,
                fontsize=self.config.subtitle_fontsize,
                color=getattr(self.config, 'subtitle_color', '#6B7280'),
                ha='left',
                va='top',
                transform=fig.transFigure,
            )

    def add_caption(self, fig, lines: List[str], y_start: float = 0.035):
        """Add centered italic caption lines beneath the axes."""
        if not lines:
            return

        spacing = getattr(self.config, 'caption_line_spacing', 0.024)
        fontsize = getattr(self.config, 'caption_fontsize', self.config.subtitle_fontsize)
        default_color = getattr(self.config, 'caption_color', '#6B7280')

        for idx, text in enumerate(lines):
            if not text:
                continue
            fig.text(
                0.5,
                y_start - idx * spacing,
                text,
                ha='center',
                va='bottom',
                fontsize=fontsize,
                color=default_color,
                style='italic'
            )
    
    def set_axis_labels(self, ax, xlabel: str = None, ylabel: str = None, fontsize: int = None):
        """
        Set consistent axis labels with bold formatting.
        
        Args:
            ax: matplotlib axis object
            xlabel: X-axis label text
            ylabel: Y-axis label text
            fontsize: Font size for labels (default: from config)
        """
        if fontsize is None:
            fontsize = self.config.label_fontsize
        
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold')
    
    def style_axis(self, ax, grid: bool = True):
        """
        Apply consistent styling to axis: remove spines, add grid, set background.
        
        Args:
            ax: matplotlib axis object
            grid: Whether to show grid (default: True)
        """
        # Remove all spines (the black border lines)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add subtle grid
        if grid:
            ax.grid(True, alpha=self.config.grid_alpha, 
                   linestyle=self.config.grid_linestyle, 
                   linewidth=self.config.grid_linewidth)
            ax.set_axisbelow(True)
    
    def add_legend(self, ax, legend_type: str = 'gender', location: str = 'upper right', **kwargs):
        """
        Add a consistent legend with color boxes.
        
        Args:
            ax: matplotlib axis object
            legend_type: 'gender' (Men/Women) or 'couples' (Couple 1/2/3)
            location: Legend location (default: 'upper right')
            **kwargs: Additional arguments to pass to ax.legend()
        """
        from matplotlib.patches import Rectangle
        
        if legend_type == 'gender':
            legend_elements = [
                Rectangle((0, 0), 1, 1, facecolor=self.config.gender_colors['Men'], label='Men'),
                Rectangle((0, 0), 1, 1, facecolor=self.config.gender_colors['Women'], label='Women')
            ]
        elif legend_type == 'couples':
            legend_elements = [
                Rectangle((0, 0), 1, 1, facecolor=self.config.couple_colors['couple1'], label='Couple 1'),
                Rectangle((0, 0), 1, 1, facecolor=self.config.couple_colors['couple2'], label='Couple 2'),
                Rectangle((0, 0), 1, 1, facecolor=self.config.couple_colors['couple3'], label='Couple 3')
            ]
        else:
            raise ValueError(f"Unknown legend_type: {legend_type}")
        
        # Default styling
        default_kwargs = {
            'loc': location,
            'fontsize': self.config.legend_fontsize,
            'frameon': True,
            'fancybox': True,
            'shadow': True
        }
        default_kwargs.update(kwargs)
        
        legend = ax.legend(handles=legend_elements, **default_kwargs)
        
        # Consistent legend styling
        if legend:
            legend.get_frame().set_facecolor(self.config.legend_bg)
            legend.get_frame().set_alpha(self.config.legend_alpha)
        
        return legend


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def _norm_name(value: Optional[str]) -> str:
    """Normalize name for lookup."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def _load_couples_gender_mapping(processed_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load couples/gender mapping from JSON file."""
    mapping_path = processed_dir / MAPPING_FILENAME
    if not mapping_path.exists() or mapping_path.stat().st_size == 0:
        default_mapping = {
            "Example Person": {
                "gender": "unknown",
                "couple": "single",
                "aliases": ["example-alias"]
            },
            "Unknown": {
                "gender": "unknown",
                "couple": "single",
                "aliases": ["unknown"]
            }
        }
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("w", encoding="utf-8") as mp:
            json.dump(default_mapping, mp, indent=4)
        print(
            f"⚠️  Created default mapping template at {mapping_path}. "
            "Please update it with your couples and aliases before rerunning."
        )

    try:
        with mapping_path.open("r", encoding="utf-8") as mp:
            raw_mapping = json.load(mp)
    except json.JSONDecodeError as exc:
        backup_path = mapping_path.with_suffix(".invalid.json")
        mapping_path.replace(backup_path)
        mapping_path.write_text("{}", encoding="utf-8")
        raise ValueError(
            f"Invalid JSON in {backup_path}. A blank template was created; "
            "please fix the file and rerun."
        ) from exc

    if not isinstance(raw_mapping, dict):
        raise ValueError("couples_gender_mapping.json must contain an object at the top level")

    real_lookup: Dict[str, Dict[str, Any]] = {}
    alias_lookup: Dict[str, Dict[str, Any]] = {}

    for name, info in raw_mapping.items():
        if not isinstance(info, dict):
            continue

        normalized_info = {
            "gender": str(info.get("gender", "unknown") or "unknown"),
            "couple": str(info.get("couple", "single") or "single"),
        }

        real_lookup[_norm_name(name)] = normalized_info

        aliases_field = info.get("aliases", [])
        if isinstance(aliases_field, str):
            aliases_field = [aliases_field]
        elif not isinstance(aliases_field, list):
            aliases_field = []

        for alias in aliases_field:
            if alias:
                alias_lookup[_norm_name(alias)] = normalized_info

    return {"real": real_lookup, "alias": alias_lookup}


def _lookup_mapping_field(name: Optional[str], field: str, mapping: Dict[str, Dict[str, Dict[str, Any]]]) -> Optional[str]:
    """Look up gender or couple from mapping."""
    if not name:
        return None
    key = _norm_name(name)
    info = mapping["real"].get(key) or mapping["alias"].get(key)
    if info:
        value = info.get(field)
        if isinstance(value, str):
            return value
    return None


def run_preprocessing() -> Path:
    """
    Run the full preprocessing pipeline.
    Returns the path to the processed output CSV.
    """
    print("\n" + "=" * 70)
    print("RUNNING PREPROCESSING")
    print("=" * 70)
    
    # Load config
    with CONFIG_FILE.open("rb") as f:
        config = tomllib.load(f)

    # Build path to raw input file
    raw_dir = Path(config.get("raw", "data/raw")).resolve()
    input_file = config.get("input", "_chat.txt")
    datafile = raw_dir / input_file
    
    processed_dir = Path(config.get("processed", "data/processed")).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if not datafile.exists():
        print(f"⚠️  Raw input file not found: {datafile}")
        print(f"   Looking for existing processed file instead...")
        return processed_dir / "output.csv"

    # Run wa_analyzer preprocessing
    try:
        import wa_analyzer.preprocess as preprocessor
        preprocessor.main(["--device", "old"], standalone_mode=False)
    except Exception as e:
        print(f"Preprocessor failed: {e}")
        return processed_dir / "output.csv"

    output_csv = processed_dir / "output.csv"
    if not output_csv.exists():
        # Backward compatibility: fall back to timestamped file if present
        try:
            latest_csv = max(processed_dir.glob("whatsapp-*.csv"), key=lambda p: p.stat().st_mtime)
            output_csv = latest_csv
        except ValueError:
            return processed_dir / "output.csv"

    # Load data
    out_df = pd.read_csv(output_csv)
    
    # Preserve original author
    if "author" in out_df.columns and "author_orig" not in out_df.columns:
        out_df["author_orig"] = out_df["author"].astype(str)

    # Load mapping
    mapping_data = _load_couples_gender_mapping(processed_dir)

    # Add gender and couples columns
    if "author_orig" in out_df.columns:
        gender_series = out_df["author_orig"].apply(lambda name: _lookup_mapping_field(name, "gender", mapping_data))
        couples_series = out_df["author_orig"].apply(lambda name: _lookup_mapping_field(name, "couple", mapping_data))
    else:
        gender_series = pd.Series([None] * len(out_df), index=out_df.index, dtype=object)
        couples_series = pd.Series([None] * len(out_df), index=out_df.index, dtype=object)

    out_df["gender"] = gender_series.fillna("unknown").replace("", "unknown")
    out_df["couples"] = couples_series.fillna("single").replace("", "single")

    # Clean message text
    timestamp_pattern = r"^\s*\d{1,2}/\d{1,2}/\d{2},\s*\d{1,2}:\d{2}\s*-\s*"
    author_pattern = r"^\s*\d{1,2}/\d{1,2}/\d{2},\s*\d{1,2}:\d{2}\s*-\s*[^:]+:\s*"
    
    out_df["message"] = (
        out_df.get("message", pd.Series(dtype=str)).astype(str)
        .str.replace(author_pattern, "", regex=True)
        .str.replace(timestamp_pattern, "", regex=True)
        .str.replace(r"\s*\n+\s*$", "", regex=True)
        .str.replace(r"^\s+|\s+$", "", regex=True)
        .str.strip()
    )

    # Anonymization
    if "author" in out_df.columns:
        authors_before = set(out_df["author"].dropna().unique().tolist())

        ref_path = processed_dir / "anon_reference.json"
        if ref_path.exists():
            try:
                with ref_path.open("r", encoding="utf-8") as rf:
                    ref_raw = json.load(rf)
                
                def _norm(s: str) -> str:
                    return re.sub(r"\s+", " ", str(s).strip()).lower()

                candB_real_to_pseudo = {_norm(k): v for k, v in ref_raw.items()}
                authors_norm = {_norm(a) for a in authors_before}
                replB = sum(1 for a in authors_norm if a in candB_real_to_pseudo)

                if replB > 0:
                    out_df["author"] = [candB_real_to_pseudo.get(_norm(a), a) for a in out_df["author"].astype(str)]
            except Exception as e:
                print(f"Anonymization failed: {e}")
        else:
            adjectives = [
                "agile","brisk","calm","dapper","eager","fiery","gentle","hazy","icy","jaunty",
                "keen","lively","mellow","nimble","odd","plucky","quirky","rosy","spry","tidy",
                "urbane","vivid","witty","young","zesty","bold","clever","daring","elegant","frosty"
            ]
            animals = [
                "ant","beagle","cat","dolphin","eagle","falcon","gecko","heron","ibis","jaguar",
                "koala","llama","manatee","narwhal","otter","panda","quail","raccoon","seal","tiger",
                "urchin","viper","walrus","yak","zebra","goshawk","trout","magpie","jackal","duck"
            ]

            def make_pseudo(name: str) -> str:
                if not name or name.lower() == "unknown":
                    return "unknown"
                if "-" in name and name.count(" ") == 0:
                    return name.lower()
                h = hashlib.sha256(name.encode("utf-8")).hexdigest()
                adj = adjectives[int(h[0:2], 16) % len(adjectives)]
                ani = animals[int(h[2:4], 16) % len(animals)]
                return f"{adj}-{ani}"

            out_df["author"] = out_df["author"].astype(str).apply(make_pseudo)

    # Emoji detection
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )

    def has_emoji(text):
        if not isinstance(text, str) or not text:
            return False
        return bool(emoji_pattern.search(text))

    out_df["has_emoji"] = out_df["message"].apply(has_emoji)

    # Write output
    csv_out = processed_dir / "output.csv"
    parquet_path = processed_dir / "output.parq"
    
    if "author_orig" in out_df.columns:
        out_df_clean = out_df.drop(columns=["author_orig"])
    else:
        out_df_clean = out_df
    
    out_df_clean.to_csv(csv_out, index=False)
    try:
        out_df_clean.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    # Remove intermediate files
    removed = 0
    for fp in processed_dir.glob("whatsapp-*.csv"):
        try:
            fp.unlink()
            removed += 1
        except OSError:
            pass

    print(f"✅ Cleaned & anonymized messages written to {csv_out}")
    print("=" * 70 + "\n")
    
    return csv_out


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================

def extract_emojis(text):
    """Extract all emojis from text."""
    if not isinstance(text, str):
        return []
    return [char for char in text if char in emoji.EMOJI_DATA]


class ResponseTimeAnalyzer:
    """Analyzes response times between messages."""
    
    def __init__(self, config: MasterConfig):
        self.config = config
        self.col = config.columns
        self.analysis = config.analysis
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and filter data for analysis."""
        data = df[df[self.col.couples].isin(self.analysis.couples_list)].copy()
        data = data[data[self.col.message] != self.analysis.omitted_message_text]
        data[self.col.timestamp] = pd.to_datetime(data[self.col.timestamp])
        data[self.col.gender_mapped] = data[self.col.gender].map(self.analysis.gender_map)
        data = data.dropna(subset=[self.col.gender_mapped])
        data = data.sort_values([self.col.timestamp]).reset_index(drop=True)
        return data
    
    def calculate_response_times(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate response times for each gender."""
        response_times = {'Men': [], 'Women': []}
        
        for couple in self.analysis.couples_list:
            couple_msgs = data[data[self.col.couples] == couple].reset_index(drop=True)
            
            for i in range(1, len(couple_msgs)):
                curr_msg = couple_msgs.iloc[i]
                prev_msg = couple_msgs.iloc[i-1]
                
                if curr_msg[self.col.author] != prev_msg[self.col.author]:
                    time_diff_minutes = (
                        curr_msg[self.col.timestamp] - prev_msg[self.col.timestamp]
                    ).total_seconds() / 60
                    
                    if 0 < time_diff_minutes < self.analysis.max_response_minutes:
                        gender = curr_msg[self.col.gender_mapped]
                        response_times[gender].append(time_diff_minutes)
        
        return response_times


def distribution(df: pd.DataFrame, output_dir: Path, config: MasterConfig = None) -> Path:
    """Distribution of response times: Men reply quicker on average."""
    
    if config is None:
        config = DEFAULT_CONFIG
    
    # Analyze data
    analyzer = ResponseTimeAnalyzer(config)
    prepared_data = analyzer.prepare_data(df)
    response_times = analyzer.calculate_response_times(prepared_data)
    
    medians = {group: np.median(times) if times else 0 for group, times in response_times.items()}
    
    # Create styling helper
    style = FigureStyle(config.style)
    
    # Create figure
    fig, ax = style.setup_figure(figsize=(16, 8))
    
    # Histogram setup: bucket response times per minute within the 6 hour window
    max_minutes = int(config.distribution.time_max_hours * 60)
    minute_range = np.arange(max_minutes)
    men_counts = np.zeros(max_minutes, dtype=float)
    women_counts = np.zeros(max_minutes, dtype=float)

    def add_counts(times: List[float], target_array: np.ndarray):
        for t in times:
            if t < 0:
                continue
            minute_index = int(min(max_minutes - 1, max(0, np.floor(t))))
            target_array[minute_index] += 1

    add_counts(response_times.get('Men', []), men_counts)
    add_counts(response_times.get('Women', []), women_counts)

    men_total = len(response_times.get('Men', []))
    women_total = len(response_times.get('Women', []))

    x_positions = minute_range / 60.0
    bar_width = 1.0 / 60.0  # one minute in hours for x-axis scaling

    ax.bar(x_positions, men_counts, width=bar_width, color=config.style.gender_colors['Men'], label='Men')
    ax.bar(x_positions, women_counts, width=bar_width,
           bottom=men_counts, color=config.style.gender_colors['Women'], label='Women')
    
    # Labels and formatting
    style.set_axis_labels(ax, xlabel='Time', ylabel='Responses (count)')
    ax.set_xlim(0, config.distribution.time_max_hours)
    ax.set_xticks(range(int(config.distribution.time_max_hours) + 1))
    ax.set_xticklabels(config.distribution.time_labels)
    ax.set_ylim(bottom=0)
    
    style.add_title(fig, 'Women reply quicker on average')
    style.add_legend(ax, legend_type='gender')
    style.style_axis(ax)

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    subtitle_text = (
        f"Replies — Men: {men_total}, Women: {women_total} | "
        f"Median response (min) — Men: {medians.get('Men', 0):.1f}, Women: {medians.get('Women', 0):.1f}"
    )
    style.add_caption(fig, [subtitle_text])
    
    output_path = output_dir / config.output.distribution_filename
    plt.savefig(output_path, dpi=config.output.dpi, bbox_inches=config.output.bbox_inches,
                facecolor=config.output.facecolor, edgecolor=config.output.edgecolor)
    plt.close(fig)
    
    print(f"    ✅ Saved: {output_path}")
    return output_path


class EmojiResponseAnalyzer:
    """Analyzes relationship between emoji usage and response rates."""
    
    def __init__(self, config: MasterConfig):
        self.config = config
        self.col = config.columns
        self.analysis = config.analysis
        self.rel_config = config.relationship
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with emoji extraction."""
        data = df[df[self.col.couples].isin(self.analysis.couples_list)].copy()
        data = data[data[self.col.message] != self.analysis.omitted_message_text]
        data[self.col.timestamp] = pd.to_datetime(data[self.col.timestamp])
        
        # Extract emojis
        data[self.col.emojis] = data[self.col.message].apply(extract_emojis)
        data[self.col.emoji_count] = data[self.col.emojis].apply(len)
        
        # Map gender
        data[self.col.gender_mapped] = data[self.col.gender].map(self.analysis.gender_map)
        data = data.dropna(subset=[self.col.gender_mapped])
        
        data = data.sort_values([self.col.timestamp]).reset_index(drop=True)
        return data
    
    def calculate_response_rates(self, data: pd.DataFrame) -> List[Dict]:
        """Calculate response rates for each emoji count category."""
        emoji_categories = self.rel_config.emoji_categories + ['5+']
        response_data = []
        
        for emoji_cat in emoji_categories:
            stats = {'Men': {'total': 0, 'responded': 0}, 'Women': {'total': 0, 'responded': 0}}
            
            for couple in self.analysis.couples_list:
                couple_msgs = data[data[self.col.couples] == couple].reset_index(drop=True)
                
                for i in range(len(couple_msgs) - 1):
                    curr_msg = couple_msgs.iloc[i]
                    next_msg = couple_msgs.iloc[i + 1]
                    
                    # Check emoji category match
                    if emoji_cat == '5+':
                        matches = curr_msg[self.col.emoji_count] >= self.rel_config.emoji_group_threshold
                    else:
                        matches = curr_msg[self.col.emoji_count] == emoji_cat
                    
                    if matches:
                        gender = curr_msg[self.col.gender_mapped]
                        stats[gender]['total'] += 1
                        
                        # Check if there was a response
                        if next_msg[self.col.author] != curr_msg[self.col.author]:
                            time_diff = (next_msg[self.col.timestamp] - curr_msg[self.col.timestamp]).total_seconds() / 60
                            if 0 < time_diff <= self.analysis.max_response_minutes:
                                stats[gender]['responded'] += 1
            
            # Calculate rates
            result = {'emoji_count': emoji_cat}
            for gender in ['Men', 'Women']:
                total = stats[gender]['total']
                responded = stats[gender]['responded']
                rate = (responded / total * 100) if total > 0 else 0
                
                result[f'{gender.lower()}_rate'] = rate
                result[f'{gender.lower()}_total'] = total
                result[f'{gender.lower()}_responded'] = responded
            
            response_data.append(result)
        
        return response_data


def relationship(df: pd.DataFrame, output_dir: Path, config: MasterConfig = None) -> Path:
    """Prove that heavy emoji use reduces response rates."""
    
    if config is None:
        config = DEFAULT_CONFIG
    
    # Analyze data
    analyzer = EmojiResponseAnalyzer(config)
    prepared_data = analyzer.prepare_data(df)
    response_data = analyzer.calculate_response_rates(prepared_data)
    
    # Create styling helper
    style = FigureStyle(config.style)
    
    # Create visualization
    fig, ax = style.setup_figure(figsize=(16, 10))
    
    x_positions = list(range(len(response_data)))
    x_labels = [str(item['emoji_count']) for item in response_data]
    
    # Calculate combined response rates
    combined_rates = []
    sample_sizes = []
    for item in response_data:
        total_msgs = item['men_total'] + item['women_total']
        total_responded = item['men_responded'] + item['women_responded']
        combined_rate = (total_responded / total_msgs * 100) if total_msgs > 0 else 0
        combined_rates.append(combined_rate)
        sample_sizes.append(total_msgs)
    
    y_axis_min, y_axis_max = 10, 30

    # Plot rigid line through actual points (no smoothing)
    x_dense = np.linspace(x_positions[0], x_positions[-1], num=400)
    y_dense = np.interp(x_dense, x_positions, combined_rates)

    fill_color = config.style.gender_colors['Men']
    point_color = fill_color

    ax.fill_between(
        x_dense,
        y_dense,
        0,
        alpha=1.0,
        color=fill_color,
        zorder=2
    )
    ax.plot(
        x_positions,
        combined_rates,
        linewidth=3,
        color=fill_color,
        alpha=1.0,
        zorder=3
    )
    
    # Add actual data points in blue
    ax.scatter(
        x_positions,
        combined_rates,
        s=config.relationship.point_size,
        color=point_color,
        edgecolors='white',
        linewidths=config.relationship.point_edge_width,
        zorder=5
    )
    
    # Add percentage labels on data points
    for i, (pos, rate, size) in enumerate(zip(x_positions, combined_rates, sample_sizes)):
        label_y = min(max(rate + 1.2, y_axis_min + 1), y_axis_max - 1.5)
        ax.text(
            pos,
            label_y,
            f'{rate:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold',
            color=point_color
        )
        ax.text(
            pos,
            y_axis_min - 0.8,
            f'n={size}',
            ha='center',
            va='top',
            fontsize=9,
            color='#666666',
            style='italic'
        )
    
    # Labels and formatting
    style.set_axis_labels(ax, xlabel='Number of Emojis in Message', ylabel='Response Rate (%)', fontsize=16)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=13)
    ax.set_ylim(y_axis_min, y_axis_max)
    
    style.style_axis(ax)
    style.add_title(fig, "Friends respond more\nif you use emoji's", y_pos=0.93)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.82])
    
    output_path = output_dir / config.output.relationship_filename
    plt.savefig(output_path, dpi=config.output.dpi, bbox_inches=config.output.bbox_inches,
                facecolor=config.output.facecolor, edgecolor=config.output.edgecolor)
    plt.close(fig)
    
    print(f"    ✅ Saved: {output_path}")
    return output_path


def _author_trigram_documents_limited(
    df: pd.DataFrame,
    config: MasterConfig,
    *,
    max_chunks_per_author: int = None,
    min_tokens: int = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate trigram documents with dynamic chunk sizing to limit chunks per author."""
    if max_chunks_per_author is None:
        max_chunks_per_author = config.tsne.max_chunks_per_author
    if min_tokens is None:
        min_tokens = config.tsne.min_tokens
    
    col = config.columns
    
    filtered = df.copy()
    filtered = filtered[filtered[col.author].notna()]
    filtered = filtered[filtered[col.message].notna()]
    filtered = filtered[filtered[col.message] != config.analysis.omitted_message_text]
    if col.couples in filtered.columns:
        filtered = filtered[filtered[col.couples].isin(config.analysis.couples_list)]

    if filtered.empty:
        raise ValueError("No author messages available for trigram analysis.")

    filtered[col.timestamp] = pd.to_datetime(filtered[col.timestamp])
    filtered.sort_values([col.author, col.timestamp], inplace=True)

    meta_rows: list[dict] = []
    documents: list[str] = []

    for author, group in filtered.groupby(col.author):
        group = group.reset_index(drop=True)
        if group.empty:
            continue

        # Calculate total characters for this author
        total_chars = group[col.message].astype(str).map(len).sum()
        
        # Calculate chunk size to get approximately max_chunks_per_author chunks
        chunk_size = max(int(total_chars / max_chunks_per_author), config.tsne.min_chunk_chars)
        
        # Chunk by CHARACTER count with dynamic chunk size
        message_lengths = group[col.message].astype(str).map(len)
        chunk_ids: list[int] = []
        current_chunk = 0
        current_total = 0
        
        for length in message_lengths:
            chunk_ids.append(current_chunk)
            current_total += int(length)
            while current_total >= chunk_size:
                current_total -= chunk_size
                current_chunk += 1
        
        group = group.assign(chunk_id=np.array(chunk_ids, dtype=int), message_length=message_lengths)

        for chunk_id, chunk_df in group.groupby('chunk_id'):
            text = " ".join(chunk_df[col.message].astype(str))
            token_count = len(text.split())
            if token_count < min_tokens:
                continue

            couple_value = chunk_df[col.couples].iloc[0] if col.couples in chunk_df.columns else 'single'

            meta_rows.append({
                'author': str(author),
                'couples': str(couple_value) if pd.notna(couple_value) else 'single',
                'chunk_index': int(chunk_id),
                'messages_in_chunk': int(len(chunk_df)),
                'characters_in_chunk': int(chunk_df['message_length'].sum()),
                'first_timestamp': chunk_df[col.timestamp].min(),
                'last_timestamp': chunk_df[col.timestamp].max(),
                'token_count': int(token_count),
            })
            documents.append(text)

    if not documents:
        raise ValueError("Not enough textual data to build trigram documents.")

    metadata = pd.DataFrame(meta_rows)
    metadata.reset_index(drop=True, inplace=True)
    return metadata, documents


class TSNEAnalyzer:
    """Analyzes text patterns using t-SNE dimensionality reduction."""
    
    def __init__(self, config: MasterConfig):
        self.config = config
        self.col = config.columns
        self.tsne_config = config.tsne
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data for couples analysis."""
        couples_df = df.copy()
        if self.col.couples in couples_df.columns:
            couples_df = couples_df[couples_df[self.col.couples].isin(self.config.analysis.couples_list)]
        return couples_df
    
    def generate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate t-SNE embeddings from text data."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.manifold import TSNE
        
        # Generate documents
        meta_docs, documents = _author_trigram_documents_limited(df, self.config)
        
        if not documents:
            raise ValueError("No documents generated")
        
        # Vectorize using character n-grams
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=self.tsne_config.ngram_range,
            lowercase=True,
            min_df=1,
            max_features=self.tsne_config.max_features,
        )
        matrix = vectorizer.fit_transform(documents)
        
        if matrix.shape[0] < 2:
            raise ValueError("Not enough documents")
        
        matrix_dense = matrix.toarray()
        
        # Apply t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(self.tsne_config.perplexity, matrix_dense.shape[0] - 1),
            random_state=self.tsne_config.random_state,
            max_iter=self.tsne_config.max_iter,
            verbose=0,
        )
        coords = tsne.fit_transform(matrix_dense)
        
        meta_docs = meta_docs.assign(tsne_x=coords[:, 0], tsne_y=coords[:, 1])
        return meta_docs
    
    def merge_gender_info(self, meta_docs: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Merge gender information from original dataframe."""
        author_lookup = (
            original_df[[self.col.author, self.col.gender]]
            .dropna(subset=[self.col.author])
            .drop_duplicates(self.col.author)
            .astype({self.col.author: str})
        )
        return meta_docs.merge(author_lookup, on=self.col.author, how='left')
    
    def filter_outliers(self, meta_docs: pd.DataFrame) -> pd.DataFrame:
        """Filter outliers using IQR method."""
        q25_x, q75_x = meta_docs['tsne_x'].quantile([0.25, 0.75])
        q25_y, q75_y = meta_docs['tsne_y'].quantile([0.25, 0.75])
        iqr_x = q75_x - q25_x
        iqr_y = q75_y - q25_y
        
        lower_x = q25_x - self.tsne_config.iqr_multiplier * iqr_x
        upper_x = q75_x + self.tsne_config.iqr_multiplier * iqr_x
        lower_y = q25_y - self.tsne_config.iqr_multiplier * iqr_y
        upper_y = q75_y + self.tsne_config.iqr_multiplier * iqr_y
        
        filtered = meta_docs[
            (meta_docs['tsne_x'] >= lower_x) & (meta_docs['tsne_x'] <= upper_x) &
            (meta_docs['tsne_y'] >= lower_y) & (meta_docs['tsne_y'] <= upper_y)
        ].copy()
        
        return filtered


def tSNE(df: pd.DataFrame, output_dir: Path, config: MasterConfig = None) -> Path:
    """t-SNE visualization focused on couples clustering: Each couple has their own language."""
    
    if config is None:
        config = DEFAULT_CONFIG

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.manifold import TSNE
    except ImportError:
        print("    ❌ scikit-learn required. Install with: pip install scikit-learn")
        return None

    # Analyze data
    analyzer = TSNEAnalyzer(config)
    couples_df = analyzer.prepare_data(df)
    
    if couples_df.empty:
        print("    ❌ No couple data found")
        return None

    try:
        meta_docs = analyzer.generate_embeddings(couples_df)
        meta_docs = analyzer.merge_gender_info(meta_docs, couples_df)
        filtered = analyzer.filter_outliers(meta_docs)
        
        # Rename columns for compatibility with plotting function
        couple_meta = filtered.rename(columns={'tsne_x': 'pca_x', 'tsne_y': 'pca_y'})
        
    except ValueError as e:
        print(f"    ❌ {e}")
        return None

    # Compute centroid distances between couples in t-SNE space
    couples = sorted(couple_meta['couples'].unique())

    distance_parts: List[str] = []
    cross_distance_parts: List[str] = []
    couple_centroids: Dict[str, np.ndarray] = {}
    for couple in couples:
        subset = couple_meta[couple_meta['couples'] == couple]
        label = couple.replace('couple', 'Couple ')

        # Attempt to split by author and compute within-couple distances
        if 'author' in subset.columns and subset['author'].nunique() >= 2:
            author_centroids = subset.groupby('author')[['pca_x', 'pca_y']].mean()
            if len(author_centroids) >= 2:
                # Choose first two authors (assumes couple pair)
                first_two = author_centroids.index[:2]
                dist = np.linalg.norm(author_centroids.loc[first_two[0]] - author_centroids.loc[first_two[1]])
                distance_parts.append(f"{label}: {dist:.1f}")
                couple_centroids[couple] = author_centroids.iloc[:2].mean().to_numpy()
                continue

        # Fallback to average distance between points with author info
        if len(subset) >= 2:
            coords = subset[['pca_x', 'pca_y']].to_numpy()
            distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
            upper = distances[np.triu_indices_from(distances, k=1)]
            if upper.size > 0:
                mean_dist = float(np.mean(upper))
                distance_parts.append(f"{label}: {mean_dist:.1f}")
                couple_centroids[couple] = coords.mean(axis=0)
        elif len(subset) == 1:
            couple_centroids[couple] = subset[['pca_x', 'pca_y']].to_numpy().squeeze()

    for c1, c2 in combinations(couple_centroids.keys(), 2):
        dist = float(np.linalg.norm(couple_centroids[c1] - couple_centroids[c2]))
        label_a = c1.replace('couple', 'Couple ')
        label_b = c2.replace('couple', 'Couple ')
        cross_distance_parts.append(f"{label_a}–{label_b}: {dist:.1f}")

    within_text = (
        "Within-couple t-SNE distance — " + " | ".join(distance_parts)
        if distance_parts else "Within-couple distances unavailable"
    )
    cross_text = (
        "Cross-couple t-SNE distance — " + " | ".join(cross_distance_parts)
        if cross_distance_parts else "Cross-couple distances unavailable"
    )

    # Create styling helper
    style = FigureStyle(config.style)

    # Create visualization
    fig, ax = style.setup_figure(figsize=(14, 10))

    def _author_variants(base_hex: str, count: int) -> List[str]:
        return [base_hex] * max(1, count)

    # Plot couples with density contours
    for couple in couples:
        subset = couple_meta[couple_meta['couples'] == couple]
        base_color = config.style.couple_colors.get(couple, '#6B7280')
        
        # Format couple label
        couple_num = couple.replace('couple', 'Couple ')
        
        unique_authors = [a for a in sorted(subset['author'].dropna().unique().tolist())]
        if not unique_authors:
            unique_authors = [None]
        author_palette = _author_variants(base_color, len(unique_authors))
        author_color_map = dict(zip(unique_authors, author_palette))

        for author in unique_authors:
            author_points = subset if author is None else subset[subset['author'] == author]
            ax.scatter(
                author_points['pca_x'],
                author_points['pca_y'],
                c=author_color_map.get(author, base_color),
                s=120,
                alpha=0.75,
                edgecolors='white',
                linewidth=1.4,
                zorder=3,
            )
        
        # Add KDE contours if enough points
        if len(subset) > 3:
            try:
                import seaborn as sns
                sns.kdeplot(
                    x=subset['pca_x'],
                    y=subset['pca_y'],
                    ax=ax,
                    color=base_color,
                    alpha=config.tsne.kde_alpha,
                    levels=config.tsne.kde_levels,
                    fill=True,
                    zorder=1,
                )
            except:
                pass

    # Styling
    style.set_axis_labels(ax, xlabel='t-SNE 1', ylabel='t-SNE 2')
    style.style_axis(ax)
    style.add_legend(ax, legend_type='couples')
    style.add_title(fig, 'Each couple has their own language', y_pos=0.97)

    fig.tight_layout(rect=[0, 0.07, 1, 0.90])
    style.add_caption(fig, [within_text, cross_text], y_start=0.045)

    output_path = output_dir / config.output.tsne_filename
    plt.savefig(output_path, dpi=config.output.dpi, bbox_inches=config.output.bbox_inches,
                facecolor=config.output.facecolor, edgecolor=config.output.edgecolor)
    plt.close(fig)
    
    print(f"    ✅ Saved: {output_path}")
    return output_path


class HourlyPatternAnalyzer:
    """Analyzes hourly messaging patterns by gender."""
    
    def __init__(self, config: MasterConfig):
        self.config = config
        self.col = config.columns
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with hour extraction."""
        data = df[df[self.col.couples].isin(self.config.analysis.couples_list)].copy()
        data = data[data[self.col.message] != self.config.analysis.omitted_message_text]
        data[self.col.timestamp] = pd.to_datetime(data[self.col.timestamp])
        data['hour'] = data[self.col.timestamp].dt.hour
        
        # Map gender to lowercase for internal processing
        data['gender_mapped'] = data[self.col.gender].map({
            'male': 'male', 'female': 'female', 'M': 'male', 'F': 'female'
        })
        data = data.dropna(subset=['gender_mapped'])
        
        return data
    
    def calculate_hourly_percentages(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate percentage of messages sent each hour by gender."""
        hour_bins = np.arange(24)
        male_percentages = []
        female_percentages = []
        
        total_male = data[data['gender_mapped'] == 'male'].shape[0]
        total_female = data[data['gender_mapped'] == 'female'].shape[0]
        
        for hour_bin in hour_bins:
            bin_data = data[data['hour'] == hour_bin]
            male_count = len(bin_data[bin_data['gender_mapped'] == 'male'])
            female_count = len(bin_data[bin_data['gender_mapped'] == 'female'])
            
            male_pct = (male_count / total_male) * 100 if total_male > 0 else 0
            female_pct = (female_count / total_female) * 100 if total_female > 0 else 0
            
            male_percentages.append(male_pct)
            female_percentages.append(female_pct)
        
        return {
            'male': np.array(male_percentages, dtype=float),
            'female': np.array(female_percentages, dtype=float)
        }


def time_series(df: pd.DataFrame, output_dir: Path, config: MasterConfig = None) -> Path:
    """Who texts when? Distribution of texting patterns throughout the day."""
    
    if config is None:
        config = DEFAULT_CONFIG

    # Analyze data
    analyzer = HourlyPatternAnalyzer(config)
    prepared_data = analyzer.prepare_data(df)
    percentages = analyzer.calculate_hourly_percentages(prepared_data)
    
    male_series = percentages['male']
    female_series = percentages['female']

    # Extend data for smooth wraparound
    hour_bins = np.arange(24)
    extended_hours = np.concatenate([hour_bins, hour_bins + 24])
    extended_male = np.concatenate([male_series, male_series])
    extended_female = np.concatenate([female_series, female_series])

    # Create smooth splines
    male_spline = make_interp_spline(extended_hours, extended_male, k=3)
    female_spline = make_interp_spline(extended_hours, extended_female, k=3)

    smooth_hours = np.linspace(2, 24, 500)
    male_smoothed = male_spline(smooth_hours)
    female_smoothed = female_spline(smooth_hours)
    
    # Create styling helper
    style = FigureStyle(config.style)
    
    # Create visualization
    fig, ax = style.setup_figure(figsize=(14, 8))
    
    # Plot lines
    ax.plot(smooth_hours, male_smoothed, 
            color=config.style.gender_colors['Men'], 
            linewidth=3, 
            label='Men',
            alpha=1.0)
    
    ax.plot(smooth_hours, female_smoothed, 
        color=config.style.gender_colors['Women'], 
        linewidth=3, 
        label='Women',
        alpha=1.0)
    
    # Fill areas between lines
    ax.fill_between(smooth_hours, male_smoothed, female_smoothed, 
                    where=(male_smoothed >= female_smoothed),
                    color=config.style.gender_colors['Men'], alpha=1.0, interpolate=True)
    ax.fill_between(smooth_hours, male_smoothed, female_smoothed, 
                    where=(female_smoothed > male_smoothed),
                    color=config.style.gender_colors['Women'], alpha=1.0, interpolate=True)
    
    style.set_axis_labels(ax, xlabel='Time', ylabel='Messages')
    style.add_title(fig, 'Who texts when?')
    
    # X-axis formatting
    ax.set_xlim(2, 24)
    xticks = list(range(2, 25, 2))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{hour % 24}h" if hour < 24 else "0h" for hour in xticks])
    
    # Y-axis formatting
    max_pct = max(float(np.max(male_smoothed)) if male_smoothed.size else 0, 
                  float(np.max(female_smoothed)) if female_smoothed.size else 0)
    min_pct = min(float(np.min(male_smoothed)) if male_smoothed.size else 0, 
                  float(np.min(female_smoothed)) if female_smoothed.size else 0)
    span = max(max_pct - min_pct, 1.0)
    lower_bound = max(-0.5, min_pct - span * 0.12)
    upper_bound = max_pct + span * 0.12
    ax.set_ylim(lower_bound, upper_bound)

    style.add_legend(ax, legend_type='gender', bbox_to_anchor=(0.98, 1.02))
    style.style_axis(ax)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    
    output_path = output_dir / config.output.timeseries_filename
    plt.savefig(output_path, dpi=config.output.dpi, bbox_inches=config.output.bbox_inches,
                facecolor=config.output.facecolor, edgecolor=config.output.edgecolor)
    plt.close(fig)
    
    print(f"    ✅ Saved: {output_path}")
    return output_path


class QuarterlyDominanceAnalyzer:
    """Analyzes quarterly message dominance patterns by couple and gender."""
    
    def __init__(self, config: MasterConfig):
        self.config = config
        self.col = config.columns
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with quarterly grouping."""
        df[self.col.timestamp] = pd.to_datetime(df[self.col.timestamp])
        couples_data = df[df[self.col.couples].isin(self.config.analysis.couples_list)].copy()
        
        if couples_data.empty:
            raise ValueError("No couple data found in dataframe.")
        
        couples_data['quarter'] = couples_data[self.col.timestamp].dt.to_period('Q')
        return couples_data
    
    def calculate_quarterly_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate message counts by quarter, couple, and gender."""
        return data.groupby(['quarter', self.col.couples, self.col.gender]).size().reset_index(name='message_count')


def comparing_categories(df: pd.DataFrame, output_dir: Path, config: MasterConfig = None) -> Path:
    """Comparing message patterns between men and women."""
    
    if config is None:
        config = DEFAULT_CONFIG

    # Analyze data
    analyzer = QuarterlyDominanceAnalyzer(config)
    couples_data = analyzer.prepare_data(df)
    quarter_counts = analyzer.calculate_quarterly_stats(couples_data)

    # Create styling helper
    style = FigureStyle(config.style)
    
    # Create visualization with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10), sharey=True)
    fig.patch.set_facecolor(config.style.figure_bg)
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(config.style.axis_bg)

    couple_labels = {'couple1': 'Couple 1', 'couple2': 'Couple 2', 'couple3': 'Couple 3'}

    all_quarters = sorted(quarter_counts['quarter'].unique())
    quarter_positions = range(len(all_quarters))
    couples = config.analysis.couples_list
    axes = [ax1, ax2, ax3]
    bar_height = 1.0

    for idx, (couple, ax) in enumerate(zip(couples, axes)):
        male_pct_list, female_pct_list = [], []
        for q in all_quarters:
            qd = quarter_counts[quarter_counts['quarter'] == q]
            cd = qd[qd[config.columns.couples] == couple]
            m = cd[cd[config.columns.gender] == 'male']['message_count'].sum()
            f = cd[cd[config.columns.gender] == 'female']['message_count'].sum()
            tot = m + f
            if tot > 0:
                male_pct_list.append(m * 100 / tot)
                female_pct_list.append(f * 100 / tot)
            else:
                male_pct_list.append(0)
                female_pct_list.append(0)

        # Plot horizontal stacked bars
        ax.barh(
            quarter_positions,
            male_pct_list,
            height=bar_height,
            color=config.style.gender_colors['Men'],
            alpha=1.0,
            edgecolor='none',
            linewidth=0,
        )
        ax.barh(
            quarter_positions,
            female_pct_list,
            height=bar_height,
            left=male_pct_list,
            color=config.style.gender_colors['Women'],
            alpha=1.0,
            edgecolor='none',
            linewidth=0,
        )

        # Add legend to middle subplot
        if idx == 2:
            style.add_legend(ax, legend_type='gender', bbox_to_anchor=(1.0, 1.15))

        ax.set_xlabel('')
        if idx == 0:
            ax.set_ylabel('')
            ax.set_yticks(list(quarter_positions))
            ax.set_yticklabels([str(q) for q in all_quarters])
            ax.set_ylim(len(all_quarters)-0.5, -0.5)
        else:
            ax.set_ylim(len(all_quarters)-0.5, -0.5)
        ax.margins(y=0)
        ax.set_title(couple_labels[couple], fontsize=18, fontweight='bold', pad=30)
        
        style.style_axis(ax, grid=False)
        
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 100])
        ax.set_xticklabels(['0', '100'])

    style.add_title(fig, 'One voice per couple always dominates\nWhatsApp messages over time')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.75, wspace=0.1)
    
    output_path = output_dir / config.output.comparing_filename
    plt.savefig(output_path, dpi=config.output.dpi, bbox_inches=config.output.bbox_inches,
                facecolor=config.output.facecolor, edgecolor=config.output.edgecolor)
    plt.close(fig)
    
    print(f"    ✅ Saved: {output_path}")
    return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("5 KEY FIGURES GENERATION - STANDALONE")
    print("=" * 70)
    
    # Step 1: Run preprocessing
    csv_path = run_preprocessing()
    
    # Step 2: Load processed data
    print("\n" + "=" * 70)
    print("LOADING PROCESSED DATA")
    print("=" * 70)
    
    if not csv_path.exists():
        print(f"❌ ERROR: {csv_path} does not exist!")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} messages from {csv_path}")
    
    # Step 3: Generate figures
    print("\n" + "=" * 70)
    print("GENERATING 5 KEY FIGURES")
    print("=" * 70 + "\n")
    
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    
    figures = [
        ("distribution", distribution),
        ("relationship", relationship),
        ("tSNE", tSNE),
        ("time_series", time_series),
        ("comparing_categories", comparing_categories),
    ]
    
    generated = []
    for name, func in figures:
        try:
            result = func(df, img_dir)
            if result:
                generated.append(result)
        except Exception as e:
            print(f"  ❌ ERROR generating {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print(f"✅ COMPLETED: Generated {len(generated)}/{len(figures)} figures")
    print("=" * 70)
    print("\nGenerated files:")
    for path in generated:
        print(f"  📈 {path}")
    print()


if __name__ == "__main__":
    main()
