import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
import seaborn as sns
import tomllib
import re
from scipy.interpolate import make_interp_spline


HERE = Path(__file__).resolve()
SRC_DIR = HERE.parent
REPO_ROOT = SRC_DIR.parent


@dataclass
class Settings:
    processed_dir: Path
    current: Optional[str] = None
    time_col: str = "timestamp"
    couples: List[str] = field(default_factory=lambda: ["couple1", "couple2", "couple3"])
    palette: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "Couple1": {"male": "#1E3A8A", "female": "#FF69B4"},
        "Couple2": {"male": "#1E3A8A", "female": "#FF69B4"},
        "Couple3": {"male": "#1E3A8A", "female": "#FF69B4"},
    })


def load_settings() -> Settings:
    cfg_path = REPO_ROOT / "config.toml"
    processed_dir = REPO_ROOT / "data/processed"
    current = None
    if cfg_path.exists():
        try:
            with cfg_path.open("rb") as f:
                cfg = tomllib.load(f)
            if isinstance(cfg.get("processed"), str):
                processed_dir = REPO_ROOT / cfg.get("processed")
            current = cfg.get("current")
        except Exception:
            pass
    return Settings(processed_dir=processed_dir, current=current)


def _load_processed_df(settings: Optional[Settings] = None) -> pd.DataFrame:
    if settings is None:
        settings = load_settings()
    processed_dir = settings.processed_dir

    candidates: list[Path] = [
        processed_dir / "output.csv",
        processed_dir / "output.parq",
    ]

    if settings.current:
        current_path = processed_dir / settings.current if not Path(settings.current).is_absolute() else Path(settings.current)
        candidates.insert(0, current_path)

    for p in candidates:
        if p and isinstance(p, Path) and p.exists() and p.is_file():
            if p.suffix.lower() in {".parq", ".parquet"}:
                return pd.read_parquet(p)
            return pd.read_csv(p)

    if processed_dir.exists():
        files = list(processed_dir.glob("**/*"))
        files = [f for f in files if f.is_file() and f.suffix.lower() in {".csv", ".parq", ".parquet"}]
        if files:
            newest = max(files, key=lambda fp: fp.stat().st_mtime)
            if newest.suffix.lower() in {".parq", ".parquet"}:
                return pd.read_parquet(newest)
            return pd.read_csv(newest)

    raise FileNotFoundError("Processed data not found. Expected data/processed/output.csv|.parq or path from config.toml.")


def plot_gauge_couples(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    """Render the three semicircular gauge charts and save to img/comparing_categories_1.png.

    If df is None, load data from data/processed/output.csv (or output.parq).
    Returns the path to the saved image.
    """
    sns.set(style="whitegrid")
    if settings is None:
        settings = load_settings()
    if df is None:
        df = _load_processed_df(settings)

    if "author" not in df.columns or "message" not in df.columns:
        raise ValueError("Dataframe must contain 'author' and 'message' columns.")

    author_counts = df.groupby("author").size().reset_index(name="message_count")

    if "couples" in df.columns:
        couples_map = df.drop_duplicates("author").set_index("author")["couples"].to_dict()
        author_counts["couples"] = author_counts["author"].map(couples_map).fillna("single")
    else:
        author_counts["couples"] = "single"

    if "gender" in df.columns:
        gender_map = df.drop_duplicates("author").set_index("author")["gender"].to_dict()
        author_counts["gender"] = author_counts["author"].map(gender_map).fillna("unknown")
    else:
        author_counts["gender"] = "unknown"

    couple_shades = settings.palette

    def _draw_gauge(ax, a_count, b_count, a_name, b_name, a_gender, b_gender, couple_label, colors):
        total = a_count + b_count
        a_pct = (a_count / total) if total > 0 else 0.5
        b_pct = (b_count / total) if total > 0 else 0.5

        ax.clear()

        a_color = colors["male"] if a_gender == "male" else colors["female"]
        b_color = colors["male"] if b_gender == "male" else colors["female"]

        bar_width = 0.38
        ax.add_patch(Wedge((0, 0), 1.0, 0, a_pct * 180, width=bar_width, facecolor=a_color, edgecolor='none', alpha=1.0))
        ax.add_patch(Wedge((0, 0), 1.0, a_pct * 180, 180, width=bar_width, facecolor=b_color, edgecolor='none', alpha=1.0))

        if a_pct > 0:
            ang = a_pct * np.pi / 2
            ax.text(1.3 * np.cos(ang), 1.3 * np.sin(ang), f"{a_count}\n({a_pct:.1%})",
                    ha="center", va="center", fontsize=10, color=a_color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        if b_pct > 0:
            ang = a_pct * np.pi + (b_pct * np.pi / 2)
            ax.text(1.3 * np.cos(ang), 1.3 * np.sin(ang), f"{b_count}\n({b_pct:.1%})",
                    ha="center", va="center", fontsize=10, color=b_color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.text(0, -0.2, f"{couple_label}\n{total} messages", ha="center", va="center", fontsize=12, fontweight="bold")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('One Voice Always Dominates\nPer Couple', fontsize=32, fontweight='bold', y=0.95)

    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#1E3A8A', alpha=0.8, label='Men'),
        Rectangle((0, 0), 1, 1, facecolor='#FF69B4', alpha=0.8, label='Women')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88),
               title_fontsize=12, fontsize=10, frameon=True, fancybox=True, shadow=True)

    for i, couple_id in enumerate(settings.couples):
        subset = author_counts[author_counts["couples"] == couple_id]
        if len(subset) >= 2:
            pair = subset.nlargest(2, "message_count")
            a, b = pair.iloc[0], pair.iloc[1]
            a_name, b_name = a["author"], b["author"]
            a_count, b_count = a["message_count"], b["message_count"]
            a_gender, b_gender = a.get("gender", "unknown"), b.get("gender", "unknown")
        elif len(subset) == 1:
            a = subset.iloc[0]
            a_name, a_count = a["author"], a["message_count"]
            a_gender = a.get("gender", "unknown")
            b_name, b_count, b_gender = "No partner", 0, "unknown"
        else:
            a_name = b_name = "No data"
            a_count = b_count = 0
            a_gender = b_gender = "unknown"

        colors = couple_shades.get(f"Couple{i+1}", {"light": "#BBBBBB", "dark": "#777777"})
        _draw_gauge(axes[i], a_count, b_count, a_name, b_name, a_gender, b_gender, f"Couple {i+1}", colors)

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    root = REPO_ROOT
    img_dir = root / "img"
    img_dir.mkdir(exist_ok=True)
    output_path = img_dir / "comparing_categories_1.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return output_path


def plot_timeline_couples_bars(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    """Render a horizontal diverging Men vs Women character count distribution chart.

    - Y-axis: Character count bins (0-10, 10-20, 20-30, etc.)
    - X-axis: Percentage distribution Men (left, blue) vs Women (right, pink)
    - Center line at x=0, showing percentage split per character count bin
    """
    if settings is None:
        settings = load_settings()
    if df is None:
        df = _load_processed_df(settings)

    couples_data = df[df['couples'].isin(settings.couples)].copy()
    if couples_data.empty:
        raise ValueError("No couple data found in dataframe.")
    
    couples_data = couples_data[~couples_data['message'].str.contains('<Media omitted>', na=False)].copy()
    couples_data['char_count'] = couples_data['message'].str.len().fillna(0)
    
    bins = list(range(0, 150))  # 0, 1, 2, 3, ..., 148, 149
    
    couples_data = couples_data[couples_data['char_count'] < 150].copy()
    couples_data['char_bin'] = pd.cut(couples_data['char_count'], bins=bins, right=False, include_lowest=True)
    
    bin_counts = couples_data.groupby(['char_bin', 'gender'], observed=True).size().reset_index(name='message_count')
    
    bin_char_totals = couples_data.groupby(['char_bin', 'gender'], observed=True)['char_count'].sum().reset_index(name='total_chars')
    
    bin_totals = bin_counts.groupby('char_bin', observed=True)['message_count'].sum().reset_index()
    bin_percentages = bin_counts.merge(bin_totals, on='char_bin', suffixes=('', '_total'))
    bin_percentages['percentage'] = (bin_percentages['message_count'] / bin_percentages['message_count_total']) * 100
    
    total_chat_chars = couples_data['char_count'].sum()
    bin_char_percentages = bin_char_totals.copy()
    bin_char_percentages['char_percentage'] = (bin_char_percentages['total_chars'] / total_chat_chars) * 100
    
    all_bins = sorted([b for b in bin_counts['char_bin'].unique() if pd.notna(b)])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    gender_colors = {
        'male': '#1E3A8A',      # Dark Blue
        'female': '#FF69B4'     # Pink
    }
    gender_labels = {
        'male': 'Men',
        'female': 'Women'
    }

    bin_positions = range(len(all_bins))
    bar_height = 1.0
    
    total_messages = len(couples_data)
    
    for bi, char_bin in enumerate(all_bins):
        bd = bin_percentages[bin_percentages['char_bin'] == char_bin]
        
        male_pct = bd[bd['gender'] == 'male']['percentage'].iloc[0] if not bd[bd['gender'] == 'male'].empty else 0
        female_pct = bd[bd['gender'] == 'female']['percentage'].iloc[0] if not bd[bd['gender'] == 'female'].empty else 0
        
        bin_message_total = bin_counts[bin_counts['char_bin'] == char_bin]['message_count'].sum()
        bin_message_pct = (bin_message_total / total_messages) * 100 if total_messages > 0 else 0
        
        male_width = (male_pct / 100) * bin_message_pct
        female_width = (female_pct / 100) * bin_message_pct
        
        if male_width > 0:
            label = gender_labels['male'] if bi == 0 else ""
            ax.barh(bi, male_width, height=bar_height, left=-male_width, 
                   color=gender_colors['male'], alpha=1.0, edgecolor='none', label=label)
        
        if female_width > 0:
            label = gender_labels['female'] if bi == 0 else ""
            ax.barh(bi, female_width, height=bar_height, left=0,
                   color=gender_colors['female'], alpha=1.0, edgecolor='none', label=label)

    half_span = 1.5  # 1.5% on each side for focused view
    
    ax.set_xlabel('% of Total Messages (Bar width = Message Volume)')
    ax.set_ylabel('Character Count')
    fig.suptitle('Men vs Women\nCharacter Distribution', 
                 fontsize=32, fontweight='bold', x=0.06, y=0.95, ha='left')
    
    tick_positions = []
    tick_labels = []
    for i, char_bin in enumerate(all_bins):
        if int(char_bin.left) % 25 == 0:  # Show every 25th character count
            tick_positions.append(i)
            tick_labels.append(str(int(char_bin.left)))
    
    tick_positions.append(len(all_bins))
    tick_labels.append("150+")
    
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    ax.set_xlim(-half_span, half_span)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.7, zorder=5)
    
    ax.set_xticks([-1.5, -0.75, 0, 0.75, 1.5])
    ax.set_xticklabels(['1.5%', '0.75%', '0%', '0.75%', '1.5%'])
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.15), frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    out = img_dir / 'distribution_categories.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return out


def plot_timeline_couples_bars_hours(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    """Render a horizontal diverging Men vs Women hour-of-day distribution chart.

    - Y-axis: Hour bins (0, 1, 2, ..., 23)
    - X-axis: Percentage distribution Men (left, blue) vs Women (right, pink)
    - Center line at x=0, showing percentage split per hour bin
    """
    if settings is None:
        settings = load_settings()
    if df is None:
        df = _load_processed_df(settings)

    couples_data = df[df['couples'].isin(settings.couples)].copy()
    
    couples_data = couples_data[couples_data['message'] != '<Media omitted>']
    
    couples_data[settings.time_col] = pd.to_datetime(couples_data[settings.time_col])
    couples_data['hour'] = couples_data[settings.time_col].dt.hour
    
    couples_data['gender_mapped'] = couples_data['gender'].map({
        'male': 'male',
        'female': 'female',
        'M': 'male',
        'F': 'female'
    })
    
    couples_data = couples_data.dropna(subset=['gender_mapped'])
    
    total_messages = len(couples_data)
    
    hour_bins = list(range(24))  # 0 to 23
    all_bins = hour_bins.copy()
    
    bin_counts = []
    for hour_bin in hour_bins:
        bin_data = couples_data[couples_data['hour'] == hour_bin]
        bin_total = len(bin_data)
        
        male_count = len(bin_data[bin_data['gender_mapped'] == 'male'])
        female_count = len(bin_data[bin_data['gender_mapped'] == 'female'])
        
        bin_counts.append({
            'hour_bin': hour_bin,
            'message_count': bin_total,
            'male_count': male_count,
            'female_count': female_count
        })
    
    bin_counts = pd.DataFrame(bin_counts)
    
    fig, ax = plt.subplots(figsize=(12, 16))
    
    gender_colors = {
        'male': '#1E3A8A',    # Dark blue
        'female': '#FF69B4'   # Pink
    }
    
    gender_labels = {
        'male': 'Men',
        'female': 'Women'
    }
    
    bar_height = 0.8
    
    for bi, hour_bin in enumerate(all_bins):
        bin_message_total = bin_counts[bin_counts['hour_bin'] == hour_bin]['message_count'].iloc[0]
        
        if bin_message_total == 0:
            continue
            
        male_count = bin_counts[bin_counts['hour_bin'] == hour_bin]['male_count'].iloc[0]
        female_count = bin_counts[bin_counts['hour_bin'] == hour_bin]['female_count'].iloc[0]
        
        male_pct = (male_count / bin_message_total * 100) if bin_message_total > 0 else 0
        female_pct = (female_count / bin_message_total * 100) if bin_message_total > 0 else 0
        
        bin_message_pct = (bin_message_total / total_messages) * 100 if total_messages > 0 else 0
        
        male_width = (male_pct / 100) * bin_message_pct
        female_width = (female_pct / 100) * bin_message_pct
        
        if male_width > 0:
            label = gender_labels['male'] if bi == 0 else ""
            ax.barh(bi, male_width, height=bar_height, left=-male_width, 
                   color=gender_colors['male'], alpha=1.0, edgecolor='none', label=label)
        
        if female_width > 0:
            label = gender_labels['female'] if bi == 0 else ""
            ax.barh(bi, female_width, height=bar_height, left=0,
                   color=gender_colors['female'], alpha=1.0, edgecolor='none', label=label)

    half_span = 8.0  # 8% on each side for wider view
    
    ax.set_xlabel('% of Total Messages (Bar width = Message Volume)')
    ax.set_ylabel('Hour of Day')
    fig.suptitle('Men vs Women\nHour Distribution', 
                 fontsize=32, fontweight='bold', x=0.06, y=0.95, ha='left')
    
    tick_positions = list(range(len(all_bins)))
    tick_labels = [f"{hour}h" for hour in all_bins]
    
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    ax.set_xlim(-half_span, half_span)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.7, zorder=5)
    
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_xticklabels(['8%', '4%', '0%', '4%', '8%'])
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.15), frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    out = img_dir / 'distribution_categories_2.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return out


def plot_timeline_couples_columns(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    """Render the 3-column 100% distributions and save to img/comparing_categories_2.png."""
    if settings is None:
        settings = load_settings()
    if df is None:
        df = _load_processed_df(settings)

    df[settings.time_col] = pd.to_datetime(df[settings.time_col])
    couples_data = df[df['couples'].isin(settings.couples)].copy()
    if couples_data.empty:
        raise ValueError("No couple data found in dataframe.")
    couples_data['quarter'] = couples_data[settings.time_col].dt.to_period('Q')
    quarter_counts = couples_data.groupby(['quarter', 'couples', 'gender']).size().reset_index(name='message_count')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10), sharey=True)
    fig.patch.set_facecolor('white')
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('#fafafa')

    couple_colors = {
        'couple1_male': '#1E3A8A', 'couple1_female': '#FF69B4',
        'couple2_male': '#1E3A8A', 'couple2_female': '#FF69B4',
        'couple3_male': '#1E3A8A', 'couple3_female': '#FF69B4'
    }
    couple_labels = {'couple1': 'Couple 1', 'couple2': 'Couple 2', 'couple3': 'Couple 3'}

    all_quarters = sorted(quarter_counts['quarter'].unique())
    quarter_positions = range(len(all_quarters))
    couples = settings.couples
    axes = [ax1, ax2, ax3]
    bar_height = 1.0

    for idx, (couple, ax) in enumerate(zip(couples, axes)):
        male_pct_list, female_pct_list = [], []
        for q in all_quarters:
            qd = quarter_counts[quarter_counts['quarter'] == q]
            cd = qd[qd['couples'] == couple]
            m = cd[cd['gender'] == 'male']['message_count'].sum()
            f = cd[cd['gender'] == 'female']['message_count'].sum()
            tot = m + f
            if tot > 0:
                male_pct_list.append(m * 100 / tot)
                female_pct_list.append(f * 100 / tot)
            else:
                male_pct_list.append(0)
                female_pct_list.append(0)

        ax.barh(
            quarter_positions,
            male_pct_list,
            height=bar_height,
            color=couple_colors[f'{couple}_male'],
            alpha=1.0,
            edgecolor='none',
            linewidth=0,
        )
        ax.barh(
            quarter_positions,
            female_pct_list,
            height=bar_height,
            left=male_pct_list,
            color=couple_colors[f'{couple}_female'],
            alpha=1.0,
            edgecolor='none',
            linewidth=0,
        )

        if idx == 2:
            legend_elements = [Rectangle((0,0),1,1, facecolor='#1E3A8A', label='Men'), Rectangle((0,0),1,1, facecolor='#FF69B4', label='Women')]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.15), frameon=True, fancybox=True, shadow=True, fontsize=12)

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
        ax.grid(False)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 100])
        ax.set_xticklabels(['0', '100'])

    fig.suptitle('One voice per couple always dominates\nWhatsApp messages over time', fontsize=32, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.75, wspace=0.1)
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    out = img_dir / 'comparing_categories_2.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return out


def plot_time_series(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    """Create a time series chart showing mentions of child/family-related words over time.
    
    - X-axis: Time periods (quarters)
    - Y-axis: Number of mentions
    - Two smooth lines: Men (blue) vs Women (pink)
    - Tracks words: baby, child, children, kids, play, school, toys, Jokūbas, Jokubas, Arianna, Ettore
    """
    if settings is None:
        settings = load_settings()
    if df is None:
        df = _load_processed_df(settings)

    df[settings.time_col] = pd.to_datetime(df[settings.time_col])
    couples_data = df[df['couples'].isin(settings.couples)].copy()
    if couples_data.empty:
        raise ValueError("No couple data found in dataframe.")
    
    target_words = [
        'baby', 'child', 'children', 'kids', 'play', 'school', 'toys',
        'babysit', 'stroller', 'strollers', 'crib', 'cribs', 'nap', 'napping',
        'playground', 'daycare', 'grandparents', 'sleepover', 'bedtime',
        'parents', 'Jokūbas', 'Jokubas', 'Arianna', 'Ettore'
    ]
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in target_words) + r')\b'
    regex = re.compile(pattern, re.IGNORECASE)
    
    couples_data['mentions'] = couples_data['message'].str.findall(regex).str.len().fillna(0)
    
    mentions_data = couples_data[couples_data['mentions'] > 0].copy()
    
    couples_data['quarter'] = couples_data[settings.time_col].dt.to_period('Q')
    mentions_data['quarter'] = mentions_data[settings.time_col].dt.to_period('Q')
    quarter_mentions = mentions_data.groupby(['quarter', 'gender'])['mentions'].sum().reset_index()
    
    all_quarters_in_data = sorted(couples_data['quarter'].unique())
    all_quarters = all_quarters_in_data
    
    actual_data_quarters = sorted(couples_data['quarter'].unique())
    men_data = quarter_mentions[quarter_mentions['gender'] == 'male'].set_index('quarter')['mentions'].reindex(actual_data_quarters, fill_value=0)
    women_data = quarter_mentions[quarter_mentions['gender'] == 'female'].set_index('quarter')['mentions'].reindex(actual_data_quarters, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    x_numeric_data = np.arange(len(actual_data_quarters))
    x_numeric_all = np.arange(len(all_quarters))
    
    men_color = '#1E3A8A'    # Dark blue
    women_color = '#FF69B4'  # Pink
    
    if len(actual_data_quarters) >= 3:
        x_smooth = np.linspace(x_numeric_data.min(), x_numeric_data.max(), 300)
        
        men_smooth = make_interp_spline(x_numeric_data, men_data.values, k=min(3, len(actual_data_quarters)-1))
        men_y_smooth = men_smooth(x_smooth)
        
        women_smooth = make_interp_spline(x_numeric_data, women_data.values, k=min(3, len(actual_data_quarters)-1))
        women_y_smooth = women_smooth(x_smooth)
        
        data_start_idx = all_quarters.index(actual_data_quarters[0])
        ax.plot(x_smooth + data_start_idx, men_y_smooth, color=men_color, linewidth=3, label='Men', alpha=0.9)
        ax.plot(x_smooth + data_start_idx, women_y_smooth, color=women_color, linewidth=3, label='Women', alpha=0.9)
        
        data_positions = [all_quarters.index(q) for q in actual_data_quarters]
        ax.scatter(data_positions, men_data.values, color=men_color, s=60, zorder=5, alpha=0.8)
        ax.scatter(data_positions, women_data.values, color=women_color, s=60, zorder=5, alpha=0.8)
    else:
        data_positions = [all_quarters.index(q) for q in actual_data_quarters]
        ax.plot(data_positions, men_data.values, color=men_color, linewidth=3, marker='o', markersize=8, label='Men')
        ax.plot(data_positions, women_data.values, color=women_color, linewidth=3, marker='o', markersize=8, label='Women')
    
    baby_consumated_1_date = pd.Period('2023-Q1')  # Q1 2023 (February 2023, ~9 months before birth)
    baby_born_2_date = pd.Period('2023-Q4')  # Q4 2023 (November 22, 2023)
    baby_consumated_2_date = pd.Period('2025-Q1')  # Q1 2025 (March 2025)
    
    reference_lines = [
        (baby_consumated_1_date, 'Baby Consumated', '#FF6B6B', '--'),
        (baby_born_2_date, 'Baby Born', '#4ECDC4', '-'),
        (baby_consumated_2_date, 'Baby Consumated', '#FF6B6B', '--')
    ]
    
    for ref_date, label, color, linestyle in reference_lines:
        if ref_date in all_quarters:
            x_pos = list(all_quarters).index(ref_date)
            ax.axvline(x=x_pos, color=color, linestyle=linestyle, linewidth=2, alpha=0.7, zorder=3)
            ax.text(x_pos, ax.get_ylim()[1] * 0.95, label, rotation=90, 
                   ha='right', va='top', fontsize=10, color=color, fontweight='bold')
    
    ax.set_xlabel('')
    ax.set_ylabel('Baby/Child talk', fontsize=14, fontweight='bold')
    ax.set_title('Women contribute more to baby/child talk than men', 
                 fontsize=24, fontweight='bold', pad=20, loc='left')
    
    ax.set_xticks(x_numeric_all)
    ax.set_xticklabels([str(q) for q in all_quarters], rotation=45, ha='right')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    fig.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1], 
               loc='upper right', bbox_to_anchor=(0.98, 0.95), frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    output_path = img_dir / 'time_series.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return output_path


def plot_time_series_couples(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    """Create a time series chart showing mentions of child/family-related words over time by couples.
    
    - X-axis: Time periods (quarters)
    - Y-axis: Number of mentions
    - Three smooth lines: Couple 1 (blue - no child), Couple 2 (light pink - with child), Couple 3 (dark pink - with child)
    - Legend: Pink - With child, Blue - No child
    """
    if settings is None:
        settings = load_settings()
    if df is None:
        df = _load_processed_df(settings)

    df[settings.time_col] = pd.to_datetime(df[settings.time_col])
    couples_data = df[df['couples'].isin(settings.couples)].copy()
    if couples_data.empty:
        raise ValueError("No couple data found in dataframe.")
    
    target_words = [
        'baby', 'child', 'children', 'kids', 'play', 'school', 'toys',
        'babysit', 'stroller', 'strollers', 'crib', 'cribs', 'nap', 'napping',
        'playground', 'daycare', 'grandparents', 'sleepover', 'bedtime',
        'parents', 'Jokūbas', 'Jokubas', 'Arianna', 'Ettore'
    ]
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in target_words) + r')\b'
    regex = re.compile(pattern, re.IGNORECASE)
    
    couples_data['mentions'] = couples_data['message'].str.findall(regex).str.len().fillna(0)
    
    mentions_data = couples_data[couples_data['mentions'] > 0].copy()
    
    couples_data['quarter'] = couples_data[settings.time_col].dt.to_period('Q')
    mentions_data['quarter'] = mentions_data[settings.time_col].dt.to_period('Q')
    quarter_mentions = mentions_data.groupby(['quarter', 'couples'])['mentions'].sum().reset_index()
    
    all_quarters_in_data = sorted(couples_data['quarter'].unique())
    all_quarters = all_quarters_in_data
    
    actual_data_quarters = sorted(couples_data['quarter'].unique())
    no_children_data = quarter_mentions[quarter_mentions['couples'] == 'couple1'].set_index('quarter')['mentions'].reindex(actual_data_quarters, fill_value=0)
    
    with_children_couples = quarter_mentions[quarter_mentions['couples'].isin(['couple2', 'couple3'])]
    with_children_data = with_children_couples.groupby('quarter')['mentions'].sum().reindex(actual_data_quarters, fill_value=0) / 2
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    x_numeric_data = np.arange(len(actual_data_quarters))
    x_numeric_all = np.arange(len(all_quarters))
    
    no_children_color = '#1E3A8A'    # Dark blue (no children)
    with_children_color = '#FF69B4'  # Pink (with children)
    
    if len(actual_data_quarters) >= 3:
        x_smooth = np.linspace(x_numeric_data.min(), x_numeric_data.max(), 300)
        
        no_children_smooth = make_interp_spline(x_numeric_data, no_children_data.values, k=min(3, len(actual_data_quarters)-1))
        no_children_y_smooth = no_children_smooth(x_smooth)
        
        with_children_smooth = make_interp_spline(x_numeric_data, with_children_data.values, k=min(3, len(actual_data_quarters)-1))
        with_children_y_smooth = with_children_smooth(x_smooth)
        
        data_start_idx = all_quarters.index(actual_data_quarters[0])
        ax.plot(x_smooth + data_start_idx, no_children_y_smooth, color=no_children_color, linewidth=3, alpha=0.9, label='No children')
        ax.plot(x_smooth + data_start_idx, with_children_y_smooth, color=with_children_color, linewidth=3, alpha=0.9, label='With children')
        
        data_positions = [all_quarters.index(q) for q in actual_data_quarters]
        ax.scatter(data_positions, no_children_data.values, color=no_children_color, s=60, zorder=5, alpha=0.8)
        ax.scatter(data_positions, with_children_data.values, color=with_children_color, s=60, zorder=5, alpha=0.8)
    else:
        data_positions = [all_quarters.index(q) for q in actual_data_quarters]
        ax.plot(data_positions, no_children_data.values, color=no_children_color, linewidth=3, marker='o', markersize=8, label='No children')
        ax.plot(data_positions, with_children_data.values, color=with_children_color, linewidth=3, marker='o', markersize=8, label='With children')
    
    baby_consumated_1_date = pd.Period('2023-Q1')  # Q1 2023 (February 2023, ~9 months before birth)
    baby_born_2_date = pd.Period('2023-Q4')  # Q4 2023 (November 22, 2023)
    baby_consumated_2_date = pd.Period('2025-Q1')  # Q1 2025 (March 2025)
    
    reference_lines = [
        (baby_consumated_1_date, 'Baby Consumated', '#FF6B6B', '--'),
        (baby_born_2_date, 'Baby Born', '#4ECDC4', '-'),
        (baby_consumated_2_date, 'Baby Consumated', '#FF6B6B', '--')
    ]
    
    for ref_date, label, color, linestyle in reference_lines:
        if ref_date in all_quarters:
            x_pos = list(all_quarters).index(ref_date)
            ax.axvline(x=x_pos, color=color, linestyle=linestyle, linewidth=2, alpha=0.7, zorder=3)
            ax.text(x_pos, ax.get_ylim()[1] * 0.95, label, rotation=90, 
                   ha='right', va='top', fontsize=10, color=color, fontweight='bold')
    
    ax.set_xlabel('')
    ax.set_ylabel('Baby/Child talk', fontsize=14, fontweight='bold')
    ax.set_title('Childfree couples show similar child talk patterns', 
                 fontsize=24, fontweight='bold', pad=20, loc='left')
    
    ax.set_xticks(x_numeric_all)
    ax.set_xticklabels([str(q) for q in all_quarters], rotation=45, ha='right')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#FF69B4', linewidth=3, label='With children'),
        Line2D([0], [0], color='#1E3A8A', linewidth=3, label='No children')
    ]
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), 
               frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    plt.tight_layout()
    
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    output_path = img_dir / 'time_series_2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return output_path


def plot_time_series_combined(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    """Create a time series chart showing combined mentions of child/family-related words over time.
    
    - X-axis: Time periods (quarters)
    - Y-axis: Number of mentions
    - One smooth line: Combined men and women mentions
    """
    if settings is None:
        settings = load_settings()
    if df is None:
        df = _load_processed_df(settings)

    df[settings.time_col] = pd.to_datetime(df[settings.time_col])
    couples_data = df[df['couples'].isin(settings.couples)].copy()
    if couples_data.empty:
        raise ValueError("No couple data found in dataframe.")
    
    target_words = [
        'baby', 'child', 'children', 'kids', 'play', 'school', 'toys',
        'babysit', 'stroller', 'strollers', 'crib', 'cribs', 'nap', 'napping',
        'playground', 'daycare', 'grandparents', 'sleepover', 'bedtime',
        'parents', 'Jokūbas', 'Jokubas', 'Arianna', 'Ettore'
    ]
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in target_words) + r')\b'
    regex = re.compile(pattern, re.IGNORECASE)
    
    couples_data['mentions'] = couples_data['message'].str.findall(regex).str.len().fillna(0)
    
    mentions_data = couples_data[couples_data['mentions'] > 0].copy()
    
    couples_data['quarter'] = couples_data[settings.time_col].dt.to_period('Q')
    mentions_data['quarter'] = mentions_data[settings.time_col].dt.to_period('Q')
    quarter_mentions = mentions_data.groupby(['quarter'])['mentions'].sum().reset_index()
    
    all_quarters_in_data = sorted(couples_data['quarter'].unique())
    all_quarters = all_quarters_in_data
    
    actual_data_quarters = sorted(couples_data['quarter'].unique())
    combined_data = quarter_mentions.set_index('quarter')['mentions'].reindex(actual_data_quarters, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    x_numeric_data = np.arange(len(actual_data_quarters))
    x_numeric_all = np.arange(len(all_quarters))
    
    combined_color = '#8B5CF6'  # Purple
    
    if len(actual_data_quarters) >= 3:
        x_smooth = np.linspace(x_numeric_data.min(), x_numeric_data.max(), 300)
        
        combined_smooth = make_interp_spline(x_numeric_data, combined_data.values, k=min(3, len(actual_data_quarters)-1))
        combined_y_smooth = combined_smooth(x_smooth)
        
        data_start_idx = all_quarters.index(actual_data_quarters[0])
        ax.plot(x_smooth + data_start_idx, combined_y_smooth, color=combined_color, linewidth=3, alpha=0.9, label='All couples')
        
        data_positions = [all_quarters.index(q) for q in actual_data_quarters]
        ax.scatter(data_positions, combined_data.values, color=combined_color, s=60, zorder=5, alpha=0.8)
    else:
        data_positions = [all_quarters.index(q) for q in actual_data_quarters]
        ax.plot(data_positions, combined_data.values, color=combined_color, linewidth=3, marker='o', markersize=8, label='All couples')
    
    baby_consumated_1_date = pd.Period('2023-Q1')  # Q1 2023 (February 2023, ~9 months before birth)
    baby_born_2_date = pd.Period('2023-Q4')  # Q4 2023 (November 22, 2023)
    baby_consumated_2_date = pd.Period('2025-Q1')  # Q1 2025 (March 2025)
    
    reference_lines = [
        (baby_consumated_1_date, 'Baby Consumated', '#FF6B6B', '--'),
        (baby_born_2_date, 'Baby Born', '#4ECDC4', '-'),
        (baby_consumated_2_date, 'Baby Consumated', '#FF6B6B', '--')
    ]
    
    for ref_date, label, color, linestyle in reference_lines:
        if ref_date in all_quarters:
            x_pos = list(all_quarters).index(ref_date)
            ax.axvline(x=x_pos, color=color, linestyle=linestyle, linewidth=2, alpha=0.7, zorder=3)
            ax.text(x_pos, ax.get_ylim()[1] * 0.95, label, rotation=90, 
                   ha='right', va='top', fontsize=10, color=color, fontweight='bold')
    
    ax.set_xlabel('')
    ax.set_ylabel('Baby/Child talk', fontsize=14, fontweight='bold')
    ax.set_title('Absent baby/child talk strong indicator of upcomming pregnancy', 
                 fontsize=24, fontweight='bold', pad=20, loc='left')
    
    ax.set_xticks(x_numeric_all)
    ax.set_xticklabels([str(q) for q in all_quarters], rotation=45, ha='right')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    fig.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1], 
               loc='upper right', bbox_to_anchor=(0.98, 0.95), frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    plt.tight_layout()
    
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    output_path = img_dir / 'time_series_3.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    dframe = _load_processed_df()
    paths = [
        plot_gauge_couples(dframe),
        plot_timeline_couples_bars(dframe),
        plot_timeline_couples_bars_hours(dframe),
        plot_timeline_couples_columns(dframe),
        plot_time_series(dframe),
        plot_time_series_couples(dframe),
        plot_time_series_combined(dframe),
    ]
    print("Generated:")
    for p in paths:
        print(f" - {p}")
