import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import tomllib
import re
import math
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


def comparing_categories_1(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
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
            ax.text(1.3 * np.cos(ang), 1.3 * np.sin(ang), f"{a_name}\n{a_count}\n({a_pct:.1%})",
                    ha="center", va="center", fontsize=10, color=a_color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        if b_pct > 0:
            ang = a_pct * np.pi + (b_pct * np.pi / 2)
            ax.text(1.3 * np.cos(ang), 1.3 * np.sin(ang), f"{b_name}\n{b_count}\n({b_pct:.1%})",
                    ha="center", va="center", fontsize=10, color=b_color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.text(0, -0.2, f"{couple_label}\n{total} messages", ha="center", va="center", fontsize=12, fontweight="bold")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

    n_couples = len(settings.couples) if settings.couples else 0
    cols = min(3, max(1, n_couples))
    rows = math.ceil(n_couples / cols) if n_couples > 0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    fig.suptitle('One Voice Always Dominates\nPer Couple', fontsize=32, fontweight='bold', y=0.95)
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#1E3A8A', alpha=0.8, label='Men'),
        Rectangle((0, 0), 1, 1, facecolor='#FF69B4', alpha=0.8, label='Women')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88),
               title_fontsize=12, fontsize=10, frameon=True, fancybox=True, shadow=True)

    for i, couple_id in enumerate(settings.couples):
        ax = axes_flat[i] if i < len(axes_flat) else None
        if ax is None:
            continue
        subset = author_counts[author_counts["couples"] == couple_id]

        male_total = int(subset[subset['gender'] == 'male']['message_count'].sum())
        female_total = int(subset[subset['gender'] == 'female']['message_count'].sum())

        # If we have no gender-aggregated counts, fall back to top-two authors
        if male_total + female_total == 0:
            if len(subset) >= 2:
                pair = subset.nlargest(2, "message_count")
                a, b = pair.iloc[0], pair.iloc[1]
                a_name, b_name = a["author"], b["author"]
                a_count, b_count = a["message_count"], b["message_count"]
                a_gender, b_gender = a.get("gender", "unknown"), b.get("gender", "unknown")
                male_total = a_count if a_gender == 'male' else 0
                female_total = a_count if a_gender == 'female' else 0
                male_total += b_count if b_gender == 'male' else 0
                female_total += b_count if b_gender == 'female' else 0
            elif len(subset) == 1:
                a = subset.iloc[0]
                a_name, a_count = a["author"], a["message_count"]
                a_gender = a.get("gender", "unknown")
                if a_gender == 'male':
                    male_total = int(a_count)
                elif a_gender == 'female':
                    female_total = int(a_count)
            else:
                male_total = female_total = 0

        colors = couple_shades.get(f"Couple{i+1}", {"male": "#1E3A8A", "female": "#FF69B4"})
        _draw_gauge(ax, male_total, female_total, 'Men', 'Women', 'male', 'female', f"Couple {i+1}", colors)

    # Hide any leftover axes
    for j in range(len(settings.couples), len(axes_flat)):
        try:
            axes_flat[j].axis('off')
        except Exception:
            pass

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    root = REPO_ROOT
    img_dir = root / "img"
    img_dir.mkdir(exist_ok=True)
    output_path = img_dir / "comparing_categories_1.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return output_path


def distribution_categories_2(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
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
    
    hour_bins = np.arange(24)
    
    male_percentages = []
    female_percentages = []
    total_counts = []

    total_male_messages = couples_data[couples_data['gender_mapped'] == 'male'].shape[0]
    total_female_messages = couples_data[couples_data['gender_mapped'] == 'female'].shape[0]
    
    for hour_bin in hour_bins:
        bin_data = couples_data[couples_data['hour'] == hour_bin]

        male_count = len(bin_data[bin_data['gender_mapped'] == 'male'])
        female_count = len(bin_data[bin_data['gender_mapped'] == 'female'])

        male_pct = (male_count / total_male_messages) * 100 if total_male_messages > 0 else 0
        female_pct = (female_count / total_female_messages) * 100 if total_female_messages > 0 else 0

        male_percentages.append(male_pct)
        female_percentages.append(female_pct)
        total_counts.append(male_count + female_count)

    male_series = np.array(male_percentages, dtype=float)
    female_series = np.array(female_percentages, dtype=float)

    extended_hours = np.concatenate([hour_bins, hour_bins + 24])
    extended_male = np.concatenate([male_series, male_series])
    extended_female = np.concatenate([female_series, female_series])

    male_spline = make_interp_spline(extended_hours, extended_male, k=3)
    female_spline = make_interp_spline(extended_hours, extended_female, k=3)

    smooth_hours = np.linspace(2, 24, 500)
    male_smoothed = male_spline(smooth_hours)
    female_smoothed = female_spline(smooth_hours)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    gender_colors = {
        'male': '#1E3A8A',
        'female': '#FF69B4'
    }
    
    gender_labels = {
        'male': 'Men',
        'female': 'Women'
    }
    
    ax.plot(smooth_hours, male_smoothed, 
            color=gender_colors['male'], 
            linewidth=3, 
            label=gender_labels['male'],
            alpha=0.8)
    
    ax.plot(smooth_hours, female_smoothed, 
        color=gender_colors['female'], 
        linewidth=3, 
        label=gender_labels['female'],
        alpha=0.8)
    
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Messages', fontsize=14)
    fig.suptitle('Who texts when?',
                 fontsize=32, fontweight='bold', x=0.06, y=0.95, ha='left')
    
    ax.set_xlim(2, 24)
    xticks = list(range(2, 25, 2))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{hour % 24}h" if hour < 24 else "0h" for hour in xticks])
    
    max_pct = max(float(np.max(male_smoothed)) if male_smoothed.size else 0, float(np.max(female_smoothed)) if female_smoothed.size else 0)
    min_pct = min(float(np.min(male_smoothed)) if male_smoothed.size else 0, float(np.min(female_smoothed)) if female_smoothed.size else 0)
    span = max(max_pct - min_pct, 1.0)
    lower_bound = max(-0.5, min_pct - span * 0.12)
    upper_bound = max_pct + span * 0.12
    ax.set_ylim(lower_bound, upper_bound)

    def _display_hour(hour: int) -> float:
        return float(hour if hour >= 2 else hour + 24)

    counts_array = np.array(total_counts, dtype=float)
    day_hours = [h for h in range(24) if 8 <= h < 20]
    night_hours = [h for h in range(24) if h not in day_hours]

    def _peak_hour(candidates: list[int]) -> int:
        valid = [h for h in candidates if h < counts_array.size]
        if not valid:
            return int(np.argmax(counts_array)) if counts_array.size else 0
        peak = max(valid, key=lambda h: counts_array[h])
        if counts_array[peak] == 0 and counts_array.sum() > 0:
            non_zero = [h for h in valid if counts_array[h] > 0]
            if non_zero:
                peak = max(non_zero, key=lambda h: counts_array[h])
        return peak

    day_peak_hour = _peak_hour(day_hours)
    night_peak_hour = _peak_hour(night_hours)

    def _peak_value(hour: int) -> float:
        x_val = _display_hour(hour)
        values = []
        try:
            values.append(float(male_spline(x_val)))
        except Exception:
            pass
        try:
            values.append(float(female_spline(x_val)))
        except Exception:
            pass
        if not values:
            return float(max_pct)
        return max(values)

    def _annotate_peak(hour: int, label: str, color: str) -> None:
        y_val = _peak_value(hour)
        ax.axhline(y=y_val, color=color, linestyle='--', linewidth=1.8, alpha=0.6)
        text_x = ax.get_xlim()[1] - 0.4
        text_y = min(upper_bound - span * 0.03, y_val + span * 0.04)
        ax.text(
            text_x,
            text_y,
            label,
            color=color,
            fontsize=12,
            fontweight='semibold',
            va='bottom',
            ha='right',
            backgroundcolor='white',
            alpha=0.85
        )

    _annotate_peak(day_peak_hour, 'Men rule the day', gender_colors['male'])
    _annotate_peak(night_peak_hour, 'Women rule the night', gender_colors['female'])
    
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 1.02), frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
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


def distribution_categories_4(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    if settings is None:
        settings = load_settings()
    if df is None:
        df = _load_processed_df(settings)

    df[settings.time_col] = pd.to_datetime(df[settings.time_col])
    data = df.copy()
    data = data[data['message'] != '<Media omitted>']
    if 'gender' not in data.columns:
        raise ValueError('Dataframe must contain gender column for distribution_categories_4')

    data['weekday'] = data[settings.time_col].dt.dayofweek

    weekdays = data[data['weekday'] < 5]
    weekends = data[data['weekday'] >= 5]

    def _percentages(segment: pd.DataFrame) -> tuple[float, float]:
        if segment.empty:
            return 0.0, 0.0
        genders = segment['gender'].astype(str).str.lower()
        male_count = genders.str.startswith('m').sum()
        female_count = genders.str.startswith('f').sum()
        total = male_count + female_count
        if total == 0:
            return 0.0, 0.0
        return male_count * 100.0 / total, female_count * 100.0 / total

    male_weekday, female_weekday = _percentages(weekdays)
    male_weekend, female_weekend = _percentages(weekends)

    labels = ['Weekdays', 'Weekends']
    male_vals = [male_weekday, male_weekend]
    female_vals = [female_weekday, female_weekend]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#fafafa')
    male_color = '#1E3A8A'
    female_color = '#FF69B4'

    x = np.arange(len(labels))
    width = 0.35

    bars_m = ax.bar(x - width / 2, male_vals, width, color=male_color, label='Men')
    bars_f = ax.bar(x + width / 2, female_vals, width, color=female_color, label='Women')

    for rect in bars_m:
        height = rect.get_height()
        if height > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height * 0.5,
                f"{height:.0f}%",
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold',
                color='white',
            )
    for rect in bars_f:
        height = rect.get_height()
        if height > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height * 0.5,
                f"{height:.0f}%",
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold',
                color='white',
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlim(-0.4, len(labels) - 0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel('')
    ax.set_yticks([])
    fig.suptitle('Women contribute more to chat\nover the weekend.', fontsize=32, fontweight='bold', x=0.05, y=0.99, ha='left')

    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0.3)
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    out = img_dir / 'distribution_categories_4.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return out
def distribution_categories_1(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    if settings is None:
        settings = load_settings()
    if df is None:
        df = _load_processed_df(settings)

    couples_data = df[df['couples'].isin(settings.couples)].copy()
    if couples_data.empty:
        raise ValueError("No couple data found in dataframe.")
    
    couples_data = couples_data[couples_data['message'] != '<Media omitted>']
    
    couples_data[settings.time_col] = pd.to_datetime(couples_data[settings.time_col])
    couples_data['weekday'] = couples_data[settings.time_col].dt.dayofweek
    
    couples_data['gender_mapped'] = couples_data['gender'].map({
        'male': 'male',
        'female': 'female',
        'M': 'male',
        'F': 'female'
    })
    
    couples_data = couples_data.dropna(subset=['gender_mapped'])
    
    total_messages = len(couples_data)
    
    weekday_bins = list(range(7))
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    male_percentages = []
    female_percentages = []
    
    for weekday_bin in weekday_bins:
        bin_data = couples_data[couples_data['weekday'] == weekday_bin]
        
        male_count = len(bin_data[bin_data['gender_mapped'] == 'male'])
        female_count = len(bin_data[bin_data['gender_mapped'] == 'female'])
        
        male_pct = (male_count / total_messages) * 100 if total_messages > 0 else 0
        female_pct = (female_count / total_messages) * 100 if total_messages > 0 else 0
        
        male_percentages.append(male_pct)
        female_percentages.append(female_pct)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('#fafafa')
    gender_colors = {
        'male': '#1E3A8A',
        'female': '#FF69B4'
    }
    
    gender_labels = {
        'male': 'Men',
        'female': 'Women'
    }
    
    x_positions = np.arange(len(weekday_names))
    bar_width = 0.35
    
    ax.bar(x_positions - bar_width/2, male_percentages, 
           bar_width, label=gender_labels['male'], 
           color=gender_colors['male'], alpha=1.0, edgecolor='none')
    
    ax.bar(x_positions + bar_width/2, female_percentages, 
           bar_width, label=gender_labels['female'], 
           color=gender_colors['female'], alpha=1.0, edgecolor='none')
    
    ax.set_xlabel('Day of Week', fontsize=14)
    ax.set_ylabel('% of Total Messages', fontsize=14)
    fig.suptitle('Women Socialize More\nThan Men Over Weekends', 
                 fontsize=32, fontweight='bold', x=0.06, y=0.95, ha='left')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(weekday_names)
    
    max_pct = max(max(male_percentages), max(female_percentages))
    ax.set_ylim(0, max_pct * 1.1)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.15), frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    out = img_dir / 'distribution_categories_1.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return out


    


def comparing_categories_2(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
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


def time_series(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
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
    
    men_color = '#1E3A8A'
    women_color = '#FF69B4'
    
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
    
    baby_consumated_1_date = pd.Period('2023-Q1')
    baby_born_2_date = pd.Period('2023-Q4')
    baby_consumated_2_date = pd.Period('2025-Q1')
    
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


def time_series_2(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
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
    
    no_children_color = '#1E3A8A'
    with_children_color = '#FF69B4'
    
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
    
    baby_consumated_1_date = pd.Period('2023-Q1')
    baby_born_2_date = pd.Period('2023-Q4')
    baby_consumated_2_date = pd.Period('2025-Q1')
    
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


def time_series_3(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
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
    
    combined_color = '#8B5CF6'
    
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
    
    baby_consumated_1_date = pd.Period('2023-Q1')
    baby_born_2_date = pd.Period('2023-Q4')
    baby_consumated_2_date = pd.Period('2025-Q1')
    
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


def distribution_categories_5(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    """Slope chart showing change in gender share from weekdays to weekends."""
    if settings is None:
        settings = load_settings()
    if df is None:
        df = _load_processed_df(settings)

    required_cols = {'message', 'gender', settings.time_col}
    missing = required_cols - set(df.columns)
    if missing:
        missing_list = ', '.join(sorted(missing))
        raise ValueError(f"Dataframe must contain columns: {missing_list}")

    data = df.copy()
    data = data[data['message'] != '<Media omitted>']
    data = data[data['gender'].notna()]
    if data.empty:
        raise ValueError('No gender-labelled textual messages available for distribution_categories_5')

    data[settings.time_col] = pd.to_datetime(data[settings.time_col])
    data['weekday'] = data[settings.time_col].dt.dayofweek

    def _share_for_segment(segment: pd.DataFrame) -> tuple[float, float]:
        if segment.empty:
            return 0.0, 0.0
        genders = segment['gender'].astype(str).str.lower()
        male_count = genders.str.startswith('m').sum()
        female_count = genders.str.startswith('f').sum()
        total = male_count + female_count
        if total == 0:
            return 0.0, 0.0
        male_share = male_count * 100.0 / total
        female_share = female_count * 100.0 / total
        return male_share, female_share

    weekday_segment = data[data['weekday'] < 5]
    weekend_segment = data[data['weekday'] >= 5]

    male_weekday, female_weekday = _share_for_segment(weekday_segment)
    male_weekend, female_weekend = _share_for_segment(weekend_segment)

    periods = ['Weekdays', 'Weekends']
    x_positions = np.arange(len(periods))

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F9FAFB')

    male_values = np.array([male_weekday, male_weekend])
    female_values = np.array([female_weekday, female_weekend])

    male_color = '#1E3A8A'
    female_color = '#FF69B4'

    ax.plot(x_positions, male_values, color=male_color, linewidth=3.2, marker='o', markersize=12, label='Men')
    ax.plot(x_positions, female_values, color=female_color, linewidth=3.2, marker='o', markersize=12, label='Women')

    for x, y in zip(x_positions, male_values):
        ax.text(
            x - 0.06,
            min(69, y + 1.6),
            f"{y:.0f}%",
            color=male_color,
            fontsize=12,
            fontweight='bold',
            ha='right'
        )
    for x, y in zip(x_positions, female_values):
        ax.text(
            x + 0.06,
            max(31, y - 2.8),
            f"{y:.0f}%",
            color=female_color,
            fontsize=12,
            fontweight='bold',
            ha='left'
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(periods, fontsize=13, fontweight='bold')
    ax.set_xlim(-0.15, len(periods) - 0.85)
    ax.set_ylim(30, 70)
    ax.set_ylabel('')

    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}%"))
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title('Weekends flip who\nspeaks more', fontsize=28, fontweight='bold', loc='left', pad=18)

    plt.tight_layout(pad=0.6)
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    out = img_dir / 'distribution_categories_5.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return out



if __name__ == "__main__":
    dframe = _load_processed_df()
    paths = [
        comparing_categories_1(dframe),
        distribution_categories_2(dframe),
        distribution_categories_4(dframe),
        distribution_categories_5(dframe),
        distribution_categories_1(dframe),
        comparing_categories_2(dframe),
        time_series(dframe),
        time_series_2(dframe),
        time_series_3(dframe),
    ]
    print("Generated:")
    for p in paths:
        print(f" - {p}")
