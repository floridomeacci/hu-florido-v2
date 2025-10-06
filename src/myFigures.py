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
from scipy.stats import skewnorm, poisson, lognorm


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
    
    hour_bins = list(range(24))
    
    male_percentages = []
    female_percentages = []
    
    for hour_bin in hour_bins:
        bin_data = couples_data[couples_data['hour'] == hour_bin]
        
        male_count = len(bin_data[bin_data['gender_mapped'] == 'male'])
        female_count = len(bin_data[bin_data['gender_mapped'] == 'female'])
        
        male_pct = (male_count / total_messages) * 100 if total_messages > 0 else 0
        female_pct = (female_count / total_messages) * 100 if total_messages > 0 else 0
        
        male_percentages.append(male_pct)
        female_percentages.append(female_pct)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    gender_colors = {
        'male': '#1E3A8A',
        'female': '#FF69B4'
    }
    
    gender_labels = {
        'male': 'Men',
        'female': 'Women'
    }
    
    ax.plot(hour_bins, male_percentages, 
            color=gender_colors['male'], 
            linewidth=3, 
            marker='o', 
            markersize=6,
            label=gender_labels['male'],
            alpha=0.8)
    
    ax.plot(hour_bins, female_percentages, 
            color=gender_colors['female'], 
            linewidth=3, 
            marker='o', 
            markersize=6,
            label=gender_labels['female'],
            alpha=0.8)
    
    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('% of Total Messages', fontsize=14)
    fig.suptitle('Men vs Women\nHour Distribution', 
                 fontsize=32, fontweight='bold', x=0.06, y=0.95, ha='left')
    
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{hour}h" for hour in range(0, 24, 2)])
    
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
    out = img_dir / 'distribution_categories_2.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return out


def distribution_categories_3(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    if settings is None:
        settings = load_settings()
    if df is None:
        df = _load_processed_df(settings)

    couples_data = df[df['couples'].isin(settings.couples)].copy()
    couples_data = couples_data[couples_data['message'] != '<Media omitted>']
    couples_data[settings.time_col] = pd.to_datetime(couples_data[settings.time_col])
    couples_data['hour'] = couples_data[settings.time_col].dt.hour
    couples_data['gender_mapped'] = couples_data['gender'].map({'male': 'male', 'female': 'female', 'M': 'male', 'F': 'female'})
    couples_data = couples_data.dropna(subset=['gender_mapped'])

    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(14, 8))

    gender_colors = {'male': '#1E3A8A', 'female': '#FF69B4'}
    gender_labels = {'male': 'Men', 'female': 'Women'}

    counts_male = np.array([(couples_data[(couples_data['hour'] == h) & (couples_data['gender_mapped'] == 'male')].shape[0]) for h in hours], dtype=float)
    counts_female = np.array([(couples_data[(couples_data['hour'] == h) & (couples_data['gender_mapped'] == 'female')].shape[0]) for h in hours], dtype=float)

    total_messages = len(couples_data)
    male_percentages = (counts_male / total_messages) * 100.0 if total_messages > 0 else np.zeros_like(counts_male)
    female_percentages = (counts_female / total_messages) * 100.0 if total_messages > 0 else np.zeros_like(counts_female)

    total_male = counts_male.sum()
    total_female = counts_female.sum()
    lam_male = float((hours * counts_male).sum() / total_male) if total_male > 0 else float('nan')
    lam_female = float((hours * counts_female).sum() / total_female) if total_female > 0 else float('nan')
    male_total_pct = float(male_percentages.sum())
    female_total_pct = float(female_percentages.sum())

    def scaled_hour_poisson(lam: float, total_pct: float) -> np.ndarray:
        if not np.isfinite(lam) or lam <= 0:
            return np.zeros_like(hours, dtype=float)
        pmf = poisson.pmf(hours, mu=lam)
        return (pmf / pmf.sum() * total_pct) if pmf.sum() > 0 else np.zeros_like(hours, dtype=float)

    male_fit = scaled_hour_poisson(lam_male, male_total_pct)
    female_fit = scaled_hour_poisson(lam_female, female_total_pct)

    label_m = f"Men (λ={lam_male:.2f}h)" if np.isfinite(lam_male) else "Men (λ=–)"
    label_f = f"Women (λ={lam_female:.2f}h)" if np.isfinite(lam_female) else "Women (λ=–)"

    ax.plot(hours, male_fit, color=gender_colors['male'], linewidth=3, marker='o', markersize=6, label=label_m, alpha=0.9)
    ax.plot(hours, female_fit, color=gender_colors['female'], linewidth=3, marker='o', markersize=6, label=label_f, alpha=0.9)

    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('% of Total Messages (Poisson fit)', fontsize=14)
    fig.suptitle('Men vs Women\nHour Distribution (Poisson fit)', fontsize=32, fontweight='bold', x=0.06, y=0.95, ha='left')

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{hour}h" for hour in range(0, 24, 2)])

    max_pct = max(male_fit.max() if male_fit.size else 0, female_fit.max() if female_fit.size else 0)
    ax.set_ylim(0, max_pct * 1.1 if max_pct > 0 else 1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.15), frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    out = img_dir / 'distribution_categories_3.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    try:
        print(f"distribution_categories_3 lambdas: Men={lam_male:.3f}h, Women={lam_female:.3f}h")
    except Exception:
        pass
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



if __name__ == "__main__":
    dframe = _load_processed_df()
    paths = [
        comparing_categories_1(dframe),
        distribution_categories_2(dframe),
    distribution_categories_3(dframe),
        distribution_categories_1(dframe),
        comparing_categories_2(dframe),
        time_series(dframe),
        time_series_2(dframe),
        time_series_3(dframe),
    ]
    print("Generated:")
    for p in paths:
        print(f" - {p}")
