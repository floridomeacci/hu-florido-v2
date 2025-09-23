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


# Paths
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
        "Couple1": {"light": "#FDBE85", "dark": "#E6550D"},
        "Couple2": {"light": "#9ECAE1", "dark": "#3182BD"},
        "Couple3": {"light": "#A1D99B", "dark": "#31A354"},
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

    # 1) Common defaults
    candidates: list[Path] = [
        processed_dir / "output.csv",
        processed_dir / "output.parq",
    ]

    # 2) config.toml (processed/current)
    if settings.current:
        current_path = processed_dir / settings.current if not Path(settings.current).is_absolute() else Path(settings.current)
        candidates.insert(0, current_path)

    # 3) Try candidates in order
    for p in candidates:
        if p and isinstance(p, Path) and p.exists() and p.is_file():
            if p.suffix.lower() in {".parq", ".parquet"}:
                return pd.read_parquet(p)
            return pd.read_csv(p)

    # 4) Fallback: newest file in processed dir
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
    """Render the three semicircular gauge charts and save to img/gauge_charts.png.

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

    # Helper: draw gauge on a provided axis
    def _draw_gauge(ax, a_count, b_count, a_name, b_name, a_gender, b_gender, couple_label, colors):
        total = a_count + b_count
        a_pct = (a_count / total) if total > 0 else 0.5
        b_pct = (b_count / total) if total > 0 else 0.5

        ax.clear()
        # Background semi-annular band
        ax.add_patch(Wedge((0, 0), 1.0, 0, 180, width=0.2, facecolor="#E0E0E0", edgecolor='none', alpha=0.3))

        a_color = colors["dark"] if a_gender == "male" else colors["light"]
        b_color = colors["dark"] if b_gender == "male" else colors["light"]

        # Annular wedges using Wedge (avoids mask/artifact lines)
        ax.add_patch(Wedge((0, 0), 1.0, 0, a_pct * 180, width=0.2, facecolor=a_color, edgecolor='none', alpha=0.8))
        ax.add_patch(Wedge((0, 0), 1.0, a_pct * 180, 180, width=0.2, facecolor=b_color, edgecolor='none', alpha=0.8))

        angle = a_pct * np.pi
        ax.plot([0, 0.9 * np.cos(angle)], [0, 0.9 * np.sin(angle)], color="black", linewidth=3, alpha=0.8)
        ax.plot(0, 0, "ko", markersize=8)

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

        for t in [0, 0.25, 0.5, 0.75, 1.0]:
            ang = t * np.pi
            ax.plot([np.cos(ang), 1.05 * np.cos(ang)], [np.sin(ang), 1.05 * np.sin(ang)], 'k-', alpha=0.3)

        ax.text(0, -0.2, f"{couple_label}\n{total} messages", ha="center", va="center", fontsize=12, fontweight="bold")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('One Voice Always Dominates\nPer Couple', fontsize=32, fontweight='bold', y=0.95)

    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#333333', alpha=0.8, label='Male (Dark)'),
        Rectangle((0, 0), 1, 1, facecolor='#CCCCCC', alpha=0.8, label='Female (Light)')
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
    output_path = img_dir / "gauge_charts.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return output_path


def plot_timeline_couples_bars(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    """Render a diverging bar per quarter with each side occupying half the total width.

    - Left side (negative x): male segments, stacked by absolute message counts
    - Right side (positive x): female segments, stacked by absolute message counts
    - Each side's full span equals half of the global max total messages in any quarter, with a minimum of 250 for breathing room
    - No numbers inside bars; center reference line at x=0
    """
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

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    couple_colors = {
        'couple1_male': '#E6550D',
        'couple1_female': '#FD8D3C',
        'couple2_male': '#3182BD',
        'couple2_female': '#6BAED6',
        'couple3_male': '#31A354',
        'couple3_female': '#74C476'
    }
    couple_labels = {
        'couple1_male': 'Couple 1 - Male',
        'couple1_female': 'Couple 1 - Female',
        'couple2_male': 'Couple 2 - Male',
        'couple2_female': 'Couple 2 - Female',
        'couple3_male': 'Couple 3 - Male',
        'couple3_female': 'Couple 3 - Female'
    }

    # Fixed couple order; we split by gender into two sides
    couples_order = ['couple1', 'couple2', 'couple3']

    all_quarters = sorted(quarter_counts['quarter'].unique())
    quarter_positions = range(len(all_quarters))
    bar_height = 1.0

    # First pass: compute per-quarter totals and global max total to set symmetric x-limits
    quarter_totals: Dict[pd.Period, int] = {}
    max_total = 0
    for quarter in all_quarters:
        qd = quarter_counts[quarter_counts['quarter'] == quarter]
        male_total = qd[qd['gender'] == 'male']['message_count'].sum()
        female_total = qd[qd['gender'] == 'female']['message_count'].sum()
        tot = int(male_total + female_total)
        quarter_totals[quarter] = tot
        if tot > max_total:
            max_total = tot

    # Half-span per side: ensure at least 250 for breathing room
    half_span = max(250, max_total / 2 if max_total > 0 else 1)

    def _fmt_tick(val: float) -> str:
        v = abs(int(round(val)))
        if v >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if v >= 1_000:
            return f"{v/1_000:.1f}k"
        return f"{v}"

    for qi, quarter in enumerate(all_quarters):
        qd = quarter_counts[quarter_counts['quarter'] == quarter]
        # Compute counts per (couple, gender)
        male_pairs: List[tuple[str, int]] = []  # (couple, count)
        female_pairs: List[tuple[str, int]] = []
        for cpl in couples_order:
            m = int(qd[(qd['couples'] == cpl) & (qd['gender'] == 'male')]['message_count'].sum())
            f = int(qd[(qd['couples'] == cpl) & (qd['gender'] == 'female')]['message_count'].sum())
            male_pairs.append((cpl, m))
            female_pairs.append((cpl, f))

        # Sort so the largest segments sit closest to the center (by absolute counts)
        male_pairs.sort(key=lambda x: x[1], reverse=True)
        female_pairs.sort(key=lambda x: x[1], reverse=True)

        # Stack male from center to the left using absolute counts
        cum = 0
        for cpl, c_abs in male_pairs:
            if c_abs <= 0:
                continue
            key = f"{cpl}_male"
            label = couple_labels[key] if qi == 0 else ""
            left = -cum - c_abs
            ax.barh(qi, c_abs, height=bar_height, left=left, color=couple_colors[key], alpha=1.0, edgecolor='none', linewidth=0, label=label)
            cum += c_abs

        # Stack female from center to the right using absolute counts
        cum = 0
        for cpl, c_abs in female_pairs:
            if c_abs <= 0:
                continue
            key = f"{cpl}_female"
            label = couple_labels[key] if qi == 0 else ""
            left = cum
            ax.barh(qi, c_abs, height=bar_height, left=left, color=couple_colors[key], alpha=1.0, edgecolor='none', linewidth=0, label=label)
            cum += c_abs

    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.suptitle('Male vs Female messages per quarter (diverging counts, equal half-width)', fontsize=28, fontweight='bold', x=0.06, y=0.95, ha='left')
    ax.set_yticks(list(quarter_positions))
    ax.set_yticklabels([str(q) for q in all_quarters])
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.15), frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=2)
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Diverging scale centered at 0 with equal halves in absolute counts (min ±250)
    ax.set_xlim(-half_span, half_span)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.7, zorder=5)
    ax.set_xticks([-half_span, -half_span/2, 0, half_span/2, half_span])
    ax.set_xticklabels([_fmt_tick(-half_span), _fmt_tick(-half_span/2), '0', _fmt_tick(half_span/2), _fmt_tick(half_span)])
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    out = img_dir / 'timeline_couples_bars.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return out


def plot_timeline_couples_columns(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    """Render the 3-column 100% distributions and save to img/timeline_couples_columns.png."""
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
        'couple1_male': '#E6550D', 'couple1_female': '#FD8D3C',
        'couple2_male': '#3182BD', 'couple2_female': '#6BAED6',
        'couple3_male': '#31A354', 'couple3_female': '#74C476'
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
            legend_elements = [Rectangle((0,0),1,1, facecolor='#525252', label='Male'), Rectangle((0,0),1,1, facecolor='#969696', label='Female')]
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
    out = img_dir / 'timeline_couples_columns.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return out


if __name__ == "__main__":
    # Optional: run all figures at once
    dframe = _load_processed_df()
    paths = [
        plot_gauge_couples(dframe),
        plot_timeline_couples_bars(dframe),
        plot_timeline_couples_columns(dframe),
    ]
    print("Generated:")
    for p in paths:
        print(f" - {p}")
