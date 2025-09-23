import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

try:
    import tomllib as toml_loader
except Exception:
    try:
        import tomli as toml_loader
    except Exception:
        toml_loader = None

COUPLES = {
    "coltish-magpie": "Couple1",
    "translucent-dinosaur": "Couple1",
    "foamy-jackal": "Couple2",
    "kidding-goshawk": "Couple2",
    "scintillating-trout": "Couple3",
    "jesting-duck": "Couple3",
}


COUPLE_ORDER = ["Couple1", "Couple2", "Couple3"]
COUPLE_AUTHOR_ORDER = {
    "Couple1": ["coltish-magpie", "translucent-dinosaur"],
    "Couple2": ["foamy-jackal", "kidding-goshawk"],
    "Couple3": ["scintillating-trout", "jesting-duck"],
}

couple_shades = {
    "Couple1": {"light": "#FDBE85", "dark": "#E6550D"},
    "Couple2": {"light": "#9ECAE1", "dark": "#3182BD"},
    "Couple3": {"light": "#A1D99B", "dark": "#31A354"},
}

authors_by_couple = {c: COUPLE_AUTHOR_ORDER.get(c, []) for c in COUPLE_ORDER}
ordered_all = [a for c in COUPLE_ORDER for a in authors_by_couple.get(c, [])]

author_colors = {}
for couple in COUPLE_ORDER:
    shades = couple_shades.get(couple, {"light": "#BBBBBB", "dark": "#777777"})
    for i, a in enumerate(authors_by_couple.get(couple, [])):
        author_colors[a] = shades["dark"] if i % 2 == 0 else shades["light"]

def order_authors(author_index_like):
    seen = pd.Index(author_index_like)
    known = [a for a in ordered_all if a in seen]
    unknown = [a for a in seen if a not in author_colors]
    return known + unknown

def palette_for_authors(author_series_or_index):
    uniq = pd.Index(author_series_or_index).unique()
    return {a: author_colors.get(a, "#999999") for a in uniq}

 

COUPLE_LABELS = {"Couple1": "Couple 1", "Couple2": "Couple 2", "Couple3": "Couple 3"}

def annotate_couple_brackets(
    ax,
    y_author_order,
    *,
    x_offset=-0.02,
    label_pad=0.02,
    tick_len=0.02,
    label_side="left",
):
    labels = [t.get_text() for t in ax.get_yticklabels()]
    positions = ax.get_yticks()
    pos = {lab: y for lab, y in zip(labels, positions)}

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    x = x_offset

    for couple in COUPLE_ORDER:
        couple_authors = [a for a in COUPLE_AUTHOR_ORDER.get(couple, []) if a in pos]
        if len(couple_authors) < 2:
            continue
        y_min = pos[couple_authors[0]]
        y_max = pos[couple_authors[-1]]
        color = couple_shades[couple]["dark"]

        ax.plot([x, x], [y_min, y_max], transform=trans, clip_on=False, color=color, linewidth=2)
        ax.plot([x, x + tick_len], [y_min, y_min], transform=trans, clip_on=False, color=color, linewidth=2)
        ax.plot([x, x + tick_len], [y_max, y_max], transform=trans, clip_on=False, color=color, linewidth=2)
        if label_side == "right":
            ax.text(
                x + label_pad,
                (y_min + y_max) / 2.0,
                COUPLE_LABELS.get(couple, couple),
                transform=trans,
                va="center",
                ha="left",
                fontsize=10,
                color=color,
            )
        else:
            ax.text(
                x - label_pad,
                (y_min + y_max) / 2.0,
                COUPLE_LABELS.get(couple, couple),
                transform=trans,
                va="center",
                ha="right",
                fontsize=10,
                color=color,
            )

def load_config(root: Path) -> dict:
    if toml_loader is None:
        raise RuntimeError(
            "No TOML loader available. Use Python 3.11+ (tomllib) or install 'tomli'."
        )
    with (root / "config.toml").open("rb") as f:
        return toml_loader.load(f)

def resolve_paths():
    here = Path(__file__).resolve()
    root = here.parent.parent
    cfg = load_config(root)
    processed = root / Path(cfg["processed"])
    datafile = processed / cfg["current"]
    return root, datafile

def create_gauge_chart(ax, person_a_count, person_b_count, person_a_name, person_b_name, 
                      person_a_gender, person_b_gender, couple_name, colors):
    total = person_a_count + person_b_count
    if total == 0:
        person_a_pct = 0.5
        person_b_pct = 0.5
    else:
        person_a_pct = person_a_count / total
        person_b_pct = person_b_count / total
    
    ax.clear()
    
    theta_bg = np.linspace(0, np.pi, 100)
    x_bg = np.cos(theta_bg)
    y_bg = np.sin(theta_bg)
    ax.fill_between(x_bg, y_bg, 0, color='#E0E0E0', alpha=0.3)
    
    person_a_color = colors["dark"] if person_a_gender == "male" else colors["light"]
    person_b_color = colors["dark"] if person_b_gender == "male" else colors["light"]
    
    theta_a = np.linspace(0, person_a_pct * np.pi, 50)
    x_a = 0.8 * np.cos(theta_a)
    y_a = 0.8 * np.sin(theta_a)
    x_a_outer = np.cos(theta_a)
    y_a_outer = np.sin(theta_a)
    
    vertices_a = list(zip(x_a, y_a)) + list(zip(x_a_outer[::-1], y_a_outer[::-1]))
    person_a_patch = patches.Polygon(vertices_a, closed=True, color=person_a_color, alpha=0.8)
    ax.add_patch(person_a_patch)
    
    theta_b = np.linspace(person_a_pct * np.pi, np.pi, 50)
    x_b = 0.8 * np.cos(theta_b)
    y_b = 0.8 * np.sin(theta_b)
    x_b_outer = np.cos(theta_b)
    y_b_outer = np.sin(theta_b)
    
    vertices_b = list(zip(x_b, y_b)) + list(zip(x_b_outer[::-1], y_b_outer[::-1]))
    person_b_patch = patches.Polygon(vertices_b, closed=True, color=person_b_color, alpha=0.8)
    ax.add_patch(person_b_patch)
    
    needle_angle = person_a_pct * np.pi
    needle_x = 0.9 * np.cos(needle_angle)
    needle_y = 0.9 * np.sin(needle_angle)
    ax.plot([0, needle_x], [0, needle_y], color='black', linewidth=3, alpha=0.8)
    ax.plot(0, 0, 'ko', markersize=8)
    
    if person_a_pct > 0:
        label_a_angle = person_a_pct * np.pi / 2
        label_a_x = 1.3 * np.cos(label_a_angle)
        label_a_y = 1.3 * np.sin(label_a_angle)
        ax.text(label_a_x, label_a_y, f'{person_a_count}\n({person_a_pct:.1%})', 
                ha='center', va='center', fontsize=10, color=colors["dark"], fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    if person_b_pct > 0:
        label_b_angle = person_a_pct * np.pi + (person_b_pct * np.pi / 2)
        label_b_x = 1.3 * np.cos(label_b_angle)
        label_b_y = 1.3 * np.sin(label_b_angle)
        ax.text(label_b_x, label_b_y, f'{person_b_count}\n({person_b_pct:.1%})', 
                ha='center', va='center', fontsize=10, color=colors["light"], fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    for i in [0, 0.25, 0.5, 0.75, 1.0]:
        angle = i * np.pi
        x_tick = 1.05 * np.cos(angle)
        y_tick = 1.05 * np.sin(angle)
        ax.plot([np.cos(angle), x_tick], [np.sin(angle), y_tick], 'k-', alpha=0.3)
    
    ax.text(0, -0.2, f'{couple_name}\n{total} messages', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='black')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

def main():
    sns.set(style="whitegrid")
    root, datafile = resolve_paths()
    
    csv_file = root / "data/processed/output.csv"
    parquet_file = root / "data/processed/output.parq"
    
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"Loaded data from {csv_file}")
    elif parquet_file.exists():
        df = pd.read_parquet(parquet_file)
        print(f"Loaded data from {parquet_file}")
    elif datafile.exists():
        try:
            df = pd.read_parquet(datafile)
        except:
            df = pd.read_csv(datafile)
        print(f"Loaded data from {datafile}")
    else:
        print(f"[WARN] No data files found. Looking for: {csv_file}, {parquet_file}, {datafile}", file=sys.stderr)
        sys.exit(1)

    if "author" not in df.columns or "message" not in df.columns:
        print("[ERROR] Dataframe must have 'author' and 'message' columns.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}")
        sys.exit(2)
    
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
    
    print("Author message counts:")
    print(author_counts.sort_values("message_count", ascending=False))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('One Voice Always Dominates\nPer Couple', 
                 fontsize=32, fontweight='bold', y=0.95)
    
    legend_elements = [
        Rectangle((0,0),1,1, facecolor='#333333', alpha=0.8, label='Male (Dark)'),
        Rectangle((0,0),1,1, facecolor='#CCCCCC', alpha=0.8, label='Female (Light)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88),
              title_fontsize=12, fontsize=10, frameon=True, 
              fancybox=True, shadow=True)
    
    couple_names = ["couple1", "couple2", "couple3"]
    
    for i, couple_id in enumerate(couple_names):
        couple_authors = author_counts[author_counts["couples"] == couple_id]
        
        if len(couple_authors) >= 2:
            couple_authors = couple_authors.nlargest(2, "message_count")
            person_a = couple_authors.iloc[0]
            person_b = couple_authors.iloc[1]
            
            person_a_name = person_a["author"]
            person_b_name = person_b["author"] 
            person_a_count = person_a["message_count"]
            person_b_count = person_b["message_count"]
            person_a_gender = person_a["gender"] if "gender" in person_a else "unknown"
            person_b_gender = person_b["gender"] if "gender" in person_b else "unknown"
            
        elif len(couple_authors) == 1:
            person_a = couple_authors.iloc[0]
            person_a_name = person_a["author"]
            person_a_count = person_a["message_count"]
            person_a_gender = person_a["gender"] if "gender" in person_a else "unknown"
            person_b_name = "No partner"
            person_b_count = 0
            person_b_gender = "unknown"
            
        else:
            person_a_name = "No data"
            person_b_name = "No data"
            person_a_count = 0
            person_b_count = 0
            person_a_gender = "unknown"
            person_b_gender = "unknown"
        
        colors = couple_shades.get(f"Couple{i+1}", {"light": "#BBBBBB", "dark": "#777777"})
        
        create_gauge_chart(
            axes[i], 
            person_a_count, person_b_count,
            person_a_name, person_b_name,
            person_a_gender, person_b_gender,
            f"Couple {i+1}",
            colors
        )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    img_dir = repo_root / "img"
    img_dir.mkdir(exist_ok=True)
    
    output_path = img_dir / "gauge_charts.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Chart saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()