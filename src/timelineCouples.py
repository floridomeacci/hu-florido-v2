import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import seaborn as sns

try:
    from scipy.interpolate import make_interp_spline
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, falling back to basic plots")

def resolve_paths():
    here = Path(__file__).resolve()
    root = here.parent.parent  
    return root

def load_data(root):
    csv_file = root / "data/processed/output.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"Loaded data from {csv_file}")
        return df
    else:
        print(f"[ERROR] No data file found at {csv_file}", file=sys.stderr)
        sys.exit(1)

def create_timeline_chart(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    couples_data = df[df['couples'].isin(['couple1', 'couple2', 'couple3'])].copy()
    
    if couples_data.empty:
        print("No couple data found!")
        return
    
    couples_data['quarter'] = couples_data['timestamp'].dt.to_period('Q')
    
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
    
    all_quarters = sorted(quarter_counts['quarter'].unique())
    
    bar_height = 1.0  
    quarter_positions = range(len(all_quarters))
    
    all_segment_data = []
    
    for quarter_idx, quarter in enumerate(all_quarters):
        quarter_data = quarter_counts[quarter_counts['quarter'] == quarter]
        
        individuals = []
        
        c1_male = quarter_data[(quarter_data['couples'] == 'couple1') & (quarter_data['gender'] == 'male')]['message_count'].sum()
        c1_female = quarter_data[(quarter_data['couples'] == 'couple1') & (quarter_data['gender'] == 'female')]['message_count'].sum()
        
        c2_male = quarter_data[(quarter_data['couples'] == 'couple2') & (quarter_data['gender'] == 'male')]['message_count'].sum()
        c2_female = quarter_data[(quarter_data['couples'] == 'couple2') & (quarter_data['gender'] == 'female')]['message_count'].sum()
        
        c3_male = quarter_data[(quarter_data['couples'] == 'couple3') & (quarter_data['gender'] == 'male')]['message_count'].sum()
        c3_female = quarter_data[(quarter_data['couples'] == 'couple3') & (quarter_data['gender'] == 'female')]['message_count'].sum()
        
        individuals = [
            ('couple1', 'male', c1_male),
            ('couple1', 'female', c1_female),
            ('couple2', 'male', c2_male),
            ('couple2', 'female', c2_female),
            ('couple3', 'male', c3_male),
            ('couple3', 'female', c3_female)
        ]
        
        individuals.sort(key=lambda x: x[2], reverse=True)
        
        quarter_segments = []
        for couple_name, gender, message_count in individuals:
            quarter_segments.append((couple_name, gender, message_count))
        
        all_segment_data.append(quarter_segments)
    
    for quarter_idx, quarter_segments in enumerate(all_segment_data):
        cumulative_pos = 0
        
        for segment_couple, segment_gender, message_count in quarter_segments:
            color_key = f'{segment_couple}_{segment_gender}'
            label = couple_labels[color_key] if quarter_idx == 0 else ""  
            
            ax.barh(quarter_idx, message_count, height=bar_height, 
                   left=cumulative_pos, color=couple_colors[color_key], alpha=0.8, 
                   label=label)
            
            cumulative_pos += message_count
    
    ax.set_xlabel('')  
    ax.set_ylabel('')  
    
    fig.suptitle('One of two people always dominate\nWhatsApp messages over time', 
                fontsize=36, fontweight='bold', x=0.08, y=0.95, ha='left')
    
    quarter_labels = [str(quarter) for quarter in all_quarters]
    ax.set_yticks(quarter_positions)
    ax.set_yticklabels(quarter_labels)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.15), frameon=True, 
              fancybox=True, shadow=True, fontsize=10, ncol=2)
    
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.axvline(x=8, color='black', linestyle='-', linewidth=2, alpha=0.7, zorder=5)
    
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)  
    
    img_dir = 'img'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    output_path = os.path.join(img_dir, 'timeline_couples_bars.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Timeline bar chart saved to {output_path}")

def main():
    try:
        df = pd.read_csv('data/processed/output.csv')
    except FileNotFoundError:
        print("output.csv not found! Please run myPreprocess.py first.")
        return
    
    print(f"Loaded {len(df)} messages")
    
    create_timeline_chart(df)

if __name__ == "__main__":
    main()