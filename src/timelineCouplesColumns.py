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

def create_timeline_chart(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    couples_data = df[df['couples'].isin(['couple1', 'couple2', 'couple3'])].copy()
    
    if couples_data.empty:
        print("No couple data found!")
        return
    
    couples_data['quarter'] = couples_data['timestamp'].dt.to_period('Q')
    
    quarter_counts = couples_data.groupby(['quarter', 'couples', 'gender']).size().reset_index(name='message_count')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10), sharey=True)
    
    fig.patch.set_facecolor('white')
    for ax in [ax1, ax2, ax3]:
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
        'couple1': 'Couple 1',
        'couple2': 'Couple 2', 
        'couple3': 'Couple 3'
    }
    
    all_quarters = sorted(quarter_counts['quarter'].unique())
    quarter_positions = range(len(all_quarters))
    
    couples = ['couple1', 'couple2', 'couple3']
    axes = [ax1, ax2, ax3]
    
    bar_height = 1.0  
    
    for couple_idx, (couple, ax) in enumerate(zip(couples, axes)):
        male_percentages = []
        female_percentages = []
        
        for quarter in all_quarters:
            quarter_data = quarter_counts[quarter_counts['quarter'] == quarter]
            couple_data = quarter_data[quarter_data['couples'] == couple]
            
            male_count = couple_data[couple_data['gender'] == 'male']['message_count'].sum()
            female_count = couple_data[couple_data['gender'] == 'female']['message_count'].sum()
            
            couple_total = male_count + female_count
            
            if couple_total > 0:
                male_pct = (male_count / couple_total) * 100
                female_pct = (female_count / couple_total) * 100
            else:
                male_pct = female_pct = 0
            
            male_percentages.append(male_pct)
            female_percentages.append(female_pct)
        
        bars_male = ax.barh(quarter_positions, male_percentages, height=bar_height, 
                           color=couple_colors[f'{couple}_male'], alpha=0.8)
        
        bars_female = ax.barh(quarter_positions, female_percentages, height=bar_height, 
                             left=male_percentages, color=couple_colors[f'{couple}_female'], alpha=0.8)
        
        if couple_idx == 2:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#525252', label='Dark grey'),
                Patch(facecolor='#969696', label='Light grey')
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.15), 
                     frameon=True, fancybox=True, shadow=True, fontsize=12)
        
        ax.set_xlabel('')  
        if couple_idx == 0:  
            ax.set_ylabel('')
            quarter_labels = [str(quarter) for quarter in all_quarters]
            ax.set_yticks(quarter_positions)
            ax.set_yticklabels(quarter_labels)
        
        ax.set_title(couple_labels[couple], fontsize=18, fontweight='bold', pad=30)
        
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 100])
        ax.set_xticklabels(['0', '100'])
    
    fig.suptitle('One voice per couple always dominates\nWhatsApp messages over time', 
                fontsize=32, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.75, wspace=0.1)  
    
    img_dir = 'img'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    output_path = os.path.join(img_dir, 'timeline_couples_columns.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Timeline columns chart saved to {output_path}")

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