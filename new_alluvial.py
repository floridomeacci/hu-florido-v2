def plot_alluvial_responses(df: Optional[pd.DataFrame] = None, settings: Optional[Settings] = None) -> Path:
    """Create an alluvial/flow diagram showing 6 Men (left) talking to 6 Women (right) with 6 lines each."""
    if settings is None:
        settings = load_settings()
    if df is None:
        df = _load_processed_df(settings)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    # Define positions - place them further apart
    left_x = 0.1
    right_x = 0.8
    rect_width = 0.08
    rect_height = 0.11  # Height for each individual rectangle
    spacing = 0.04      # Gap between rectangles
    
    # Create 6 individuals per side (duplicate each couple to get 6 total)
    men_data = []
    women_data = []
    
    couples_list = sorted(settings.couples)
    
    # Create 6 men and 6 women by doubling each couple
    for i in range(6):
        couple_idx = i // 2  # Each couple appears twice
        person_num = i % 2 + 1
        couple_name = couples_list[couple_idx]
        
        men_data.append(f'{couple_name}_male{person_num}')
        women_data.append(f'{couple_name}_female{person_num}')
    
    # Calculate total height needed and center vertically
    total_height = 6 * rect_height + 5 * spacing
    start_y = 0.5 - total_height / 2
    
    # Draw 6 men rectangles on the left
    men_positions = {}
    current_y = start_y
    
    for i, man in enumerate(men_data):
        rect = Rectangle((left_x, current_y), rect_width, rect_height, 
                        facecolor='#1E3A8A', alpha=0.8, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        
        # Add label - show which couple and person number
        couple_num = (i // 2) + 1
        person_num = (i % 2) + 1
        label = f'M{couple_num}.{person_num}'
        
        ax.text(left_x + rect_width/2, current_y + rect_height/2, label, 
                ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        
        men_positions[man] = (current_y, rect_height)
        current_y += rect_height + spacing
    
    # Draw 6 women rectangles on the right
    women_positions = {}
    current_y = start_y
    
    for i, woman in enumerate(women_data):
        rect = Rectangle((right_x, current_y), rect_width, rect_height,
                        facecolor='#FF69B4', alpha=0.8, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        
        # Add label - show which couple and person number
        couple_num = (i // 2) + 1
        person_num = (i % 2) + 1
        label = f'W{couple_num}.{person_num}'
        
        ax.text(right_x + rect_width/2, current_y + rect_height/2, label, 
                ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        
        women_positions[woman] = (current_y, rect_height)
        current_y += rect_height + spacing
    
    # Create exactly 6 flows from each man to all 6 women
    for man_idx, man in enumerate(men_data):
        man_couple = (man_idx // 2) + 1  # Which couple this man belongs to
        
        # Flow thickness - each gets 1/6 of the rectangle height (equal thickness)
        flow_thickness = rect_height / 6
        
        # Starting position for flows from this man
        man_y, man_height = men_positions[man]
        current_flow_y = man_y
        
        # Connect to all 6 women
        for woman_idx, woman in enumerate(women_data):
            woman_couple = (woman_idx // 2) + 1  # Which couple this woman belongs to
            
            # Calculate start and end positions
            start_y = current_flow_y + flow_thickness/2
            
            # For women, distribute incoming flows evenly across their height
            woman_y, woman_height = women_positions[woman]
            # Each woman receives 6 flows, so divide their height by 6
            incoming_flow_height = woman_height / 6
            end_y = woman_y + (man_idx * incoming_flow_height) + incoming_flow_height/2
            
            # Create smooth connection curve
            x_vals = np.linspace(left_x + rect_width, right_x, 100)
            t = np.linspace(0, 1, 100)
            
            # Smooth S-curve using cubic interpolation
            y_vals = start_y + (end_y - start_y) * (3*t**2 - 2*t**3)
            
            # Color: blue if same couple, pink if different couple
            if man_couple == woman_couple:
                color = '#1E3A8A'  # Blue for same couple
                alpha = 0.7
            else:
                color = '#FF69B4'  # Pink for cross-couple
                alpha = 0.5
            
            # Draw the flow with consistent thickness
            thickness = min(flow_thickness, incoming_flow_height) * 0.8  # Slightly thinner to avoid overlap
            ax.fill_between(x_vals, y_vals - thickness/2, y_vals + thickness/2, 
                           alpha=alpha, color=color, linewidth=0, edgecolor='none')
            
            # Move to next flow position on the sending side
            current_flow_y += flow_thickness
    
    # Add headers
    ax.text(left_x + rect_width/2, 0.95, 'MEN (6 INDIVIDUALS)', ha='center', va='center', 
            fontweight='bold', fontsize=16, color='#1E3A8A')
    ax.text(right_x + rect_width/2, 0.95, 'WOMEN (6 INDIVIDUALS)', ha='center', va='center', 
            fontweight='bold', fontsize=16, color='#FF69B4')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1E3A8A', alpha=0.7, label='Same Couple Connection'),
        Patch(facecolor='#FF69B4', alpha=0.5, label='Cross-Couple Connection')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=2, frameon=True, fontsize=12)
    
    # Add explanation text
    ax.text(0.5, 0.08, '6 LINES FROM EACH MAN TO ALL 6 WOMEN', 
            ha='center', va='center', fontweight='bold', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    fig.suptitle('6 MEN vs 6 WOMEN: COMPLETE CROSS-GENDER COMMUNICATION', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    img_dir = REPO_ROOT / 'img'
    img_dir.mkdir(exist_ok=True)
    out = img_dir / 'alluvial_responses.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return out