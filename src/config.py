"""
Configuration models for figure generation.
All parameters and settings are centralized here using Pydantic for validation.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class DataColumnConfig(BaseModel):
    """Configuration for data column names - eliminates hardcoding."""
    timestamp: str = "timestamp"
    author: str = "author"
    message: str = "message"
    gender: str = "gender"
    couples: str = "couples"
    gender_mapped: str = "gender_mapped"
    emojis: str = "emojis"
    emoji_count: str = "emoji_count"


class AnalysisConfig(BaseModel):
    """Configuration for analysis parameters."""
    couples_list: List[str] = Field(default=["couple1", "couple2", "couple3"])
    
    gender_map: Dict[str, str] = Field(default={
        'male': 'Men',
        'female': 'Women',
        'M': 'Men',
        'F': 'Women'
    })
    
    max_response_hours: float = 6.0
    omitted_message_text: str = '<Media omitted>'
    
    @property
    def max_response_minutes(self) -> float:
        return self.max_response_hours * 60


class DistributionPlotConfig(BaseModel):
    """Configuration for distribution plots."""
    num_bins: int = 100
    smoothing_sigma: float = 2.0
    scaling_factor: float = 10.0
    time_max_hours: float = 6.0
    time_labels: List[str] = Field(default=['0h', '1h', '2h', '3h', '4h', '5h', '6h'])
    median_box_position: tuple = (0.05, 0.95)


class RelationshipPlotConfig(BaseModel):
    """Configuration for relationship plots (emoji analysis)."""
    emoji_categories: List[int] = Field(default=[0, 1, 2, 3, 4])
    emoji_group_threshold: int = 5
    smooth_points: int = 300
    spline_degree: int = 3
    point_size: int = 200
    point_edge_width: int = 3


class TSNEPlotConfig(BaseModel):
    """Configuration for t-SNE visualization."""
    max_chunks_per_author: int = 20
    min_tokens: int = 25
    min_chunk_chars: int = 200
    perplexity: int = 30
    random_state: int = 42
    max_iter: int = 1000
    ngram_range: tuple = (3, 5)
    max_features: int = 1000
    iqr_multiplier: float = 2.5
    kde_alpha: float = 0.3
    kde_levels: int = 3


class FigureStyleConfig(BaseModel):
    """Configuration for figure styling - colors, fonts, backgrounds, etc."""
    # Color palette
    gender_colors: Dict[str, str] = Field(default={
        'Men': '#1E3A8A',      # Blue
        'Women': '#FF69B4'     # Pink
    })
    
    couple_colors: Dict[str, str] = Field(default={
        'couple1': '#FF69B4',   # Pink
        'couple2': '#16A34A',   # Green
        'couple3': '#1E3A8A',   # Blue
    })
    
    # Typography
    title_fontsize: int = 32
    subtitle_fontsize: int = 13
    label_fontsize: int = 14
    legend_fontsize: int = 12
    
    # Background colors
    figure_bg: str = 'white'
    axis_bg: str = '#F9FAFB'
    legend_bg: str = 'white'
    legend_alpha: float = 0.95
    
    # Grid settings
    grid_alpha: float = 0.25
    grid_linestyle: str = '--'
    grid_linewidth: float = 0.8


class FigureOutputConfig(BaseModel):
    """Configuration for figure output."""
    dpi: int = 300
    bbox_inches: str = 'tight'
    facecolor: str = 'white'
    edgecolor: str = 'none'
    distribution_filename: str = 'distribution.png'
    relationship_filename: str = 'relationship.png'
    tsne_filename: str = 'tSNE.png'
    timeseries_filename: str = 'time_series.png'
    comparing_filename: str = 'comparing_categories.png'


class MasterConfig(BaseModel):
    """Master configuration combining all settings."""
    columns: DataColumnConfig = Field(default_factory=DataColumnConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    distribution: DistributionPlotConfig = Field(default_factory=DistributionPlotConfig)
    relationship: RelationshipPlotConfig = Field(default_factory=RelationshipPlotConfig)
    tsne: TSNEPlotConfig = Field(default_factory=TSNEPlotConfig)
    style: FigureStyleConfig = Field(default_factory=FigureStyleConfig)
    output: FigureOutputConfig = Field(default_factory=FigureOutputConfig)


# Create global default configuration instance
DEFAULT_CONFIG = MasterConfig()
