"""
hu-florido-v2 source package

WhatsApp chat analysis and visualization tools for exploring messaging dynamics
between couples, including character distributions, time series analysis,
and family word tracking.
"""

__version__ = "2.0.0"
__author__ = "Florido Meacci"

from .myFigures import (
    plot_gauge_couples,
    plot_timeline_couples_bars,
    plot_timeline_couples_bars_hours,
    plot_timeline_couples_columns,
    plot_time_series,
    plot_time_series_couples,
    plot_time_series_combined,
    plot_response_times_distribution,
    plot_response_times_skewnorm,
)

from .myPreprocess import main as preprocess

__all__ = [
    "plot_gauge_couples",
    "plot_timeline_couples_bars", 
    "plot_timeline_couples_bars_hours",
    "plot_timeline_couples_columns",
    "plot_time_series",
    "plot_time_series_couples", 
    "plot_time_series_combined",
    "plot_response_times_distribution",
    "plot_response_times_skewnorm",
    "preprocess",
]