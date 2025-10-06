"""
hu-florido-v2 source package

WhatsApp chat analysis and visualization tools for exploring messaging dynamics
between couples, including character distributions, time series analysis,
and family word tracking.
"""

__version__ = "2.0.0"
__author__ = "Florido Meacci"

from .myFigures import (
    comparing_categories_1,
    distribution_categories_2,
    distribution_categories_1,
    comparing_categories_2,
    time_series,
    time_series_2,
    time_series_3,
    response_times,
    response_times_skewnorm,
)

from .myPreprocess import main as preprocess

__all__ = [
    "comparing_categories_1",
    "distribution_categories_2", 
    "distribution_categories_1",
    "comparing_categories_2",
    "time_series",
    "time_series_2", 
    "time_series_3",
    "response_times",
    "response_times_skewnorm",
    "preprocess",
]