"""Random Forest module for wine quality prediction."""

from .model import WineQualityRF
from .utils import split_data, preprocess_features

__all__ = ["WineQualityRF", "split_data", "preprocess_features"]
