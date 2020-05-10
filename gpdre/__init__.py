"""Top-level package for Gaussian Process Density Ratio Estimation."""

__author__ = """Louis Tiao"""
__email__ = 'louistiao@gmail.com'
__version__ = '0.1.0'

from .base import DensityRatio, DensityRatioMarginals
from .gaussian_process import GaussianProcessDensityRatioEstimator

__all__ = [
    "GaussianProcessDensityRatioEstimator",
    "DensityRatio", "DensityRatioMarginals"
]
