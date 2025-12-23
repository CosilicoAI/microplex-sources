"""
Calibration methods.

Provides different algorithms for computing calibrated weights:
- EntropyCalibrator: Minimize KL divergence from original weights
- (Future) RakingCalibrator: Iterative proportional fitting
- (Future) LinearCalibrator: Linear regression adjustment
"""

from .entropy import EntropyCalibrator

__all__ = ["EntropyCalibrator"]
