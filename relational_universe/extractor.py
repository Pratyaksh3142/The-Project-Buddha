"""
Level 0: Symbolic Continuum Extractor

Translates raw relational measurements into exact mathematical invariants and LaTeX equations:
  Observable Slot 1: Incoming Flux F
  Observable Slot 2: Knot Capacity / Mass m
  Observable Slot 3: Kinematic Hop Acceleration a
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np


class ContinuumExtractor:
    """Extracts macroscopic differential equations from discrete graph knot measurements."""

    @staticmethod
    def extract_power_law_invariant(
        observations: List[Dict[str, float]]
    ) -> Tuple[float, float, float, float]:
        """
        Fits the relational scaling law:
            a = k * (F^alpha) * (m^beta)
        Via linear regression in logarithmic space:
            log(a) = log(k) + alpha * log(F) + beta * log(m)
        Returns: (alpha, beta, k, max_residual_error)
        """
        assert len(observations) >= 3, "Need at least 3 distinct measurements to solve power law."

        log_F = np.array([np.log(obs["flux"]) for obs in observations])
        log_m = np.array([np.log(obs["mass"]) for obs in observations])
        log_a = np.array([np.log(obs["acceleration"]) for obs in observations])

        # Design matrix X: [1, log(F), log(m)]
        X = np.stack([np.ones_like(log_F), log_F, log_m], axis=1)

        # Solve via least squares
        coeffs, residuals, _, _ = np.linalg.lstsq(X, log_a, rcond=None)
        log_k, alpha, beta = coeffs

        k = float(np.exp(log_k))
        alpha = float(alpha)
        beta = float(beta)

        # Calculate maximum absolute residual error
        predicted_log_a = X @ coeffs
        max_error = float(np.max(np.abs(np.exp(predicted_log_a) - np.exp(log_a))))

        return alpha, beta, k, max_error

    @staticmethod
    def format_latex_law(alpha: float, beta: float, k: float) -> str:
        """Formats the discovered exponents into human-readable LaTeX notation."""
        # Check if alpha ≈ 1.0 and beta ≈ -1.0 (Newton's 2nd Law)
        if abs(alpha - 1.0) < 1e-4 and abs(beta - (-1.0)) < 1e-4:
            return r"\mathbf{F} = m \frac{d^2 \mathbf{x}}{dt^2}"
        # General power law representation
        return rf"\mathbf{{a}} = {k:.4f} \cdot \frac{{\mathbf{{F}}^{{{alpha:.2f}}}}}{{m^{{{-beta:.2f}}}}}"
