"""
Algebraic Fitness Evaluator for Static Property Discovery (Materials & Chemistry).

Unlike dynamical ODE integration (which simulates dy/dt over time),
algebraic discovery directly maps feature vector x -> property y (e.g. material features -> Tc).
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from lmm.ast_graph import MathAST
from lmm.config import LMMConfig


def evaluate_algebraic_fitness(ast: MathAST,
                               X: np.ndarray,
                               y_true: np.ndarray,
                               cfg: LMMConfig) -> float:
    """
    Evaluates how accurately the AST computes y_pred from feature matrix X.

    Parameters:
        ast: Completed MathAST
        X: [N, D] feature matrix (e.g. [coupling_lambda, omega, mu_star])
        y_true: [N] ground-truth targets (e.g. Tc / 100K)
        cfg: Configuration containing complexity penalty

    Returns:
        fitness: float in [0.0, 1.0]
    """
    if not ast.is_complete:
        return 0.0

    try:
        ast_fn = ast.to_callable()
    except Exception:
        return 0.0

    import jax.numpy as jnp

    y_pred = []
    for i in range(len(X)):
        try:
            val = float(ast_fn(jnp.array(X[i], dtype=jnp.float32)))
            if np.isnan(val) or np.isinf(val) or abs(val) > 1e6:
                return 0.0
            y_pred.append(val)
        except Exception:
            return 0.0

    y_pred = np.array(y_pred, dtype=np.float64)

    # Normalized Mean Squared Error
    mse = float(np.mean((y_pred - y_true) ** 2))
    variance = float(np.var(y_true))
    normalized_mse = mse / max(variance, 1e-6)

    # Accuracy mapping
    accuracy = 1.0 / (1.0 + normalized_mse)

    # Complexity penalty
    penalty = cfg.complexity_penalty * ast.complexity

    fitness = max(0.0, min(1.0, accuracy - penalty))
    return fitness
