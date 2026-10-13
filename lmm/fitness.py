"""
Component 3 — The Fitness Evaluator (The Referee).

Takes a completed MathAST and a dataset of observed trajectories,
simulates the AST as an ODE, and returns a fitness score ∈ [0, 1].

Pipeline
────────
1.  Convert completed AST → callable function  f(state) → derivatives
2.  For each trajectory in the dataset:
      a. Take the initial condition
      b. Integrate the ODE using RK4  (using the AST's function)
      c. Compare the simulated trajectory to the observed trajectory
3.  Fitness = 1.0 - normalised_mse - complexity_penalty
4.  Clamp to [0, 1]

Edge cases handled:
  - NaN / Inf → fitness = 0.0
  - Division by zero → guarded in AST.to_callable()
  - Exponential blowup → clamped in AST.to_callable()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np

from lmm.ast_graph import MathAST
from lmm.config import LMMConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsDataset:
    """
    A collection of observed trajectories for a dynamical system.

    Each trajectory is a pair (initial_condition, sequence_of_states).
    The LMM must discover the ODE  dy/dt = f(y)  that generated them.
    """
    name: str
    var_names: List[str]            # e.g. ["r", "v"]
    dimension: int                  # number of state variables
    dt: float                       # timestep between observations
    trajectories: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    # Each entry: (initial_condition [D], observed_states [T, D])

    def add_trajectory(self, y0: np.ndarray, states: np.ndarray):
        """Add a trajectory: y0 is [D], states is [T, D]."""
        self.trajectories.append((np.asarray(y0), np.asarray(states)))


# ═══════════════════════════════════════════════════════════════════════════════
# RK4 Integrator (pure numpy — fast, no JAX overhead for fitness eval)
# ═══════════════════════════════════════════════════════════════════════════════

def rk4_integrate(deriv_fn: Callable,
                  y0: np.ndarray,
                  dt: float,
                  num_steps: int) -> np.ndarray:
    """
    4th-order Runge-Kutta integration.

    Parameters
    ----------
    deriv_fn  : f(y) → dy/dt   (numpy-compatible callable)
    y0        : initial state [D]
    dt        : timestep
    num_steps : how many steps to simulate

    Returns
    -------
    trajectory : [num_steps+1, D]  (includes initial condition)
    """
    import jax.numpy as jnp

    dim = y0.shape[0]
    traj = np.zeros((num_steps + 1, dim), dtype=np.float64)
    traj[0] = y0
    y = jnp.array(y0, dtype=jnp.float32)

    for i in range(num_steps):
        k1 = jnp.array(deriv_fn(y))
        k2 = jnp.array(deriv_fn(y + 0.5 * dt * k1))
        k3 = jnp.array(deriv_fn(y + 0.5 * dt * k2))
        k4 = jnp.array(deriv_fn(y + dt * k3))
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Bail on NaN / Inf
        if jnp.any(jnp.isnan(y)) or jnp.any(jnp.abs(y) > 1e10):
            traj[i + 1:] = np.nan
            return traj

        traj[i + 1] = np.array(y)

    return traj


# ═══════════════════════════════════════════════════════════════════════════════
# Fitness evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_fitness(ast: MathAST,
                     dataset: PhysicsDataset,
                     cfg: LMMConfig,
                     target_var: int = -1) -> float:
    """
    Score how well the AST's equation matches the observed data.

    Parameters
    ----------
    ast        : a completed MathAST
    dataset    : the observed trajectories
    cfg        : configuration (for complexity penalty, dt)
    target_var : which state variable the AST predicts the derivative of.
                 -1 means the last variable (e.g., velocity in [r, v]).

    Returns
    -------
    fitness : float ∈ [0, 1]
        1.0  = perfect prediction
        0.0  = complete failure / NaN
    """
    if not ast.is_complete:
        return 0.0

    # Get the AST as a callable
    try:
        ast_fn = ast.to_callable()
    except Exception:
        return 0.0

    if target_var == -1:
        target_var = dataset.dimension - 1

    total_mse = 0.0
    num_points = 0

    for y0, observed in dataset.trajectories:
        num_steps = len(observed) - 1
        if num_steps <= 0:
            continue

        # Build the full derivative function:
        #   For a 2D system [r, v]:
        #     dr/dt = v            (identity — state[1])
        #     dv/dt = ast_fn(state) (what the AST predicts)
        import jax.numpy as jnp

        def full_derivs(state, _ast_fn=ast_fn, _dim=dataset.dimension, _tv=target_var):
            derivs = []
            for i in range(_dim):
                if i == _tv:
                    derivs.append(float(_ast_fn(state)))
                else:
                    # Chain rule: d(state[i])/dt = state[i+1]
                    # (standard position-velocity-acceleration chain)
                    if i + 1 < _dim:
                        derivs.append(float(state[i + 1]))
                    else:
                        derivs.append(0.0)
            return np.array(derivs, dtype=np.float64)

        # Integrate
        simulated = rk4_integrate(full_derivs, y0, dataset.dt, num_steps)

        # Check for NaN
        if np.any(np.isnan(simulated)):
            return 0.0

        # MSE between simulated and observed
        mse = np.mean((simulated - observed) ** 2)
        total_mse += mse
        num_points += 1

    if num_points == 0:
        return 0.0

    avg_mse = total_mse / num_points

    # Normalise: use 1 / (1 + MSE) to map [0, ∞) → (0, 1]
    accuracy = 1.0 / (1.0 + avg_mse)

    # Occam's Razor: penalise complex equations
    penalty = cfg.complexity_penalty * ast.complexity

    fitness = max(0.0, min(1.0, accuracy - penalty))
    return fitness


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: generate synthetic datasets for testing
# ═══════════════════════════════════════════════════════════════════════════════

def make_gravity_dataset() -> PhysicsDataset:
    """
    dr/dt = v
    dv/dt = -1/r²

    3 trajectories from different initial conditions.
    """
    ds = PhysicsDataset(
        name="Gravity (Inverse Square)",
        var_names=["r", "v"],
        dimension=2,
        dt=0.05,
    )

    def true_derivs(y):
        r, v = y
        return np.array([v, -1.0 / (r ** 2)])

    for y0 in [np.array([2.0, 0.0]),
               np.array([2.5, -0.4]),
               np.array([1.8, 0.2])]:
        states = rk4_integrate(true_derivs, y0, ds.dt, num_steps=40)
        if not np.any(np.isnan(states)):
            ds.add_trajectory(y0, states)

    return ds


def make_spring_dataset() -> PhysicsDataset:
    """
    dx/dt = v
    dv/dt = -x    (Hooke's law with k=1, m=1)

    Simple harmonic motion.
    """
    ds = PhysicsDataset(
        name="Spring (Hooke's Law)",
        var_names=["x", "v"],
        dimension=2,
        dt=0.05,
    )

    def true_derivs(y):
        x, v = y
        return np.array([v, -x])

    for y0 in [np.array([1.0, 0.0]),
               np.array([0.0, 1.0]),
               np.array([0.5, -0.5])]:
        states = rk4_integrate(true_derivs, y0, ds.dt, num_steps=40)
        ds.add_trajectory(y0, states)

    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from lmm.ast_graph import NodeType

    print("=" * 60)
    print("  FITNESS EVALUATOR — SELF TEST")
    print("=" * 60)

    cfg = LMMConfig(max_vars=2)

    # ── Test 1: Perfect equation for gravity ───────────────────────────────
    # Build AST:  -(1 / (r ^ 2))  =  -1/r²
    ast = MathAST(max_nodes=20, num_vars=2)
    for action in [NodeType.NEG, NodeType.DIV, NodeType.CONST_1,
                   NodeType.POW, NodeType.VAR_0, NodeType.CONST_2]:
        ast = ast.apply_action(int(action))

    ds = make_gravity_dataset()
    fitness = evaluate_fitness(ast, ds, cfg)
    print(f"\n  Gravity dataset: {ds.name}")
    print(f"  Equation:        {ast.to_string(ds.var_names)}")
    print(f"  Fitness:         {fitness:.6f}  (should be near 1.0)")
    assert fitness > 0.9, f"Gravity fitness too low: {fitness}"

    # ── Test 2: Wrong equation (should get low fitness) ────────────────────
    # Build AST:  VAR_0   (just "r" — completely wrong derivative for dv/dt)
    bad_ast = MathAST(max_nodes=20, num_vars=2)
    bad_ast = bad_ast.apply_action(int(NodeType.VAR_0))

    bad_fitness = evaluate_fitness(bad_ast, ds, cfg)
    print(f"\n  Wrong equation:  {bad_ast.to_string(ds.var_names)}")
    print(f"  Fitness:         {bad_fitness:.6f}  (should be near 0.0)")
    assert bad_fitness < 0.5, f"Bad fitness too high: {bad_fitness}"

    # ── Test 3: Perfect equation for spring ────────────────────────────────
    # Build AST:  NEG(VAR_0)  =  -x   (Hooke's law)
    spring_ast = MathAST(max_nodes=20, num_vars=2)
    spring_ast = spring_ast.apply_action(int(NodeType.NEG))
    spring_ast = spring_ast.apply_action(int(NodeType.VAR_0))

    spring_ds = make_spring_dataset()
    spring_fitness = evaluate_fitness(spring_ast, spring_ds, cfg)
    print(f"\n  Spring dataset:  {spring_ds.name}")
    print(f"  Equation:        {spring_ast.to_string(spring_ds.var_names)}")
    print(f"  Fitness:         {spring_fitness:.6f}  (should be near 1.0)")
    assert spring_fitness > 0.9, f"Spring fitness too low: {spring_fitness}"

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
