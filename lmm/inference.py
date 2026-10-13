"""
Component 6 — Inference & Autonomous Discovery Pipeline.

Provides the end-user interface:
Takes a CSV file or PhysicalDataset, configures the trained LMM Brain & MCTS,
and extracts the clean, symbolic differential equation.
"""

from __future__ import annotations

import csv
from typing import List, Optional, Tuple

import numpy as np

from lmm.config import LMMConfig
from lmm.ast_graph import MathAST
from lmm.neural_brain import MathBrain, init_brain
from lmm.fitness import PhysicsDataset, evaluate_fitness, rk4_integrate
from lmm.mcts import MCTS


def load_dataset_from_csv(csv_path: str, dt: float, name: str = "Empirical System") -> PhysicsDataset:
    """
    Parses a CSV file with trajectory data.
    Expected format:
        First row: variable names (e.g. 'r,v' or 'x,v')
        Subsequent rows: numeric measurements over time.
    """
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        var_names = [h.strip() for h in header]
        rows = []
        for r in reader:
            if r:
                rows.append([float(val.strip()) for val in r])

    states = np.array(rows, dtype=np.float64)
    dim = len(var_names)
    y0 = states[0]

    ds = PhysicsDataset(name=name, var_names=var_names, dimension=dim, dt=dt)
    ds.add_trajectory(y0, states)
    return ds


class DiscoveryEngine:
    """
    Inference controller for finding symbolic laws from data.
    """

    def __init__(self, cfg: LMMConfig, model: Optional[MathBrain] = None, params = None):
        self.cfg = cfg
        self.model = model
        self.params = params

        if self.model is None or self.params is None:
            # Initialize fresh brain if none provided
            import jax
            key = jax.random.PRNGKey(101)
            self.model, self.params = init_brain(cfg, key)

        self.mcts = MCTS(self.model, self.params, cfg)

    def discover(self,
                 dataset: PhysicsDataset,
                 num_attempts: int = 20,
                 simulations_per_move: int = 150) -> Tuple[MathAST, float, str]:
        """
        Runs deep tree searches to discover the best governing equation.
        """
        best_ast: Optional[MathAST] = None
        best_fitness: float = -1.0

        print(f"\n========================================================")
        print(f"  AUTONOMOUS DISCOVERY ENGINE: {dataset.name}")
        print(f"  Variables: {dataset.var_names} | Timestep dt: {dataset.dt}")
        print(f"========================================================")

        for attempt in range(1, num_attempts + 1):
            ast = MathAST(max_nodes=self.cfg.max_nodes, num_vars=dataset.dimension)
            # Temperature decays across attempts: high early on, greedy near the end
            temp = max(0.1, 1.0 - (attempt - 1) / float(max(1, num_attempts - 1)))

            while not ast.is_complete:
                policy_dict, _ = self.mcts.search(ast, num_simulations=simulations_per_move, add_noise=(attempt > 1))

                actions = list(policy_dict.keys())
                probs = np.array([policy_dict[a] for a in actions], dtype=np.float64)

                if temp <= 0.15:
                    action = actions[int(np.argmax(probs))]
                else:
                    scaled = np.power(probs, 1.0 / temp)
                    s_sum = np.sum(scaled)
                    if s_sum > 0:
                        scaled /= s_sum
                    else:
                        scaled = np.ones_like(scaled) / len(scaled)
                    action = int(np.random.choice(actions, p=scaled))

                ast = ast.apply_action(action)

            fitness = evaluate_fitness(ast, dataset, self.cfg)
            eq_str = ast.to_string(dataset.var_names)

            if fitness > best_fitness:
                best_fitness = fitness
                best_ast = ast
                print(f"[Attempt {attempt:02d}] NEW BEST FITNESS: {fitness:.6f} | Equation: {eq_str}")
            else:
                print(f"[Attempt {attempt:02d}] Fitness: {fitness:.6f} | Equation: {eq_str}")

            if best_fitness > 0.95:
                print(f"\n>>> Discovered exact physical law with high precision!")
                break

        final_eq = best_ast.to_string(dataset.var_names) if best_ast else "<None>"
        return best_ast, best_fitness, final_eq
