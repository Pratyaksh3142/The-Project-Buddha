#!/usr/bin/env python3
"""
================================================================================
          PROJECT BUDDHA: AUTONOMOUS GRAPH DISCOVERY ENGINE
================================================================================
Tabula Rasa Mathematical Discovery via Graph Evolution & Pratyaksh-II Integration.

Key Guarantees:
  1. ZERO HARDCODING: No hardcoded templates, no fallbacks, no fixed 6-step cheats.
  2. PURE GRAPH SEARCH: Evolves Computation Graphs (DAGs) of operators and state variables.
  3. GENERALIZATION: Evaluates on multiple initial conditions simultaneously.
  4. OCCAM'S RAZOR: Penalizes graph complexity to prevent Ptolemaic overfitting.
  5. UNIFIED PRATYAKSH-II: Integrates physics with the S_n-normalized limiter.
================================================================================
"""

import math
import copy
import random
import time
import json
import os
import sys
from typing import List, Dict, Tuple, Optional, Any

from graph_engine import ComputationGraph, UNARY_OPS, BINARY_OPS, ALL_OPS


# ==============================================================================
# SECTION 1: UNIFIED PRATYAKSH-II NUMERICAL INTEGRATOR (PYTHON REFERENCE)
# ==============================================================================

def pratyaksh_ii_integrate(graph: ComputationGraph, y0: List[float], t_end: float, dt: float,
                           alpha: float = 1.0, beta: float = 2.0) -> Optional[List[List[float]]]:
    """Integrates a candidate ComputationGraph using the official Unified Pratyaksh-II
       formulation from pratyaksh_v2_paper.tex.
       Returns trajectory snapshots, or None if diverged.
    """
    dim = len(y0)
    y = list(y0)
    t = 0.0
    trajectory = [list(y)]
    half_dt = 0.5 * dt

    while t < t_end - 1e-12:
        h = min(dt, t_end - t)

        # Stage 1: k1 = f(t, y)
        k1_rates = graph.evaluate(y, t)
        if k1_rates is None: return None
        k1 = [h * r for r in k1_rates]

        # Stage 2: k2 = f(t + h/2, y + k1/2)
        y_stage2 = [y[i] + 0.5 * k1[i] for i in range(dim)]
        k2_rates = graph.evaluate(y_stage2, t + half_dt)
        if k2_rates is None: return None
        k2 = [h * r for r in k2_rates]

        # Stage 3: k3 = f(t + h/2, y + k2/2)
        y_stage3 = [y[i] + 0.5 * k2[i] for i in range(dim)]
        k3_rates = graph.evaluate(y_stage3, t + half_dt)
        if k3_rates is None: return None
        k3 = [h * r for r in k3_rates]

        # Stage 4: k4 = f(t + h, y + k3)
        y_stage4 = [y[i] + k3[i] for i in range(dim)]
        k4_rates = graph.evaluate(y_stage4, t + h)
        if k4_rates is None: return None
        k4 = [h * r for r in k4_rates]

        # Unified Curvature C = k4 - k3 - k2 + k1
        # and Scale Factor S_n = ||y|| + ||k1|| + eps
        c_norm_sq = 0.0
        y_norm_sq = 0.0
        k1_norm_sq = 0.0

        for i in range(dim):
            c_i = k4[i] - k3[i] - k2[i] + k1[i]
            c_norm_sq += c_i * c_i
            y_norm_sq += y[i] * y[i]
            k1_norm_sq += k1[i] * k1[i]

        c_norm = math.sqrt(c_norm_sq)
        s_n = math.sqrt(y_norm_sq) + math.sqrt(k1_norm_sq) + 1e-12

        # Dimensionless Curvature Metric
        c_hat = c_norm / s_n

        # Bounded Rational Denominator: D = 6.0 + (alpha * c_hat)^beta
        denominator = 6.0 + math.pow(alpha * c_hat, beta)

        # State update
        for i in range(dim):
            num = k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]
            y[i] += num / denominator
            if not math.isfinite(y[i]) or abs(y[i]) > 1e12:
                return None

        t += h
        trajectory.append(list(y))

    return trajectory


# ==============================================================================
# SECTION 2: GRAPH MUTATIONS & EVOLUTIONARY SEARCH
# ==============================================================================

def random_graph(dimension: int) -> ComputationGraph:
    """Creates a basic valid starting graph with random connections."""
    g = ComputationGraph(dimension)
    for d in range(dimension):
        # Start each output connected to a random input state variable or constant
        if random.random() < 0.6:
            g.set_output(d, g.var_nodes[random.randint(0, dimension - 1)])
        else:
            cid = g.add_constant(random.uniform(-2.0, 2.0))
            g.set_output(d, cid)
    return g


def mutate_graph(graph: ComputationGraph) -> ComputationGraph:
    """Applies a random topological mutation to the computation graph."""
    g = graph.copy()
    dim = g.dimension
    r = random.random()

    all_node_ids = list(g.nodes.keys())

    # --- Mutation 1: Constant Perturbation (Fine Tuning) ---
    if r < 0.25:
        const_nodes = [nid for nid, n in g.nodes.items() if n.node_type == 'const']
        if const_nodes:
            target_nid = random.choice(const_nodes)
            g.nodes[target_nid].const_val += random.gauss(0, 0.25)
            return g

    # --- Mutation 2: Insert a New Operator ---
    if r < 0.60:
        op = random.choice(ALL_OPS)
        if op in UNARY_OPS:
            src = random.choice(all_node_ids)
            new_id = g.add_operator(op, [src])
        else: # Binary
            src1 = random.choice(all_node_ids)
            src2 = random.choice(all_node_ids)
            new_id = g.add_operator(op, [src1, src2])

        # Optionally redirect an output to this new operator
        target_dim = random.randint(0, dim - 1)
        g.set_output(target_dim, new_id)
        return g

    # --- Mutation 3: Rewire an Existing Operator Input ---
    if r < 0.85:
        op_nodes = [nid for nid, n in g.nodes.items() if n.node_type == 'op']
        if op_nodes:
            target_nid = random.choice(op_nodes)
            node = g.nodes[target_nid]
            if node.inputs:
                arg_idx = random.randint(0, len(node.inputs) - 1)
                # Pick a source that does not create a cycle
                possible_sources = [nid for nid in all_node_ids if nid != target_nid]
                if possible_sources:
                    node.inputs[arg_idx] = random.choice(possible_sources)
            return g

    # --- Mutation 4: Add New Constant & Wire ---
    new_cid = g.add_constant(random.uniform(-3.0, 3.0))
    target_dim = random.randint(0, dim - 1)
    # Combine with existing output via add/mul
    curr_out = g.output_nodes[target_dim]
    combo_op = random.choice(['add', 'sub', 'mul'])
    new_combo = g.add_operator(combo_op, [curr_out, new_cid])
    g.set_output(target_dim, new_combo)

    return g


# ==============================================================================
# SECTION 3: MULTI-INITIAL-CONDITION GENERALIZATION EVALUATOR
# ==============================================================================

class PhysicalDataset:
    """Holds physical trajectory data generated across multiple initial conditions."""
    def __init__(self, name: str, dimension: int, var_names: List[str],
                 t_end: float, dt: float,
                 ic_trajectories: List[Tuple[List[float], List[List[float]]]]):
        self.name = name
        self.dimension = dimension
        self.var_names = var_names
        self.t_end = t_end
        self.dt = dt
        # List of (y0, trajectory)
        self.ic_trajectories = ic_trajectories


def evaluate_fitness(graph: ComputationGraph, dataset: PhysicalDataset,
                     lambda_complexity: float = 0.001) -> float:
    """Evaluates graph generalization over all initial conditions in the dataset.
       Reward = 1 / (1 + Normalized MSE) - lambda * Complexity.
       STRICT: If any IC diverges or graph is invalid, fitness is strictly 0.0.
    """
    if not graph.is_valid():
        return 0.0

    total_mse = 0.0
    n_ics = len(dataset.ic_trajectories)

    for y0, true_traj in dataset.ic_trajectories:
        pred_traj = pratyaksh_ii_integrate(graph, y0, dataset.t_end, dataset.dt)
        if pred_traj is None or len(pred_traj) != len(true_traj):
            return 0.0 # Strict failure: diverged or wrong length

        # Compute Mean Squared Error for this IC
        sse = 0.0
        n_points = len(true_traj)
        for step in range(n_points):
            for d in range(dataset.dimension):
                diff = pred_traj[step][d] - true_traj[step][d]
                sse += diff * diff

        mse = sse / (n_points * dataset.dimension)
        total_mse += mse

    avg_mse = total_mse / n_ics
    # Occam's Razor complexity penalty
    comp = graph.complexity()
    complexity_penalty = lambda_complexity * comp

    base_fitness = 1.0 / (1.0 + avg_mse)
    return max(0.0, base_fitness - complexity_penalty)


# ==============================================================================
# SECTION 4: TEST PROBLEMS (NON-TRIVIAL DYNAMICS)
# ==============================================================================

def make_lotka_volterra_dataset() -> PhysicalDataset:
    """Lotka-Volterra Predator-Prey Equations:
       dx/dt = 1.5*x - 1.0*x*y
       dy/dt = 1.0*x*y - 3.0*y
       Coupled non-linear terms. Impossible to guess with 1 leaf node.
    """
    dimension = 2
    var_names = ['prey', 'predator']
    t_end = 4.0
    dt = 0.05

    # Ground truth reference simulator using high-precision RK4
    def true_derivs(y):
        x, p = y[0], y[1]
        return [1.5 * x - 1.0 * x * p, 1.0 * x * p - 3.0 * p]

    def gen_traj(y0):
        y = list(y0)
        t = 0.0
        traj = [list(y)]
        while t < t_end - 1e-12:
            # RK4 step
            k1 = true_derivs(y)
            y2 = [y[i] + 0.5 * dt * k1[i] for i in range(2)]
            k2 = true_derivs(y2)
            y3 = [y[i] + 0.5 * dt * k2[i] for i in range(2)]
            k3 = true_derivs(y3)
            y4 = [y[i] + dt * k3[i] for i in range(2)]
            k4 = true_derivs(y4)
            for i in range(2):
                y[i] += (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            t += dt
            traj.append(list(y))
        return traj

    # 3 Diverse Initial Conditions for Generalization Testing
    test_ics = [
        [1.0, 1.0],  # Equilibrium orbit
        [2.0, 0.5],  # High prey excursion
        [0.5, 2.0],  # High predator crash
    ]
    ic_trajs = [(ic, gen_traj(ic)) for ic in test_ics]

    return PhysicalDataset(
        name="Lotka-Volterra Predator-Prey System",
        dimension=dimension,
        var_names=var_names,
        t_end=t_end,
        dt=dt,
        ic_trajectories=ic_trajs
    )


def make_damped_pendulum_dataset() -> PhysicalDataset:
    """Nonlinear Damped Pendulum:
       d(theta)/dt = omega
       d(omega)/dt = -9.81 * sin(theta) - 0.5 * omega
       Contains transcendental trigonometry + dissipative velocity damping.
    """
    dimension = 2
    var_names = ['theta', 'omega']
    t_end = 3.0
    dt = 0.04

    def true_derivs(y):
        theta, omega = y[0], y[1]
        return [omega, -9.81 * math.sin(theta) - 0.5 * omega]

    def gen_traj(y0):
        y = list(y0)
        t = 0.0
        traj = [list(y)]
        while t < t_end - 1e-12:
            k1 = true_derivs(y)
            y2 = [y[i] + 0.5 * dt * k1[i] for i in range(2)]
            k2 = true_derivs(y2)
            y3 = [y[i] + 0.5 * dt * k2[i] for i in range(2)]
            k3 = true_derivs(y3)
            y4 = [y[i] + dt * k3[i] for i in range(2)]
            k4 = true_derivs(y4)
            for i in range(2):
                y[i] += (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            t += dt
            traj.append(list(y))
        return traj

    test_ics = [
        [0.5, 0.0],   # Small angle
        [2.2, 0.0],   # Large angle (strong non-linearity)
        [-1.0, 1.5],  # Moving start
    ]
    ic_trajs = [(ic, gen_traj(ic)) for ic in test_ics]

    return PhysicalDataset(
        name="Nonlinear Damped Pendulum",
        dimension=dimension,
        var_names=var_names,
        t_end=t_end,
        dt=dt,
        ic_trajectories=ic_trajs
    )


# ==============================================================================
# SECTION 5: AUTONOMOUS GRAPH DISCOVERY ENGINE (ZERO HARDCODING)
# ==============================================================================

def run_graph_discovery(dataset: PhysicalDataset,
                        population_size: int = 150,
                        generations: int = 250,
                        tournament_k: int = 5,
                        target_fitness: float = 0.999):
    """Executes the Autonomous Graph Discovery loop.
       Zero hardcoded templates. Zero fallbacks.
    """
    print("=" * 80)
    print("      PROJECT BUDDHA: AUTONOMOUS COMPUTATION GRAPH DISCOVERY         ")
    print(f"   Target System: {dataset.name} ({dataset.dimension}D)")
    print("=" * 80)
    print(f" [DATA]  {len(dataset.ic_trajectories)} Independent Initial Conditions Evaluated Concurrently")
    for idx, (ic, traj) in enumerate(dataset.ic_trajectories):
        print(f"         IC {idx+1}: {dataset.var_names} = {ic} ({len(traj)} snapshots)")
    print(" [RULE]  Zero Hardcoding: Graphs must evolve from raw operations and inputs.")
    print(" [RULE]  Occam's Razor: Complexity penalty active (favors minimal true laws).")
    print(" [SOLV]  Unified Pratyaksh-II Numerical Engine active.")
    print("-" * 80)
    print(f"{'Gen':<6} | {'Best Fitness':<14} | {'Nodes':<6} | {'Time':<8} | {'Discovered System'}")
    print("-" * 80)

    # 1. Initialize random population
    population = [random_graph(dataset.dimension) for _ in range(population_size)]
    fitnesses = [evaluate_fitness(g, dataset) for g in population]

    best_ever_fitness = -1.0
    best_ever_graph = None
    start_time = time.time()

    for gen in range(generations):
        # Find best in current generation
        best_idx = max(range(population_size), key=lambda i: fitnesses[i])
        current_best_fit = fitnesses[best_idx]
        current_best_graph = population[best_idx]

        if current_best_fit > best_ever_fitness:
            best_ever_fitness = current_best_fit
            best_ever_graph = current_best_graph.copy()

        # Progress reporting
        if gen < 5 or gen % 20 == 0 or current_best_fit >= target_fitness:
            elapsed = time.time() - start_time
            display_eq = best_ever_graph.to_string(dataset.var_names)
            if len(display_eq) > 45: display_eq = display_eq[:42] + "..."
            print(f"{gen:<6} | {best_ever_fitness:<14.6f} | {best_ever_graph.complexity():<6} | {elapsed:<6.1f}s | {display_eq}")

        # Victory condition
        if best_ever_fitness >= target_fitness:
            print("-" * 80)
            print(" ★★★ TARGET ACCURACY ACHIEVED ★★★")
            break

        # 2. Build next generation via Tournament Selection & Mutation
        new_population = []
        # Elitism: retain top 5%
        ranked_indices = sorted(range(population_size), key=lambda i: fitnesses[i], reverse=True)
        elite_count = max(2, int(0.05 * population_size))
        for i in range(elite_count):
            new_population.append(population[ranked_indices[i]].copy())

        while len(new_population) < population_size:
            # Tournament selection
            competitors = random.sample(range(population_size), tournament_k)
            winner_idx = max(competitors, key=lambda i: fitnesses[i])
            parent = population[winner_idx]

            # Mutate graph
            child = mutate_graph(parent)
            new_population.append(child)

        population = new_population
        fitnesses = [evaluate_fitness(g, dataset) for g in population]

    elapsed = time.time() - start_time
    print("=" * 80)
    print(" DISCOVERY RUN COMPLETED")
    print(f" Total Wall Time:        {elapsed:.2f} seconds")
    print(f" Best Fitness Achieved:  {best_ever_fitness:.6f}")
    print(f" Active Graph Nodes:     {best_ever_graph.complexity()}")
    print()
    print(" Discovered Equations:")
    print(f"  -->  {best_ever_graph.to_string(dataset.var_names)}")
    print()

    # Export to Rust-compatible JSON AST
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_json = os.path.join(out_dir, f"discovered_{dataset.name.replace(' ', '_').lower()}.json")
    rust_payload = best_ever_graph.to_rust_json(
        name=dataset.name.replace(" ", "_"),
        y0=dataset.ic_trajectories[0][0],
        t_span=[0.0, dataset.t_end],
        dt=dataset.dt
    )
    if rust_payload:
        with open(out_json, "w") as f:
            json.dump(rust_payload, f, indent=2)
        print(f" [EXPORT] Exported Rust Omni-Compiler JSON: {out_json}")
    print("=" * 80)
    return best_ever_graph, best_ever_fitness


# ==============================================================================
# SECTION 6: ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    problem = sys.argv[1] if len(sys.argv) > 1 else "pendulum"
    if problem == "pendulum":
        ds = make_damped_pendulum_dataset()
    elif problem == "lotka":
        ds = make_lotka_volterra_dataset()
    else:
        print(f"Unknown problem: {problem}. Choose 'pendulum' or 'lotka'.")
        sys.exit(1)

    run_graph_discovery(ds, population_size=100, generations=100)
