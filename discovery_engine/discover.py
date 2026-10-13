#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║              PROJECT BUDDHA — DISCOVERY ENGINE v1.0              ║
║   Automated Discovery of Physical Laws via Genetic Programming   ║
║                                                                  ║
║   The AI Brain that discovers equations.                         ║
║   The Pratyaksh Omni-Compiler is the muscle that evaluates them. ║
╚══════════════════════════════════════════════════════════════════╝

This engine uses Genetic Programming (GP) to search the space of
mathematical Abstract Syntax Trees (ASTs) for systems of ODEs that
match observed trajectory data.

Phase 1: Pure-Python RK4 evaluator (this file)
Phase 2: Pratyaksh/Rust FFI backend for stiff/exotic physics
"""

import random
import copy
import math
import time
import json
import sys
import os

# ═══════════════════════════════════════════════════════════════
# SECTION 1: AST CORE — Mathematical Expression Trees
# ═══════════════════════════════════════════════════════════════

class Const:
    """A numeric constant leaf node."""
    __slots__ = ['value']
    def __init__(self, value):
        self.value = float(value)

class Var:
    """A state variable y[index] leaf node."""
    __slots__ = ['index']
    def __init__(self, index):
        self.index = int(index)

class BinOp:
    """Binary operation: left ○ right, where ○ ∈ {+, -, *}."""
    __slots__ = ['op', 'left', 'right']
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class Neg:
    """Unary negation: -(arg)."""
    __slots__ = ['arg']
    def __init__(self, arg):
        self.arg = arg


def expr_to_str(expr, var_names=None):
    """Pretty-print an expression tree."""
    if isinstance(expr, Const):
        v = expr.value
        if abs(v - round(v)) < 0.0005 and abs(v) < 1000:
            return str(int(round(v)))
        return f"{v:.4f}"
    if isinstance(expr, Var):
        if var_names and expr.index < len(var_names):
            return var_names[expr.index]
        return f"y{expr.index}"
    if isinstance(expr, Neg):
        inner = expr_to_str(expr.arg, var_names)
        if isinstance(expr.arg, Const):
            return f"{-expr.arg.value:.4f}"
        return f"(-{inner})"
    if isinstance(expr, BinOp):
        l = expr_to_str(expr.left, var_names)
        r = expr_to_str(expr.right, var_names)
        return f"({l} {expr.op} {r})"
    return "?"


def evaluate(expr, y):
    """Evaluate an expression tree given state vector y. No protection."""
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        return y[expr.index]
    if isinstance(expr, Neg):
        return -evaluate(expr.arg, y)
    if isinstance(expr, BinOp):
        l = evaluate(expr.left, y)
        r = evaluate(expr.right, y)
        if expr.op == '+': return l + r
        if expr.op == '-': return l - r
        if expr.op == '*': return l * r
    return 0.0


def safe_evaluate(expr, y):
    """Protected evaluation — returns None on any numerical failure."""
    try:
        val = evaluate(expr, y)
        if not math.isfinite(val) or abs(val) > 1e12:
            return None
        return val
    except Exception:
        return None


def tree_size(expr):
    """Count total nodes."""
    if isinstance(expr, (Const, Var)):
        return 1
    if isinstance(expr, Neg):
        return 1 + tree_size(expr.arg)
    if isinstance(expr, BinOp):
        return 1 + tree_size(expr.left) + tree_size(expr.right)
    return 1


def tree_depth(expr):
    """Maximum depth of tree."""
    if isinstance(expr, (Const, Var)):
        return 0
    if isinstance(expr, Neg):
        return 1 + tree_depth(expr.arg)
    if isinstance(expr, BinOp):
        return 1 + max(tree_depth(expr.left), tree_depth(expr.right))
    return 0


def get_nodes_with_parent(expr, parent=None, attr=None):
    """Flatten tree into [(node, parent_obj, attr_name), ...]."""
    result = [(expr, parent, attr)]
    if isinstance(expr, Neg):
        result.extend(get_nodes_with_parent(expr.arg, expr, 'arg'))
    elif isinstance(expr, BinOp):
        result.extend(get_nodes_with_parent(expr.left, expr, 'left'))
        result.extend(get_nodes_with_parent(expr.right, expr, 'right'))
    return result


def get_const_nodes(expr):
    """Get references to all Const nodes (modifying .value modifies the tree)."""
    result = []
    if isinstance(expr, Const):
        result.append(expr)
    elif isinstance(expr, Neg):
        result.extend(get_const_nodes(expr.arg))
    elif isinstance(expr, BinOp):
        result.extend(get_const_nodes(expr.left))
        result.extend(get_const_nodes(expr.right))
    return result


def simplify(expr):
    """Basic algebraic simplification for readability."""
    if isinstance(expr, (Const, Var)):
        return expr

    if isinstance(expr, Neg):
        a = simplify(expr.arg)
        if isinstance(a, Const):
            return Const(-a.value)
        if isinstance(a, Neg):
            return a.arg
        return Neg(a)

    if isinstance(expr, BinOp):
        l = simplify(expr.left)
        r = simplify(expr.right)

        # Constant folding: if both sides are constants, compute result
        if isinstance(l, Const) and isinstance(r, Const):
            try:
                if expr.op == '+': return Const(l.value + r.value)
                if expr.op == '-': return Const(l.value - r.value)
                if expr.op == '*': return Const(l.value * r.value)
            except Exception:
                pass

        if expr.op == '+':
            if isinstance(l, Const) and abs(l.value) < 1e-10: return r
            if isinstance(r, Const) and abs(r.value) < 1e-10: return l
        elif expr.op == '-':
            if isinstance(r, Const) and abs(r.value) < 1e-10: return l
        elif expr.op == '*':
            if isinstance(l, Const) and abs(l.value) < 1e-10: return Const(0.0)
            if isinstance(r, Const) and abs(r.value) < 1e-10: return Const(0.0)
            if isinstance(l, Const) and abs(l.value - 1.0) < 1e-10: return r
            if isinstance(r, Const) and abs(r.value - 1.0) < 1e-10: return l

        return BinOp(expr.op, l, r)

    return expr


def to_rust_json(expr):
    """Convert AST to JSON dict compatible with the Rust Omni-Compiler.
    Rust uses #[serde(tag = "type", content = "data")] for Expr enum."""
    if isinstance(expr, Const):
        return {"type": "Constant", "data": expr.value}
    if isinstance(expr, Var):
        return {"type": "State", "data": expr.index}
    if isinstance(expr, Neg):
        return {"type": "Neg", "data": to_rust_json(expr.arg)}
    if isinstance(expr, BinOp):
        l = to_rust_json(expr.left)
        r = to_rust_json(expr.right)
        if expr.op == '+': return {"type": "Add", "data": [l, r]}
        if expr.op == '-': return {"type": "Sub", "data": [l, r]}
        if expr.op == '*': return {"type": "Mul", "data": [l, r]}
    return {"type": "Constant", "data": 0.0}


# ═══════════════════════════════════════════════════════════════
# SECTION 2: RANDOM TREE GENERATION
# ═══════════════════════════════════════════════════════════════

def random_terminal(n_vars):
    """Generate a random terminal (leaf) node."""
    if random.random() < 0.45:
        return Const(random.uniform(-15.0, 15.0))
    return Var(random.randint(0, n_vars - 1))


def random_expr(max_depth, n_vars, depth=0):
    """Generate a random expression tree using the 'grow' method."""
    if depth >= max_depth:
        return random_terminal(n_vars)

    r = random.random()
    if r < 0.35:
        # Terminal (early stop — keeps some trees small)
        return random_terminal(n_vars)
    elif r < 0.85:
        # Binary operator
        op = random.choice(['+', '-', '*'])
        return BinOp(
            op,
            random_expr(max_depth, n_vars, depth + 1),
            random_expr(max_depth, n_vars, depth + 1),
        )
    else:
        # Negation
        return Neg(random_expr(max_depth, n_vars, depth + 1))


# ═══════════════════════════════════════════════════════════════
# SECTION 3: PROTECTED RK4 INTEGRATOR
# ═══════════════════════════════════════════════════════════════

def rk4_integrate(equations, y0, t_end, dt):
    """
    Integrate an ODE system using fixed-step RK4 with crash protection.
    
    equations: list of Expr (one per state dimension)
    y0:        initial state vector [y0, y1, ...]
    t_end:     simulation end time
    dt:        fixed time step
    
    Returns: list of state snapshots [[y0,y1,...], ...] or None if diverged.
    """
    dim = len(y0)
    y = list(y0)
    t = 0.0
    trajectory = [list(y)]

    while t < t_end - 1e-12:
        h = min(dt, t_end - t)

        # ---- k1 ----
        k1 = []
        for eq in equations:
            v = safe_evaluate(eq, y)
            if v is None: return None
            k1.append(v)

        # ---- k2 ----
        y2 = [y[i] + 0.5 * h * k1[i] for i in range(dim)]
        k2 = []
        for eq in equations:
            v = safe_evaluate(eq, y2)
            if v is None: return None
            k2.append(v)

        # ---- k3 ----
        y3 = [y[i] + 0.5 * h * k2[i] for i in range(dim)]
        k3 = []
        for eq in equations:
            v = safe_evaluate(eq, y3)
            if v is None: return None
            k3.append(v)

        # ---- k4 ----
        y4 = [y[i] + h * k3[i] for i in range(dim)]
        k4 = []
        for eq in equations:
            v = safe_evaluate(eq, y4)
            if v is None: return None
            k4.append(v)

        # ---- Update state ----
        for i in range(dim):
            y[i] += (h / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            if not math.isfinite(y[i]) or abs(y[i]) > 1e12:
                return None

        t += h
        trajectory.append(list(y))

    return trajectory


# ═══════════════════════════════════════════════════════════════
# SECTION 4: GENETIC OPERATORS
# ═══════════════════════════════════════════════════════════════

def mutate(expr, n_vars, max_depth=4):
    """Apply one random mutation to an expression tree."""
    expr = copy.deepcopy(expr)
    nodes = get_nodes_with_parent(expr)
    r = random.random()

    # --- Constant perturbation (fine-tuning) ---
    if r < 0.30:
        consts = [(n, p, a) for n, p, a in nodes if isinstance(n, Const)]
        if consts:
            node, parent, attr = random.choice(consts)
            # Small Gaussian nudge
            new_val = node.value + random.gauss(0, 1.5)
            new_node = Const(new_val)
            if parent is None:
                return new_node
            setattr(parent, attr, new_node)
            return expr

    # --- Constant replacement (large jump) ---
    if r < 0.42:
        consts = [(n, p, a) for n, p, a in nodes if isinstance(n, Const)]
        if consts:
            node, parent, attr = random.choice(consts)
            new_node = Const(random.uniform(-15, 15))
            if parent is None:
                return new_node
            setattr(parent, attr, new_node)
            return expr

    # --- Shrink mutation (promotes simplicity / Occam's razor) ---
    if r < 0.60:
        non_terms = [(n, p, a) for n, p, a in nodes
                     if not isinstance(n, (Const, Var)) and p is not None]
        if non_terms:
            _, parent, attr = random.choice(non_terms)
            setattr(parent, attr, random_terminal(n_vars))
            return expr

    # --- Point mutation (replace a random node with a new subtree) ---
    if r < 0.82:
        node, parent, attr = random.choice(nodes)
        new_node = random_expr(max(1, max_depth - 1), n_vars)
        if parent is None:
            return new_node
        setattr(parent, attr, new_node)
        return expr

    # --- Subtree mutation (replace random internal node) ---
    if len(nodes) > 1:
        _, parent, attr = random.choice(nodes[1:])
        setattr(parent, attr, random_expr(2, n_vars))
        return expr

    return random_expr(max_depth, n_vars)


def crossover(p1, p2, n_vars):
    """Subtree crossover: swap a random subtree between two parents."""
    p1 = copy.deepcopy(p1)
    p2 = copy.deepcopy(p2)
    nodes1 = get_nodes_with_parent(p1)
    nodes2 = get_nodes_with_parent(p2)

    n1, par1, a1 = random.choice(nodes1)
    n2, par2, a2 = random.choice(nodes2)

    # Inject subtree from p2 into p1
    if par1 is None:
        return copy.deepcopy(n2)
    setattr(par1, a1, copy.deepcopy(n2))
    return p1


def tournament_select(population, fitnesses, k=7):
    """Select the best individual from k random competitors."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


# ═══════════════════════════════════════════════════════════════
# SECTION 5: FITNESS EVALUATION & CONSTANT REFINEMENT
# ═══════════════════════════════════════════════════════════════

def compute_fitness(individual, target_traj, y0, t_end, dt):
    """
    Fitness = 1 / (1 + MSE).
    Range: (0, 1]. Higher is better. 1.0 = perfect match.
    """
    traj = rk4_integrate(individual, y0, t_end, dt)
    if traj is None:
        return 0.0

    n = min(len(traj), len(target_traj))
    dim = len(y0)
    if n < 2:
        return 0.0

    sse = 0.0
    for i in range(n):
        for d in range(dim):
            diff = traj[i][d] - target_traj[i][d]
            sse += diff * diff

    mse = sse / (n * dim)
    return 1.0 / (1.0 + mse)


def refine_constants(individual, target_traj, y0, t_end, dt, n_iters=3):
    """
    Hill-climb on all constants in the individual's expression trees.
    Greedy coordinate descent: try small perturbations, keep improvements.
    """
    best_fit = compute_fitness(individual, target_traj, y0, t_end, dt)
    deltas = [2.0, -2.0, 0.5, -0.5, 0.1, -0.1, 0.02, -0.02]

    for _ in range(n_iters):
        for eq in individual:
            for cnode in get_const_nodes(eq):
                original = cnode.value
                for delta in deltas:
                    cnode.value = original + delta
                    new_fit = compute_fitness(individual, target_traj, y0, t_end, dt)
                    if new_fit > best_fit:
                        best_fit = new_fit
                        original = cnode.value  # Accept improvement
                    else:
                        cnode.value = original   # Revert
    return best_fit


# ═══════════════════════════════════════════════════════════════
# SECTION 6: TEST PROBLEMS
# ═══════════════════════════════════════════════════════════════

def make_freefall_problem():
    """
    Free fall under gravity.
    True equations: dx/dt = v, dv/dt = -9.81
    """
    G = 9.81
    x0, v0 = 10.0, 0.0
    t_end, dt = 1.0, 0.05
    y0 = [x0, v0]
    var_names = ['x', 'v']

    # Generate ground-truth trajectory analytically
    target = []
    t = 0.0
    while t <= t_end + 1e-12:
        x = x0 + v0 * t - 0.5 * G * t * t
        v = v0 - G * t
        target.append([x, v])
        t += dt

    return {
        'name':       'Free Fall (Newton\'s Law: F = ma)',
        'var_names':  var_names,
        'n_vars':     2,
        'y0':         y0,
        't_end':      t_end,
        'dt':         dt,
        'target':     target,
        'true_eqs':   ['v', '-9.81'],
        'true_g':     G,
    }


def make_harmonic_problem():
    """
    Simple Harmonic Oscillator.
    True equations: dx/dt = v, dv/dt = -x
    """
    x0, v0 = 1.0, 0.0
    t_end, dt = 6.28, 0.05
    y0 = [x0, v0]
    var_names = ['x', 'v']

    target = []
    t = 0.0
    while t <= t_end + 1e-12:
        x = x0 * math.cos(t) + v0 * math.sin(t)
        v = -x0 * math.sin(t) + v0 * math.cos(t)
        target.append([x, v])
        t += dt

    return {
        'name':       'Harmonic Oscillator',
        'var_names':  var_names,
        'n_vars':     2,
        'y0':         y0,
        't_end':      t_end,
        'dt':         dt,
        'target':     target,
        'true_eqs':   ['v', '-x'],
    }


def make_decay_problem():
    """
    Exponential decay: dy/dt = -2*y.
    """
    y_init = 5.0
    t_end, dt = 2.0, 0.05
    y0 = [y_init]
    var_names = ['y']

    target = []
    t = 0.0
    while t <= t_end + 1e-12:
        target.append([y_init * math.exp(-2.0 * t)])
        t += dt

    return {
        'name':       'Exponential Decay',
        'var_names':  var_names,
        'n_vars':     1,
        'y0':         y0,
        't_end':      t_end,
        'dt':         dt,
        'target':     target,
        'true_eqs':   ['-2 * y'],
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 7: MAIN DISCOVERY LOOP
# ═══════════════════════════════════════════════════════════════

def discover(problem, seed=42,
             pop_size=500, max_gen=500, max_depth=4,
             elite_frac=0.05, crossover_rate=0.60,
             tournament_k=7, fitness_goal=0.9999,
             refine_top_n=8, refine_iters=3):
    """
    Run the full Genetic Programming discovery loop.

    Returns: (best_individual, best_fitness, generation_found)
    """
    random.seed(seed)

    name      = problem['name']
    var_names = problem['var_names']
    n_vars    = problem['n_vars']
    y0        = problem['y0']
    t_end     = problem['t_end']
    dt        = problem['dt']
    target    = problem['target']

    # ── Banner ──
    print("═" * 66)
    print(f"  PROJECT BUDDHA — DISCOVERY ENGINE v1.0")
    print(f"  Target: {name}")
    print("═" * 66)
    print()
    print(f"  [DATA]   {len(target)} trajectory snapshots over {t_end:.2f} seconds")
    print(f"  [DATA]   Initial state: {dict(zip(var_names, y0))}")
    print(f"  [DATA]   True equations: {problem['true_eqs']}")
    print()
    print(f"  [SEARCH] Population: {pop_size}  |  Max depth: {max_depth}")
    print(f"  [SEARCH] Grammar: {{const, {', '.join(var_names)}, +, -, ×, neg}}")
    print(f"  [SEARCH] Constant refinement: top {refine_top_n}, {refine_iters} iters")
    print()

    # ── Initialize population ──
    population = []
    for _ in range(pop_size):
        ind = [random_expr(max_depth, n_vars) for _ in range(n_vars)]
        population.append(ind)

    best_ever_fit = 0.0
    best_ever_ind = None
    best_ever_gen = 0
    stagnation_counter = 0
    prev_best = 0.0

    wall_start = time.time()

    for gen in range(max_gen):
        # ── Evaluate fitness ──
        fitnesses = []
        for ind in population:
            fitnesses.append(compute_fitness(ind, target, y0, t_end, dt))

        # ── Sort and find best ──
        ranked = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        gen_best_idx = ranked[0]
        gen_best_fit = fitnesses[gen_best_idx]
        gen_best_ind = population[gen_best_idx]

        # ── Constant refinement on top individuals ──
        for rank in range(min(refine_top_n, len(ranked))):
            idx = ranked[rank]
            refined_fit = refine_constants(
                population[idx], target, y0, t_end, dt, refine_iters
            )
            fitnesses[idx] = refined_fit
            if refined_fit > gen_best_fit:
                gen_best_fit = refined_fit
                gen_best_ind = population[idx]
                gen_best_idx = idx

        # ── Update global best ──
        if gen_best_fit > best_ever_fit:
            best_ever_fit = gen_best_fit
            best_ever_ind = copy.deepcopy(gen_best_ind)
            best_ever_gen = gen

        # ── Stagnation detection ──
        if abs(gen_best_fit - prev_best) < 1e-8:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        prev_best = gen_best_fit

        # ── Print progress ──
        should_print = (
            gen < 5
            or gen % 10 == 0
            or gen_best_fit > fitness_goal
            or stagnation_counter == 0 and gen_best_fit > best_ever_fit - 0.001
        )
        if should_print:
            elapsed = time.time() - wall_start
            avg_fit = sum(fitnesses) / len(fitnesses)
            eq_strs = [expr_to_str(simplify(eq), var_names) for eq in gen_best_ind]
            eq_display = " | ".join(
                f"d{var_names[d]}/dt = {eq_strs[d]}" for d in range(n_vars)
            )
            print(
                f"  Gen {gen:4d} │ Best {gen_best_fit:.6f} │ "
                f"Avg {avg_fit:.4f} │ {elapsed:5.1f}s │ {eq_display}"
            )

        # ── Victory condition ──
        if gen_best_fit > fitness_goal:
            elapsed = time.time() - wall_start
            print()
            print("═" * 66)
            print("  ★  DISCOVERY ACHIEVED  ★")
            print("═" * 66)
            _print_discovery(best_ever_ind, best_ever_fit, best_ever_gen,
                             elapsed, var_names, n_vars, problem)
            _export_to_json(best_ever_ind, problem)
            return best_ever_ind, best_ever_fit, best_ever_gen

        # ── Inject random immigrants on stagnation ──
        if stagnation_counter >= 40:
            n_immigrants = int(0.25 * pop_size)
            for i in range(n_immigrants):
                idx = ranked[-(i + 1)]  # Replace worst individuals
                population[idx] = [random_expr(max_depth, n_vars)
                                   for _ in range(n_vars)]
            stagnation_counter = 0
            print(f"  ** Stagnation detected — injected {n_immigrants} immigrants **")

        # ── Build next generation ──
        new_pop = []

        # Elitism: keep top individuals unchanged
        n_elite = max(1, int(elite_frac * pop_size))
        for i in range(n_elite):
            new_pop.append(copy.deepcopy(population[ranked[i]]))

        # Fill rest with crossover and mutation
        while len(new_pop) < pop_size:
            if random.random() < crossover_rate:
                p1 = tournament_select(population, fitnesses, tournament_k)
                p2 = tournament_select(population, fitnesses, tournament_k)
                child = []
                for d in range(n_vars):
                    c = crossover(p1[d], p2[d], n_vars)
                    # Depth guard
                    if tree_depth(c) > max_depth + 2:
                        c = random_expr(max_depth, n_vars)
                    child.append(c)
                new_pop.append(child)
            else:
                parent = tournament_select(population, fitnesses, tournament_k)
                child = [mutate(copy.deepcopy(eq), n_vars, max_depth)
                         for eq in parent]
                new_pop.append(child)

        population = new_pop[:pop_size]

    # ── Did not converge ──
    elapsed = time.time() - wall_start
    print()
    print("═" * 66)
    print(f"  Search ended after {max_gen} generations ({elapsed:.1f}s)")
    print(f"  Best fitness achieved: {best_ever_fit:.8f}")
    print("═" * 66)
    _print_discovery(best_ever_ind, best_ever_fit, best_ever_gen,
                     elapsed, var_names, n_vars, problem)
    _export_to_json(best_ever_ind, problem)
    return best_ever_ind, best_ever_fit, best_ever_gen


def _print_discovery(individual, fitness, gen, elapsed, var_names, n_vars, problem):
    """Pretty-print the discovered equations."""
    print()
    for d in range(n_vars):
        eq_str = expr_to_str(simplify(individual[d]), var_names)
        true_str = problem['true_eqs'][d]
        print(f"  d{var_names[d]}/dt = {eq_str:30s}  (true: {true_str})")
    print()
    total_nodes = sum(tree_size(eq) for eq in individual)
    print(f"  Fitness:      {fitness:.8f}")
    print(f"  Complexity:   {total_nodes} nodes")
    print(f"  Generation:   {gen}")
    print(f"  Wall time:    {elapsed:.2f} seconds")

    # If it's the freefall problem, extract discovered g
    if 'true_g' in problem:
        try:
            discovered_g = abs(evaluate(simplify(individual[1]), [0.0, 0.0]))
            true_g = problem['true_g']
            error_pct = abs(discovered_g - true_g) / true_g * 100
            print()
            print(f"  Discovered g = {discovered_g:.4f} m/s²")
            print(f"  True g       = {true_g:.4f} m/s²")
            print(f"  Error        = {error_pct:.4f}%")
        except Exception:
            pass

    print()
    print("═" * 66)


def _export_to_json(individual, problem):
    """Export discovered equations to JSON for the Rust Omni-Compiler."""
    export_dir = os.path.dirname(os.path.abspath(__file__))
    export_path = os.path.join(export_dir, "discovered_system.json")

    system = {
        "name": problem['name'].replace("'", "").replace(" ", "_"),
        "dimension": problem['n_vars'],
        "is_pde": False,
        "equations": [to_rust_json(simplify(eq)) for eq in individual],
        "params": [],
        "default_y0": problem['y0'],
        "default_t_span": [0.0, problem['t_end']],
        "default_dt": problem['dt'],
    }

    with open(export_path, 'w') as f:
        json.dump(system, f, indent=2)

    print(f"  [EXPORT] Rust-compatible JSON → {export_path}")
    print()


# ═══════════════════════════════════════════════════════════════
# SECTION 8: ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    test = sys.argv[1] if len(sys.argv) > 1 else "freefall"

    if test == "freefall":
        problem = make_freefall_problem()
    elif test == "harmonic":
        problem = make_harmonic_problem()
    elif test == "decay":
        problem = make_decay_problem()
    elif test == "all":
        for name in ["freefall", "harmonic", "decay"]:
            print("\n\n")
            if name == "freefall":    p = make_freefall_problem()
            elif name == "harmonic":  p = make_harmonic_problem()
            else:                     p = make_decay_problem()
            discover(p)
        return
    else:
        print(f"Unknown test: {test}")
        print("Usage: python discover.py [freefall|harmonic|decay|all]")
        return

    discover(problem)


if __name__ == "__main__":
    main()
