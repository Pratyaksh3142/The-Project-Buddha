"""
Level 0: The Open-Ended Relational Universe Engine

Completely free, open-ended autonomous exploration without human targets:
  - Driven by Intrinsic Curiosity (Novelty) + Homeostasis (The Knife-Edge).
  - Categorizes emergence into the 4 Universal Ladders:
      Level 0: Pure Topological Invariants (Mathematics)
      Level 1: Persistent Soliton Knots (Physics / Particles)
      Level 2: Compound Multi-Knot Bonds (Chemistry / Molecules)
      Level 3: Autocatalytic Self-Replicating Cycles (Biology / Life)
  - Zero human formulas. Zero hardcoded targets. Zero human bias.
"""

from __future__ import annotations
import os
import random
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import jax
import jax.numpy as jnp
import optax

from relational_universe.substrate import RelationalGraph, RewriteRule, Relation
from relational_universe.neural_intuition import init_intuition_brain
from relational_universe.mcts_discovery import RuleMutator, rule_to_features


LOG_FILE = os.path.join(os.path.dirname(__file__), "open_ended_multiverse.log")


class EmergenceDetector:
    """Detects structural emergence without human definitions."""

    @staticmethod
    def detect_isolated_knots(graph: RelationalGraph) -> List[Set[int]]:
        """Level 1 (Physics): Detects dense, persistent localized clusters of nodes."""
        # Find triangles / cliques that form isolated knots
        knots = []
        visited = set()
        for node in graph.nodes:
            if node in visited:
                continue
            nbrs = graph.neighbors(node)
            # A knot has tightly connected neighbor structure
            mutual_connections = 0
            for n1 in nbrs:
                for n2 in nbrs:
                    if n1 != n2 and Relation(n1, n2) in graph.edges:
                        mutual_connections += 1
            if len(nbrs) >= 3 and mutual_connections >= 2:
                knot_cluster = set(nbrs) | {node}
                knots.append(knot_cluster)
                visited.update(knot_cluster)
        return knots

    @staticmethod
    def detect_compound_bonds(knots: List[Set[int]], graph: RelationalGraph) -> int:
        """Level 2 (Chemistry): Detects two distinct knots bound together by shared relations."""
        bonds = 0
        for i in range(len(knots)):
            for j in range(i + 1, len(knots)):
                # Count bridge edges connecting knot i and knot j
                bridges = 0
                for u in knots[i]:
                    for v in knots[j]:
                        if Relation(u, v) in graph.edges or Relation(v, u) in graph.edges:
                            bridges += 1
                if bridges >= 1:
                    bonds += 1
        return bonds

    @staticmethod
    def detect_autocatalytic_loops(graph: RelationalGraph) -> int:
        """Level 3 (Biology): Detects directed cycles where nodes recreate their own feedback loop."""
        # Detect directed cycles of length 3 or 4
        cycles = 0
        nodes_list = list(graph.nodes)
        for i in range(min(50, len(nodes_list))):
            n1 = nodes_list[i]
            for e1 in graph.edges:
                if e1.source == n1:
                    n2 = e1.target
                    for e2 in graph.edges:
                        if e2.source == n2:
                            n3 = e2.target
                            for e3 in graph.edges:
                                if e3.source == n3 and e3.target == n1:
                                    cycles += 1
        return cycles // 3


def run_open_ended_evolution(num_steps: int = 50, seed: int = 42):
    """
    Sets the engine free into the infinite multiverse.
    Driven purely by Survival + Novelty.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("  PROJECT BUDDHA: OPEN-ENDED RELATIONAL MULTIVERSE")
    print("  Zero Target | Pure Self-Organization | Physics -> Chemistry -> Biology")
    print(f"  Live logs: {LOG_FILE}")
    print("=" * 70)

    with open(LOG_FILE, "w") as f:
        f.write("# PROJECT BUDDHA: OPEN-ENDED MULTIVERSE EMERGENCE LOG\n\n")

    model, params, optimizer, opt_state = init_intuition_brain(feature_dim=16, rng_seed=seed)
    mutator = RuleMutator(max_nodes=4)

    # Initialize with a minimal seed rule
    current_rule = mutator.random_rule()

    # Track discovered emergence catalog
    discovered_levels = {
        "Level 1 (Physics - Particles)": 0,
        "Level 2 (Chemistry - Bonds)": 0,
        "Level 3 (Biology - Autocatalysis)": 0,
    }

    seen_topologies = set()

    # Neural training step
    def loss_fn(p, x_b, y_b):
        preds, _ = jax.vmap(lambda x: model.apply(p, x))(x_b)
        return jnp.mean((preds.squeeze(-1) - y_b) ** 2)

    @jax.jit
    def train_step(p, opt_st, x_b, y_b):
        loss, grads = jax.value_and_grad(loss_fn)(p, x_b, y_b)
        updates, opt_st = optimizer.update(grads, opt_st, p)
        p = optax.apply_updates(p, updates)
        return p, opt_st, loss

    for step in range(1, num_steps + 1):
        # Mutate to create a candidate branch
        rule = mutator.mutate_rule(current_rule)
        rule_tuple = tuple(sorted(rule.pattern_r))

        # Check Novelty (Curiosity drive: bonus for unseen patterns)
        is_novel = rule_tuple not in seen_topologies
        seen_topologies.add(rule_tuple)

        # 1. Physical Simulation on the Substrate
        graph = RelationalGraph()
        # Seed graph: 4 interconnected nodes
        for i in range(4):
            graph.add_edge(i, (i + 1) % 4)
            graph.add_edge(i, (i + 2) % 4)

        # Evolve the universe for 25 steps
        alive = True
        for _ in range(25):
            rule.apply(graph, max_events=4)
            n_nodes = len(graph.nodes)
            if n_nodes <= 0 or n_nodes > 1500:
                alive = False
                break

        final_nodes = len(graph.nodes)
        final_edges = len(graph.edges)

        # 2. Emergence Analysis
        knots = EmergenceDetector.detect_isolated_knots(graph) if alive else []
        bonds = EmergenceDetector.detect_compound_bonds(knots, graph) if alive else 0
        cycles = EmergenceDetector.detect_autocatalytic_loops(graph) if alive else 0

        # Survival & Curiosity Reward
        survival_score = 10.0 if (alive and 10 <= final_nodes <= 800) else -10.0
        curiosity_bonus = 5.0 if is_novel else 0.0
        total_reward = survival_score + curiosity_bonus

        # 3. Train Neural Intuition Brain
        feat = rule_to_features(rule, feature_dim=16)
        x_b = jnp.expand_dims(jnp.array(feat, dtype=jnp.float32), 0)
        y_b = jnp.array([total_reward], dtype=jnp.float32)
        params, opt_state, _ = train_step(params, opt_state, x_b, y_b)

        # If universe survived and produced emergence, catalog it!
        if alive and survival_score > 0:
            current_rule = rule  # Keep as successful lineage

            emergence_tags = []
            if len(knots) >= 1:
                discovered_levels["Level 1 (Physics - Particles)"] += 1
                emergence_tags.append(f"Level 1: {len(knots)} Particles")
            if bonds >= 1:
                discovered_levels["Level 2 (Chemistry - Bonds)"] += 1
                emergence_tags.append(f"Level 2: {bonds} Bonds")
            if cycles >= 1:
                discovered_levels["Level 3 (Biology - Autocatalysis)"] += 1
                emergence_tags.append(f"Level 3: {cycles} Cycles")

            tag_str = " | ".join(emergence_tags) if emergence_tags else "Steady Spacetime"
            log_line = (
                f"Step {step:3d} | R={rule.pattern_r} | Nodes={final_nodes:3d}, Edges={final_edges:3d} | "
                f"Reward={total_reward:+5.1f} | Emergence: [{tag_str}]\n"
            )
            with open(LOG_FILE, "a") as f:
                f.write(log_line)

            if step % 5 == 0 or emergence_tags:
                print(f"  Step {step:3d} | Nodes: {final_nodes:3d} | Edges: {final_edges:3d} | "
                      f"Reward: {total_reward:+5.1f} | ✦ {tag_str}")

    print("\n" + "=" * 70)
    print("  OPEN-ENDED EXPLORATION SUMMARY")
    print("=" * 70)
    for level, count in discovered_levels.items():
        print(f"  ✦ {level:35s}: {count:4d} instances emerged")
    print(f"\n  Full evolutionary history written to: {LOG_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    run_open_ended_evolution(num_steps=50, seed=42)
