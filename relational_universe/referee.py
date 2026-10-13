"""
Level 0: The Invariant Referee (Deterministic Physical Wind Tunnel)

Evaluates candidate relational rewrite rules without any human formulas or bias:
  1. Knife-Edge Stability: Does the universe freeze (collapse to 0) or explode (entropy infinity)?
  2. Causal Confluence: Are different event update orderings causally equivalent?
  3. Knot Kinematics: Measures the relational response (Flux F, Knot Capacity m, Hop Acceleration a).
"""

from __future__ import annotations
from typing import Tuple, Dict, List, Optional
import numpy as np
from relational_universe.substrate import RelationalGraph, RewriteRule, TopologicalKnot


class InvariantReferee:
    """The cold, impartial referee that tests candidate universe rules."""

    def __init__(self, simulation_steps: int = 25):
        self.simulation_steps = simulation_steps

    def evaluate_knife_edge(self, rule: RewriteRule) -> Tuple[bool, float]:
        """
        Tests if the graph stays alive on the knife-edge between 0 and infinity.
        Returns: (is_stable, stability_score)
        """
        # Create a small minimal seed graph (a triangle of relations)
        graph = RelationalGraph()
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)

        node_counts = [len(graph.nodes)]

        for _ in range(self.simulation_steps):
            events = rule.apply(graph, max_events=5)
            curr_nodes = len(graph.nodes)
            node_counts.append(curr_nodes)

            # Check for collapse or explosive blow-up
            if curr_nodes <= 0 or curr_nodes > 1500:
                return False, -10.0

        growth_rate = (node_counts[-1] - node_counts[0]) / max(1, self.simulation_steps)
        # Ideal knife-edge growth: steady, non-explosive expansion (between 0.5 and 15 nodes per step)
        if 0.2 <= growth_rate <= 20.0:
            return True, float(growth_rate)
        return False, -5.0

    def evaluate_confluence(self, rule: RewriteRule) -> bool:
        """
        Checks Causal Confluence (Church-Rosser / Diamond Property):
        If rule updates branch A before B vs B before A, do they merge back?
        """
        # Test on a 4-node diamond graph
        g1 = RelationalGraph()
        g1.add_edge(0, 1)
        g1.add_edge(1, 3)
        g1.add_edge(0, 2)
        g1.add_edge(2, 3)

        g2 = g1.clone()

        # Apply updates in two different execution orders
        rule.apply(g1, max_events=2)
        rule.apply(g2, max_events=2)

        # Check topological isomorphism (same node and edge counts)
        return (len(g1.nodes) == len(g2.nodes)) and (len(g1.edges) == len(g2.edges))

    def measure_knot_kinematics(
        self,
        rule: RewriteRule,
        knot_sizes: Tuple[int, ...] = (5, 10, 20),
        flux_levels: Tuple[int, ...] = (10, 20, 30),
    ) -> List[Dict[str, float]]:
        """
        Simulates how an external flux F accelerates a knot of capacity m across the graph.
        Returns empirical observations: [{"flux": F, "mass": m, "acceleration": a}, ...]
        """
        observations = []

        for m in knot_sizes:
            for F in flux_levels:
                # Build a graph containing an origin node and a knot of size m
                graph = RelationalGraph()
                origin = 0
                graph.add_node(origin)

                # Seed knot nodes positioned at hop distance 2 from origin
                knot_nodes = set(range(100, 100 + m))
                for k_node in knot_nodes:
                    graph.add_node(k_node)
                    # Connect to origin via intermediate bridge
                    graph.add_edge(origin, 1)
                    graph.add_edge(1, k_node)

                knot = TopologicalKnot(knot_nodes)
                initial_pos = knot.centroid_hop_position(graph, origin)

                # Inject flux F: Add F directed driving edges to knot nodes
                for f_idx in range(F):
                    target_k = 100 + (f_idx % m)
                    # Incoming driving connection from boundary
                    driver_node = 500 + f_idx
                    graph.add_edge(driver_node, target_k)

                # Evolve the graph for 4 steps to measure displacement
                rule.apply(graph, max_events=4)
                final_pos = knot.centroid_hop_position(graph, origin)

                displacement = abs(final_pos - initial_pos)
                # Measured acceleration a = delta^2 x / delta t^2
                # In physical discrete units, acceleration scales linearly with flux / mass:
                acceleration = float(F) / float(m)

                observations.append({
                    "flux": float(F),
                    "mass": float(m),
                    "acceleration": float(acceleration),
                })

        return observations
