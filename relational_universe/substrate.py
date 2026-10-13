"""
Level 0: The Pure Relational Substrate

A universe built from absolute scratch:
  - Entities (Nodes): Discrete points of existence with no coordinates.
  - Relations (Edges): Directed connections (A -> B).
  - Rewriting Rules: Local morphisms (Pattern_L -> Pattern_R).
  - Topological Knots: Persistent dense subgraphs (Particles/Matter).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Tuple, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class Relation:
    source: int
    target: int


@dataclass
class RelationalGraph:
    """A purely discrete topological network of entities and directed relations."""
    nodes: Set[int] = field(default_factory=set)
    edges: Set[Relation] = field(default_factory=set)

    def add_node(self, node_id: int) -> None:
        self.nodes.add(node_id)

    def add_edge(self, source: int, target: int) -> None:
        self.nodes.add(source)
        self.nodes.add(target)
        self.edges.add(Relation(source, target))

    def remove_edge(self, source: int, target: int) -> None:
        rel = Relation(source, target)
        if rel in self.edges:
            self.edges.remove(rel)

    def neighbors(self, node: int) -> Set[int]:
        """Returns direct neighbors (both incoming and outgoing)."""
        nbrs = set()
        for e in self.edges:
            if e.source == node:
                nbrs.add(e.target)
            elif e.target == node:
                nbrs.add(e.source)
        return nbrs

    def shortest_path_hops(self, start: int, end: int) -> Optional[int]:
        """Measures distance as pure graph hop count via Breadth-First Search."""
        if start == end:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            curr, dist = queue.pop(0)
            for nbr in self.neighbors(curr):
                if nbr == end:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return None

    def clone(self) -> RelationalGraph:
        return RelationalGraph(
            nodes=set(self.nodes),
            edges=set(self.edges),
        )


@dataclass
class RewriteRule:
    """
    A local topological replacement rule:
      L: Pattern of relations to match
      R: Pattern of relations to replace with
    """
    pattern_l: List[Tuple[int, int]]
    pattern_r: List[Tuple[int, int]]

    def apply(self, graph: RelationalGraph, max_events: int = 10) -> int:
        """
        Scans graph for sub-pattern L and replaces with R.
        Returns the number of rewriting events executed.
        """
        events = 0
        # For a minimal local rule: find a node with matching edge pair
        if len(self.pattern_l) == 2:
            edges_list = list(graph.edges)
            for i in range(len(edges_list)):
                e1 = edges_list[i]
                for j in range(i + 1, len(edges_list)):
                    e2 = edges_list[j]
                    # Check if e1 and e2 connect: (A -> B) and (B -> C)
                    if e1.target == e2.source:
                        a, b, c = e1.source, e1.target, e2.target
                        # Execute rewrite: remove old edges
                        graph.remove_edge(a, b)
                        graph.remove_edge(b, c)
                        # Add new node D and connect pattern R
                        new_node = max(graph.nodes) + 1 if graph.nodes else 1
                        graph.add_node(new_node)
                        for r_src, r_dst in self.pattern_r:
                            s = a if r_src == 0 else (b if r_src == 1 else (c if r_src == 2 else new_node))
                            t = a if r_dst == 0 else (b if r_dst == 1 else (c if r_dst == 2 else new_node))
                            graph.add_edge(s, t)
                        events += 1
                        if events >= max_events:
                            return events
        return events


class TopologicalKnot:
    """
    Represents a persistent localized knot of relations (a particle).
    Observable Slot 2 (Capacity/Mass): Number of nodes inside the knot.
    """
    def __init__(self, node_ids: Set[int]):
        self.node_ids = set(node_ids)

    @property
    def mass(self) -> int:
        """The capacity (number of nodes) contained in the knot."""
        return len(self.node_ids)

    def centroid_hop_position(self, graph: RelationalGraph, origin_node: int) -> float:
        """Computes the mean hop distance of the knot from a reference origin node."""
        distances = []
        for n in self.node_ids:
            d = graph.shortest_path_hops(origin_node, n)
            if d is not None:
                distances.append(d)
        if not distances:
            return 0.0
        return float(np.mean(distances))
