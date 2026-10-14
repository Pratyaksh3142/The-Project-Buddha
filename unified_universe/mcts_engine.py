"""
Unified MCTS Engine for Multi-Physical & Mathematical Discovery.

Explores relational graph mutations and AlphaFold 3-style spatial relaxation
guided by the Neural Pairformer Brain and Universal Action Referee.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import math
from unified_universe.substrate import UniversalSubstrate
from unified_universe.action_referee import UniversalActionReferee
from unified_universe.neural_pairformer import UnifiedIntuitionEngine


class UnifiedMCTSNode:
    def __init__(self, substrate: UniversalSubstrate, parent=None, action_taken: Optional[int] = None, prior: float = 0.0):
        self.substrate = substrate
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, UnifiedMCTSNode] = {}
        self.is_expanded = False
        self.eval_result: Optional[Dict] = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class UnifiedMCTSEngine:
    """
    Directs Monte Carlo Tree Search over the universal relational substrate.
    """

    def __init__(
        self,
        referee: UniversalActionReferee,
        brain: UnifiedIntuitionEngine,
        num_simulations: int = 32,
        max_depth: int = 10,
        c_puct: float = 1.4
    ):
        self.referee = referee
        self.brain = brain
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.discovered_phases: List[Dict[str, Any]] = []

    def evaluate_node(self, node: UnifiedMCTSNode) -> Tuple[float, bool]:
        """
        Evaluates node using the Universal Action Referee.
        """
        if node.eval_result is None:
            node.eval_result = self.referee.compute_universal_action(node.substrate)

        res = node.eval_result
        tot_act = res["total_action"]

        # Reward minimization of action / free energy
        # Action lower -> reward higher
        reward = -np.tanh(tot_act / 20.0)

        # Bonus for non-trivial emergent physical phases
        qm = res["quantum"]
        sp = res["spatial"]
        alg = res["algebra"]
        bio = res["biology"]

        is_breakthrough = False
        if qm["is_superconducting"] or sp.get("is_compact_fold", False) or (alg["jacobi_violation"] == 0 and np.sum(np.abs(node.substrate.Z)) > 0) or bio["has_autocatalysis"]:
            is_breakthrough = True
            reward += 5.0
            self.discovered_phases.append({
                "action": tot_act,
                "quantum": qm,
                "spatial": sp,
                "algebra": alg,
                "biology": bio,
                "substrate": node.substrate.clone()
            })

        return float(reward), is_breakthrough

    def search(self, root_substrate: UniversalSubstrate, temperature: float = 0.5) -> Tuple[int, np.ndarray, float]:
        root = UnifiedMCTSNode(substrate=root_substrate.clone())

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            depth = 0

            # 1. SELECT
            while node.is_expanded and depth < self.max_depth:
                action, node = self._select_child(node)
                search_path.append(node)
                depth += 1

            # 2. EVALUATE & EXPAND
            leaf = search_path[-1]
            reward, is_breakthrough = self.evaluate_node(leaf)

            if depth < self.max_depth and not leaf.is_expanded:
                self._expand_node(leaf)

            # 3. BACKPROPAGATE
            for n in reversed(search_path):
                n.visit_count += 1
                n.value_sum += reward

        # Visit distribution
        visit_counts = np.zeros(self.brain.num_actions, dtype=np.float32)
        for a, child in root.children.items():
            visit_counts[a] = child.visit_count

        if np.sum(visit_counts) > 0:
            if temperature < 1e-3:
                best_action = int(np.argmax(visit_counts))
                visit_probs = visit_counts / np.sum(visit_counts)
            else:
                counts_temp = visit_counts ** (1.0 / temperature)
                visit_probs = counts_temp / np.sum(counts_temp)
                best_action = int(np.random.choice(len(visit_probs), p=visit_probs))
        else:
            best_action = 0
            visit_probs = np.ones(self.brain.num_actions, dtype=np.float32) / self.brain.num_actions

        return best_action, visit_probs, root.q_value

    def _select_child(self, node: UnifiedMCTSNode) -> Tuple[int, UnifiedMCTSNode]:
        best_score = -float('inf')
        best_action = 0
        best_child = None
        sqrt_parent = math.sqrt(node.visit_count + 1)

        for a, child in node.children.items():
            u = child.q_value + self.c_puct * child.prior * (sqrt_parent / (1 + child.visit_count))
            if u > best_score:
                best_score = u
                best_action = a
                best_child = child

        return best_action, best_child

    def _expand_node(self, node: UnifiedMCTSNode):
        pol_probs, _, delta_x = self.brain.predict(node.substrate)
        node.is_expanded = True
        N = node.substrate.N

        # Decode actions: mutating pair (i, j, channel)
        action_idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                for ch in range(min(node.substrate.d, 2)):
                    for delta in [-1.0, 1.0]:
                        if action_idx < self.brain.num_actions:
                            next_sub = node.substrate.clone()
                            next_sub.mutate_pair_tensor(i, j, ch, delta)
                            # AlphaFold 3-style coordinate flow
                            if next_sub.is_spatial:
                                next_sub.X += delta_x * 0.05

                            prior = float(pol_probs[action_idx])
                            child = UnifiedMCTSNode(substrate=next_sub, parent=node, action_taken=action_idx, prior=prior)
                            node.children[action_idx] = child
                            action_idx += 1
