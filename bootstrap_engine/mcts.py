"""
Monte Carlo Tree Search for Axiomatic Bootstrap Discovery.

Searches the discrete tensor space for non-trivial configurations that satisfy
pure algebraic axioms (Jacobi identity, Associativity) with EXACTLY ZERO violation.
Zero hardcoded targets. Zero human formulas.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import math
from bootstrap_engine.generator import CandidateTensorSpace
from bootstrap_engine.axioms import evaluate_algebraic_axioms
from bootstrap_engine.neural_brain import NeuralIntuitionEngine


class MCTSNode:
    def __init__(self, tensor: np.ndarray, parent=None, action_taken: Optional[int] = None, prior: float = 0.0):
        self.tensor = tensor
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, MCTSNode] = {}
        self.is_expanded = False

        # Axiomatic evaluation cache
        self.eval_result: Optional[Dict] = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def tensor_key(self) -> bytes:
        return self.tensor.tobytes()


class AxiomaticMCTS:
    """
    Directs neural-guided tree search to discover self-consistent algebraic structures.
    """

    def __init__(
        self,
        space: CandidateTensorSpace,
        brain: NeuralIntuitionEngine,
        axiom_system: str = "lie",
        target_mode: str = "any",
        c_puct: float = 1.4,
        num_simulations: int = 64,
        max_depth: int = 12
    ):
        self.space = space
        self.brain = brain
        self.axiom_system = axiom_system
        self.target_mode = target_mode
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.max_depth = max_depth

        # Track all unique genuine discoveries
        self.discovered_algebras: List[Dict] = []
        self._seen_discovered_keys: Set[bytes] = set()

    def evaluate_node(self, node: MCTSNode) -> Tuple[float, bool]:
        """
        Computes the axiomatic reward for a node.
        Zero hardcoded targets. Rewards pure consistency and structural non-degeneracy.
        """
        if node.eval_result is None:
            node.eval_result = evaluate_algebraic_axioms(node.tensor, self.axiom_system)

        res = node.eval_result
        is_consistent = res["is_consistent"]
        viol = res["total_violation"]

        if is_consistent:
            # Check semi-simplicity if requested
            if self.target_mode == "semisimple":
                from bootstrap_engine.axioms import compute_killing_metric
                kap = compute_killing_metric(node.tensor)
                det_k = float(np.real(np.linalg.det(kap)))
                rank_k = int(np.linalg.matrix_rank(kap))
                if abs(det_k) > 1e-5:
                    key = node.tensor_key()
                    if key not in self._seen_discovered_keys:
                        self._seen_discovered_keys.add(key)
                        self.discovered_algebras.append({
                            "tensor": np.copy(node.tensor),
                            "eval_result": res,
                            "depth": 0
                        })
                    return 10.0, True
                else:
                    # Consistent but solvable: reward based on rank of Cartan-Killing metric
                    return 2.0 + 2.0 * rank_k, False

            key = node.tensor_key()
            if key not in self._seen_discovered_keys:
                self._seen_discovered_keys.add(key)
                self.discovered_algebras.append({
                    "tensor": np.copy(node.tensor),
                    "eval_result": res,
                    "depth": 0
                })
            # Huge reward for exact mathematical consistency
            return 10.0, True
        else:
            # Scaled negative violation reward in [-1.0, 0.0]
            reward = -min(1.0, viol / 50.0)
            return reward, False

    def search(self, root_tensor: np.ndarray, last_action: Optional[int] = None, temperature: float = 0.5) -> Tuple[int, np.ndarray, float]:
        """
        Runs num_simulations of MCTS starting from root_tensor.
        Returns:
            best_action: Action selected from visit distribution
            visit_probs: Policy target for neural training
            root_value: Estimated value of root state
        """
        root = MCTSNode(tensor=np.copy(root_tensor), action_taken=last_action)

        for sim_idx in range(self.num_simulations):
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
            reward, is_consistent = self.evaluate_node(leaf)

            if not is_consistent and depth < self.max_depth and not leaf.is_expanded:
                self._expand_node(leaf)

            # 3. BACKPROPAGATE
            for n in reversed(search_path):
                n.visit_count += 1
                n.value_sum += reward

        # Compute visit distribution from root
        visit_counts = np.zeros(self.space.action_size, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        if np.sum(visit_counts) > 0:
            if temperature < 1e-3:
                best_action = int(np.argmax(visit_counts))
                visit_probs = visit_counts / np.sum(visit_counts)
            else:
                counts_temp = visit_counts ** (1.0 / temperature)
                visit_probs = counts_temp / np.sum(counts_temp)
                best_action = int(np.random.choice(len(visit_probs), p=visit_probs))
        else:
            valid_actions = self.space.get_valid_actions(root_tensor, last_action=last_action)
            best_action = valid_actions[0] if valid_actions else 0
            visit_probs = np.zeros(self.space.action_size, dtype=np.float32)
            if valid_actions:
                visit_probs[valid_actions] = 1.0 / len(valid_actions)
            else:
                visit_probs[best_action] = 1.0

        return best_action, visit_probs, root.q_value

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """
        Selects child using PUCT formula.
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None

        sqrt_parent_visits = math.sqrt(node.visit_count + 1)

        for action, child in node.children.items():
            u_score = child.q_value + self.c_puct * child.prior * (sqrt_parent_visits / (1 + child.visit_count))
            if u_score > best_score:
                best_score = u_score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand_node(self, node: MCTSNode):
        """
        Expands node using neural policy priors on valid actions only.
        Filters out no-op mutations and immediate inverse oscillations.
        """
        policy_probs, _ = self.brain.predict(node.tensor)
        node.is_expanded = True
        valid_actions = self.space.get_valid_actions(node.tensor, last_action=node.action_taken)

        if not valid_actions:
            return

        # Normalize prior over valid actions
        valid_priors = np.array([policy_probs[a] for a in valid_actions], dtype=np.float32)
        sum_p = np.sum(valid_priors)
        if sum_p > 1e-9:
            valid_priors /= sum_p
        else:
            valid_priors = np.ones(len(valid_actions), dtype=np.float32) / len(valid_actions)

        for a, prior in zip(valid_actions, valid_priors):
            next_tensor = self.space.apply_action(node.tensor, a)
            child = MCTSNode(
                tensor=next_tensor,
                parent=node,
                action_taken=a,
                prior=float(prior)
            )
            node.children[a] = child
