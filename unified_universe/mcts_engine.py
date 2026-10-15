"""
Unified MCTS Engine for Relational Discovery.

AlphaZero/MuZero-style Monte Carlo Tree Search with:
  - Dynamic min-max Q-value normalization (MuZero)
  - Dirichlet noise at root for exploration
  - Neural value head as primary leaf evaluator
  - Complete action space: off-diagonal couplings (i < j) AND on-site potentials (i == j)
  - Top-K expansion and progressive widening for deep tree search
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import math
from unified_universe.substrate import UniversalSubstrate
from unified_universe.action_referee import UniversalActionReferee
from unified_universe.neural_pairformer import UnifiedIntuitionEngine


class MCTSNode:
    __slots__ = ['substrate', 'parent', 'action_taken', 'prior',
                 'visit_count', 'value_sum', 'children', 'is_expanded', 'relaxed_eval']

    def __init__(self, substrate: UniversalSubstrate, parent=None,
                 action_taken: Optional[int] = None, prior: float = 0.0):
        self.substrate = substrate
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, 'MCTSNode'] = {}
        self.is_expanded = False
        self.relaxed_eval: Optional[Dict[str, Any]] = None

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


def compute_num_pairs(N: int) -> int:
    """Number of unique entity pairs including diagonals (i <= j): N*(N+1)/2."""
    return N * (N + 1) // 2


def compute_num_actions(N: int, K: int) -> int:
    """Total discrete actions: pairs (i <= j) * K channels * 2 deltas."""
    return compute_num_pairs(N) * K * 2


def decode_action(action_idx: int, N: int, K: int) -> Tuple[int, int, int, float]:
    """Decode flat action index -> (i, j, channel, delta) with i <= j."""
    acts_per_pair = K * 2
    pair_idx = action_idx // acts_per_pair
    remainder = action_idx % acts_per_pair
    ch = remainder // 2
    delta = -1.0 if (remainder % 2 == 0) else 1.0

    curr = 0
    for i in range(N):
        for j in range(i, N):
            if curr == pair_idx:
                return i, j, ch, delta
            curr += 1
    return 0, 0, 0, 1.0


class MinMaxStats:
    """Tracks running min/max Q-values for dynamic normalization (MuZero)."""
    def __init__(self):
        self.minimum = float('inf')
        self.maximum = float('-inf')

    def update(self, value: float):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        span = self.maximum - self.minimum
        if span > 1e-6:
            return float(np.clip((value - self.minimum) / span, 0.0, 1.0))
        return 0.5


class UnifiedMCTSEngine:
    """AlphaZero/MuZero-style MCTS over the universal relational substrate."""

    def __init__(
        self,
        referee: UniversalActionReferee,
        brain: UnifiedIntuitionEngine,
        num_simulations: int = 64,
        max_depth: int = 8,
        c_puct: float = 1.25,
        c_puct_log_base: float = 19652.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        bilevel: bool = True,
        relax_steps: int = 15,
    ):
        self.referee = referee
        self.brain = brain
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.c_puct_log_base = c_puct_log_base
        self.dir_alpha = dirichlet_alpha
        self.dir_frac = dirichlet_frac
        self.bilevel = bilevel
        self.relax_steps = relax_steps

    def search(
        self,
        root_substrate: UniversalSubstrate,
        temperature: float = 1.0
    ) -> Tuple[int, np.ndarray, float]:
        """Run MCTS. Returns (best_action, visit_probs, root_q)."""
        num_actions = self.brain.num_actions
        root = MCTSNode(substrate=root_substrate.clone())
        min_max = MinMaxStats()

        # Expand root with Dirichlet exploration noise
        self._expand_node(root, add_noise=True)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            depth = 0

            # 1. SELECT (PUCT with dynamic min-max normalization)
            while node.is_expanded and len(node.children) > 0 and depth < self.max_depth:
                _, node = self._select_child(node, min_max)
                search_path.append(node)
                depth += 1

            # 2. EVALUATE: Bilevel Differentiable Inner Solver + Neural Intuition
            if node.relaxed_eval is None:
                node.relaxed_eval = self.referee.compute_action(
                    node.substrate, relax=self.bilevel, steps=self.relax_steps
                )
            act_dict = node.relaxed_eval

            if act_dict.get("is_stable", True):
                phys_val = -np.tanh(act_dict["total_action"] / 10.0)
            else:
                phys_val = -1.0  # Vacuum instability penalty (rejects runaway theories)

            _, net_val, _ = self.brain.predict(node.substrate)
            value = 0.5 * float(net_val) + 0.5 * float(phys_val)

            # 3. EXPAND if not at leaf boundary
            if depth < self.max_depth and not node.is_expanded:
                self._expand_node(node, add_noise=False)

            # 4. BACKPROPAGATE
            for n in reversed(search_path):
                n.visit_count += 1
                n.value_sum += value
                min_max.update(n.q_value)

        # Extract visit distribution
        visit_counts = np.zeros(num_actions, dtype=np.float32)
        for a, child in root.children.items():
            if a < num_actions:
                visit_counts[a] = child.visit_count

        total = np.sum(visit_counts)
        if total > 0:
            # Raw visit probabilities are the true AlphaZero training target
            raw_probs = visit_counts / total

            if temperature < 0.01:
                best = int(np.argmax(visit_counts))
            else:
                # Temperature is used ONLY to sample the move during play
                log_c = np.log(visit_counts + 1e-8) / max(temperature, 0.01)
                log_c -= np.max(log_c)
                sample_probs = np.exp(log_c)
                sample_probs /= np.sum(sample_probs)
                best = int(np.random.choice(len(sample_probs), p=sample_probs))
        else:
            best = int(np.random.randint(num_actions))
            raw_probs = np.ones(num_actions, dtype=np.float32) / num_actions

        return best, raw_probs, root.q_value

    def _select_child(self, node: MCTSNode, min_max: MinMaxStats) -> Tuple[int, MCTSNode]:
        """PUCT with dynamic min-max Q normalization (MuZero)."""
        best_score = -float('inf')
        best_action = 0
        best_child = None

        total_visits = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(total_visits + 1)

        # Dynamic c_puct (MuZero scaling)
        c = math.log((total_visits + self.c_puct_log_base + 1) / self.c_puct_log_base) + self.c_puct

        for a, child in node.children.items():
            # First-Play Urgency (FPU): unvisited nodes inherit parent Q value
            q_val = child.q_value if child.visit_count > 0 else node.q_value
            q_norm = min_max.normalize(q_val)
            exploration = c * child.prior * sqrt_total / (1 + child.visit_count)
            score = q_norm + exploration

            if score > best_score:
                best_score = score
                best_action = a
                best_child = child

        return best_action, best_child

    def _expand_node(self, node: MCTSNode, add_noise: bool = False):
        """Expand node with neural prior + optional Dirichlet noise."""
        pol_probs, _, delta_x = self.brain.predict(node.substrate)
        node.is_expanded = True

        N = node.substrate.N
        K = node.substrate.K
        num_actions = self.brain.num_actions

        if add_noise:
            noise = np.random.dirichlet([self.dir_alpha] * num_actions)
            pol_probs = (1 - self.dir_frac) * pol_probs + self.dir_frac * noise

        # Progressive widening / top-K expansion to ensure deep tree search
        if num_actions <= 64:
            actions_to_expand = range(num_actions)
        else:
            # Expand top 40% of actions or minimum 32 actions
            top_k = max(int(num_actions * 0.4), 32)
            actions_to_expand = np.argsort(pol_probs)[-top_k:]

        for idx in actions_to_expand:
            if idx >= num_actions:
                continue
            i, j, ch, delta = decode_action(idx, N, K)
            if ch >= K:
                continue

            child_sub = node.substrate.clone()
            child_sub.mutate_pair_tensor(i, j, ch, delta)
            if child_sub.is_spatial:
                child_sub.X += delta_x * 0.05

            child = MCTSNode(
                substrate=child_sub,
                parent=node,
                action_taken=idx,
                prior=float(pol_probs[idx])
            )
            node.children[idx] = child
