"""
Component 4 — The Search Engine: Monte Carlo Tree Search (MCTS) with PUCT.

Integrates the Neural Brain with the AST Graph Engine:
1. Selection: Traverse down the tree using PUCT until an unexpanded node is reached.
2. Expansion & Evaluation: Query MathBrain for policy priors and value.
3. Masking: Strictly mask illegal actions to -inf so invalid AST nodes are never picked.
4. Backpropagation: Propagate value estimates up the tree to update Q-values and visit counts.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from lmm.ast_graph import MathAST
from lmm.config import LMMConfig
from lmm.neural_brain import MathBrain


class MCTSNode:
    """Represents a node in the MCTS tree (a specific AST state)."""

    def __init__(self, state: MathAST, parent: Optional[MCTSNode] = None, prior: float = 0.0):
        self.state: MathAST = state
        self.parent: Optional[MCTSNode] = parent
        self.prior: float = prior

        self.children: Dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_terminal(self) -> bool:
        return self.state.is_complete

    def puct_score(self, child_action: int, c_puct: float, total_parent_visits: int) -> float:
        child = self.children.get(child_action)
        if child is None or child.visit_count == 0:
            q = 0.0
            visits = 0
            prior = self.children[child_action].prior if child else 0.0
        else:
            q = child.q_value
            visits = child.visit_count
            prior = child.prior

        u = c_puct * prior * (math.sqrt(total_parent_visits) / (1.0 + visits))
        return q + u


class MCTS:
    """
    Monte Carlo Tree Search guided by the MathBrain dual-head neural network.
    """

    def __init__(self, model: MathBrain, params, cfg: LMMConfig):
        self.model = model
        self.params = params
        self.cfg = cfg
        self._jitted_apply = jax.jit(self.model.apply)

    def _evaluate_brain(self, ast: MathAST) -> Tuple[np.ndarray, float]:
        """
        Evaluate AST with MathBrain.
        Returns:
            masked_probs: np.ndarray [action_space_size] over legal actions
            value: float scalar in [-1, 1]
        """
        feat, adj = ast.to_tensors()
        feat_batch = jnp.expand_dims(jnp.array(feat), axis=0)
        adj_batch = jnp.expand_dims(jnp.array(adj), axis=0)

        logits, val = self._jitted_apply(self.params, feat_batch, adj_batch)
        logits = np.array(logits[0])
        value = float(val[0, 0])

        legal = ast.legal_actions()
        if not legal:
            return np.zeros(self.cfg.action_space_size, dtype=np.float32), value

        # Mask illegal actions with -inf
        masked_logits = np.full(self.cfg.action_space_size, -1e9, dtype=np.float32)
        for a in legal:
            if a < self.cfg.action_space_size:
                masked_logits[a] = logits[a]

        # Stable softmax
        max_logit = np.max(masked_logits[legal])
        exp_logits = np.exp(masked_logits - max_logit)
        mask = np.zeros(self.cfg.action_space_size, dtype=np.float32)
        mask[legal] = 1.0
        probs = (exp_logits * mask)
        sum_probs = np.sum(probs)
        if sum_probs > 0:
            probs = probs / sum_probs
        else:
            probs[legal] = 1.0 / len(legal)

        return probs, value

    def search(self, root_ast: MathAST, num_simulations: Optional[int] = None, add_noise: bool = True) -> Tuple[Dict[int, float], MCTSNode]:
        """
        Runs MCTS rollouts starting from root_ast.
        Returns:
            policy_distribution: Dict[action, probability] normalized by visit counts
            root_node: The root MCTSNode containing subtrees
        """
        if num_simulations is None:
            num_simulations = self.cfg.mcts_simulations

        root = MCTSNode(state=root_ast)

        # Initial expansion of root
        probs, val = self._evaluate_brain(root.state)
        legal_actions = root.state.legal_actions()

        # Add Dirichlet noise at root for exploration (AlphaZero standard)
        if add_noise and len(legal_actions) > 1:
            noise = np.random.dirichlet([0.3] * len(legal_actions))
            for i, a in enumerate(legal_actions):
                probs[a] = 0.75 * probs[a] + 0.25 * float(noise[i])

        for action in legal_actions:
            next_state = root.state.apply_action(action)
            root.children[action] = MCTSNode(state=next_state, parent=root, prior=float(probs[action]))
        root.is_expanded = True

        for _ in range(num_simulations):
            node = root

            # 1. Selection
            while node.is_expanded and not node.is_terminal() and len(node.children) > 0:
                best_action = -1
                best_score = -float("inf")
                total_visits = sum(c.visit_count for c in node.children.values())

                for a in node.children.keys():
                    score = node.puct_score(a, self.cfg.c_puct, total_visits)
                    if score > best_score:
                        best_score = score
                        best_action = a

                node = node.children[best_action]

            # 2. Expansion & Evaluation
            if node.is_terminal():
                # Leaf node: terminal equation built, value defaults to neutral until fitness is assessed
                value = 0.0
            else:
                probs, value = self._evaluate_brain(node.state)
                legal = node.state.legal_actions()
                for action in legal:
                    if action not in node.children:
                        next_state = node.state.apply_action(action)
                        node.children[action] = MCTSNode(
                            state=next_state,
                            parent=node,
                            prior=float(probs[action])
                        )
                node.is_expanded = True

            # 3. Backpropagation
            curr = node
            while curr is not None:
                curr.visit_count += 1
                curr.total_value += value
                curr = curr.parent

        # Compute visit count distribution at root
        visits = {a: child.visit_count for a, child in root.children.items()}
        total_visits = sum(visits.values())
        if total_visits > 0:
            policy_distribution = {a: v / total_visits for a, v in visits.items()}
        else:
            policy_distribution = {a: 1.0 / len(visits) for a in visits.keys()}

        return policy_distribution, root


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  MCTS SEARCH ENGINE — SELF TEST")
    print("=" * 60)

    cfg = LMMConfig(max_vars=2, mcts_simulations=50)
    key = jax.random.PRNGKey(101)

    # Init Brain
    from lmm.neural_brain import init_brain
    model, params = init_brain(cfg, key)

    mcts = MCTS(model=model, params=params, cfg=cfg)

    # Start from empty root AST
    root_ast = MathAST(max_nodes=cfg.max_nodes, num_vars=cfg.max_vars)
    print(f"Initial state: {root_ast}")

    policy, root_node = mcts.search(root_ast, num_simulations=30)

    print("\nPolicy distribution from root after 30 simulations:")
    for a, p in sorted(policy.items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"  Action {a:2d} -> Visit prob: {p:.4f} (Visits: {root_node.children[a].visit_count})")

    assert len(policy) > 0, "MCTS failed to generate legal policy"
    assert math.isclose(sum(policy.values()), 1.0, abs_tol=1e-5), "Policy probs do not sum to 1"

    print("\n" + "=" * 60)
    print("  MCTS SELF TEST PASSED ✓")
    print("=" * 60)
