"""
Level 0: Tensor Monte Carlo Tree Search (Tensor MCTS)

Explores the space of rank-1 tensor decompositions using PUCT search:
  - State Node: Holds the residual tensor S_t.
  - Edges: Rank-1 actions (u (x) v (x) w).
  - Rollouts & Value: Evaluated using the TensorBrain neural network.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from tensor_discovery.environment import TensorGameEnv, TensorAction


class MCTSNode:
    """A search tree node in the Tensor Decomposition Game."""

    def __init__(self, env: TensorGameEnv, parent: Optional[MCTSNode] = None, action_from_parent: Optional[TensorAction] = None):
        self.env = env
        self.parent = parent
        self.action_from_parent = action_from_parent

        self.children: Dict[TensorAction, MCTSNode] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 1.0
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_terminal(self) -> bool:
        return self.env.is_terminal


class TensorMCTS:
    """Monte Carlo Tree Search specialized for 3D Tensor Decomposition."""

    def __init__(
        self,
        model,
        params,
        c_puct: float = 1.4,
        num_simulations: int = 40,
        coeff_map: Tuple[int, ...] = (-1, 0, 1),
    ):
        self.model = model
        self.params = params
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.coeff_map = coeff_map

    def sample_actions_from_policy(
        self,
        policy_logits: Dict[str, jnp.ndarray],
        num_candidates: int = 8,
        rng_key: Optional[jax.Array] = None,
        temperature: float = 1.0,
        current_tensor: Optional[np.ndarray] = None,
    ) -> List[Tuple[TensorAction, float]]:
        """
        Samples candidate (u, v, w) rank-1 actions from the neural policy logits.
        Returns a list of (TensorAction, prior_probability).
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(np.random.randint(0, 1000000))

        k_u, k_v, k_w = jax.random.split(rng_key, 3)

        # Apply temperature
        p_u = jax.nn.softmax(policy_logits["u"] / max(temperature, 1e-4), axis=-1)
        p_v = jax.nn.softmax(policy_logits["v"] / max(temperature, 1e-4), axis=-1)
        p_w = jax.nn.softmax(policy_logits["w"] / max(temperature, 1e-4), axis=-1)

        dim_u = p_u.shape[0]
        dim_v = p_v.shape[0]
        dim_w = p_w.shape[0]

        candidates = []

        # 1. Coordinate-Targeted Proposal: Target non-zero entries in current residual tensor
        if current_tensor is not None:
            non_zeros = np.argwhere(current_tensor != 0)
            if len(non_zeros) > 0:
                target_count = min(len(non_zeros), num_candidates // 2)
                chosen_indices = np.random.choice(len(non_zeros), size=target_count, replace=False)
                for idx in chosen_indices:
                    i, j, k = non_zeros[idx]
                    sign = int(np.sign(current_tensor[i, j, k]))
                    vec_u = tuple(sign if x == i else 0 for x in range(dim_u))
                    vec_v = tuple(1 if x == j else 0 for x in range(dim_v))
                    vec_w = tuple(1 if x == k else 0 for x in range(dim_w))
                    action = TensorAction(u=vec_u, v=vec_v, w=vec_w)
                    candidates.append((action, 0.5))

        # 2. Neural Policy Guided Sampling
        while len(candidates) < num_candidates:
            k_u, subkey_u = jax.random.split(k_u)
            k_v, subkey_v = jax.random.split(k_v)
            k_w, subkey_w = jax.random.split(k_w)

            idx_u = [int(jax.random.categorical(subkey_u, policy_logits["u"][i])) for i in range(dim_u)]
            idx_v = [int(jax.random.categorical(subkey_v, policy_logits["v"][j])) for j in range(dim_v)]
            idx_w = [int(jax.random.categorical(subkey_w, policy_logits["w"][k])) for k in range(dim_w)]

            vec_u = tuple(self.coeff_map[idx] for idx in idx_u)
            vec_v = tuple(self.coeff_map[idx] for idx in idx_v)
            vec_w = tuple(self.coeff_map[idx] for idx in idx_w)

            # Skip all-zero vectors
            if all(x == 0 for x in vec_u) or all(x == 0 for x in vec_v) or all(x == 0 for x in vec_w):
                continue

            action = TensorAction(u=vec_u, v=vec_v, w=vec_w)
            candidates.append((action, 0.2))

        return candidates

    def search(self, root_env: TensorGameEnv) -> Tuple[TensorAction, Dict[TensorAction, int]]:
        """
        Runs MCTS simulations from the current root environment and returns the best action.
        """
        root = MCTSNode(env=root_env.clone())

        for _ in range(self.num_simulations):
            node = root

            # 1. SELECT
            while node.is_expanded and not node.is_terminal and len(node.children) > 0:
                best_score = -float("inf")
                best_child = None

                for action, child in node.children.items():
                    # PUCT Formula
                    u_score = self.c_puct * child.prior * (math.sqrt(node.visit_count) / (1 + child.visit_count))
                    score = child.q_value + u_score
                    if score > best_score:
                        best_score = score
                        best_child = child

                if best_child is None:
                    break
                node = best_child

            # 2. EXPAND & EVALUATE
            value = 0.0
            if not node.is_terminal:
                tensor_input = jnp.array(node.env.current_tensor, dtype=jnp.float32)
                val_pred, policy_logits = self.model.apply(self.params, tensor_input)
                value = float(val_pred[0] if val_pred.ndim > 0 else val_pred)

                candidates = self.sample_actions_from_policy(
                    policy_logits,
                    num_candidates=6,
                    current_tensor=node.env.current_tensor,
                )
                for action, prior in candidates:
                    if action not in node.children:
                        next_env = node.env.clone()
                        _, reward, _, _ = next_env.step(action)
                        child_node = MCTSNode(env=next_env, parent=node, action_from_parent=action)
                        child_node.prior = prior
                        node.children[action] = child_node
                node.is_expanded = True
            else:
                value = 10.0 if node.env.is_solved else -5.0

            # 3. BACKPROPAGATE
            curr: Optional[MCTSNode] = node
            while curr is not None:
                curr.visit_count += 1
                curr.total_value += value
                curr = curr.parent

        # Select action with highest visit count
        visit_counts = {act: child.visit_count for act, child in root.children.items()}
        if not visit_counts:
            # Fallback if no children were expanded
            dummy_u = tuple(1 if i == 0 else 0 for i in range(root_env.dim_u))
            dummy_v = tuple(1 if i == 0 else 0 for i in range(root_env.dim_v))
            dummy_w = tuple(1 if i == 0 else 0 for i in range(root_env.dim_w))
            fallback_action = TensorAction(u=dummy_u, v=dummy_v, w=dummy_w)
            return fallback_action, {fallback_action: 1}

        best_action = max(visit_counts, key=visit_counts.get)
        return best_action, visit_counts
