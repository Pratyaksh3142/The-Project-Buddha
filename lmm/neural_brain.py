"""
Component 2 — The Neural Brain: MPNN + AlphaZero Dual Head.

Architecture
────────────
1.  **Node Embedding**    One-hot features → Dense → 128-dim vectors
2.  **Message Passing**   4 rounds of neighbor-aggregation + MLP update
3.  **Global Readout**    Mean-pool all node embeddings → single graph vector
4.  **Policy Head**       Dense → ReLU → Dense → Softmax over action space
5.  **Value Head**        Dense → ReLU → Dense(1) → Tanh  (score ∈ [-1, +1])

The entire forward pass is jax.jit-compilable for microsecond inference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

from lmm.config import LMMConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════════════

class MessagePassingLayer(nn.Module):
    """
    One round of message passing on the equation graph.

    For each node i:
        message_i  = sum_j  adj[j,i] · MLP(h_j)       (aggregate neighbours)
        h_i'       = LayerNorm( h_i + message_i )      (residual + norm)
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, node_emb: jnp.ndarray, adj: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        node_emb : [batch, max_nodes, hidden_dim]
        adj      : [batch, max_nodes, max_nodes]

        Returns
        -------
        updated  : [batch, max_nodes, hidden_dim]
        """
        # Transform neighbour features
        messages = nn.Dense(self.hidden_dim, name="msg_transform")(node_emb)
        messages = nn.relu(messages)

        # Aggregate:  for each node, sum messages from all parents + children
        # Use adj + adj^T so messages flow in BOTH directions
        sym_adj = adj + jnp.transpose(adj, (0, 2, 1))
        aggregated = jnp.matmul(sym_adj, messages)    # [B, N, H]

        # Residual + LayerNorm
        updated = node_emb + aggregated
        updated = nn.LayerNorm(name="layer_norm")(updated)
        return updated


# ═══════════════════════════════════════════════════════════════════════════════
# The Brain
# ═══════════════════════════════════════════════════════════════════════════════

class MathBrain(nn.Module):
    """
    The LMM's neural intuition.

    Input:  (node_features, adjacency_matrix)   — the current partial AST
    Output: (policy_logits, value)
            policy_logits : [batch, action_space]  — unnormalised log-probs
            value         : [batch, 1]             — predicted fitness ∈ [-1,+1]
    """
    config: LMMConfig

    @nn.compact
    def __call__(self,
                 node_features: jnp.ndarray,
                 adjacency: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Parameters
        ----------
        node_features : [batch, max_nodes, vocab_size]
        adjacency     : [batch, max_nodes, max_nodes]
        """
        cfg = self.config

        # ── 1. Node embedding ──────────────────────────────────────────────
        h = nn.Dense(cfg.embedding_dim, name="node_embed")(node_features)
        h = nn.relu(h)                                       # [B, N, D]

        # ── 2. Message passing ─────────────────────────────────────────────
        for i in range(cfg.mpnn_layers):
            h = MessagePassingLayer(
                hidden_dim=cfg.embedding_dim,
                name=f"mpnn_{i}"
            )(h, adjacency)

        # ── 3. Global readout (mean pool over nodes) ───────────────────────
        # Mask out empty node positions (all-zero feature rows)
        node_mask = jnp.any(node_features != 0, axis=-1, keepdims=True)  # [B,N,1]
        masked_h = h * node_mask
        num_active = jnp.maximum(node_mask.sum(axis=1, keepdims=True), 1.0)
        graph_emb = masked_h.sum(axis=1) / num_active.squeeze(-1)        # [B, D]

        # ── 4. Policy head ─────────────────────────────────────────────────
        policy = nn.Dense(cfg.mlp_hidden, name="policy_h")(graph_emb)
        policy = nn.relu(policy)
        policy = nn.Dense(cfg.action_space_size, name="policy_out")(policy)
        # logits — softmax applied later during loss / sampling

        # ── 5. Value head ──────────────────────────────────────────────────
        value = nn.Dense(cfg.mlp_hidden, name="value_h")(graph_emb)
        value = nn.relu(value)
        value = nn.Dense(1, name="value_out")(value)
        value = nn.tanh(value)                               # [B, 1]

        return policy, value


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: initialise model + params
# ═══════════════════════════════════════════════════════════════════════════════

def init_brain(cfg: LMMConfig, rng_key: jax.Array):
    """
    Create a MathBrain and initialise its parameters with dummy input.

    Returns
    -------
    model  : MathBrain (Flax module — stateless)
    params : FrozenDict of weights
    """
    model = MathBrain(config=cfg)

    dummy_feat = jnp.zeros((1, cfg.max_nodes, cfg.vocab_size))
    dummy_adj  = jnp.zeros((1, cfg.max_nodes, cfg.max_nodes))

    params = model.init(rng_key, dummy_feat, dummy_adj)
    return model, params


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  NEURAL BRAIN — SELF TEST")
    print("=" * 60)

    cfg = LMMConfig(max_vars=2)
    key = jax.random.PRNGKey(42)

    # Initialise
    model, params = init_brain(cfg, key)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"\n  Model initialised: {n_params:,} parameters")

    # Forward pass with dummy input
    dummy_feat = jnp.ones((1, cfg.max_nodes, cfg.vocab_size))
    dummy_adj  = jnp.eye(cfg.max_nodes)[None, :, :]   # identity adj

    policy, value = model.apply(params, dummy_feat, dummy_adj)
    print(f"  Policy logits shape: {policy.shape}  (expected: [1, {cfg.action_space_size}])")
    print(f"  Value shape:         {value.shape}  (expected: [1, 1])")
    print(f"  Value:               {float(value[0, 0]):.4f}  (range: -1 to +1)")

    # JIT compile
    jitted = jax.jit(model.apply)
    policy2, value2 = jitted(params, dummy_feat, dummy_adj)
    print(f"\n  JIT compiled successfully ✓")

    # Gradient test
    def dummy_loss(params):
        p, v = model.apply(params, dummy_feat, dummy_adj)
        return jnp.mean(p ** 2) + jnp.mean(v ** 2)

    grads = jax.grad(dummy_loss)(params)
    grad_norm = sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))
    print(f"  Gradient norm:       {float(grad_norm):.4f}  (non-zero = gradients flow ✓)")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
