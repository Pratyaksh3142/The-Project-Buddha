#!/usr/bin/env python3
"""
================================================================================
          PROJECT BUDDHA: ALPHA-MATH NEURAL NETWORK (JAX/FLAX)
================================================================================
The "Brain" of the autonomous discovery engine.
This is a Graph Transformer designed to process mathematical Computation Graphs.

Inputs:
  - node_features: (Batch, Max_Nodes, Feature_Dim) - One-hot encodings of operators
  - adjacency_mask: (Batch, Max_Nodes, Max_Nodes) - Graph wiring

Outputs:
  - policy_logits: Probability distribution over next graph actions
  - value: Predicted physical fitness [0, 1] of the partial graph
================================================================================
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Any

# Vocabulary sizes
NUM_NODE_TYPES = 15      # var, const, add, mul, sin, exp, etc.
MAX_NODES = 32           # Maximum complexity of a discovered equation
ACTION_SPACE_SIZE = 128  # Number of distinct legal graph mutations


class GraphTransformerBlock(nn.Module):
    """A single layer of Graph Attention."""
    embed_dim: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, adj_mask, deterministic=True):
        # 1. Multi-Head Self Attention using the Graph Adjacency as a Mask
        # This allows operators to pass messages along the computational edges
        # JAX attention requires mask to broadcast with (batch, num_heads, q_len, kv_len)
        # adj_mask is (batch, max_nodes, max_nodes)
        expanded_mask = jnp.expand_dims(adj_mask, axis=-3)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.embed_dim, 
            out_features=self.embed_dim
        )(inputs_q=x, inputs_kv=x, mask=expanded_mask)
        
        attn_out = nn.Dropout(rate=self.dropout_rate)(attn_out, deterministic=deterministic)
        x = nn.LayerNorm()(x + attn_out)

        # 2. Feed Forward / MLP
        mlp_out = nn.Dense(self.embed_dim * 4)(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(self.embed_dim)(mlp_out)
        mlp_out = nn.Dropout(rate=self.dropout_rate)(mlp_out, deterministic=deterministic)
        x = nn.LayerNorm()(x + mlp_out)

        return x


class AlphaMathBrain(nn.Module):
    """The full AlphaZero-style Dual-Head Neural Architecture."""
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    action_space: int = ACTION_SPACE_SIZE

    @nn.compact
    def __call__(self, node_features, adjacency_mask, deterministic=True):
        # 1. Node Embedding
        x = nn.Dense(self.embed_dim)(node_features)

        # 2. Graph Transformer Layers (Message Passing)
        for _ in range(self.num_layers):
            x = GraphTransformerBlock(
                embed_dim=self.embed_dim, 
                num_heads=self.num_heads
            )(x, adjacency_mask, deterministic=deterministic)

        # 3. Global Graph Representation (Mean Pooling across valid nodes)
        # We mask out padding nodes (where node feature == 0)
        valid_nodes_mask = jnp.any(node_features != 0, axis=-1, keepdims=True)
        graph_embedding = jnp.sum(x * valid_nodes_mask, axis=1) / (jnp.sum(valid_nodes_mask, axis=1) + 1e-8)

        # 4. POLICY HEAD (Predicts the best next mathematical action)
        policy_logits = nn.Dense(self.action_space, name="policy_head")(graph_embedding)

        # 5. VALUE HEAD (Predicts the final physical reward of this graph [0, 1])
        value_hidden = nn.Dense(64)(graph_embedding)
        value_hidden = nn.gelu(value_hidden)
        value_out = nn.Dense(1, name="value_head")(value_hidden)
        value_out = nn.sigmoid(value_out)  # Bounded to [0, 1] for Pratyaksh fitness

        return policy_logits, value_out

# Helper function to initialize the network
def initialize_model(rng_key):
    model = AlphaMathBrain()
    dummy_nodes = jnp.zeros((1, MAX_NODES, NUM_NODE_TYPES))
    dummy_adj = jnp.zeros((1, MAX_NODES, MAX_NODES))
    variables = model.init(rng_key, dummy_nodes, dummy_adj, deterministic=True)
    return model, variables

if __name__ == "__main__":
    print("Testing AlphaMathBrain JAX Compilation...")
    key = jax.random.PRNGKey(42)
    model, params = initialize_model(key)
    
    dummy_n = jax.random.normal(key, (2, MAX_NODES, NUM_NODE_TYPES))
    dummy_a = jnp.ones((2, MAX_NODES, MAX_NODES))
    
    logits, val = model.apply(params, dummy_n, dummy_a)
    print(f"Policy Output Shape: {logits.shape} (Expected: 2, {ACTION_SPACE_SIZE})")
    print(f"Value Output Shape:  {val.shape} (Expected: 2, 1)")
    print("Neural Network successfully initialized in JAX/Flax!")
