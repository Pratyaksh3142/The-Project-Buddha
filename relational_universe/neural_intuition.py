"""
Level 0: Neural Intuition Brain (JAX/Flax)

Develops the subconscious geometric intuition of which relational rewriting rules
produce stable, living universes versus explosive dead noise.
"""

from __future__ import annotations
from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class RuleIntuitionBrain(nn.Module):
    hidden_dim: int = 64
    num_actions: int = 8  # Discrete mutation moves on rule topologies

    @nn.compact
    def __call__(self, rule_features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Input: rule_features [batch, feature_dim] or [feature_dim]
        Output:
          value: [batch, 1] - predicted stability score (-10 to +10)
          policy_logits: [batch, num_actions] - action priors
        """
        is_unbatched = (rule_features.ndim == 1)
        if is_unbatched:
            rule_features = jnp.expand_dims(rule_features, 0)

        # Shared representation layers
        x = nn.Dense(self.hidden_dim)(rule_features)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)

        # 1. Value Head (Intuition on survival)
        v = nn.Dense(self.hidden_dim // 2)(x)
        v = nn.gelu(v)
        value = nn.Dense(1)(v)

        # 2. Policy Head (Prior over topological mutations)
        p = nn.Dense(self.hidden_dim // 2)(x)
        p = nn.gelu(p)
        logits = nn.Dense(self.num_actions)(p)

        val_out = value[0] if is_unbatched else value
        logits_out = logits[0] if is_unbatched else logits
        return val_out, logits_out


def init_intuition_brain(feature_dim: int = 16, rng_seed: int = 42):
    """Initializes the Neural Intuition Brain and Optax optimizer."""
    rng = jax.random.PRNGKey(rng_seed)
    model = RuleIntuitionBrain()
    dummy_in = jnp.zeros((feature_dim,), dtype=jnp.float32)
    params = model.init(rng, dummy_in)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(0.01),
    )
    opt_state = optimizer.init(params)
    return model, params, optimizer, opt_state
