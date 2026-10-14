"""
JAX/Flax Neural Pairformer Brain (Inspired by AlphaFold 3 & GNoME).

Processes node tokens S, relational pair tensors Z, and 3D spatial coordinates X.
Updates node-pair representations via invariant attention and outputs:
1. Value Head: Expected action potential of the universe
2. Policy Head: Probability distribution over relational graph mutations
3. Spatial Flow Head: Cartesian displacement vectors Delta X for coordinate relaxation
"""

from typing import Dict, Any, Tuple
import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
import optax
import numpy as np


class PairformerBlock(nn.Module):
    dim: int = 32

    @nn.compact
    def __call__(self, s: jnp.ndarray, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # s: (batch, N, dim), z: (batch, N, N, dim)
        batch, N, _ = s.shape

        # 1. Outer Product Mean (Nodes update Pairs): z_ij += Linear(s_i (x) s_j)
        s_i = s[:, :, None, :]  # (B, N, 1, d)
        s_j = s[:, None, :, :]  # (B, 1, N, d)
        pair_update = nn.Dense(self.dim)(s_i * s_j)  # element-wise product projected
        z = z + nn.LayerNorm()(pair_update)

        # 2. Triangular/Pair-biased Self-Attention (Pairs update Nodes):
        # Attention scores: Q_i * K_j / sqrt(d) + bias(z_ij)
        q = nn.Dense(self.dim)(s)
        k = nn.Dense(self.dim)(s)
        v = nn.Dense(self.dim)(s)

        attn_logits = jnp.einsum('bid, bjd -> bij', q, k) / jnp.sqrt(self.dim)
        z_bias = nn.Dense(1)(z).squeeze(-1)  # (B, N, N)
        attn_weights = jax.nn.softmax(attn_logits + z_bias, axis=-1)

        node_update = jnp.einsum('bij, bjd -> bid', attn_weights, v)
        s = s + nn.LayerNorm()(nn.Dense(self.dim)(node_update))

        return s, z


class UnifiedBrainModel(nn.Module):
    N: int
    num_actions: int
    dim: int = 32

    @nn.compact
    def __call__(self, s_in: jnp.ndarray, z_in: jnp.ndarray, x_in: jnp.ndarray, train: bool = False):
        # Embed inputs
        s = nn.Dense(self.dim)(s_in)
        z = nn.Dense(self.dim)(z_in)

        # Pairformer Stack (2 layers)
        s, z = PairformerBlock(dim=self.dim)(s, z)
        s, z = PairformerBlock(dim=self.dim)(s, z)

        # Head 1: Value Head (Total Action Potential in R)
        s_pooled = jnp.mean(s, axis=1)  # (B, dim)
        z_pooled = jnp.mean(z, axis=(1, 2))  # (B, dim)
        pooled = jnp.concatenate([s_pooled, z_pooled], axis=-1)
        h_val = nn.gelu(nn.Dense(32)(pooled))
        value = nn.Dense(1)(h_val).squeeze(-1)

        # Head 2: Policy Head (Distribution over discrete graph mutations)
        h_pol = nn.gelu(nn.Dense(64)(pooled))
        policy_logits = nn.Dense(self.num_actions)(h_pol)

        # Head 3: Spatial Flow Head (AlphaFold 3 Coordinate Displacements Delta X in R^{N x 3})
        delta_x = nn.Dense(3)(s)  # (B, N, 3)

        return policy_logits, value, delta_x


class UnifiedIntuitionEngine:
    """
    Manages JAX compilation, parameter updates, and inference for the Unified Substrate.
    """

    def __init__(self, N: int, node_dim: int, pair_dim: int, num_actions: int, learning_rate: float = 1e-3, seed: int = 42):
        self.N = N
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.num_actions = num_actions
        self.rng = jax.random.PRNGKey(seed)

        self.model = UnifiedBrainModel(N=N, num_actions=num_actions)
        dummy_s = jnp.zeros((1, N, node_dim), dtype=jnp.float32)
        dummy_z = jnp.zeros((1, N, N, pair_dim), dtype=jnp.float32)
        dummy_x = jnp.zeros((1, N, 3), dtype=jnp.float32)

        self.params = self.model.init(self.rng, dummy_s, dummy_z, dummy_x)
        self.tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
        self.opt_state = self.tx.init(self.params)

        self._init_jits()

    def _init_jits(self):
        @jax.jit
        def _forward(params, s, z, x):
            return self.model.apply(params, s, z, x, train=False)

        @jax.jit
        def _train_step(params, opt_state, s, z, x, target_p, target_v):
            def loss_fn(p):
                pol_logits, value, _ = self.model.apply(p, s, z, x, train=True)
                val_loss = jnp.mean(jnp.square(value - target_v))
                log_probs = jax.nn.log_softmax(pol_logits, axis=-1)
                pol_loss = -jnp.mean(jnp.sum(target_p * log_probs, axis=-1))
                return val_loss + pol_loss, (val_loss, pol_loss)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, aux), grads = grad_fn(params)
            updates, new_opt_state = self.tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss, aux

        self._forward_jit = _forward
        self._train_step_jit = _train_step

    def predict(self, substrate) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Runs inference on a UniversalSubstrate.
        Returns:
            policy_probs: (num_actions,) probability prior
            value: scalar estimated potential
            delta_x: (N, 3) spatial relaxation displacement
        """
        s_in = jnp.array(substrate.S, dtype=jnp.float32)[None, ...]
        z_in = jnp.array(substrate.Z, dtype=jnp.float32)[None, ...]
        x_in = jnp.array(substrate.X, dtype=jnp.float32)[None, ...]

        logits, val, delta_x = self._forward_jit(self.params, s_in, z_in, x_in)
        pol_probs = np.array(jax.nn.softmax(logits[0], axis=-1))
        val_scalar = float(val[0])
        dx = np.array(delta_x[0])

        return pol_probs, val_scalar, dx

    def train_on_batch(self, s_batch: np.ndarray, z_batch: np.ndarray, x_batch: np.ndarray, target_p: np.ndarray, target_v: np.ndarray) -> Dict[str, float]:
        sb = jnp.array(s_batch, dtype=jnp.float32)
        zb = jnp.array(z_batch, dtype=jnp.float32)
        xb = jnp.array(x_batch, dtype=jnp.float32)
        tp = jnp.array(target_p, dtype=jnp.float32)
        tv = jnp.array(target_v, dtype=jnp.float32)

        self.params, self.opt_state, total_loss, (val_loss, pol_loss) = self._train_step_jit(
            self.params, self.opt_state, sb, zb, xb, tp, tv
        )

        return {"total_loss": float(total_loss), "value_loss": float(val_loss), "policy_loss": float(pol_loss)}

    def save_checkpoint(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(serialization.to_bytes(self.params))

    def load_checkpoint(self, filepath: str) -> bool:
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                self.params = serialization.from_bytes(self.params, f.read())
            return True
        return False
