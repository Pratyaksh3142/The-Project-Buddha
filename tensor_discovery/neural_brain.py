"""
Level 0: Tensor Brain (Axial 3D Tensor Value & Policy Network in JAX/Flax)

Processes a 3D tensor state S in R^{U x V x W} and produces:
  1. Value Head V(S): Predicts the expected steps to zero (solution likelihood).
  2. Policy Head P(u, v, w | S): Predicts probability distributions over coefficients
     {-1, 0, 1} for each mode vector u in R^U, v in R^V, w in R^W.
"""

from __future__ import annotations
from typing import Tuple, Dict
import jax
import jax.numpy as jnp
import flax.linen as nn


class TensorBrain(nn.Module):
    dim_u: int
    dim_v: int
    dim_w: int
    num_coeffs: int = 3  # Allowed choices: -1 (idx 0), 0 (idx 1), +1 (idx 2)
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, tensor: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Args:
            tensor: [batch, dim_u, dim_v, dim_w] or [dim_u, dim_v, dim_w]
        Returns:
            value: [batch, 1] scalar value estimate
            policy_logits: {
                "u": [batch, dim_u, num_coeffs],
                "v": [batch, dim_v, num_coeffs],
                "w": [batch, dim_w, num_coeffs],
            }
        """
        # Ensure batch dimension
        is_unbatched = (tensor.ndim == 3)
        if is_unbatched:
            tensor = jnp.expand_dims(tensor, 0)

        batch_size = tensor.shape[0]

        # Mode-wise projections (Mode-1, Mode-2, Mode-3 slices)
        # Slices along U: [batch, dim_u, dim_v * dim_w]
        flat_u = tensor.reshape((batch_size, self.dim_u, self.dim_v * self.dim_w))
        # Slices along V: [batch, dim_v, dim_u * dim_w]
        flat_v = jnp.swapaxes(tensor, 1, 2).reshape((batch_size, self.dim_v, self.dim_u * self.dim_w))
        # Slices along W: [batch, dim_w, dim_u * dim_v]
        flat_w = jnp.swapaxes(tensor, 1, 3).reshape((batch_size, self.dim_w, self.dim_u * self.dim_v))

        # Shared Global Representation
        flat_all = tensor.reshape((batch_size, -1))
        h_global = nn.Dense(self.hidden_dim)(flat_all)
        h_global = nn.gelu(h_global)
        h_global = nn.LayerNorm()(h_global)
        h_global = nn.Dense(self.hidden_dim)(h_global)
        h_global = nn.gelu(h_global)

        # 1. Value Head
        v_hidden = nn.Dense(self.hidden_dim // 2)(h_global)
        v_hidden = nn.gelu(v_hidden)
        value = nn.Dense(1)(v_hidden)  # [batch, 1]

        # 2. Policy Heads for vectors u, v, w
        # Vector u policy: conditioned on mode-u features + global representation
        h_u = nn.Dense(self.hidden_dim // 2)(flat_u)
        h_u = h_u + jnp.expand_dims(nn.Dense(self.hidden_dim // 2)(h_global), 1)
        h_u = nn.gelu(h_u)
        logits_u = nn.Dense(self.num_coeffs)(h_u)  # [batch, dim_u, num_coeffs]

        # Vector v policy
        h_v = nn.Dense(self.hidden_dim // 2)(flat_v)
        h_v = h_v + jnp.expand_dims(nn.Dense(self.hidden_dim // 2)(h_global), 1)
        h_v = nn.gelu(h_v)
        logits_v = nn.Dense(self.num_coeffs)(h_v)  # [batch, dim_v, num_coeffs]

        # Vector w policy
        h_w = nn.Dense(self.hidden_dim // 2)(flat_w)
        h_w = h_w + jnp.expand_dims(nn.Dense(self.hidden_dim // 2)(h_global), 1)
        h_w = nn.gelu(h_w)
        logits_w = nn.Dense(self.num_coeffs)(h_w)  # [batch, dim_w, num_coeffs]

        policy_logits = {
            "u": logits_u[0] if is_unbatched else logits_u,
            "v": logits_v[0] if is_unbatched else logits_v,
            "w": logits_w[0] if is_unbatched else logits_w,
        }
        val_out = value[0] if is_unbatched else value

        return val_out, policy_logits


def init_tensor_brain(dim_u: int, dim_v: int, dim_w: int, rng_key: jax.Array):
    """Factory function to initialize the TensorBrain network and parameters."""
    model = TensorBrain(dim_u=dim_u, dim_v=dim_v, dim_w=dim_w)
    dummy_input = jnp.zeros((dim_u, dim_v, dim_w), dtype=jnp.float32)
    params = model.init(rng_key, dummy_input)
    return model, params
