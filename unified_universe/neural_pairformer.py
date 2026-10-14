"""
JAX/Flax Neural Pairformer Brain.

Architecture inspired by DeepMind AlphaFold 3 Pairformer + AlphaZero dual heads:
  - Outer Product Mean updates (nodes -> pairs)
  - Triangular Multiplicative updates (pair -> pair composition)
  - Pair-Biased Self-Attention (pairs -> nodes)
  - Invariant Euclidean distance injection into edge representations

Outputs:
  1. Value Head: V(state) in R - estimated action potential
  2. Policy Head: pi(a|state) - probability distribution over graph mutations
  3. Spatial Flow Head: Delta X in R^{N x 3} - coordinate relaxation displacements

All three heads are trained end-to-end:
  - Value Loss: MSE(value, target_v)
  - Policy Loss: Categorical Cross-Entropy(target_p, log_probs)
  - Spatial Flow Loss: MSE(delta_x, target_dx) - Physical Force Matching
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
    """Single Pairformer layer: outer-product update + triangular update + pair-biased attention."""
    dim: int = 64

    @nn.compact
    def __call__(self, s: jnp.ndarray, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # s: (B, N, dim), z: (B, N, N, dim)
        B, N, D = s.shape

        # ── 1. Outer Product Mean: nodes -> pairs ──
        s_norm = nn.LayerNorm()(s)
        s_i = s_norm[:, :, None, :]  # (B, N, 1, D)
        s_j = s_norm[:, None, :, :]  # (B, 1, N, D)
        pair_update = nn.Dense(D)(s_i * s_j)
        z = z + pair_update

        # ── 2. Triangular Multiplicative Update: z_ij += sum_k z_ik * z_kj ──
        z_norm = nn.LayerNorm()(z)
        gate = jax.nn.sigmoid(nn.Dense(D)(z_norm))
        z_tri = jnp.einsum('bikd,bkjd->bijd', z_norm, z_norm) / jnp.sqrt(float(N))
        z = z + gate * nn.Dense(D)(z_tri)

        # ── 3. Pair-Biased Self-Attention: pairs -> nodes ──
        s_norm2 = nn.LayerNorm()(s)
        q = nn.Dense(D)(s_norm2)
        k = nn.Dense(D)(s_norm2)
        v = nn.Dense(D)(s_norm2)

        attn_logits = jnp.einsum('bid,bjd->bij', q, k) / jnp.sqrt(float(D))
        z_bias = nn.Dense(1)(nn.LayerNorm()(z)).squeeze(-1)  # (B, N, N)
        attn_weights = jax.nn.softmax(attn_logits + z_bias, axis=-1)

        node_update = jnp.einsum('bij,bjd->bid', attn_weights, v)
        s = s + nn.Dense(D)(nn.gelu(nn.Dense(D)(node_update)))

        return s, z


class UnifiedBrainModel(nn.Module):
    """Full Model: Input Embedding -> Pairformer Stack -> 3 Output Heads."""
    N: int
    num_actions: int
    dim: int = 64
    num_layers: int = 3

    @nn.compact
    def __call__(self, s_in, z_in, x_in, dist_in, train=False):
        D = self.dim

        # Input Embeddings
        s = nn.Dense(D)(s_in)                        # (B, N, D)
        z = nn.Dense(D)(z_in)                        # (B, N, N, D)

        # Inject invariant 3D spatial distances into edge representations
        dist_feat = nn.Dense(D)(dist_in[..., None])  # (B, N, N, D)
        z = z + dist_feat

        # Pairformer Stack
        for _ in range(self.num_layers):
            s, z = PairformerBlock(dim=D)(s, z)

        # Pooling for global graph representations
        s_pool = jnp.mean(s, axis=1)                 # (B, D)
        z_pool = jnp.mean(z, axis=(1, 2))            # (B, D)
        pooled = jnp.concatenate([s_pool, z_pool], axis=-1)  # (B, 2D)

        # Head 1: Value Head (Total Action Potential in R)
        h_v = nn.gelu(nn.Dense(D)(pooled))
        h_v = nn.gelu(nn.Dense(D // 2)(h_v))
        value = nn.Dense(1)(h_v).squeeze(-1)         # (B,)

        # Head 2: Policy Head (Distribution over discrete mutations)
        h_p = nn.gelu(nn.Dense(D)(pooled))
        h_p = nn.gelu(nn.Dense(D)(h_p))
        policy_logits = nn.Dense(self.num_actions)(h_p)  # (B, A)

        # Head 3: Spatial Flow Head (AlphaFold 3 Coordinate Displacements Delta X in R^{N x 3})
        h_x = nn.gelu(nn.Dense(D)(s))
        delta_x = nn.Dense(3)(h_x)                   # (B, N, 3)

        return policy_logits, value, delta_x


class UnifiedIntuitionEngine:
    """
    Manages JAX compilation, parameter updates, inference, and checkpointing.
    """

    def __init__(
        self,
        N: int,
        node_dim: int,
        pair_dim: int,
        num_actions: int,
        dim: int = 64,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
        seed: int = 42
    ):
        self.N = N
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.num_actions = num_actions
        self.rng = jax.random.PRNGKey(seed)

        self.model = UnifiedBrainModel(
            N=N, num_actions=num_actions, dim=dim, num_layers=num_layers
        )

        dummy_s = jnp.zeros((1, N, node_dim), dtype=jnp.float32)
        dummy_z = jnp.zeros((1, N, N, pair_dim), dtype=jnp.float32)
        dummy_x = jnp.zeros((1, N, 3), dtype=jnp.float32)
        dummy_d = jnp.zeros((1, N, N), dtype=jnp.float32)

        self.params = self.model.init(self.rng, dummy_s, dummy_z, dummy_x, dummy_d)

        # AdamW with gradient clipping and cosine schedule
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate * 0.1,
            peak_value=learning_rate,
            warmup_steps=100,
            decay_steps=50000,
            end_value=learning_rate * 0.01,
        )
        self.tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule, weight_decay=1e-4),
        )
        self.opt_state = self.tx.init(self.params)
        self.train_steps = 0

        self._compile_jits()

    def _compile_jits(self):
        @jax.jit
        def _forward(params, s, z, x, d):
            return self.model.apply(params, s, z, x, d, train=False)

        @jax.jit
        def _train_step(params, opt_state, s, z, x, d, target_p, target_v, target_dx):
            def loss_fn(p):
                logits, value, delta_x = self.model.apply(p, s, z, x, d, train=True)
                # 1. Value Loss (MSE)
                v_loss = jnp.mean(jnp.square(value - target_v))
                # 2. Policy Loss (Cross-Entropy)
                log_p = jax.nn.log_softmax(logits, axis=-1)
                p_loss = -jnp.mean(jnp.sum(target_p * log_p, axis=-1))
                # 3. Spatial Flow Loss (Physical Force Matching: MSE against analytic relaxation flow)
                x_loss = jnp.mean(jnp.square(delta_x - target_dx))

                total_loss = v_loss + p_loss + 0.5 * x_loss
                return total_loss, (v_loss, p_loss, x_loss)

            (loss, (vl, pl, xl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt = self.tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt, loss, vl, pl, xl

        self._fwd = _forward
        self._step = _train_step

    def predict(self, substrate) -> Tuple[np.ndarray, float, np.ndarray]:
        """Run inference on a substrate. Returns (policy_probs, value, delta_x)."""
        s = jnp.array(substrate.S, dtype=jnp.float32)[None]
        z = jnp.array(substrate.Z, dtype=jnp.float32)[None]
        x = jnp.array(substrate.X, dtype=jnp.float32)[None]
        d = jnp.array(substrate.compute_distance_matrix(), dtype=jnp.float32)[None]

        logits, val, dx = self._fwd(self.params, s, z, x, d)
        probs = np.array(jax.nn.softmax(logits[0]))
        return probs, float(val[0]), np.array(dx[0])

    def train_on_batch(
        self,
        s_b: np.ndarray,
        z_b: np.ndarray,
        x_b: np.ndarray,
        d_b: np.ndarray,
        target_p: np.ndarray,
        target_v: np.ndarray,
        target_dx: np.ndarray
    ) -> Dict[str, float]:
        """One gradient step on a batch, including force matching on spatial flow."""
        sb = jnp.array(s_b, dtype=jnp.float32)
        zb = jnp.array(z_b, dtype=jnp.float32)
        xb = jnp.array(x_b, dtype=jnp.float32)
        db = jnp.array(d_b, dtype=jnp.float32)
        tp = jnp.array(target_p, dtype=jnp.float32)
        tv = jnp.array(target_v, dtype=jnp.float32)
        tdx = jnp.array(target_dx, dtype=jnp.float32)

        self.params, self.opt_state, loss, vl, pl, xl = self._step(
            self.params, self.opt_state, sb, zb, xb, db, tp, tv, tdx
        )
        self.train_steps += 1
        return {
            "total_loss": float(loss),
            "value_loss": float(vl),
            "policy_loss": float(pl),
            "spatial_loss": float(xl)
        }

    def save_checkpoint(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        checkpoint_data = {
            "metadata": {
                "N": self.N,
                "node_dim": self.node_dim,
                "pair_dim": self.pair_dim,
                "num_actions": self.num_actions,
            },
            "params": serialization.to_state_dict(self.params),
            "opt_state": serialization.to_state_dict(self.opt_state),
            "train_steps": self.train_steps
        }
        with open(filepath, "wb") as f:
            f.write(serialization.to_bytes(checkpoint_data))

    def load_checkpoint(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "rb") as f:
                raw_bytes = f.read()
            data = serialization.from_bytes(dict, raw_bytes)
            if "metadata" in data:
                meta = data["metadata"]
                if (meta.get("node_dim") != self.node_dim or
                    meta.get("pair_dim") != self.pair_dim or
                    meta.get("num_actions") != self.num_actions):
                    print(f"  [Checkpoint Skip] Dimensions in {filepath} (node_dim={meta.get('node_dim')}) do not match current config (node_dim={self.node_dim}). Initializing fresh.")
                    return False

            if "params" in data:
                loaded_params = serialization.from_state_dict(self.params, data["params"])
                # Validate array shapes match current model
                flat_current, _ = jax.tree_util.tree_flatten(self.params)
                flat_loaded, _ = jax.tree_util.tree_flatten(loaded_params)
                if any(c.shape != l.shape for c, l in zip(flat_current, flat_loaded)):
                    print(f"  [Checkpoint Skip] Parameter tensor shapes in {filepath} do not match model. Initializing fresh.")
                    return False

                self.params = loaded_params
                if "opt_state" in data:
                    self.opt_state = serialization.from_state_dict(self.opt_state, data["opt_state"])
                if "train_steps" in data:
                    self.train_steps = int(data["train_steps"])
                return True
        except Exception as e:
            print(f"  [Checkpoint Warning] Failed to restore {filepath}: {e}. Initializing fresh.")
            return False
        return False
