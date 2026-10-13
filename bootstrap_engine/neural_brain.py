"""
JAX/Flax Neural Intuition Brain for Axiomatic Bootstrap.

Takes a candidate structure tensor C, embeds its algebraic configuration,
and outputs:
1. Value Head: Expected consistency potential [-1, +1]
2. Policy Head: Probability distribution over discrete mutation actions
"""

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np


class AxiomBrainModel(nn.Module):
    action_size: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Flatten candidate tensor: shape (batch_size, N*N*N)
        flat = x.reshape((x.shape[0], -1))

        # Dense trunk with LayerNorm
        h = nn.Dense(features=self.hidden_dim)(flat)
        h = nn.LayerNorm()(h)
        h = nn.gelu(h)

        h = nn.Dense(features=self.hidden_dim)(h)
        h = nn.LayerNorm()(h)
        h = nn.gelu(h)

        # Residual block
        res = nn.Dense(features=self.hidden_dim)(h)
        res = nn.LayerNorm()(res)
        res = nn.gelu(res)
        h = h + res

        # Policy Head (logits over possible mutation actions)
        policy_logits = nn.Dense(features=self.action_size)(h)

        # Value Head (scalar potential evaluation in [-1, +1])
        v = nn.Dense(features=64)(h)
        v = nn.gelu(v)
        value = nn.tanh(nn.Dense(features=1)(v)).squeeze(-1)

        return policy_logits, value


class NeuralIntuitionEngine:
    """
    Manages parameters, forward inference, and gradient updates with Optax.
    """

    def __init__(self, tensor_shape: Tuple[int, int, int], action_size: int, learning_rate: float = 1e-3, seed: int = 42):
        self.tensor_shape = tensor_shape
        self.action_size = action_size
        self.rng = jax.random.PRNGKey(seed)

        self.model = AxiomBrainModel(action_size=action_size)
        dummy_input = jnp.zeros((1, *tensor_shape), dtype=jnp.float32)
        self.params = self.model.init(self.rng, dummy_input)

        self.tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
        self.opt_state = self.tx.init(self.params)

        # Compile jitted forward and train step
        self._init_jits()

    def _init_jits(self):
        @jax.jit
        def _forward(params, x):
            return self.model.apply(params, x, train=False)

        @jax.jit
        def _train_step(params, opt_state, x, target_policy, target_value):
            def loss_fn(p):
                policy_logits, value = self.model.apply(p, x, train=True)
                # Value loss: Mean squared error
                value_loss = jnp.mean(jnp.square(value - target_value))
                # Policy loss: Cross-entropy
                log_probs = jax.nn.log_softmax(policy_logits, axis=-1)
                policy_loss = -jnp.mean(jnp.sum(target_policy * log_probs, axis=-1))
                total_loss = value_loss + policy_loss
                return total_loss, (value_loss, policy_loss)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, aux), grads = grad_fn(params)
            updates, new_opt_state = self.tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss, aux

        self._forward_jit = _forward
        self._train_step_jit = _train_step

    def predict(self, tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Runs neural inference on a single candidate tensor.
        Returns policy prior probabilities (sum=1) and scalar value [-1, +1].
        """
        batch = jnp.array(tensor, dtype=jnp.float32)[None, ...]
        logits, value = self._forward_jit(self.params, batch)
        policy_probs = np.array(jax.nn.softmax(logits[0], axis=-1))
        val = float(value[0])
        return policy_probs, val

    def train_on_batch(self, tensors: np.ndarray, target_policies: np.ndarray, target_values: np.ndarray) -> Dict[str, float]:
        """
        Executes one gradient update step.
        """
        x = jnp.array(tensors, dtype=jnp.float32)
        tp = jnp.array(target_policies, dtype=jnp.float32)
        tv = jnp.array(target_values, dtype=jnp.float32)

        self.params, self.opt_state, total_loss, (val_loss, pol_loss) = self._train_step_jit(
            self.params, self.opt_state, x, tp, tv
        )

        return {
            "total_loss": float(total_loss),
            "value_loss": float(val_loss),
            "policy_loss": float(pol_loss)
        }
