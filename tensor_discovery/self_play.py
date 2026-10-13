"""
Level 0: Self-Play Training Loop for 3D Tensor Discovery

Orchestrates autonomous self-play, MCTS exploration, and neural network updates:
  - Plays decomposition games starting from target tensor S_0 = T.
  - Updates TensorBrain weights using Optax.
  - When ||S_t|| == 0 is achieved, stores the verified exact algorithm.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
import optax

from tensor_discovery.environment import TensorGameEnv, TensorAction
from tensor_discovery.neural_brain import init_tensor_brain, TensorBrain
from tensor_discovery.mcts import TensorMCTS
from tensor_discovery.extractor import DiscoveredTensorLaw


def run_self_play_discovery(
    target_tensor: np.ndarray,
    target_name: str = "Target Operation",
    num_episodes: int = 50,
    mcts_sims: int = 30,
    lr: float = 0.005,
    seed: int = 42,
) -> Optional[DiscoveredTensorLaw]:
    """
    Runs self-play discovery loop on a target tensor.
    Returns the discovered exact DiscoveredTensorLaw if solved.
    """
    dim_u, dim_v, dim_w = target_tensor.shape
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    # Initialize Brain
    model, params = init_tensor_brain(dim_u, dim_v, dim_w, init_rng)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    opt_state = optimizer.init(params)

    # Loss function for training the brain on visited trajectories
    def compute_loss(p, batch_tensors, batch_values):
        val_preds, _ = jax.vmap(lambda s: model.apply(p, s))(batch_tensors)
        return jnp.mean((val_preds.squeeze(-1) - batch_values) ** 2)

    @jax.jit
    def train_step(p, opt_st, batch_t, batch_v):
        loss, grads = jax.value_and_grad(compute_loss)(p, batch_t, batch_v)
        updates, opt_st = optimizer.update(grads, opt_st, p)
        p = optax.apply_updates(p, updates)
        return p, opt_st, loss

    best_law: Optional[DiscoveredTensorLaw] = None
    min_norm = np.sum(np.abs(target_tensor))

    print("=" * 70)
    print(f"  LEVEL 0: TENSOR DECOMPOSITION DISCOVERY ENGINE")
    print(f"  Target: {target_name} | Shape: {target_tensor.shape}")
    print(f"  Initial L1 Norm: {min_norm} | Goal: Reach exact norm = 0")
    print("=" * 70)

    for ep in range(1, num_episodes + 1):
        env = TensorGameEnv(target_tensor=target_tensor, max_steps=12)
        mcts = TensorMCTS(model=model, params=params, num_simulations=mcts_sims)

        states: List[np.ndarray] = []
        actions: List[TensorAction] = []

        while not env.is_terminal:
            states.append(np.copy(env.current_tensor))
            action, _ = mcts.search(env)
            actions.append(action)
            _, reward, is_term, info = env.step(action)

        current_norm = env.l1_norm
        if current_norm < min_norm:
            min_norm = current_norm

        # Check if solved
        if env.is_solved:
            law = DiscoveredTensorLaw(target_tensor, actions, name=target_name)
            if law.is_exact:
                print(f"\n  [✦ SUCCESS] Discovered Exact Solution in Episode {ep}!")
                print(f"  Discovered Rank: R = {law.rank}")
                print(f"  Exact Residual Error: {np.sum(np.abs(law.residual))}")
                best_law = law
                break

        # Train brain on the trajectory
        if len(states) > 0:
            terminal_val = 10.0 if env.is_solved else -float(current_norm)
            targets = np.full((len(states),), terminal_val, dtype=np.float32)
            batch_tensors = jnp.array(np.stack(states), dtype=jnp.float32)
            batch_targets = jnp.array(targets, dtype=jnp.float32)

            params, opt_state, loss_val = train_step(params, opt_state, batch_tensors, batch_targets)

        if ep % 5 == 0 or ep == 1:
            print(f"  Episode {ep:3d}/{num_episodes} | Steps: {env.steps:2d} | "
                  f"Residual Norm: {current_norm:3d} (Best: {min_norm:3d}) | "
                  f"Solved: {env.is_solved}")

    return best_law
