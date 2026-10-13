"""
Component 5 — The Training Loop: Self-Play RL for Mathematical Discovery.

Implements the AlphaZero-style training pipeline:
1. Self-Play Episode: MCTS builds an AST equation step-by-step.
2. Fitness Assessment: The completed equation is evaluated against the physical dataset.
3. Value Assignment: All intermediate AST states receive the empirical fitness reward.
4. Experience Replay: Samples batches from ReplayBuffer.
5. Optimization: JAX-jitted gradient descent on Policy (Cross-Entropy) and Value (MSE).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax

from lmm.config import LMMConfig
from lmm.ast_graph import MathAST
from lmm.neural_brain import MathBrain, init_brain
from lmm.fitness import PhysicsDataset, evaluate_fitness
from lmm.mcts import MCTS


# ═══════════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Stores experience tuples (features, adjacency, target_policy, target_value)."""

    def __init__(self, capacity: int = 10_000):
        self.capacity: int = capacity
        self.buffer: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        self.position: int = 0

    def add(self, feat: np.ndarray, adj: np.ndarray, policy: np.ndarray, value: float):
        entry = (feat, adj, policy, float(value))
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.position] = entry
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        feats = jnp.array(np.stack([b[0] for b in batch], axis=0))
        adjs  = jnp.array(np.stack([b[1] for b in batch], axis=0))
        pis   = jnp.array(np.stack([b[2] for b in batch], axis=0))
        vals  = jnp.array([[b[3]] for b in batch], dtype=jnp.float32)
        return feats, adjs, pis, vals

    def __len__(self) -> int:
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# JAX JIT Loss & Train Step
# ═══════════════════════════════════════════════════════════════════════════════

def create_train_step(model: MathBrain, optimizer: optax.GradientTransformation):
    """Factory creating jitted train_step function."""

    @jax.jit
    def train_step(params, opt_state, b_feat, b_adj, b_target_pi, b_target_val):
        def loss_fn(p):
            logits, val_pred = model.apply(p, b_feat, b_adj)

            # 1. Policy loss: categorical cross entropy with target distribution
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            # Clip target probabilities to avoid 0 * -inf = nan
            clipped_target = jnp.clip(b_target_pi, 1e-7, 1.0)
            policy_loss = -jnp.mean(jnp.sum(b_target_pi * log_probs, axis=-1))

            # 2. Value loss: Mean squared error
            value_loss = jnp.mean((val_pred - b_target_val) ** 2)

            total_loss = policy_loss + value_loss
            return total_loss, (policy_loss, value_loss)

        (loss, (p_loss, v_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss, p_loss, v_loss

    return train_step


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Play Episode Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(mcts: MCTS,
                dataset: PhysicsDataset,
                cfg: LMMConfig,
                temperature: float = 1.0) -> Tuple[MathAST, float, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Executes a single self-play episode to build an equation from root to completion.

    Returns:
        final_ast: The discovered MathAST
        fitness: Fitness score in [0, 1]
        trajectory: List of (features, adj, policy_target) before value backfill
    """
    ast = MathAST(max_nodes=cfg.max_nodes, num_vars=dataset.dimension)
    trajectory = []

    while not ast.is_complete:
        policy_dict, _ = mcts.search(ast, num_simulations=cfg.mcts_simulations)

        # Build full dense policy array
        pi_arr = np.zeros(cfg.action_space_size, dtype=np.float32)
        actions = list(policy_dict.keys())
        probs = np.array([policy_dict[a] for a in actions], dtype=np.float64)

        for a, p in policy_dict.items():
            pi_arr[a] = p

        feat, adj = ast.to_tensors()
        trajectory.append((feat, adj, pi_arr))

        # Sample action with temperature
        if temperature <= 1e-2:
            action = actions[int(np.argmax(probs))]
        else:
            # Temperature scaling
            scaled_probs = np.power(probs, 1.0 / temperature)
            sum_p = np.sum(scaled_probs)
            if sum_p > 0:
                scaled_probs /= sum_p
            else:
                scaled_probs = np.ones_like(scaled_probs) / len(scaled_probs)
            action = int(np.random.choice(actions, p=scaled_probs))

        ast = ast.apply_action(action)

    fitness = evaluate_fitness(ast, dataset, cfg)
    return ast, fitness, trajectory


# ═══════════════════════════════════════════════════════════════════════════════
# High-Level Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class LMMTrainer:
    """Manages the self-play discovery loop, replay buffer, and neural optimization."""

    def __init__(self, cfg: LMMConfig, dataset: PhysicsDataset, seed: int = 42):
        self.cfg = cfg
        self.dataset = dataset
        self.rng_key = jax.random.PRNGKey(seed)

        # Brain setup
        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.model, self.params = init_brain(cfg, subkey)

        # Optax optimizer
        self.optimizer = optax.adam(learning_rate=cfg.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        self.train_step = create_train_step(self.model, self.optimizer)

        # Search & Memory
        self.mcts = MCTS(self.model, self.params, cfg)
        self.buffer = ReplayBuffer(capacity=cfg.replay_buffer_size)

        self.best_equation: Optional[MathAST] = None
        self.best_fitness: float = -1.0

    def train_episodes(self, num_episodes: int, print_interval: int = 5):
        """Runs the self-play training loop for a specified number of episodes."""
        print(f"Starting LMM Training on target system: {self.dataset.name}")
        print(f"Variables: {self.dataset.var_names} | Action Space: {self.cfg.action_space_size}")
        print("=" * 80)

        for ep in range(1, num_episodes + 1):
            temp = 1.0 if ep < 100 else 0.5
            ast, fitness, trajectory = run_episode(self.mcts, self.dataset, self.cfg, temperature=temp)

            # Map fitness [0, 1] to value target [-1, 1]
            val_target = float(2.0 * fitness - 1.0)

            # Add steps to buffer
            for feat, adj, pi in trajectory:
                self.buffer.add(feat, adj, pi, val_target)

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_equation = ast

            loss_info = ""
            # Train step if enough samples in buffer
            if len(self.buffer) >= self.cfg.batch_size:
                b_feat, b_adj, b_pi, b_val = self.buffer.sample(self.cfg.batch_size)
                self.params, self.opt_state, loss, p_loss, v_loss = self.train_step(
                    self.params, self.opt_state, b_feat, b_adj, b_pi, b_val
                )
                self.mcts.params = self.params  # Update search model with updated weights
                loss_info = f" | Loss: {float(loss):.4f} (Pol: {float(p_loss):.4f}, Val: {float(v_loss):.4f})"

            if ep % print_interval == 0 or ep == 1 or fitness > 0.8:
                eq_str = ast.to_string(self.dataset.var_names)
                if len(eq_str) > 40:
                    eq_str = eq_str[:37] + "..."
                print(f"Ep {ep:4d}/{num_episodes} | Fitness: {fitness:.4f} | Best: {self.best_fitness:.4f} | Eq: {eq_str:<40}{loss_info}")

        print("=" * 80)
        print("Training finished.")
        if self.best_equation is not None:
            print(f"Best Discovered Equation: {self.best_equation.to_string(self.dataset.var_names)}")
            print(f"Highest Fitness Achieved: {self.best_fitness:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from lmm.fitness import make_spring_dataset

    print("=" * 60)
    print("  TRAINER & RL LOOP — SELF TEST")
    print("=" * 60)

    cfg = LMMConfig(max_vars=2, mcts_simulations=15, batch_size=4)
    ds = make_spring_dataset()

    trainer = LMMTrainer(cfg, ds, seed=42)
    # Run a quick 3-episode trial
    trainer.train_episodes(num_episodes=3, print_interval=1)

    print("\n" + "=" * 60)
    print("  TRAINER SELF TEST PASSED ✓")
    print("=" * 60)
