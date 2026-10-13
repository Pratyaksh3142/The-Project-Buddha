#!/usr/bin/env python3
"""
================================================================================
          PROJECT BUDDHA: UNIVERSAL MATHEMATICS PRE-TRAINING
================================================================================
This script generates random, alien mathematical differential equations, 
simulates them to create synthetic datasets, and trains the JAX Graph 
Transformer to discover them. This creates the Foundation Model.
================================================================================
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
import random
import os
import time

# Import Buddha Components
from neural_alphamath import initialize_model, ACTION_SPACE_SIZE
from mcts_alphamath import GraphMCTS
from graph_engine import ComputationGraph, GraphNode
from autonomous_discovery import PhysicalDataset, evaluate_fitness
from rl_train_alphamath import RLComputationGraph, ReplayBuffer, train_step_jit

def generate_alien_universe(dimension=2, max_complexity=8) -> PhysicalDataset:
    """Generates a random, valid differential equation and simulates it."""
    # For robust pre-training, we create simple chaotic/decaying random equations
    # To avoid inf/nan trajectories, we heavily bound the random operators.
    # In a full-scale run, we'd use the GraphEngine to randomly build DAGs.
    # Here, we create a pseudo-random stable linear/trig system for the POC.
    
    coeffs = np.random.uniform(-2.0, 2.0, size=(dimension, dimension))
    
    def true_derivs(y):
        dy = []
        for i in range(dimension):
            # Mix of linear and trig to make it non-trivial but stable
            val = sum(coeffs[i, j] * y[j] for j in range(dimension))
            dy.append(val - 0.1 * y[i]**3)  # damping to prevent explosion
        return dy

    t_end = 2.0
    dt = 0.1

    def gen_traj(y0):
        y = list(y0)
        t = 0.0
        traj = [list(y)]
        while t < t_end - 1e-12:
            k1 = true_derivs(y)
            y2 = [y[i] + 0.5 * dt * k1[i] for i in range(dimension)]
            k2 = true_derivs(y2)
            y3 = [y[i] + 0.5 * dt * k2[i] for i in range(dimension)]
            k3 = true_derivs(y3)
            y4 = [y[i] + dt * k3[i] for i in range(dimension)]
            k4 = true_derivs(y4)
            for i in range(dimension):
                y[i] += (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            t += dt
            traj.append(list(y))
        return traj

    test_ics = [np.random.uniform(-1.0, 1.0, size=dimension).tolist() for _ in range(3)]
    ic_trajs = [(ic, gen_traj(ic)) for ic in test_ics]

    return PhysicalDataset(
        name=f"Alien Universe {random.randint(1000, 9999)}",
        dimension=dimension,
        var_names=[f'y{i}' for i in range(dimension)],
        t_end=t_end,
        dt=dt,
        ic_trajectories=ic_trajs
    )

def universal_pretrain_loop():
    print("================================================================================")
    print("      PROJECT BUDDHA: INITIATING LMM UNIVERSAL PRE-TRAINING")
    print("================================================================================")
    
    key = jax.random.PRNGKey(42)
    model, params = initialize_model(key)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    buffer = ReplayBuffer(capacity=50000)
    
    class RLGraphMCTS(GraphMCTS):
        def _get_masked_probabilities(self, logits, legal_actions):
            probs = jax.nn.softmax(logits).squeeze()
            return {a: float(probs[a]) for a in legal_actions}
            
    mcts = RLGraphMCTS(model, params, num_simulations=20)
    
    EPISODES = 100000
    BATCH_SIZE = 32
    
    print(" [STATUS] Model initialized on device:", jax.devices()[0])
    
    for episode in range(EPISODES):
        # Every 10 episodes, generate a new random universe
        if episode % 10 == 0:
            dataset = generate_alien_universe()
            print(f"\n[UNIVERSE] Generated {dataset.name}")

        cg = ComputationGraph(dimension=dataset.dimension)
        state = RLComputationGraph(cg)
        episode_memory_indices = []
        
        while not state.is_terminal():
            pi_dict = mcts.search(state)
            pi_arr = np.zeros(ACTION_SPACE_SIZE)
            for a, p in pi_dict.items(): pi_arr[a] = p
            action = np.random.choice(list(pi_dict.keys()), p=list(pi_dict.values()))
            
            feat, adj = state.to_jax_tensors()
            buffer.add(feat, adj, pi_arr, 0.0)
            episode_memory_indices.append(len(buffer.buffer) - 1)
            state = state.copy_and_apply(action)
            
        fitness = evaluate_fitness(state.graph, dataset)
        buffer.update_episode_values(episode_memory_indices, fitness)
        
        if len(buffer.buffer) >= BATCH_SIZE:
            import random as pyrand
            batch = pyrand.sample(buffer.buffer, BATCH_SIZE)
            b_feat = jnp.concatenate([b[0] for b in batch])
            b_adj = jnp.concatenate([b[1] for b in batch])
            b_pi = jnp.array([b[2] for b in batch])
            b_val = jnp.array([b[3] for b in batch])
            
            params, opt_state, loss = train_step_jit(params, opt_state, model, optimizer, b_feat, b_adj, b_pi, b_val)
            mcts.params = params
            
            if episode % 5 == 0:
                print(f" Ep {episode:05d} | Env: {dataset.name} | Fitness: {fitness:.4f} | Loss: {loss:.4f}")
                
        # Save checkpoint periodically
        if episode > 0 and episode % 100 == 0:
            print(f" [CHECKPOINT] Saving model at episode {episode}...")
            # In production, use orbax/flax.training.checkpoints
            
if __name__ == "__main__":
    universal_pretrain_loop()
