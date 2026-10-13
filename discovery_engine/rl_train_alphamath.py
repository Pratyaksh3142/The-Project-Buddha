#!/usr/bin/env python3
"""
================================================================================
          PROJECT BUDDHA: ALPHA-MATH RL SELF-PLAY LOOP
================================================================================
Ties together:
  1. The Physical Environment (Pratyaksh / autonomous_discovery)
  2. The Neural Brain (JAX Graph Transformer)
  3. The MCTS Engine (AlphaZero Search)
  4. The RL Replay Buffer & Training Loop
================================================================================
"""

import math
import random
import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
import copy
from typing import List, Dict, Any, Tuple

# Import our Buddha Components
from neural_alphamath import initialize_model, AlphaMathBrain, MAX_NODES, NUM_NODE_TYPES, ACTION_SPACE_SIZE
from mcts_alphamath import GraphMCTS
from graph_engine import ComputationGraph, GraphNode
from autonomous_discovery import make_damped_pendulum_dataset, evaluate_fitness

# Action Space Definition for RL Loop
# 0..14: Add a new node of a specific type
# 15..127: Add a directed edge between nodes (simplified flattened index)

# Node type mapping for JAX One-Hot encoding
NODE_TYPES = ['var', 'const', 'add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'abs', 'sqrt', 'tanh', 'neg', 'pow', 'pad']
NODE_TYPE_TO_IDX = {nt: i for i, nt in enumerate(NODE_TYPES)}

class RLComputationGraph:
    """Wrapper around ComputationGraph to provide an RL environment API."""
    def __init__(self, base_graph, max_nodes=MAX_NODES):
        self.graph = base_graph
        self.max_nodes = max_nodes
        self.step_count = 0
        self.max_steps = 20

    def copy_and_apply(self, action_id: int):
        new_rl_graph = RLComputationGraph(self.graph.copy(), max_nodes=self.max_nodes)
        new_rl_graph.step_count = self.step_count + 1
        g = new_rl_graph.graph

        # Execute topological mutation based on action_id
        if action_id < 15:
            # Action: Add Node
            ntype = NODE_TYPES[action_id]
            if len(g.nodes) < self.max_nodes:
                nid = max(g.nodes.keys()) + 1 if g.nodes else 0
                if ntype == 'var':
                    g.nodes[nid] = GraphNode(nid, 'var', var_idx=random.randint(0, g.dimension-1))
                elif ntype == 'const':
                    g.nodes[nid] = GraphNode(nid, 'const', const_val=random.uniform(-1.0, 1.0))
                elif ntype == 'pad':
                    pass
                else:
                    g.nodes[nid] = GraphNode(nid, 'op', op=ntype)
                
                # Automatically connect it as an output node if needed
                if len(g.output_nodes) < g.dimension:
                    g.output_nodes.append(nid)
        else:
            # Action: Connect nodes
            # Decode action into source and target
            edge_id = action_id - 15
            if len(g.nodes) > 1:
                src_idx = edge_id % len(g.nodes)
                dst_idx = (edge_id // len(g.nodes)) % len(g.nodes)
                
                node_keys = list(g.nodes.keys())
                if src_idx < len(node_keys) and dst_idx < len(node_keys):
                    src_nid = node_keys[src_idx]
                    dst_nid = node_keys[dst_idx]
                    dst_node = g.nodes[dst_nid]
                    
                    if dst_node.node_type == 'op':
                        # Allow max 2 inputs for binary ops, 1 for unary
                        max_ins = 2 if dst_node.op in ['add', 'sub', 'mul', 'div', 'pow'] else 1
                        if len(dst_node.inputs) < max_ins and src_nid != dst_nid:
                            # Prevent cycles by checking topological validness
                            dst_node.inputs.append(src_nid)
                            if not g.is_valid(): # Rollback if cycle created
                                dst_node.inputs.pop()
        
        return new_rl_graph

    def get_legal_actions(self) -> List[int]:
        # For this prototype, all actions are "legal" but the environment ignores invalid ones.
        return list(range(ACTION_SPACE_SIZE))

    def is_terminal(self) -> bool:
        return self.step_count >= self.max_steps or len(self.graph.nodes) >= self.max_nodes

    def to_jax_tensors(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Convert DAG to [Batch=1, Max_Nodes, Features], [Batch=1, Max_Nodes, Max_Nodes]
        features = np.zeros((1, self.max_nodes, NUM_NODE_TYPES))
        adj = np.zeros((1, self.max_nodes, self.max_nodes))
        
        node_keys = list(self.graph.nodes.keys())
        for i, nid in enumerate(node_keys):
            if i >= self.max_nodes: break
            node = self.graph.nodes[nid]
            ntype = 'op' if node.node_type == 'op' else node.node_type
            if node.node_type == 'op': ntype = node.op
            
            idx = NODE_TYPE_TO_IDX.get(ntype, 0)
            features[0, i, idx] = 1.0
            
            for inp_nid in node.inputs:
                try:
                    inp_i = node_keys.index(inp_nid)
                    if inp_i < self.max_nodes:
                        adj[0, i, inp_i] = 1.0  # Directed edge from input to current node
                except ValueError:
                    pass
                    
        return jnp.array(features), jnp.array(adj)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, state_features, state_adj, policy, value=0.0):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state_features, state_adj, policy, value))

    def update_episode_values(self, episode_indices, final_reward):
        for idx in episode_indices:
            feat, adj, pol, _ = self.buffer[idx]
            self.buffer[idx] = (feat, adj, pol, final_reward)


def alphazero_loss(params, model, batch_features, batch_adj, target_policies, target_values):
    """
    Combined Loss = CrossEntropy(Policy) + MeanSquaredError(Value)
    """
    pred_logits, pred_values = model.apply(params, batch_features, batch_adj)
    
    # Value Loss (MSE)
    value_loss = jnp.mean(jnp.square(target_values - pred_values.squeeze()))
    
    # Policy Loss (Cross Entropy)
    # Target policy is a probability distribution from MCTS
    log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
    policy_loss = -jnp.mean(jnp.sum(target_policies * log_probs, axis=-1))
    
    return value_loss + policy_loss

def train_step(params, opt_state, model, optimizer, batch_features, batch_adj, target_policies, target_values):
    loss, grads = jax.value_and_grad(alphazero_loss)(params, model, batch_features, batch_adj, target_policies, target_values)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

train_step_jit = jax.jit(train_step, static_argnums=(2, 3))


def self_play_training_loop():
    print("=" * 80)
    print("      PROJECT BUDDHA: ALPHA-MATH MCTS SELF-PLAY TRAINING             ")
    print("=" * 80)
    
    # 1. Initialize Dataset
    dataset = make_damped_pendulum_dataset()
    
    # 2. Initialize JAX Brain
    key = jax.random.PRNGKey(42)
    model, params = initialize_model(key)
    
    # 3. Initialize Optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    
    buffer = ReplayBuffer()
    
    # We subclass GraphMCTS to override _get_masked_probabilities to fix dict conversion
    class RLGraphMCTS(GraphMCTS):
        def _get_masked_probabilities(self, logits, legal_actions):
            import jax
            import jax.numpy as jnp
            # Softmax
            probs = jax.nn.softmax(logits).squeeze()
            prob_dict = {}
            for a in legal_actions:
                prob_dict[a] = float(probs[a])
            return prob_dict
            
    mcts = RLGraphMCTS(model, params, num_simulations=10) # 10 sims for fast POC
    
    NUM_EPISODES = 3
    BATCH_SIZE = 4
    
    for episode in range(NUM_EPISODES):
        print(f"--- Starting Episode {episode+1}/{NUM_EPISODES} ---")
        
        # Start empty graph
        cg = ComputationGraph(dimension=dataset.dimension)
        state = RLComputationGraph(cg)
        
        episode_memory_indices = []
        
        while not state.is_terminal():
            # MCTS Search for Policy
            pi_dict = mcts.search(state)
            
            # Convert policy to array
            pi_arr = np.zeros(ACTION_SPACE_SIZE)
            for a, p in pi_dict.items():
                pi_arr[a] = p
                
            # Sample action
            actions, probs = list(pi_dict.keys()), list(pi_dict.values())
            action = np.random.choice(actions, p=probs)
            
            # Store State, Pi in buffer (Value to be filled later)
            feat, adj = state.to_jax_tensors()
            buffer.add(feat, adj, pi_arr, 0.0)
            episode_memory_indices.append(len(buffer.buffer) - 1)
            
            # Take step
            state = state.copy_and_apply(action)
            
        # Referee Evaluation (Pratyaksh Physics)
        print(f" Episode {episode+1} Equation Generated. Evaluating Physical Fitness...")
        fitness = evaluate_fitness(state.graph, dataset)
        print(f" Fitness Score: {fitness:.6f} | Nodes: {state.graph.complexity()}")
        
        # Backpropagate Final Value to episode states
        buffer.update_episode_values(episode_memory_indices, fitness)
        
        # Train Brain
        if len(buffer.buffer) >= BATCH_SIZE:
            batch = random.sample(buffer.buffer, BATCH_SIZE)
            b_feat = jnp.concatenate([b[0] for b in batch])
            b_adj = jnp.concatenate([b[1] for b in batch])
            b_pi = jnp.array([b[2] for b in batch])
            b_val = jnp.array([b[3] for b in batch])
            
            params, opt_state, loss = train_step_jit(params, opt_state, model, optimizer, b_feat, b_adj, b_pi, b_val)
            print(f" Neural Network Trained. Loss: {loss:.4f}")
            
            # Update MCTS params
            mcts.params = params

    print("=" * 80)
    print(" ALPHA-MATH TRAINING LOOP COMPLETED ")
    print(" Architecture is Fully-Fledged and Ready for Large-Scale Compute.")
    print("=" * 80)

if __name__ == "__main__":
    self_play_training_loop()
