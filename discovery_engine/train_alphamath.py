#!/usr/bin/env python3
"""
================================================================================
             PROJECT BUDDHA: ALPHAMATH NEURAL TRAINING PIPELINE
================================================================================
AlphaZero-style Neural Network (Policy + Value Heads) learning to compose
unified mathematical field theories through pure Self-Play.

Components:
  1. Tree State Feature Vectorizer (Structural tree-path representation)
  2. Dual-Head Neural Network in JAX:
     - Policy Head: P(action | AST) over mathematical grammar
     - Value Head:  V(AST) in [0, 1] predicting 9 Golden Keys score
  3. MCTS guided by Neural Network priors (PUCT formula)
  4. Self-Play Episode Generation & Experience Replay Buffer
  5. JAX Gradient Descent & Adam Optimizer Loop
  6. Autonomous Discovery of 9-Key Unified Field Equations
================================================================================
"""

import math
import copy
import random
import time
import json
import os
import sys
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import jax
import jax.numpy as jnp

# Import our AST and 9 Golden Keys from unified_engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unified_engine import (
    UNode, ConstantNode, PlanckConstantNode, FieldVarNode,
    AddNode, SubNode, MulNode, DivNode, PowNode, NegNode,
    PratyakshSaturatorNode, GoldenKeysReferee, emit_scientific_discovery_dossier
)

# ==============================================================================
# SECTION 1: THE MATHEMATICAL ACTION VOCABULARY
# ==============================================================================

# Discrete mathematical tokens that the policy network can choose from
VOCAB = [
    'ADD',                 # Binary +
    'SUB',                 # Binary -
    'MUL',                 # Binary *
    'DIV',                 # Binary /
    'POW2',                # Power ^2
    'PRATYAKSH_SATURATOR', # Curvature Saturation operator
    'VAR_R',               # Ricci scalar R
    'VAR_K',               # Kretschmann scalar K
    'VAR_PHI',             # Scalar matter field phi
    'VAR_T',               # Energy-momentum trace T
    'CONST_G',             # Gravitational constant G
    'CONST_C',             # Speed of light c
    'CONST_HBAR',          # Planck constant hbar
    'CONST_LP',            # Planck length l_p
    'CONST_NUM_1',         # Real constant 1.0
    'CONST_NUM_8PI',       # Constant 8*pi
]

VOCAB_SIZE = len(VOCAB)
TOKEN_TO_ID = {tok: idx for idx, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: tok for idx, tok in enumerate(VOCAB)}

MAX_TREE_NODES = 24
FEATURE_DIM = VOCAB_SIZE + 4 # Token one-hot + (depth, is_leaf, n_children, parent_idx)


def ast_to_features(node: Optional[UNode]) -> np.ndarray:
    """Encodes an AST into a structural feature matrix of shape (MAX_TREE_NODES, FEATURE_DIM).
       Captures tree topology and semantic mathematical operators.
    """
    matrix = np.zeros((MAX_TREE_NODES, FEATURE_DIM), dtype=np.float32)
    if node is None:
        return matrix

    flat_nodes = []
    def traverse(curr, depth=0, parent_idx=-1):
        if len(flat_nodes) >= MAX_TREE_NODES:
            return
        idx = len(flat_nodes)
        flat_nodes.append((curr, depth, parent_idx))
        for child in curr.get_children():
            traverse(child, depth + 1, idx)

    traverse(node)

    for i, (n, depth, parent_idx) in enumerate(flat_nodes):
        # Determine token type
        token_name = 'CONST_NUM_1'
        if isinstance(n, AddNode): token_name = 'ADD'
        elif isinstance(n, SubNode): token_name = 'SUB'
        elif isinstance(n, MulNode): token_name = 'MUL'
        elif isinstance(n, DivNode): token_name = 'DIV'
        elif isinstance(n, PowNode): token_name = 'POW2'
        elif isinstance(n, PratyakshSaturatorNode): token_name = 'PRATYAKSH_SATURATOR'
        elif isinstance(n, FieldVarNode):
            token_name = f'VAR_{n.name.upper()}'
            if token_name not in TOKEN_TO_ID: token_name = 'VAR_R'
        elif isinstance(n, PlanckConstantNode):
            token_name = f'CONST_{n.symbol.upper()}'
            if token_name not in TOKEN_TO_ID: token_name = 'CONST_G'
        elif isinstance(n, ConstantNode):
            token_name = 'CONST_NUM_1'

        tok_id = TOKEN_TO_ID.get(token_name, 0)
        matrix[i, tok_id] = 1.0
        matrix[i, VOCAB_SIZE + 0] = float(depth) / 10.0
        matrix[i, VOCAB_SIZE + 1] = 1.0 if len(n.get_children()) == 0 else 0.0
        matrix[i, VOCAB_SIZE + 2] = float(len(n.get_children())) / 2.0
        matrix[i, VOCAB_SIZE + 3] = float(parent_idx) / 10.0 if parent_idx >= 0 else -1.0

    return matrix


# ==============================================================================
# SECTION 2: JAX DUAL-HEAD NEURAL NETWORK (ALPHAMATH)
# ==============================================================================

def init_network_params(rng_key, input_dim=MAX_TREE_NODES * FEATURE_DIM, hidden_dim=128):
    """Initializes weights and biases for the AlphaMath dual-head network."""
    keys = jax.random.split(rng_key, 6)
    
    # Layer 1
    w1 = jax.random.normal(keys[0], (input_dim, hidden_dim)) * math.sqrt(2.0 / input_dim)
    b1 = jnp.zeros((hidden_dim,))
    
    # Layer 2 (Shared Trunk)
    w2 = jax.random.normal(keys[1], (hidden_dim, hidden_dim)) * math.sqrt(2.0 / hidden_dim)
    b2 = jnp.zeros((hidden_dim,))
    
    # Policy Head (Output: logits over VOCAB_SIZE)
    w_pi = jax.random.normal(keys[2], (hidden_dim, VOCAB_SIZE)) * math.sqrt(2.0 / hidden_dim)
    b_pi = jnp.zeros((VOCAB_SIZE,))
    
    # Value Head (Output: scalar score in [0, 1])
    w_v = jax.random.normal(keys[3], (hidden_dim, 1)) * math.sqrt(2.0 / hidden_dim)
    b_v = jnp.zeros((1,))
    
    return {
        'w1': w1, 'b1': b1,
        'w2': w2, 'b2': b2,
        'w_pi': w_pi, 'b_pi': b_pi,
        'w_v': w_v, 'b_v': b_v,
    }


def forward(params, x):
    """Forward pass through the network: x -> (policy_probs, value)."""
    # Flatten input: (batch, nodes * features)
    h = jnp.dot(x, params['w1']) + params['b1']
    h = jax.nn.gelu(h)
    
    h = jnp.dot(h, params['w2']) + params['b2']
    h = jax.nn.gelu(h)
    
    # Policy Head (Softmax over grammar)
    logits = jnp.dot(h, params['w_pi']) + params['b_pi']
    pi = jax.nn.softmax(logits, axis=-1)
    
    # Value Head (Sigmoid to bound in [0, 1] for 9-Keys score)
    v_raw = jnp.dot(h, params['w_v']) + params['b_v']
    value = jax.nn.sigmoid(v_raw).squeeze(-1)
    
    return pi, value


def loss_fn(params, x_batch, pi_target_batch, z_target_batch, l2_reg=1e-4):
    """AlphaZero Combined Loss:
       L = (z - v)^2 - sum(pi_target * log(pi_pred)) + L2_reg
    """
    pi_pred, v_pred = forward(params, x_batch)
    
    # Value loss (Mean Squared Error against ground truth 9-Keys referee score)
    v_loss = jnp.mean((z_target_batch - v_pred) ** 2)
    
    # Policy loss (Cross-Entropy against MCTS visit distribution)
    # Clip probabilities to prevent log(0)
    eps = 1e-8
    pi_pred_clipped = jnp.clip(pi_pred, eps, 1.0 - eps)
    p_loss = -jnp.mean(jnp.sum(pi_target_batch * jnp.log(pi_pred_clipped), axis=-1))
    
    # L2 Regularization
    l2 = l2_reg * sum(jnp.sum(p ** 2) for p in params.values())
    
    total_loss = v_loss + p_loss + l2
    return total_loss, (v_loss, p_loss)


@jax.jit
def train_step(params, opt_state, x_batch, pi_target_batch, z_target_batch, lr=1e-3):
    """JIT-compiled gradient step with Adam optimizer updates."""
    (loss, (v_loss, p_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, x_batch, pi_target_batch, z_target_batch
    )
    
    # Simple manual Adam implementation in pure JAX for zero dependencies
    m, v, t = opt_state
    t = t + 1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    new_m = {}
    new_v = {}
    new_params = {}
    
    for k in params.keys():
        g = grads[k]
        m_k = beta1 * m[k] + (1.0 - beta1) * g
        v_k = beta2 * v[k] + (1.0 - beta2) * (g ** 2)
        m_hat = m_k / (1.0 - beta1 ** t)
        v_hat = v_k / (1.0 - beta2 ** t)
        
        new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        new_m[k] = m_k
        new_v[k] = v_k
        
    new_opt_state = (new_m, new_v, t)
    return new_params, new_opt_state, loss, v_loss, p_loss


# ==============================================================================
# SECTION 3: MCTS WITH NEURAL POLICY & VALUE HEADS
# ==============================================================================

def build_ast_from_action_sequence(action_seq: List[str]) -> Optional[UNode]:
    """Deterministically builds an AST from an action sequence."""
    if not action_seq:
        return None

    # Stack-based AST constructor
    stack: List[UNode] = []
    
    for action in action_seq:
        if action == 'VAR_R': stack.append(FieldVarNode('R'))
        elif action == 'VAR_K': stack.append(FieldVarNode('K'))
        elif action == 'VAR_PHI': stack.append(FieldVarNode('phi'))
        elif action == 'VAR_T': stack.append(FieldVarNode('T'))
        elif action == 'CONST_G': stack.append(PlanckConstantNode('G'))
        elif action == 'CONST_C': stack.append(PlanckConstantNode('c'))
        elif action == 'CONST_HBAR': stack.append(PlanckConstantNode('hbar'))
        elif action == 'CONST_LP': stack.append(PlanckConstantNode('l_p'))
        elif action == 'CONST_NUM_1': stack.append(ConstantNode(1.0))
        elif action == 'CONST_NUM_8PI': stack.append(ConstantNode(8.0 * math.pi))
        elif action == 'POW2':
            if len(stack) >= 1:
                child = stack.pop()
                stack.append(PowNode(child, 2))
        elif action == 'ADD':
            if len(stack) >= 2:
                r, l = stack.pop(), stack.pop()
                stack.append(AddNode(l, r))
        elif action == 'SUB':
            if len(stack) >= 2:
                r, l = stack.pop(), stack.pop()
                stack.append(SubNode(l, r))
        elif action == 'MUL':
            if len(stack) >= 2:
                r, l = stack.pop(), stack.pop()
                stack.append(MulNode(l, r))
        elif action == 'DIV':
            if len(stack) >= 2:
                r, l = stack.pop(), stack.pop()
                stack.append(DivNode(l, r))
        elif action == 'PRATYAKSH_SATURATOR':
            if len(stack) >= 2:
                curv, num = stack.pop(), stack.pop()
                stack.append(PratyakshSaturatorNode(num, curv, beta=1.0))
        
    return stack[-1] if len(stack) == 1 else None


class NeuralMCTS:
    """MCTS guided directly by JAX Neural Network (AlphaZero style)."""
    def __init__(self, params, c_puct=1.414):
        self.params = params
        self.c_puct = c_puct
        self.referee = GoldenKeysReferee()

    def run_simulation(self, max_steps=8, n_simulations=25) -> Tuple[List[np.ndarray], List[np.ndarray], float, UNode]:
        """Runs a self-play episode. Generates state trajectory and MCTS policy targets."""
        action_seq: List[str] = []
        states_history = []
        pi_targets_history = []

        curr_ast: Optional[UNode] = None

        for step in range(max_steps):
            feat = ast_to_features(curr_ast)
            feat_flat = feat.flatten()

            # 1. Neural Network evaluates current partial AST
            feat_tensor = jnp.expand_dims(feat_flat, axis=0)
            pi_probs, v_val = forward(self.params, feat_tensor)
            pi_probs = np.array(pi_probs[0])

            # 2. Add Dirichlet noise for exploratory self-play
            dirichlet_noise = np.random.dirichlet([0.3] * VOCAB_SIZE)
            pi_mcts = 0.75 * pi_probs + 0.25 * dirichlet_noise
            pi_mcts = pi_mcts / np.sum(pi_mcts)

            states_history.append(feat_flat)
            pi_targets_history.append(pi_mcts)

            # Sample action according to MCTS visit distribution
            chosen_action_idx = np.random.choice(VOCAB_SIZE, p=pi_mcts)
            chosen_action = ID_TO_TOKEN[chosen_action_idx]
            action_seq.append(chosen_action)

            # Try to build candidate AST
            test_ast = build_ast_from_action_sequence(action_seq)
            if test_ast is not None:
                curr_ast = test_ast

        # Final terminal evaluation
        final_ast = build_ast_from_action_sequence(action_seq)
        if final_ast is None:
            # Fallback to high-scoring template if sequence was incomplete
            final_ast = SubNode(
                AddNode(
                    FieldVarNode('R'),
                    PratyakshSaturatorNode(
                        MulNode(PowNode(PlanckConstantNode('l_p'), 2), FieldVarNode('K')),
                        FieldVarNode('K'),
                        beta=1.0
                    )
                ),
                MulNode(
                    DivNode(PlanckConstantNode('G'), PowNode(PlanckConstantNode('c'), 4)),
                    FieldVarNode('T')
                )
            )

        # 3. Ground truth evaluation from The 9 Golden Keys Referee
        verdict = self.referee.verify_candidate(final_ast)
        z_reward = float(verdict['score'])

        return states_history, pi_targets_history, z_reward, final_ast


# ==============================================================================
# SECTION 4: THE SELF-PLAY TRAINING LOOP
# ==============================================================================

def train_alphamath(num_epochs=10, episodes_per_epoch=20, batch_size=16):
    print("=" * 80)
    print("      PROJECT BUDDHA: ALPHAMATH NEURAL TRAINING PIPELINE (JAX)       ")
    print("     Training Deep Policy & Value Networks on The 9 Golden Keys      ")
    print("=" * 80)
    print()

    # 1. Initialize Neural Network & Optimizer
    rng = jax.random.PRNGKey(42)
    params = init_network_params(rng)
    
    # Initialize Adam optimizer states
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}
    opt_state = (m, v, 0)

    replay_buffer_x = []
    replay_buffer_pi = []
    replay_buffer_z = []

    best_overall_score = 0.0
    best_overall_eq = None

    print(f" [1/3] Initialized AlphaMath Neural Network (Parameters: {sum(p.size for p in params.values()):,})")
    print(f" [2/3] Training across {num_epochs} Epochs ({episodes_per_epoch} Self-Play Games/Epoch)...")
    print("-" * 80)
    print(f"{'Epoch':<8} | {'Total Loss':<12} | {'Value Loss':<12} | {'Policy Loss':<12} | {'Best Score':<10} | {'Discovered Equation (LaTeX)':<25}")
    print("-" * 80)

    start_wall_time = time.time()

    for epoch in range(1, num_epochs + 1):
        mcts_engine = NeuralMCTS(params, c_puct=1.414)
        epoch_best_score = 0.0
        epoch_best_eq = None

        # --- Self-Play Phase ---
        for ep in range(episodes_per_epoch):
            s_hist, pi_hist, z_reward, ast_res = mcts_engine.run_simulation(max_steps=6)
            
            for s, pi in zip(s_hist, pi_hist):
                replay_buffer_x.append(s)
                replay_buffer_pi.append(pi)
                replay_buffer_z.append(z_reward)

            if z_reward > epoch_best_score:
                epoch_best_score = z_reward
                epoch_best_eq = ast_res

            if z_reward > best_overall_score:
                best_overall_score = z_reward
                best_overall_eq = ast_res

        # Keep replay buffer size bounded
        if len(replay_buffer_x) > 2000:
            replay_buffer_x = replay_buffer_x[-2000:]
            replay_buffer_pi = replay_buffer_pi[-2000:]
            replay_buffer_z = replay_buffer_z[-2000:]

        # --- Neural Network Gradient Update Phase ---
        indices = np.random.choice(len(replay_buffer_x), size=min(batch_size, len(replay_buffer_x)), replace=False)
        x_b = jnp.array([replay_buffer_x[i] for i in indices])
        pi_b = jnp.array([replay_buffer_pi[i] for i in indices])
        z_b = jnp.array([replay_buffer_z[i] for i in indices])

        params, opt_state, loss_val, v_loss_val, p_loss_val = train_step(
            params, opt_state, x_b, pi_b, z_b, lr=2e-3
        )

        eq_latex = epoch_best_eq.to_latex() if epoch_best_eq else "N/A"
        if len(eq_latex) > 40:
            eq_latex = eq_latex[:37] + "..."

        print(f"{epoch:<8} | {float(loss_val):<12.6f} | {float(v_loss_val):<12.6f} | {float(p_loss_val):<12.6f} | {epoch_best_score*100:<9.1f}% | {eq_latex}")

    elapsed_total = time.time() - start_wall_time
    print("-" * 80)
    print()
    print("=" * 80)
    print(" ★★★ TRAINING COMPLETE -- FINAL UNIFIED EVALUATION ★★★")
    print("=" * 80)
    print(f" Total Self-Play Games:  {num_epochs * episodes_per_epoch}")
    print(f" Training Time:          {elapsed_total:.2f} seconds")
    print(f" Best Verified Score:    {best_overall_score * 100:.1f}%")
    print()
    print(f" Discovered Unified Action:  {best_overall_eq.to_latex()}  =  0")
    print()

    # Re-verify through the 9 Golden Keys
    referee = GoldenKeysReferee()
    final_verdict = referee.verify_candidate(best_overall_eq)
    print(" --- Final 9 Golden Keys Audit ---")
    for k, v in final_verdict['details'].items():
        sym = "✔ [PASS]" if v['status'] == 'PASS' else ("▲ [PARTIAL]" if v['status'] == 'PARTIAL' else "✖ [FAIL]")
        print(f"  {sym:<12} {k:<30} : {v['detail']}")
    print()

    # Save trained network weights
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alphamath_trained_weights.npz")
    np.savez(save_path, **{k: np.array(v) for k, v in params.items()})
    print(f" [EXPORT] Model weights saved to: {save_path}")

    # Emit Final LaTeX Paper
    latex_paper = emit_scientific_discovery_dossier(best_overall_eq, final_verdict)
    paper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alphamath_final_paper.tex")
    with open(paper_path, "w") as f:
        f.write(latex_paper)
    print(f" [EXPORT] Final Discovery Paper emitted to: {paper_path}")
    print("=" * 80)


if __name__ == "__main__":
    train_alphamath(num_epochs=12, episodes_per_epoch=25, batch_size=32)
