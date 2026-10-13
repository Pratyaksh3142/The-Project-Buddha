"""
Configuration: All hyperparameters for the Large Mathematical Model in one place.
"""

from dataclasses import dataclass


@dataclass
class LMMConfig:
    """Central configuration for the entire LMM pipeline."""

    # --- Graph ---
    max_nodes: int = 20          # Max nodes in any equation AST
    max_vars: int = 4            # Max state variables (e.g., r, v, theta, omega)

    # --- Neural Network ---
    embedding_dim: int = 128     # Latent dimension for graph embeddings
    mpnn_layers: int = 4         # Message-passing rounds
    mlp_hidden: int = 128        # Hidden dim in Policy/Value heads

    # --- MCTS ---
    mcts_simulations: int = 100  # Tree search rollouts per move
    c_puct: float = 1.5          # Exploration constant (PUCT formula)

    # --- Training ---
    replay_buffer_size: int = 10_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    training_episodes: int = 50_000
    checkpoint_interval: int = 500

    # --- Fitness Evaluator ---
    complexity_penalty: float = 0.01   # Occam penalty per node
    integration_dt: float = 0.01       # RK4 timestep

    # --- Vocabulary sizes (derived) ---
    NUM_BINARY_OPS: int = 5   # ADD SUB MUL DIV POW
    NUM_UNARY_OPS: int = 5    # SIN COS EXP LOG NEG
    NUM_CONSTANTS: int = 3    # 0.5, 1.0, 2.0

    @property
    def vocab_size(self) -> int:
        """Total node types the neural net must distinguish."""
        return self.NUM_BINARY_OPS + self.NUM_UNARY_OPS + self.NUM_CONSTANTS + self.max_vars

    @property
    def action_space_size(self) -> int:
        """Each action = place one of these node types at the next open slot."""
        return self.vocab_size
