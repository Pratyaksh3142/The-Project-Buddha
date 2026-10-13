"""
Project Buddha — Level 0: Tensor Decomposition Discovery Engine
AlphaTensor-style exact mathematical and physical law discovery from scratch.
"""

from tensor_discovery.environment import TensorGameEnv, TensorAction
from tensor_discovery.targets import (
    make_matrix_multiplication_tensor,
    make_lie_algebra_so3_tensor,
    make_symplectic_invariance_tensor,
)
from tensor_discovery.neural_brain import TensorBrain, init_tensor_brain
from tensor_discovery.mcts import TensorMCTS
from tensor_discovery.extractor import DiscoveredTensorLaw
from tensor_discovery.self_play import run_self_play_discovery

__all__ = [
    "TensorGameEnv",
    "TensorAction",
    "make_matrix_multiplication_tensor",
    "make_lie_algebra_so3_tensor",
    "make_symplectic_invariance_tensor",
    "TensorBrain",
    "init_tensor_brain",
    "TensorMCTS",
    "DiscoveredTensorLaw",
    "run_self_play_discovery",
]
