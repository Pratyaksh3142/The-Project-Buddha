"""
Level 0: The 3D Tensor Decomposition Game Environment

Converts mathematical theorem proving and physical law discovery into an
exact, discrete game:
    State:      3D Tensor S_t in R^{U x V x W} (or Z^{U x V x W})
    Action:     Rank-1 outer product (u (x) v (x) w)
    Transition: S_{t+1} = S_t - (u (x) v (x) w)
    Win:        ||S_t|| == 0  (Zero residual error, exact identity)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class TensorAction:
    """A rank-1 outer product action specified by three vectors (u, v, w)."""
    u: Tuple[int, ...]
    v: Tuple[int, ...]
    w: Tuple[int, ...]

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.u, dtype=np.int32),
            np.array(self.v, dtype=np.int32),
            np.array(self.w, dtype=np.int32),
        )

    def outer_product(self) -> np.ndarray:
        u_arr, v_arr, w_arr = self.to_arrays()
        # u (x) v (x) w -> shape (len(u), len(v), len(w))
        return np.einsum("i,j,k->ijk", u_arr, v_arr, w_arr)


class TensorGameEnv:
    """
    Exact Markov Decision Process for 3D Tensor Decomposition.
    """

    def __init__(
        self,
        target_tensor: np.ndarray,
        max_steps: int = 12,
        allowed_coeffs: Tuple[int, ...] = (-1, 0, 1),
    ):
        self.target_tensor = np.copy(target_tensor).astype(np.int32)
        self.shape = self.target_tensor.shape
        assert len(self.shape) == 3, f"Expected 3D tensor, got shape {self.shape}"
        self.dim_u, self.dim_v, self.dim_w = self.shape

        self.max_steps = max_steps
        self.allowed_coeffs = allowed_coeffs

        # State tracking
        self.current_tensor = np.copy(self.target_tensor)
        self.history: List[TensorAction] = []
        self.steps = 0

    def reset(self) -> np.ndarray:
        """Reset environment back to initial target tensor."""
        self.current_tensor = np.copy(self.target_tensor)
        self.history = []
        self.steps = 0
        return np.copy(self.current_tensor)

    @property
    def l1_norm(self) -> int:
        """Sum of absolute values of the remaining tensor elements."""
        return int(np.sum(np.abs(self.current_tensor)))

    @property
    def is_solved(self) -> bool:
        """Tensor has reached exact zero (all entries zero)."""
        return np.all(self.current_tensor == 0)

    @property
    def is_terminal(self) -> bool:
        return self.is_solved or (self.steps >= self.max_steps)

    def step(self, action: TensorAction) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute a rank-1 reduction step:
            S_{t+1} = S_t - (u (x) v (x) w)
        """
        if self.is_terminal:
            raise RuntimeError("Cannot step in a terminal state. Call reset().")

        rank1 = action.outer_product()
        prev_norm = self.l1_norm

        # Update tensor state
        self.current_tensor = self.current_tensor - rank1
        self.history.append(action)
        self.steps += 1

        new_norm = self.l1_norm
        norm_reduction = prev_norm - new_norm

        # Calculate reward
        if self.is_solved:
            # Solved bonus: higher reward for using fewer steps
            efficiency_bonus = (self.max_steps - self.steps) * 5.0
            reward = 20.0 + efficiency_bonus
        else:
            # Strong positive reward for decreasing norm, heavy penalty for increasing norm
            if norm_reduction > 0:
                reward = 1.0 * norm_reduction
            else:
                reward = -2.0 + 1.0 * norm_reduction

        info = {
            "steps": self.steps,
            "norm": new_norm,
            "solved": self.is_solved,
            "norm_reduction": norm_reduction,
        }

        return np.copy(self.current_tensor), reward, self.is_terminal, info

    def clone(self) -> TensorGameEnv:
        """Creates an independent copy of the current environment state."""
        env = TensorGameEnv(
            target_tensor=self.target_tensor,
            max_steps=self.max_steps,
            allowed_coeffs=self.allowed_coeffs,
        )
        env.current_tensor = np.copy(self.current_tensor)
        env.history = list(self.history)
        env.steps = self.steps
        return env
