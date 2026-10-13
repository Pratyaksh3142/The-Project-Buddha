"""
Stochastic Candidate Generator and Mutation Space.

Zero hardcoded targets. Candidate tensors are initialized from pure stochastic noise
or null states, and mutated through discrete transitions.
"""

from typing import List, Tuple, Optional
import numpy as np


class CandidateTensorSpace:
    """
    Defines the discrete state space of candidate rank-3 tensors C in Z^{N x N x N}.
    Supports both unconstrained and skew-symmetric subspace parameterizations.
    """

    def __init__(self, dimension: int, alphabet: Tuple[int, ...] = (-1, 0, 1), skew_symmetric: bool = True):
        self.N = dimension
        self.alphabet = alphabet
        self.skew_symmetric = skew_symmetric

        if self.skew_symmetric:
            # Independent action coordinates are pairs (i < j) and output index k
            self.action_coords = []
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    for k in range(self.N):
                        self.action_coords.append((i, j, k))
        else:
            self.action_coords = []
            for i in range(self.N):
                for j in range(self.N):
                    for k in range(self.N):
                        self.action_coords.append((i, j, k))

        self.num_coords = len(self.action_coords)
        # For each coordinate, possible delta adjustments (+1 or -1)
        self.deltas = [-1, 1]
        self.action_size = self.num_coords * len(self.deltas)

    def action_to_tuple(self, action_idx: int) -> Tuple[Tuple[int, int, int], int]:
        coord_idx = action_idx // len(self.deltas)
        delta_idx = action_idx % len(self.deltas)
        return self.action_coords[coord_idx], self.deltas[delta_idx]

    def tuple_to_action(self, coord: Tuple[int, int, int], delta: int) -> int:
        coord_idx = self.action_coords.index(coord)
        delta_idx = self.deltas.index(delta)
        return coord_idx * len(self.deltas) + delta_idx

    def create_null_tensor(self) -> np.ndarray:
        return np.zeros((self.N, self.N, self.N), dtype=np.int32)

    def create_random_tensor(self, rng: Optional[np.random.Generator] = None, sparsity: float = 0.8) -> np.ndarray:
        """
        Creates a candidate tensor from pure stochastic noise.
        Zero predefined structure.
        """
        if rng is None:
            rng = np.random.default_rng()

        C = np.zeros((self.N, self.N, self.N), dtype=np.int32)
        for (i, j, k) in self.action_coords:
            if rng.random() > sparsity:
                val = rng.choice([-1, 1])
                C[i, j, k] = val
                if self.skew_symmetric:
                    C[j, i, k] = -val

        return C

    def get_valid_actions(self, C: np.ndarray, last_action: Optional[int] = None) -> List[int]:
        """
        Returns list of valid mutation actions that do not exceed alphabet bounds
        and avoids immediate inverse oscillations.
        """
        valid = []
        min_b = min(self.alphabet)
        max_b = max(self.alphabet)

        forbidden_action = None
        if last_action is not None:
            (last_c, last_d) = self.action_to_tuple(last_action)
            forbidden_action = self.tuple_to_action(last_c, -last_d)

        for action_idx in range(self.action_size):
            if action_idx == forbidden_action:
                continue
            (i, j, k), delta = self.action_to_tuple(action_idx)
            cur_val = int(C[i, j, k])
            new_val = cur_val + delta
            if min_b <= new_val <= max_b:
                valid.append(action_idx)
        return valid

    def apply_action(self, C: np.ndarray, action_idx: int) -> np.ndarray:
        """
        Applies a discrete step mutation to the candidate tensor.
        Clamps values within alphabet bounds.
        """
        (i, j, k), delta = self.action_to_tuple(action_idx)
        new_C = np.copy(C)
        new_val = int(new_C[i, j, k] + delta)

        # Clamp to bounds [-1, 1]
        new_val = max(min(self.alphabet), min(max(self.alphabet), new_val))
        new_C[i, j, k] = new_val

        if self.skew_symmetric:
            new_C[j, i, k] = -new_val

        return new_C
