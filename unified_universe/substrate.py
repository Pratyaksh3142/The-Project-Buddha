"""
Universal Relational Substrate.

Unifies all physical, chemical, biological, and mathematical systems into a single
relational graph: G = (Nodes S, Pair Tensor Z, Coordinates X).

Zero hardcoding. Zero domain-specific silos.

The substrate is the universe's state. The AI searches over mutations to Z and X
to find ground states of the universal action functional.
"""

from typing import Optional, Tuple
import hashlib
import numpy as np


class UniversalSubstrate:
    """
    Unified state representation for any relational system:
      S ∈ ℝ^{N×d}   — Node feature embeddings (auto-computed from Z)
      Z ∈ ℝ^{N×N×K}  — Pairwise relational coupling tensor (all channels)
      X ∈ ℝ^{N×3}    — 3D Euclidean spatial coordinates (when spatial=True)
    """

    def __init__(
        self,
        num_entities: int,
        num_channels: int = 8,
        is_spatial: bool = False,
        rng: Optional[np.random.Generator] = None
    ):
        self.N = num_entities
        self.K = num_channels
        self.is_spatial = is_spatial
        self.rng = rng if rng is not None else np.random.default_rng()

        # Pair relational tensor Z: shape (N, N, K)
        self.Z = np.zeros((self.N, self.N, self.K), dtype=np.float32)

        # 3D spatial coordinates X: shape (N, 3)
        if self.is_spatial:
            self.X = self.rng.normal(0.0, 1.0, size=(self.N, 3)).astype(np.float32)
        else:
            self.X = np.zeros((self.N, 3), dtype=np.float32)

    @property
    def S(self) -> np.ndarray:
        """
        Node features auto-derived from the relational tensor Z.
        S_i = [sum_j Z[i,j,:], sum_j Z[j,i,:]] flattened.
        This ensures nodes carry structural information without hardcoding.
        Shape: (N, 2*K)
        """
        outgoing = np.sum(self.Z, axis=1)   # (N, K) — sum over j for Z[i,j,:]
        incoming = np.sum(self.Z, axis=0)   # (N, K) — sum over i for Z[i,j,:]
        return np.concatenate([outgoing, incoming], axis=-1)  # (N, 2*K)

    @property
    def node_dim(self) -> int:
        return 2 * self.K

    def clone(self) -> 'UniversalSubstrate':
        new_sub = UniversalSubstrate(self.N, self.K, self.is_spatial, self.rng)
        new_sub.Z = np.copy(self.Z)
        new_sub.X = np.copy(self.X)
        return new_sub

    def mutate_pair_tensor(self, i: int, j: int, channel: int, delta: float,
                           bounds: Tuple[float, float] = (-2.0, 2.0)):
        """
        Discrete mutation of Z[i, j, channel].
        """
        val = np.clip(self.Z[i, j, channel] + delta, bounds[0], bounds[1])
        self.Z[i, j, channel] = val

    def compute_distance_matrix(self) -> np.ndarray:
        """SE(3)-invariant pairwise Euclidean distance matrix."""
        diff = self.X[:, None, :] - self.X[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def fingerprint(self) -> str:
        """
        Deterministic cryptographic SHA-256 fingerprint for deduplication
        across process lifecycles.
        """
        z_bytes = np.round(self.Z, 3).tobytes()
        x_bytes = np.round(self.X, 3).tobytes() if self.is_spatial else b""
        digest = hashlib.sha256(z_bytes + x_bytes).hexdigest()[:16]
        return f"N{self.N}_K{self.K}_{digest}"
