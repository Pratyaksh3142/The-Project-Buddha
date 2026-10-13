"""
Tensor Discovery Engine — Target Tensor Generators

Constructs exact 3D tensors representing fundamental mathematical and physical operations:
  1. Matrix Multiplication (Bilinear forms over M_{n,m} x M_{m,p} -> M_{n,p})
  2. Lie Algebra Symmetries (so(3) Levi-Civita tensor [spin/angular momentum])
  3. Symplectic and Bilinear Invariants
"""

from __future__ import annotations
import numpy as np


def make_matrix_multiplication_tensor(n: int, m: int, p: int) -> np.ndarray:
    """
    Constructs the 3D tensor T representing matrix multiplication:
        C_{i,k} = sum_j A_{i,j} * B_{j,k}
    where A is (n x m), B is (m x p), C is (n x p).
    
    Flattening:
        Dim 1 (dim A): index i*m + j   (size n*m)
        Dim 2 (dim B): index j*p + k   (size m*p)
        Dim 3 (dim C): index i*p + k   (size n*p)
        
    T_{a, b, c} = 1 if entry (i,j) of A and entry (j,k) of B contribute to entry (i,k) of C.
    Tensor shape: (n*m, m*p, n*p).
    """
    dim_a = n * m
    dim_b = m * p
    dim_c = n * p
    
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int32)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = i * p + k
                T[idx_a, idx_b, idx_c] = 1
                
    return T


def make_lie_algebra_so3_tensor() -> np.ndarray:
    """
    Constructs the structure constant tensor f_{ijk} for the so(3) / su(2) Lie algebra:
        [J_i, J_j] = eps_{ijk} J_k
    where eps_{ijk} is the 3D completely anti-symmetric Levi-Civita tensor:
        eps_{123} = eps_{231} = eps_{312} = +1
        eps_{132} = eps_{321} = eps_{213} = -1
        all other entries = 0
    Shape: (3, 3, 3).
    This governs all quantum spin, 3D rotations, and angular momentum in physics.
    """
    eps = np.zeros((3, 3, 3), dtype=np.int32)
    eps[0, 1, 2] =  1
    eps[1, 2, 0] =  1
    eps[2, 0, 1] =  1
    eps[0, 2, 1] = -1
    eps[2, 1, 0] = -1
    eps[1, 0, 2] = -1
    return eps


def make_symplectic_invariance_tensor(dim: int = 2) -> np.ndarray:
    """
    Constructs the 3D bilinear tensor encoding the symplectic form Omega:
        Omega(u, v) = u^T J v
    representing phase-space conservation in Hamiltonian mechanics.
    Shape: (2*dim, 2*dim, 1).
    """
    n = 2 * dim
    J = np.zeros((n, n), dtype=np.int32)
    # Block matrix [[0, I], [-I, 0]]
    I = np.eye(dim, dtype=np.int32)
    J[:dim, dim:] = I
    J[dim:, :dim] = -I
    
    T = np.zeros((n, n, 1), dtype=np.int32)
    T[:, :, 0] = J
    return T
