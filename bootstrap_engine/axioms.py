"""
Universal Mathematical Axiom Referee.

Zero physics assumptions. Zero human labels. Zero coordinates. Zero predefined target tensors.
Evaluates pure algebraic consistency axioms on rank-3 structure tensors C[i, j, k]
representing bilinear compositions: e_i * e_j = sum_k C^k_{ij} e_k.
"""

from typing import Dict, Any, Tuple
import numpy as np


def compute_antisymmetry_anomaly(C: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes skew-symmetry violation tensor: S^k_{ij} = C^k_{ij} + C^k_{ji}.
    For a Lie algebra bracket [e_i, e_j] = -[e_j, e_i], S must be exactly zero.
    """
    # C shape is (N, N, N) where C[i, j, k] = C^k_{ij}
    S = C + np.swapaxes(C, 0, 1)
    violation = float(np.sum(np.abs(S)))
    return S, violation


def compute_jacobi_anomaly(C: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes Jacobi identity violation tensor for bilinear bracket:
    J^l_{ijk} = sum_m ( C^m_{ij} C^l_{mk} + C^m_{jk} C^l_{mi} + C^m_{ki} C^l_{mj} )

    This is the fundamental closure axiom of Lie algebras, continuous symmetries,
    and Hamiltonian phase flows. J must be identically zero for consistency.
    """
    N = C.shape[0]
    # Term 1: [[e_i, e_j], e_k] = sum_m C^m_{ij} C^l_{mk} -> einsum('ijm, mkl -> ijkl', C, C)
    term1 = np.einsum('ijm, mkl -> ijkl', C, C)
    # Term 2: [[e_j, e_k], e_i] = sum_m C^m_{jk} C^l_{mi} -> einsum('jkm, mil -> ijkl', C, C)
    term2 = np.einsum('jkm, mil -> ijkl', C, C)
    # Term 3: [[e_k, e_i], e_j] = sum_m C^m_{ki} C^l_{mj} -> einsum('kim, mjl -> ijkl', C, C)
    term3 = np.einsum('kim, mjl -> ijkl', C, C)

    J = term1 + term2 + term3
    violation = float(np.sum(np.abs(J)))
    return J, violation


def compute_associativity_anomaly(C: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes associator violation tensor for associative product:
    A^l_{ijk} = sum_m ( C^m_{ij} C^l_{mk} - C^m_{jk} C^l_{im} )
    representing: (e_i * e_j) * e_k - e_i * (e_j * e_k) = 0.
    """
    # (e_i * e_j) * e_k -> sum_m C^m_{ij} C^l_{mk}
    term1 = np.einsum('ijm, mkl -> ijkl', C, C)
    # e_i * (e_j * e_k) -> sum_m C^m_{jk} C^l_{im}
    term2 = np.einsum('jkm, iml -> ijkl', C, C)

    A = term1 - term2
    violation = float(np.sum(np.abs(A)))
    return A, violation


def compute_killing_metric(C: np.ndarray) -> np.ndarray:
    """
    Computes the Cartan-Killing metric tensor of an algebra:
    kappa_{ij} = sum_{a, b} C^a_{ib} C^b_{ja} = Tr(ad_{e_i} ad_{e_j})

    Non-degenerate det(kappa) != 0 signifies semi-simple symmetry (Cartan criterion).
    """
    kappa = np.einsum('iab, jba -> ij', C, C)
    return kappa


def evaluate_algebraic_axioms(C: np.ndarray, axiom_system: str = "lie") -> Dict[str, Any]:
    """
    Evaluates candidate tensor against chosen mathematical axiom system.
    Returns exact violations and boolean consistency flag.
    """
    l1_norm = float(np.sum(np.abs(C)))
    is_non_trivial = l1_norm > 0

    if axiom_system == "lie":
        _, s_viol = compute_antisymmetry_anomaly(C)
        _, j_viol = compute_jacobi_anomaly(C)
        total_violation = s_viol + j_viol
        is_consistent = (total_violation < 1e-9) and is_non_trivial
        
        return {
            "axiom_system": "lie",
            "antisymmetry_violation": s_viol,
            "jacobi_violation": j_viol,
            "total_violation": total_violation,
            "l1_norm": l1_norm,
            "is_non_trivial": is_non_trivial,
            "is_consistent": is_consistent
        }

    elif axiom_system == "associative":
        _, a_viol = compute_associativity_anomaly(C)
        total_violation = a_viol
        is_consistent = (total_violation < 1e-9) and is_non_trivial

        return {
            "axiom_system": "associative",
            "associativity_violation": a_viol,
            "total_violation": total_violation,
            "l1_norm": l1_norm,
            "is_non_trivial": is_non_trivial,
            "is_consistent": is_consistent
        }

    else:
        raise ValueError(f"Unknown axiom system: {axiom_system}")
