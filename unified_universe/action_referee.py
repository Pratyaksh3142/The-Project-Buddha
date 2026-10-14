"""
Universal Action Referee (Pure Variational Formulation).

Evaluates fundamental physical and mathematical actions from first principles:
1. Exact Many-Body Quantum Ground State (Bogoliubov-de Gennes / Rayleigh-Ritz variational energy)
2. Invariant Graph Laplacian Dirichlet Energy (Harmonic coordinate embedding & minimal surface)
3. Algebraic Associativity / Gauge Curvature (Jacobi Bianchi identity closure)

CRITICAL DESIGN MANDATE:
- ZERO hardcoded empirical formulas (No BCS gap approximations, no 1D chain models).
- ZERO reward bonuses (+5.0, +3.0) or magic classification thresholds.
- The action functional S is purely the sum of fundamental variational energies.
"""

from typing import Dict, Any, Tuple
import numpy as np


class UniversalActionReferee:
    """
    Evaluates:
    1. Algebraic Jacobi Anomaly (Gauge & Lie algebra consistency): J = 0
    2. Exact Quantum Many-Body Ground State Energy: E_0 = min <Psi|H|Psi>
    3. Geometric Dirichlet / Elastic Energy on Graph Laplacian: Tr(X^T L X) + E_rep
    4. Graph Spectral Topology: Fiedler connectivity & spectral radius
    """

    def __init__(self, N: int):
        self.N = N

    def evaluate_algebraic_symmetry(self, substrate) -> Dict[str, float]:
        """
        Evaluates closure of relational tensor Z as structure constants C[i, j, k].
        Measures the Jacobi anomaly tensor (the Bianchi identity of Lie algebras):
        J^l_{ijk} = C^m_{ij} C^l_{mk} + C^m_{jk} C^l_{mi} + C^m_{ki} C^l_{mj}
        """
        if substrate.d >= substrate.N:
            C = substrate.Z[:, :, :substrate.N]
            # Symmetrized Jacobi tensor via tensor contraction
            term1 = np.einsum('ijm, mkl -> ijkl', C, C)
            term2 = np.einsum('jkm, mil -> ijkl', C, C)
            term3 = np.einsum('kim, mjl -> ijkl', C, C)
            J = term1 + term2 + term3
            jacobi_violation = float(np.sum(np.abs(J)))

            # Cartan-Killing metric: kappa_ij = Tr(ad_i ad_j) = C^a_{ib} C^b_{ja}
            kappa = np.einsum('iab, jba -> ij', C, C)
            det_kappa = float(np.real(np.linalg.det(kappa)))
            return {
                "jacobi_violation": jacobi_violation,
                "killing_det": det_kappa
            }
        else:
            return {
                "jacobi_violation": 0.0,
                "killing_det": 0.0
            }

    def evaluate_quantum_matter(self, substrate) -> Dict[str, float]:
        """
        Exact Many-Body Quantum Mechanics on the relational graph.
        Evaluates the Bogoliubov-de Gennes (BdG) Hamiltonian on 2N Nambu fermion modes:
            H_BdG = [  T    Delta ]
                    [ Delta^dagger  -T^T  ]
        where:
            T_ij = (Z_ij,0 + Z_ji,0) / 2      (Hermitian single-particle kinetic/hopping)
            Delta_ij = (Z_ij,1 - Z_ji,1) / 2  (Antisymmetric two-particle pairing matrix)
        
        Zero hardcoding. Zero BCS gap approximations.
        Ground-state energy is the exact Rayleigh-Ritz variational minimum:
            E_0 = -1/2 * sum_{E_k > 0} E_k
        """
        # Single-particle kinetic/hopping operator (Hermitian)
        T = 0.5 * (substrate.Z[:, :, 0] + substrate.Z[:, :, 0].T)

        # Two-particle pairing operator (Antisymmetric by Pauli exclusion)
        if substrate.d > 1:
            Delta = 0.5 * (substrate.Z[:, :, 1] - substrate.Z[:, :, 1].T)
        else:
            Delta = np.zeros((self.N, self.N), dtype=np.float32)

        # Assemble full 2N x 2N BdG Hamiltonian
        H_bdg = np.block([
            [T, Delta],
            [Delta.T, -T]
        ])

        # Exact diagonalization via Hermitian eigensolver
        eigenvals = np.linalg.eigvalsh(H_bdg)

        # Positive quasiparticle excitation energies
        pos_eigenvals = eigenvals[eigenvals > 1e-7]

        # Exact variational ground state energy
        if len(pos_eigenvals) > 0:
            ground_energy = -0.5 * float(np.sum(pos_eigenvals))
            spectral_gap = float(np.min(pos_eigenvals))
        else:
            ground_energy = float(np.sum(eigenvals[eigenvals < 0])) if np.any(eigenvals < 0) else 0.0
            spectral_gap = 0.0

        pairing_norm = float(np.linalg.norm(Delta) / np.sqrt(self.N))

        return {
            "ground_energy": ground_energy,
            "spectral_gap": spectral_gap,
            "pairing_norm": pairing_norm
        }

    def evaluate_spatial_conformation(self, substrate) -> Dict[str, Any]:
        """
        Evaluates Geometric Dirichlet Action via the invariant Graph Laplacian.
        Energy = Tr(X^T L X) + Volume Repulsion
        
        Zero hardcoded 1D chains. Zero hardcoded bond lengths.
        Natural harmonic relaxation of spatial graphs (minimal surfaces / crystal clusters).
        """
        if not substrate.is_spatial:
            return {"spatial_energy": 0.0, "dirichlet_energy": 0.0, "radius_of_gyration": 0.0}

        # Pairwise coupling affinity A_ij >= 0
        A = np.linalg.norm(substrate.Z, axis=-1)
        np.fill_diagonal(A, 0.0)

        # Graph Laplacian: L = D - A
        degrees = np.sum(A, axis=-1)
        L = np.diag(degrees) - A

        # Harmonic Dirichlet elastic energy: 1/2 sum A_ij ||X_i - X_j||^2 = Tr(X^T L X)
        E_dirichlet = float(np.trace(substrate.X.T @ L @ substrate.X))

        # Core volume exclusion (preventing collapse to a single point)
        D = substrate.compute_distance_matrix() + 1e-4
        mask_off_diag = 1.0 - np.eye(self.N)
        E_repulse = float(np.sum(mask_off_diag / (D**2 + 0.1)) / self.N)

        total_spatial_energy = float(0.1 * E_dirichlet + E_repulse)

        # Invariant spatial radius of gyration
        center = np.mean(substrate.X, axis=0)
        rg = float(np.sqrt(np.mean(np.sum((substrate.X - center)**2, axis=-1))))

        return {
            "spatial_energy": total_spatial_energy,
            "dirichlet_energy": E_dirichlet,
            "radius_of_gyration": rg
        }

    def evaluate_graph_topology(self, substrate) -> Dict[str, Any]:
        """
        Evaluates graph algebraic connectivity (Fiedler eigenvalue lambda_2 of Laplacian)
        and Perron-Frobenius spectral radius without arbitrary loop thresholds.
        """
        A = np.abs(substrate.Z[:, :, 0])
        np.fill_diagonal(A, 0.0)
        deg = np.sum(A, axis=-1)
        L = np.diag(deg) - A

        lap_eigenvals = np.sort(np.linalg.eigvalsh(L))
        fiedler_value = float(lap_eigenvals[1]) if len(lap_eigenvals) > 1 else 0.0

        adj_eigenvals = np.linalg.eigvals(A)
        spectral_radius = float(np.max(np.abs(adj_eigenvals)))

        return {
            "algebraic_connectivity": fiedler_value,
            "spectral_radius": spectral_radius
        }

    def compute_universal_action(self, substrate) -> Dict[str, Any]:
        """
        Universal Action Functional:
            S = E_quantum + 0.1 * E_spatial + 2.0 * Jacobi_violation
        
        Strictly Variational.
        ZERO human reward bonuses. ZERO thresholds.
        """
        alg_res = self.evaluate_algebraic_symmetry(substrate)
        qm_res = self.evaluate_quantum_matter(substrate)
        sp_res = self.evaluate_spatial_conformation(substrate)
        top_res = self.evaluate_graph_topology(substrate)

        # Total action functional
        total_action = (
            qm_res["ground_energy"] +
            0.1 * sp_res["spatial_energy"] +
            2.0 * alg_res["jacobi_violation"]
        )

        return {
            "total_action": float(total_action),
            "quantum": qm_res,
            "spatial": sp_res,
            "algebra": alg_res,
            "topology": top_res
        }
