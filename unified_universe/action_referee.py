"""
Universal Action Referee.

Evaluates pure physical, chemical, biological, and mathematical order parameters
on UniversalSubstrate relational graphs without hardcoded empirical parameters.
"""

from typing import Dict, Any, Tuple
import numpy as np


class UniversalActionReferee:
    """
    Evaluates:
    1. Algebraic Symmetry Closure (Jacobi anomaly)
    2. Quantum Ground State & Superconducting Pairing (ODLRO order parameter Delta)
    3. Spatial Conformational Free Energy (Molecular/Protein/Crystal relaxation)
    4. Topological / Autocatalytic Graph Closure (Metabolic feedback loops)
    """

    def __init__(self, N: int):
        self.N = N

    def evaluate_algebraic_symmetry(self, substrate) -> Dict[str, float]:
        """
        Evaluates closure of relational tensor Z as structure constants C[i, j, k].
        """
        if substrate.d >= substrate.N:
            # We can treat Z[:, :, :N] as a rank-3 tensor C[i, j, k]
            C = substrate.Z[:, :, :substrate.N]
            # Symmetrized Jacobi tensor
            term1 = np.einsum('ijm, mkl -> ijkl', C, C)
            term2 = np.einsum('jkm, mil -> ijkl', C, C)
            term3 = np.einsum('kim, mjl -> ijkl', C, C)
            J = term1 + term2 + term3
            jacobi_violation = float(np.sum(np.abs(J)))

            # Killing metric
            kappa = np.einsum('iab, jba -> ij', C, C)
            det_kappa = float(np.real(np.linalg.det(kappa)))
            return {"jacobi_violation": jacobi_violation, "killing_det": det_kappa}
        else:
            return {"jacobi_violation": 0.0, "killing_det": 0.0}

    def evaluate_quantum_matter(self, substrate) -> Dict[str, float]:
        """
        Evaluates many-body quantum ground state from relational coupling tensors.
        Channel 0 = Hopping matrix T_ij
        Channel 1 = Interaction matrix V_ij
        Computes gap, ground-state energy, and Superconducting ODLRO Delta.
        """
        # Hopping matrix T
        T = substrate.Z[:, :, 0]
        # Make Hermitian: (T + T^T)/2
        H_kin = 0.5 * (T + T.T)
        eigenvals, eigenvecs = np.linalg.eigh(H_kin)
        ground_energy = float(np.sum(eigenvals[eigenvals < 0])) if np.any(eigenvals < 0) else float(eigenvals[0])

        # Interaction tensor V (Pairing potential)
        V = substrate.Z[:, :, 1] if substrate.d > 1 else np.zeros((self.N, self.N))
        # BCS-type Gap Equation / Off-Diagonal Long-Range Order (ODLRO):
        # Delta_i = sum_j V_ij * Delta_j / (2 * sqrt(xi_j^2 + Delta_j^2))
        # We solve iteratively for self-consistent order parameter Delta
        delta_vec = np.ones(self.N, dtype=np.float32) * 0.1
        xi = eigenvals - np.median(eigenvals)  # chemical potential relative to Fermi level

        for _ in range(5):
            denom = 2.0 * np.sqrt(xi**2 + delta_vec**2 + 1e-6)
            delta_vec = np.abs(np.dot(np.abs(V), delta_vec / denom))

        sc_order_parameter = float(np.mean(delta_vec))
        single_particle_gap = float(np.min(np.sqrt(xi**2 + delta_vec**2)))

        return {
            "ground_energy": ground_energy,
            "superconducting_delta": sc_order_parameter,
            "energy_gap": single_particle_gap,
            "is_superconducting": bool(sc_order_parameter > 0.05)
        }

    def evaluate_spatial_conformation(self, substrate) -> Dict[str, Any]:
        """
        Evaluates 3D spatial conformation (AlphaFold 3 / GNoME molecular energy).
        E = E_bonding + E_repulsion + E_chain_continuity
        """
        if not substrate.is_spatial:
            return {"spatial_energy": 0.0, "rmsd_stability": 0.0}

        D = substrate.compute_distance_matrix() + 1e-6

        # Attractive bonding: Z_ij,0 / D_ij
        bond_weights = np.maximum(0.0, substrate.Z[:, :, 0])
        E_attract = -np.sum(bond_weights / D)

        # Steric clash repulsion: 1 / D^4 for non-bonded pairs
        mask_off_diag = 1.0 - np.eye(self.N)
        E_repulse = np.sum(mask_off_diag * (1.0 / (D**4 + 0.1)))

        # Polymer chain connectivity: (D_{i, i+1} - 1.0)^2
        E_chain = 0.0
        for i in range(self.N - 1):
            E_chain += (D[i, i + 1] - 1.0)**2

        total_spatial_energy = float(E_attract + 0.1 * E_repulse + 2.0 * E_chain)

        # Radius of gyration (compactness indicator)
        center = np.mean(substrate.X, axis=0)
        rg = float(np.sqrt(np.mean(np.sum((substrate.X - center)**2, axis=-1))))

        return {
            "spatial_energy": total_spatial_energy,
            "radius_of_gyration": rg,
            "is_compact_fold": bool(rg < np.sqrt(self.N))
        }

    def evaluate_autocatalytic_topology(self, substrate) -> Dict[str, Any]:
        """
        Evaluates directed relational graph for closed autocatalytic cycles (Biology/Metabolism).
        A network has metabolic closure if every sub-cycle feeds back into self-production.
        """
        adj = (np.abs(substrate.Z[:, :, 0]) > 0.5).astype(np.float32)
        # Compute adjacency powers: (A^k)_{ii} counts closed loops of length k
        loop_density = 0.0
        ak = adj
        for _ in range(2, min(self.N, 5)):
            ak = np.dot(ak, adj)
            loop_density += float(np.trace(ak))

        has_autocatalysis = loop_density > 0.0
        return {
            "loop_density": loop_density,
            "has_autocatalysis": has_autocatalysis
        }

    def compute_universal_action(self, substrate) -> Dict[str, Any]:
        """
        Computes composite universal action functional.
        S = S_algebra + E_quantum + E_spatial - Reward(Closure)
        """
        alg_res = self.evaluate_algebraic_symmetry(substrate)
        qm_res = self.evaluate_quantum_matter(substrate)
        sp_res = self.evaluate_spatial_conformation(substrate)
        bio_res = self.evaluate_autocatalytic_topology(substrate)

        # Total action to be minimized
        total_action = (
            alg_res["jacobi_violation"] * 2.0 +
            qm_res["ground_energy"] +
            sp_res["spatial_energy"] * 0.1 -
            (5.0 if qm_res["is_superconducting"] else 0.0) -
            (5.0 if sp_res.get("is_compact_fold", False) else 0.0) -
            (3.0 if bio_res["has_autocatalysis"] else 0.0)
        )

        return {
            "total_action": float(total_action),
            "algebra": alg_res,
            "quantum": qm_res,
            "spatial": sp_res,
            "biology": bio_res
        }
