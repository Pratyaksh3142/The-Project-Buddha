"""
Project Buddha v2.0: Composable Tensor-Contraction Action Algebra.

Provides primitive gauge-covariant and coordinate-free mathematical building blocks:
  1. Node fields (0-forms) \\psi_i
  2. Edge connections (1-forms / gauge links) U_{ij} = Z_{ij}
  3. Plaquette curvatures (2-forms) F_{ijk} = U_{ij} U_{jk} - U_{ik}
  4. Exterior derivative d, Hodge dual *, and Traces Tr(\\cdot)
  5. Many-body fermion ground states (BdG Dirac/Nambu spectrum)
  6. Algebraic Jacobi-Bianchi closure J_{ijkl} = C_{ij}^m C_{mk}^l + cyclic

Candidate Action functionals S[Z, X] are composed as linear combinations of
typed tensor contraction primitives, allowing MCTS to DISCOVER new Lagrangians
rather than evaluating a hardcoded equation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch


class TensorContractionTerm(ABC):
    """Abstract base class for a primitive gauge/differentiable tensor contraction term."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def compute_torch(
        self,
        Z: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor,
        N: int,
        K: int,
        device: torch.device
    ) -> torch.Tensor:
        """Computes differentiable scalar energy term using PyTorch autograd."""
        pass

    @abstractmethod
    def symbolic_repr(self) -> str:
        """Returns mathematical LaTeX / ASCII representation."""
        pass


class DirichletHarmonicTerm(TensorContractionTerm):
    """
    Harmonic Dirichlet map energy: S = Tr(X^T L_A X)
    Measures spatial smoothness of the relational graph embedding.
    """
    def __init__(self, weight: float = 0.01):
        super().__init__("dirichlet_harmonic", weight)

    def compute_torch(self, Z: torch.Tensor, X: torch.Tensor, mask: torch.Tensor, N: int, K: int, device: torch.device) -> torch.Tensor:
        Z_masked = Z * mask
        # Safe norm avoiding sqrt(0) gradient singularities
        A = torch.sqrt(torch.sum(Z_masked**2, dim=-1) + 1e-8) * (1.0 - torch.eye(N, device=device))
        degrees = torch.sum(A, dim=-1)
        L = torch.diag(degrees) - A
        return torch.trace(X.T @ L @ X)

    def symbolic_repr(self) -> str:
        return r"\mathrm{Tr}(X^T L_A X)"


class VolumeExclusionTerm(TensorContractionTerm):
    """
    Core volume exclusion (Fermi-Dirac steric repulsion):
    S = (1/N) \\sum_{i < j} 1 / (||X_i - X_j||^2 + \\epsilon)
    Prevents spatial coordinate collapse to a single point.
    """
    def __init__(self, weight: float = 1.0, epsilon: float = 0.1):
        super().__init__("volume_exclusion", weight)
        self.epsilon = epsilon

    def compute_torch(self, Z: torch.Tensor, X: torch.Tensor, mask: torch.Tensor, N: int, K: int, device: torch.device) -> torch.Tensor:
        diff = X.unsqueeze(1) - X.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=-1) + self.epsilon + torch.eye(N, device=device)
        mask_off_diag = 1.0 - torch.eye(N, device=device)
        return 0.5 * torch.sum(mask_off_diag / dist_sq) / max(N, 1)

    def symbolic_repr(self) -> str:
        return r"\frac{1}{N} \sum_{i < j} \frac{1}{\|X_i - X_j\|^2 + \epsilon}"


class DiracFermionBdGTerm(TensorContractionTerm):
    """
    Exact Many-Body Nambu-Dirac Fermion Ground Energy:
    S = -0.5 \\sum_{E_k > 0} E_k(H_{BdG})
    Calculated via exact Hellmann-Feynman autograd through symmetric eigensolvers.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__("dirac_fermion_bdg", weight)

    def compute_torch(self, Z: torch.Tensor, X: torch.Tensor, mask: torch.Tensor, N: int, K: int, device: torch.device) -> torch.Tensor:
        # Channel 0: Kinetic hopping & on-site energy
        T = 0.5 * (Z[:, :, 0] + Z[:, :, 0].T) * mask[:, :, 0]
        # Channel 1: Antisymmetric pairing
        if K > 1:
            Delta = 0.5 * (Z[:, :, 1] - Z[:, :, 1].T) * mask[:, :, 1]
        else:
            Delta = torch.zeros((N, N), dtype=Z.dtype, device=device)

        H_top = torch.cat([T, Delta], dim=1)
        H_bot = torch.cat([Delta.T, -T], dim=1)
        H_bdg = torch.cat([H_top, H_bot], dim=0)

        eigs = torch.linalg.eigvalsh(H_bdg)
        pos_eigs = torch.relu(eigs)
        return -0.5 * torch.sum(pos_eigs)

    def symbolic_repr(self) -> str:
        return r"-\frac{1}{2} \sum_{E_k > 0} E_k(H_{\mathrm{BdG}})"


class YangMillsCurvatureTerm(TensorContractionTerm):
    """
    Yang-Mills Plaquette Curvature 2-Form Action:
    S = \\sum_{ijk} Tr(F_{ijk}^\\dagger F_{ijk})
    where F_{ijk} = U_{ij} U_{jk} - U_{ik} is the exterior covariant curvature.
    """
    def __init__(self, weight: float = 0.5):
        super().__init__("yang_mills_curvature", weight)

    def compute_torch(self, Z: torch.Tensor, X: torch.Tensor, mask: torch.Tensor, N: int, K: int, device: torch.device) -> torch.Tensor:
        gauge_offset = 2 if K >= 4 else 0
        M = min(N, K - gauge_offset)
        if M < 2:
            return torch.tensor(0.0, device=device)

        # Gauge field connection 1-forms
        U = Z[:M, :M, gauge_offset:gauge_offset + M] * mask[:M, :M, gauge_offset:gauge_offset + M]

        # Plaquette curvature F_{ijk} = sum_m U_{im} U_{mj} - U_{ij}
        # In multi-channel representation: contraction along matrix indices
        F = torch.einsum("imk,mjk->ijk", U, U) - U
        return torch.sum(F**2) / (M * M * M)

    def symbolic_repr(self) -> str:
        return r"\sum_{\triangle ijk} \mathrm{Tr}(F_{ijk}^\dagger F_{ijk})"


class JacobiBianchiAnomalyTerm(TensorContractionTerm):
    """
    Algebraic Jacobi-Bianchi Anomaly Closure:
    S = ||C + C^T||^2 + \\sum_{ijkl} (C_{ij}^m C_{mk}^l + cyclic)^2
    Measures breaking of Lie algebra gauge symmetry.
    """
    def __init__(self, weight: float = 2.0):
        super().__init__("jacobi_bianchi_anomaly", weight)

    def compute_torch(self, Z: torch.Tensor, X: torch.Tensor, mask: torch.Tensor, N: int, K: int, device: torch.device) -> torch.Tensor:
        gauge_offset = 2 if K >= 4 else 0
        M = min(N, K - gauge_offset)
        if M < 2:
            return torch.tensor(0.0, device=device)

        C = Z[:M, :M, gauge_offset:gauge_offset + M] * mask[:M, :M, gauge_offset:gauge_offset + M]
        S_anti = C + C.transpose(0, 1)
        antisym_loss = torch.sum(S_anti**2)

        C_skew = 0.5 * (C - C.transpose(0, 1))
        t1 = torch.einsum("ijm,mkl->ijkl", C_skew, C_skew)
        t2 = torch.einsum("jkm,mil->ijkl", C_skew, C_skew)
        t3 = torch.einsum("kim,mjl->ijkl", C_skew, C_skew)
        J = t1 + t2 + t3
        jacobi_loss = torch.sum(J**2)

        return antisym_loss + jacobi_loss

    def symbolic_repr(self) -> str:
        return r"\|C + C^T\|^2 + \sum_{ijkl} \|C_{ij}^m C_{mk}^l + \mathrm{cyclic}\|^2"


class CompactnessWellTerm(TensorContractionTerm):
    """
    Field Mass / Potential Well Anchor:
    S = 0.05 * \\sum ||Z_{ij} \\odot M_{ij}||^2
    Ensures vacuum compactness and prevents runaway infinite field amplitudes.
    """
    def __init__(self, weight: float = 0.05):
        super().__init__("compactness_well", weight)

    def compute_torch(self, Z: torch.Tensor, X: torch.Tensor, mask: torch.Tensor, N: int, K: int, device: torch.device) -> torch.Tensor:
        Z_masked = Z * mask
        return torch.sum(Z_masked**2) / (N * N)

    def symbolic_repr(self) -> str:
        return r"\frac{1}{N^2} \sum_{ij} \|Z_{ij} \odot M_{ij}\|^2"


class TopologicalChernSimonsTerm(TensorContractionTerm):
    """
    Topological Plaquette Winding / Chern-Simons 3-Form:
    S = \\sum_{ijk} \\epsilon^{abc} U_{ij,a} U_{jk,b} U_{ki,c}
    Measures topological knotting / chiral anomaly in gauge space.
    """
    def __init__(self, weight: float = 0.1):
        super().__init__("topological_chern_simons", weight)

    def compute_torch(self, Z: torch.Tensor, X: torch.Tensor, mask: torch.Tensor, N: int, K: int, device: torch.device) -> torch.Tensor:
        gauge_offset = 2 if K >= 4 else 0
        if K - gauge_offset < 3:
            return torch.tensor(0.0, device=device)

        U = Z[:, :, gauge_offset:gauge_offset + 3] * mask[:, :, gauge_offset:gauge_offset + 3]
        # eps_012 = +1, cyclic
        cs = (
            torch.einsum("ija,jkb,kia->", U[:, :, 0], U[:, :, 1], U[:, :, 2])
            - torch.einsum("ija,jkb,kia->", U[:, :, 0], U[:, :, 2], U[:, :, 1])
        )
        return cs / max(N * N * N, 1)

    def symbolic_repr(self) -> str:
        return r"\sum_{ijk} \epsilon^{abc} U_{ij}^a U_{jk}^b U_{ki}^c"


# ─────────────────────────────────────────────────────────────
# COMPOSABLE ACTION FUNCTIONAL
# ─────────────────────────────────────────────────────────────

AVAILABLE_TERMS = {
    "dirichlet_harmonic": DirichletHarmonicTerm,
    "volume_exclusion": VolumeExclusionTerm,
    "dirac_fermion_bdg": DiracFermionBdGTerm,
    "yang_mills_curvature": YangMillsCurvatureTerm,
    "jacobi_bianchi_anomaly": JacobiBianchiAnomalyTerm,
    "compactness_well": CompactnessWellTerm,
    "topological_chern_simons": TopologicalChernSimonsTerm,
}


class ComposableAction:
    """
    A candidate mathematical Action Functional S composed of a set of
    gauge-covariant tensor contraction terms with variational coupling constants.
    """

    def __init__(self, terms: Optional[List[TensorContractionTerm]] = None):
        self.terms: List[TensorContractionTerm] = terms or []

    @classmethod
    def standard_universe(
        cls,
        alpha_spatial: float = 0.1,
        beta_jacobi: float = 2.0
    ) -> "ComposableAction":
        """Instantiates standard physical action functional combining geometry, matter, and gauge."""
        return cls([
            DiracFermionBdGTerm(weight=1.0),
            DirichletHarmonicTerm(weight=alpha_spatial),
            VolumeExclusionTerm(weight=alpha_spatial),
            JacobiBianchiAnomalyTerm(weight=beta_jacobi),
            CompactnessWellTerm(weight=0.05),
        ])

    @classmethod
    def pure_gauge_yang_mills(cls, beta: float = 1.0) -> "ComposableAction":
        """Pure gauge action discovering non-abelian Yang-Mills and Lie structure."""
        return cls([
            YangMillsCurvatureTerm(weight=beta),
            JacobiBianchiAnomalyTerm(weight=2.0),
            CompactnessWellTerm(weight=0.05),
        ])

    @classmethod
    def topological_matter(cls) -> "ComposableAction":
        """Action discovering topological superconducting phases."""
        return cls([
            DiracFermionBdGTerm(weight=1.0),
            TopologicalChernSimonsTerm(weight=0.5),
            CompactnessWellTerm(weight=0.05),
        ])

    def add_term(self, term: TensorContractionTerm):
        self.terms.append(term)

    def compute_action(
        self,
        Z: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor,
        N: int,
        K: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the composite differentiable action S = sum w_k T_k.
        Returns (total_action_tensor, dict_of_term_values).
        """
        total_action = torch.tensor(0.0, dtype=Z.dtype, device=device)
        term_breakdown = {}

        for term in self.terms:
            val = term.compute_torch(Z, X, mask, N, K, device)
            weighted_val = term.weight * val
            total_action = total_action + weighted_val
            term_breakdown[term.name] = float(val.detach().cpu().item())

        return total_action, term_breakdown

    def to_latex(self) -> str:
        """Returns mathematical LaTeX representation of the Action Functional."""
        components = []
        for term in self.terms:
            w_str = f"{term.weight:g} " if abs(term.weight - 1.0) > 1e-4 else ""
            components.append(f"{w_str}{term.symbolic_repr()}")
        return r"\mathcal{S} = \int \left[ " + " + ".join(components) + r" \right]"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes action configuration for JSON export."""
        return {
            "terms": [
                {
                    "name": term.name,
                    "weight": float(term.weight),
                    "symbolic": term.symbolic_repr()
                }
                for term in self.terms
            ],
            "latex": self.to_latex()
        }

    def __repr__(self) -> str:
        terms_str = " + ".join(f"{t.weight:g}*{t.name}" for t in self.terms)
        return f"ComposableAction(S = {terms_str})"
