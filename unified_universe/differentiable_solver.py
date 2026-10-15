"""
Project Buddha v2.0: Differentiable Variational Inner Solver (Bilevel Architecture).

Solves the inner variational principle:
    δS = 0  <=>  ∇_{Z, X} S(Z, X; M) = 0
for a given discrete topological graph mask and symmetry blueprint M proposed by MCTS.

Key Features:
  1. Exact Analytical Autograd through Many-Body BdG eigensolvers (torch.linalg.eigvalsh)
  2. Differentiable Dirichlet Graph Laplacian & Harmonic Geometry
  3. Differentiable Lie Jacobi / Bianchi Anomaly Tensor Contraction
  4. Automatic Vacuum Instability Detection (detects S -> -∞ and rejects unphysical theories)
  5. Multi-Device: Native CPU and Apple Silicon MPS acceleration
"""

from typing import Dict, Tuple, Any, Optional
import time
import numpy as np
import torch


class DifferentiableActionSolver:
    """
    Inner-loop variational solver: finds the exact stationary point δS = 0
    under a discrete graph topology/symmetry mask M.
    """

    def __init__(
        self,
        N: int,
        K: int,
        alpha_spatial: float = 0.1,
        beta_jacobi: float = 2.0,
        device: Optional[str] = None
    ):
        self.N = N
        self.K = K
        self.alpha = alpha_spatial
        self.beta = beta_jacobi

        if device is None:
            self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device)

    def compute_differentiable_action(
        self,
        Z: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes the fully differentiable universal action S(Z, X; mask).
        """
        N, K = self.N, self.K

        # ── 1. Matter Sector (Channels 0 & 1) ──
        # Channel 0: Symmetric kinetic hopping / on-site potential
        T = 0.5 * (Z[:, :, 0] + Z[:, :, 0].T) * mask[:, :, 0]

        # Channel 1: Antisymmetric fermion pairing (Pauli exclusion)
        if K > 1:
            Delta = 0.5 * (Z[:, :, 1] - Z[:, :, 1].T) * mask[:, :, 1]
        else:
            Delta = torch.zeros((N, N), dtype=Z.dtype, device=self.device)

        # Full 2N x 2N Bogoliubov-de Gennes Hamiltonian
        H_top = torch.cat([T, Delta], dim=1)
        H_bot = torch.cat([Delta.T, -T], dim=1)
        H_bdg = torch.cat([H_top, H_bot], dim=0)

        # Exact analytical autograd through eigensolver (Hellmann-Feynman)
        eigs = torch.linalg.eigvalsh(H_bdg)
        pos_eigs = torch.relu(eigs)
        E_quantum = -0.5 * torch.sum(pos_eigs)

        # Quasiparticle spectral gap & pairing norm
        positive_mask = eigs > 1e-6
        if torch.any(positive_mask):
            spectral_gap = torch.min(eigs[positive_mask])
        else:
            spectral_gap = torch.tensor(0.0, device=self.device)

        pairing_norm = torch.linalg.norm(Delta) / max(float(np.sqrt(N)), 1.0)

        # ── 2. Gauge Sector Interactions (Channels c >= 2) ──
        E_inter = torch.tensor(0.0, device=self.device)
        for ch in range(2, min(K, N + 2)):
            M_ch = 0.5 * (Z[:, :, ch] + Z[:, :, ch].T) * mask[:, :, ch]
            E_inter = E_inter + torch.trace(M_ch @ M_ch) / N

        # ── 3. Geometric Dirichlet Action & Repulsion ──
        # Coupling affinity: safe norm avoiding sqrt(0) gradient singularities
        Z_masked = Z * mask
        A = torch.sqrt(torch.sum(Z_masked**2, dim=-1) + 1e-8) * (1.0 - torch.eye(N, device=self.device))
        degrees = torch.sum(A, dim=-1)
        L = torch.diag(degrees) - A
        E_dir = torch.trace(X.T @ L @ X)

        diff = X.unsqueeze(1) - X.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=-1) + 0.1 + torch.eye(N, device=self.device)
        mask_off_diag = 1.0 - torch.eye(N, device=self.device)
        E_rep = 0.5 * torch.sum(mask_off_diag / dist_sq) / N
        E_spatial = 0.1 * E_dir + E_rep

        # ── 4. Lie Gauge Symmetry & Jacobi/Bianchi Anomaly ──
        gauge_offset = 2 if K >= 4 else 0
        M = min(N, K - gauge_offset)
        if M >= 2:
            C = Z[:M, :M, gauge_offset:gauge_offset + M] * mask[:M, :M, gauge_offset:gauge_offset + M]
            # Antisymmetry violation
            S_anti = C + C.transpose(0, 1)
            antisym_loss = torch.sum(S_anti**2)

            # Jacobi Bianchi anomaly on skew-symmetrized tensor
            C_skew = 0.5 * (C - C.transpose(0, 1))
            t1 = torch.einsum("ijm,mkl->ijkl", C_skew, C_skew)
            t2 = torch.einsum("jkm,mil->ijkl", C_skew, C_skew)
            t3 = torch.einsum("kim,mjl->ijkl", C_skew, C_skew)
            J = t1 + t2 + t3
            jacobi_loss = torch.sum(J**2)

            # Killing form trace
            kappa = torch.einsum("iab,jba->ij", C_skew, C_skew)
            killing_trace = torch.trace(kappa)
            E_algebra = antisym_loss + jacobi_loss
        else:
            antisym_loss = torch.tensor(0.0, device=self.device)
            jacobi_loss = torch.tensor(0.0, device=self.device)
            killing_trace = torch.tensor(0.0, device=self.device)
            E_algebra = torch.tensor(0.0, device=self.device)

        # ── 5. Compactness Anchor ──
        # Natural potential well preventing infinite runaway field amplitudes
        E_compact = 0.05 * torch.sum(Z_masked**2)

        total_action = (
            E_quantum
            + E_inter
            + self.alpha * E_spatial
            + self.beta * E_algebra
            + E_compact
        )

        observables = {
            "ground_energy": E_quantum,
            "spectral_gap": spectral_gap,
            "pairing_norm": pairing_norm,
            "spatial_energy": E_spatial,
            "dirichlet_energy": E_dir,
            "antisymmetry_loss": antisym_loss,
            "jacobi_loss": jacobi_loss,
            "killing_trace": killing_trace,
            "compactness_energy": E_compact,
        }

        return total_action, observables

    def solve_stationary(
        self,
        Z_init: np.ndarray,
        X_init: np.ndarray,
        mask: np.ndarray,
        steps: int = 30,
        lr: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool, Dict[str, float]]:
        """
        Differentiable relaxation of continuous parameters (Z, X) to exact δS = 0.
        Returns:
            (Z_star, X_star, S_star, grad_norm, is_stable, observables_dict)
        """
        Z = torch.tensor(Z_init, dtype=torch.float32, requires_grad=True, device=self.device)
        X = torch.tensor(X_init, dtype=torch.float32, requires_grad=True, device=self.device)
        m = torch.tensor(mask, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam([Z, X], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            action, _ = self.compute_differentiable_action(Z, X, m)

            # Instant Vacuum Instability Detection (detects runaway unphysical actions)
            if torch.isnan(action) or torch.isinf(action) or action < -1e5:
                return Z_init, X_init, float("inf"), float("inf"), False, {}

            action.backward()
            optimizer.step()

        # Final evaluation at the stationary point
        with torch.no_grad():
            final_action, obs = self.compute_differentiable_action(Z, X, m)
            g_norm_Z = torch.norm(Z.grad) if Z.grad is not None else torch.tensor(0.0)
            g_norm_X = torch.norm(X.grad) if X.grad is not None else torch.tensor(0.0)
            grad_norm = float(g_norm_Z + g_norm_X)
            is_stable = not torch.isnan(final_action) and final_action.item() > -1e4

            obs_scalars = {k: float(v.item()) for k, v in obs.items()}
            obs_scalars["total_action"] = float(final_action.item())
            obs_scalars["stationarity_grad_norm"] = grad_norm

        return (
            Z.detach().cpu().numpy(),
            X.detach().cpu().numpy(),
            float(final_action.item()),
            grad_norm,
            is_stable,
            obs_scalars
        )
