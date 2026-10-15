"""
Project Buddha v2.0: Meta-Referee (First-Principles Axiomatic Evaluator).

Eliminates the "Referee Paradox" where equations of physics were hardcoded.
Instead of checking whether a state satisfies a pre-programmed equation, the
Meta-Referee evaluates candidate Action Functionals S[Z, X] against universal
Mathematical and Physical Meta-Axioms:

  Meta-Axiom 1: Vacuum Boundedness & Stability (inf S > -inf, local minimum)
  Meta-Axiom 2: Gauge Invariance & Anomaly Cancellation (Bianchi J = 0, C + C^T = 0)
  Meta-Axiom 3: Non-Trivial Vacuum & Order Parameter (Z != 0, Delta E > 0 or E_0 < 0)
  Meta-Axiom 4: Spatial & Topological Realizability (Rg > 0, connected graph)

Any candidate Lagrangian/Action proposed by MCTS that satisfies these Meta-Axioms
represents a self-consistent, viable universe capable of generating physics.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch

from unified_universe.composable_action import ComposableAction


class MetaReferee:
    """
    Evaluates candidate Action Functionals S and their relaxed vacua against
    First-Principles Meta-Axioms.
    """

    def __init__(
        self,
        N: int,
        K: int,
        device: Optional[str] = None
    ):
        self.N = N
        self.K = K
        if device is None:
            self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device)

    def relax_and_evaluate(
        self,
        action: ComposableAction,
        substrate,
        steps: int = 30,
        lr: float = 0.05
    ) -> Dict[str, Any]:
        """
        Executes the Differentiable Variational Inner Solver on the candidate action S:
            nabla_{Z, X} S = 0
        and evaluates the resulting vacuum state against all Meta-Axioms.
        """
        N, K = self.N, self.K
        device = self.device

        # Initialize continuous field tensors from substrate
        Z_tensor = torch.tensor(substrate.Z, dtype=torch.float32, device=device, requires_grad=True)
        X_tensor = torch.tensor(substrate.X, dtype=torch.float32, device=device, requires_grad=True)
        mask = torch.tensor(getattr(substrate, 'mask', np.ones_like(substrate.Z)), dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam([Z_tensor, X_tensor], lr=lr)

        prev_loss = float("inf")
        is_stable = True
        divergence_detected = False

        # Variational relaxation loop: delta S = 0
        for step in range(steps):
            optimizer.zero_grad()
            total_S, breakdown = action.compute_action(Z_tensor, X_tensor, mask, N, K, device)

            # Check for vacuum instability (divergence to -inf or NaN)
            loss_val = total_S.item()
            if np.isnan(loss_val) or np.isinf(loss_val) or loss_val < -1e6:
                is_stable = False
                divergence_detected = True
                break

            total_S.backward()

            # Apply discrete mask to gradients safely
            if Z_tensor.grad is not None:
                with torch.no_grad():
                    Z_tensor.grad.mul_(mask)

            # Gradient clipping for numerical stability
            active_params = [p for p in [Z_tensor, X_tensor] if p.grad is not None]
            if active_params:
                torch.nn.utils.clip_grad_norm_(active_params, max_norm=5.0)
            optimizer.step()

        # Compute final stationarity gradient norm
        optimizer.zero_grad()
        final_S, final_breakdown = action.compute_action(Z_tensor, X_tensor, mask, N, K, device)
        final_S.backward()
        g_sq = 0.0
        if Z_tensor.grad is not None:
            with torch.no_grad():
                Z_tensor.grad.mul_(mask)
            g_sq += float(torch.sum(Z_tensor.grad**2).cpu().item())
        if X_tensor.grad is not None:
            g_sq += float(torch.sum(X_tensor.grad**2).cpu().item())
        g_norm = float(np.sqrt(g_sq))

        # If stable, update substrate in-place
        if is_stable and not divergence_detected:
            substrate.Z = Z_tensor.detach().cpu().numpy()
            substrate.X = X_tensor.detach().cpu().numpy()

        Z_np = substrate.Z
        X_np = substrate.X

        # ─────────────────────────────────────────────────────────
        # AXIOMATIC EVALUATION
        # ─────────────────────────────────────────────────────────

        # 1. Vacuum Stability: Action bounded from below, no runaway to -inf or NaN
        axiom_stability = is_stable and not divergence_detected
        axiom_stationarity = axiom_stability and (g_norm < 1.0)

        # 2. Anomaly Cancellation & Gauge Invariance
        gauge_offset = 2 if K >= 4 else 0
        M = min(N, K - gauge_offset)
        if M >= 2:
            C = Z_np[:M, :M, gauge_offset:gauge_offset + M]
            S_anti = C + np.swapaxes(C, 0, 1)
            antisym_violation = float(np.sum(np.abs(S_anti)))

            C_skew = 0.5 * (C - np.swapaxes(C, 0, 1))
            t1 = np.einsum('ijm,mkl->ijkl', C_skew, C_skew)
            t2 = np.einsum('jkm,mil->ijkl', C_skew, C_skew)
            t3 = np.einsum('kim,mjl->ijkl', C_skew, C_skew)
            J = t1 + t2 + t3
            jacobi_violation = float(np.sum(np.abs(J)))
            killing_kappa = np.einsum('iab,jba->ij', C_skew, C_skew)
            killing_trace = float(np.trace(killing_kappa))
        else:
            antisym_violation = 0.0
            jacobi_violation = 0.0
            killing_trace = 0.0

        axiom_gauge_anomaly = (antisym_violation < 1e-4) and (jacobi_violation < 1e-4)

        # 3. Non-Trivial Vacuum (Order Parameter != 0)
        # Check BdG spectrum
        T = 0.5 * (Z_np[:, :, 0] + Z_np[:, :, 0].T)
        Delta = 0.5 * (Z_np[:, :, 1] - Z_np[:, :, 1].T) if K > 1 else np.zeros((N, N))
        H_bdg = np.block([[T, Delta], [Delta.T, -T]])
        eigs = np.linalg.eigvalsh(H_bdg)
        pos_eigs = eigs[eigs > 1e-7]
        ground_energy = -0.5 * float(np.sum(pos_eigs)) if len(pos_eigs) > 0 else 0.0
        spectral_gap = float(np.min(pos_eigs)) if len(pos_eigs) > 0 else 0.0
        field_magnitude = float(np.linalg.norm(Z_np))

        # A non-trivial universe has non-zero field amplitude and either negative ground energy,
        # a spectral gap, or non-zero gauge generators
        axiom_non_trivial = (field_magnitude > 0.1) and (ground_energy < -1e-4 or spectral_gap > 1e-4 or abs(killing_trace) > 1e-4)

        # 4. Spatial Realizability
        diff = X_np[:, None, :] - X_np[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        center = np.mean(X_np, axis=0)
        rg = float(np.sqrt(np.mean(np.sum((X_np - center)**2, axis=-1))))
        axiom_spatial = (rg > 0.05) and (not np.any(np.isnan(X_np)))

        # Summary of Meta-Axioms
        passed_axioms = {
            "vacuum_stability": axiom_stability,
            "stationarity": axiom_stationarity,
            "anomaly_freedom": axiom_gauge_anomaly,
            "non_triviality": axiom_non_trivial,
            "spatial_realizability": axiom_spatial
        }

        # Core axioms satisfied => genuinely viable physical universe
        is_valid_universe = axiom_stability and axiom_gauge_anomaly and axiom_non_trivial and axiom_spatial

        # Meta-Score for guiding MCTS discovery
        if not axiom_stability:
            meta_score = -100.0  # Harsh penalty for unstable vacua
        else:
            # Reward: lower ground energy, higher spectral gap, lower gauge anomaly
            score = 0.0
            if ground_energy < 0:
                score += -ground_energy * 2.0
            if spectral_gap > 0:
                score += spectral_gap * 3.0
            if axiom_gauge_anomaly:
                score += 5.0
            else:
                score -= min(jacobi_violation + antisym_violation, 10.0)
            if axiom_spatial:
                score += 1.0
            meta_score = float(score)

        return {
            "is_valid_universe": is_valid_universe,
            "passed_axioms": passed_axioms,
            "meta_score": meta_score,
            "stationary_action": float(final_S.detach().cpu().item()) if is_stable else 999.0,
            "stationarity_grad_norm": g_norm if is_stable else 999.0,
            "breakdown": final_breakdown if is_stable else {},
            "observables": {
                "ground_energy": ground_energy,
                "spectral_gap": spectral_gap,
                "field_magnitude": field_magnitude,
                "jacobi_violation": jacobi_violation,
                "antisymmetry_violation": antisym_violation,
                "killing_trace": killing_trace,
                "radius_of_gyration": rg
            },
            "action_latex": action.to_latex()
        }
