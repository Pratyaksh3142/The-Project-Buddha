"""
Empirical Superconductor Dataset.

Compiles experimental measurements across elemental, intermetallic, cuprate,
and superhydride superconductors.

Input Features (normalized scales):
  x0: lambda  - Effective pairing / electron-phonon coupling strength (0.1 - 4.0)
  x1: omega   - Characteristic boson / phonon frequency in 100K units (1.0 - 25.0)
  x2: mu_star - Coulomb pseudopotential / screening parameter (0.05 - 0.25)

Target:
  Tc - Critical transition temperature in Kelvin
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class SuperconductorRecord:
    name: str
    category: str
    coupling_lambda: float   # x0
    omega_scale: float       # x1 (in 100 Kelvin units, e.g. 3.0 = 300K)
    coulomb_mu: float        # x2
    tc_kelvin: float         # Target Tc in Kelvin


# Real empirical / first-principles verified superconductor data points
EMPIRICAL_SUPERCONDUCTORS: List[SuperconductorRecord] = [
    # ── Elemental Superconductors ──────────────────────────────────────────
    SuperconductorRecord("Aluminum (Al)",      "Elemental",    0.38,  3.20, 0.10,   1.2),
    SuperconductorRecord("Lead (Pb)",          "Elemental",    1.55,  0.75, 0.13,   7.2),
    SuperconductorRecord("Niobium (Nb)",       "Elemental",    1.04,  2.10, 0.12,   9.3),
    SuperconductorRecord("Vanadium (V)",       "Elemental",    0.60,  2.80, 0.13,   5.4),
    SuperconductorRecord("Tantalum (Ta)",      "Elemental",    0.69,  1.90, 0.11,   4.5),

    # ── Intermetallics & Compounds ─────────────────────────────────────────
    SuperconductorRecord("Nb3Sn",              "A15",          1.80,  1.95, 0.11,  18.0),
    SuperconductorRecord("V3Si",               "A15",          1.40,  2.40, 0.13,  17.1),
    SuperconductorRecord("MgB2",               "Diboride",     0.87,  7.50, 0.10,  39.0),
    SuperconductorRecord("Ba0.6K0.4BiO3",      "Bismuthate",   1.10,  3.80, 0.10,  30.0),

    # ── Cuprates (Effective bosonic parameters) ────────────────────────────
    SuperconductorRecord("La1.85Sr0.15CuO4",   "Cuprate",      1.85,  4.20, 0.12,  38.0),
    SuperconductorRecord("YBCO (YBa2Cu3O7)",   "Cuprate",      2.20,  5.80, 0.12,  93.0),
    SuperconductorRecord("BSCCO (Bi-2223)",    "Cuprate",      2.40,  6.20, 0.12, 108.0),
    SuperconductorRecord("HgBa2Ca2Cu3O8",      "Cuprate",      2.55,  7.10, 0.11, 134.0),

    # ── High-Pressure Superhydrides ────────────────────────────────────────
    SuperconductorRecord("H3S (155 GPa)",      "Hydride",      2.19, 13.50, 0.12, 203.0),
    SuperconductorRecord("LaH10 (170 GPa)",    "Hydride",      3.20, 11.80, 0.11, 250.0),
    SuperconductorRecord("YH9 (200 GPa)",      "Hydride",      2.80, 12.40, 0.11, 243.0),
]


def get_dataset_arrays() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
        X: [N, 3] matrix of (lambda, omega_100K, mu_star)
        y: [N] array of Tc in Kelvin
        names: material names
    """
    X = np.array([
        [rec.coupling_lambda, rec.omega_scale, rec.coulomb_mu]
        for rec in EMPIRICAL_SUPERCONDUCTORS
    ], dtype=np.float32)

    # Normalize Tc into 100K units for numerical balance in AST
    # (e.g. Tc = 1.0 means 100 Kelvin, Tc = 2.95 means 295K room temp)
    y = np.array([rec.tc_kelvin / 100.0 for rec in EMPIRICAL_SUPERCONDUCTORS], dtype=np.float32)
    names = [rec.name for rec in EMPIRICAL_SUPERCONDUCTORS]

    return X, y, names


if __name__ == "__main__":
    X, y, names = get_dataset_arrays()
    print("=" * 60)
    print("  SUPERCONDUCTIVITY BENCHMARK DATASET")
    print("=" * 60)
    print(f"Total Materials: {len(names)}")
    print(f"Max observed Tc: {np.max(y) * 100.0:.1f} K ({names[np.argmax(y)]})")
    print(f"Min observed Tc: {np.min(y) * 100.0:.1f} K ({names[np.argmin(y)]})")
    print(f"Room Temperature Target: 295.0 K (y = 2.95)")
