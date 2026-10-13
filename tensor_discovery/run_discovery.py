"""
Level 0: Tensor Discovery Master Runner — Expanding to Matrix Multiplication

Executes autonomous discovery on:
  1. Symplectic Hamiltonian Invariance Tensor (Phase-Space Conservation)
  2. Lie Algebra so(3) Levi-Civita Tensor (Quantum Spin & 3D Angular Momentum)
  3. 2x2 Matrix Multiplication Tensor (Shape 4x4x4, searching for rank <= 7)
"""

from __future__ import annotations
import numpy as np

from tensor_discovery.targets import (
    make_symplectic_invariance_tensor,
    make_lie_algebra_so3_tensor,
    make_matrix_multiplication_tensor,
)
from tensor_discovery.self_play import run_self_play_discovery


def run_discovery_suite():
    print("\n" + "#" * 70)
    print("  EXPERIMENT 1: SYMPLECTIC INVARIANCE TENSOR (HAMILTONIAN PHASE SPACE)")
    print("#" * 70)
    target_symp = make_symplectic_invariance_tensor(dim=2)  # Shape (4, 4, 1)
    law_symp = run_self_play_discovery(
        target_tensor=target_symp,
        target_name="Symplectic Phase-Space Invariance",
        num_episodes=10,
        mcts_sims=20,
    )
    if law_symp and law_symp.is_exact:
        print("\n  [VERIFIED EXACT] Discovered Symplectic Decomposition:")
        print(law_symp.to_latex())

    print("\n" + "#" * 70)
    print("  EXPERIMENT 2: LIE ALGEBRA so(3) TENSOR (QUANTUM SPIN / ROTATION)")
    print("#" * 70)
    target_so3 = make_lie_algebra_so3_tensor()  # Shape (3, 3, 3)
    law_so3 = run_self_play_discovery(
        target_tensor=target_so3,
        target_name="Lie Algebra so(3) Levi-Civita Structure",
        num_episodes=10,
        mcts_sims=25,
    )
    if law_so3 and law_so3.is_exact:
        print("\n  [VERIFIED EXACT] Discovered so(3) Decomposition:")
        print(law_so3.to_latex())

    print("\n" + "#" * 70)
    print("  EXPERIMENT 3: 2x2 MATRIX MULTIPLICATION (ALPHA-TENSOR CHALLENGE)")
    print("#" * 70)
    target_matmul = make_matrix_multiplication_tensor(2, 2, 2)  # Shape (4, 4, 4)
    law_matmul = run_self_play_discovery(
        target_tensor=target_matmul,
        target_name="2x2 Matrix Multiplication Bilinear Tensor",
        num_episodes=30,
        mcts_sims=40,
    )
    if law_matmul and law_matmul.is_exact:
        print("\n  [VERIFIED EXACT] Discovered Matrix Multiplication Algorithm:")
        print(law_matmul.to_latex())
        print("\n  [EXECUTABLE PYTHON CODE]")
        print(law_matmul.to_python_function("discovered_fast_matmul_2x2"))


if __name__ == "__main__":
    run_discovery_suite()
