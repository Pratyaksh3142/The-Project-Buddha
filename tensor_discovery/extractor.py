"""
Level 0: Symbolic Extractor & Invariant Prover

Takes a winning sequence of rank-1 tensor actions and:
  1. Formally verifies that sum_{r=1}^R u_r (x) v_r (x) w_r == Target Tensor.
  2. Generates human-readable LaTeX and executable Python code for the discovered law/algorithm.
  3. Compares discovered rank R against human baseline complexity.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from tensor_discovery.environment import TensorAction


class DiscoveredTensorLaw:
    """Represents a verified exact tensor decomposition discovered by the engine."""

    def __init__(self, target_tensor: np.ndarray, actions: List[TensorAction], name: str = "Discovered Law"):
        self.target_tensor = np.copy(target_tensor)
        self.actions = actions
        self.name = name
        self.rank = len(actions)

        # Exact Invariant Verification
        self.reconstructed = np.zeros_like(self.target_tensor)
        for act in self.actions:
            self.reconstructed += act.outer_product()

        self.residual = self.target_tensor - self.reconstructed
        self.is_exact = np.all(self.residual == 0)

    def to_latex(self) -> str:
        """Emits LaTeX representation of the discovered tensor decomposition."""
        lines = [
            f"% Discovered Exact Decomposition: {self.name}",
            f"% Discovered Tensor Rank: R = {self.rank}",
            f"% Verified Invariant Error: {np.sum(np.abs(self.residual))}",
            r"\mathcal{T} = \sum_{r=1}^{" + str(self.rank) + r"} \mathbf{u}_r \otimes \mathbf{v}_r \otimes \mathbf{w}_r",
            r"\begin{aligned}",
        ]
        for r, act in enumerate(self.actions, 1):
            u_str = f"[{', '.join(map(str, act.u))}]^T"
            v_str = f"[{', '.join(map(str, act.v))}]^T"
            w_str = f"[{', '.join(map(str, act.w))}]^T"
            lines.append(f"  M_{{{r}}} &= ({u_str}) \\otimes ({v_str}) \\otimes ({w_str}) \\\\")
        lines.append(r"\end{aligned}")
        return "\n".join(lines)

    def to_python_function(self, func_name: str = "discovered_algorithm") -> str:
        """Emits standalone Python code implementing the discovered bilinear operation."""
        code = [
            f"# Auto-Generated Discovered Algorithm: {self.name}",
            f"# Rank: {self.rank} (Multiplications)",
            f"# Exact Invariant Verified: {self.is_exact}",
            "import numpy as np\n",
            f"def {func_name}(A, B):",
            "    # Flatten inputs",
            "    a = A.flatten()",
            "    b = B.flatten()",
            f"    # Compute {self.rank} scalar factors",
        ]
        for r, act in enumerate(self.actions, 1):
            u_terms = [f"{c}*a[{i}]" if c != 1 else f"a[{i}]" for i, c in enumerate(act.u) if c != 0]
            v_terms = [f"{c}*b[{j}]" if c != 1 else f"b[{j}]" for j, c in enumerate(act.v) if c != 0]
            u_expr = " + ".join(u_terms).replace("+ -", "- ") if u_terms else "0"
            v_expr = " + ".join(v_terms).replace("+ -", "- ") if v_terms else "0"
            code.append(f"    m_{r} = ({u_expr}) * ({v_expr})")

        dim_c = len(self.actions[0].w)
        code.append("\n    # Reconstruct outputs via w combinations")
        code.append(f"    c = np.zeros({dim_c}, dtype=np.float32)")
        for k in range(dim_c):
            w_terms = [f"{act.w[k]}*m_{r}" if act.w[k] != 1 else f"m_{r}" for r, act in enumerate(self.actions, 1) if act.w[k] != 0]
            w_expr = " + ".join(w_terms).replace("+ -", "- ") if w_terms else "0"
            code.append(f"    c[{k}] = {w_expr}")
        code.append("    return c\n")
        return "\n".join(code)
