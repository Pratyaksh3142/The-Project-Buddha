"""
Component 1 — The Mathematical Language: Abstract Syntax Trees as Graphs.

Every equation the LMM builds or evaluates is stored as a tree of typed nodes:

  - Internal nodes are operators  (Add, Sin, Div, …)
  - Leaf nodes are terminals      (Var_0, Const_2, …)
  - Edges flow parent → operand

The tree is built *incrementally* by the MCTS.  It starts as a single
OPEN_SLOT and each "action" fills the next slot with a concrete node type,
possibly spawning new OPEN_SLOTs for that operator's children.

Public API
----------
MathAST(max_nodes, num_vars)
    .is_complete        → bool
    .legal_actions()    → List[int]
    .apply_action(a)    → MathAST          (returns a *new* tree — immutable)
    .to_tensors()       → (features, adj)  (for the neural net)
    .to_string(names)   → str              (human-readable algebra)
    .to_callable()      → Callable         (JAX-evaluable function)
    .complexity         → int              (number of operator nodes)
"""

from __future__ import annotations

import enum
import copy
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Node-type vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

class NodeType(enum.IntEnum):
    """Mathematical vocabulary — every possible node in the AST."""

    # Binary operators (2 children) ─────────────────────────────────────────
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    POW = 4

    # Unary operators (1 child) ────────────────────────────────────────────
    SIN = 5
    COS = 6
    EXP = 7
    LOG = 8
    NEG = 9

    # Constants (terminals — 0 children) ───────────────────────────────────
    CONST_HALF = 10     # 0.5
    CONST_1    = 11     # 1.0
    CONST_2    = 12     # 2.0

    # State variables (terminals — 0 children) ─────────────────────────────
    VAR_0 = 13
    VAR_1 = 14
    VAR_2 = 15
    VAR_3 = 16

    # Internal marker — NOT a real node type, never an action ──────────────
    OPEN_SLOT = 99


# Convenience sets ──────────────────────────────────────────────────────────
BINARY_OPS = frozenset({NodeType.ADD, NodeType.SUB, NodeType.MUL,
                        NodeType.DIV, NodeType.POW})
UNARY_OPS  = frozenset({NodeType.SIN, NodeType.COS, NodeType.EXP,
                        NodeType.LOG, NodeType.NEG})
CONSTANTS  = frozenset({NodeType.CONST_HALF, NodeType.CONST_1,
                        NodeType.CONST_2})
ALL_VARS   = frozenset({NodeType.VAR_0, NodeType.VAR_1,
                        NodeType.VAR_2, NodeType.VAR_3})
TERMINALS  = CONSTANTS | ALL_VARS

CONST_VALUES = {
    NodeType.CONST_HALF: 0.5,
    NodeType.CONST_1:    1.0,
    NodeType.CONST_2:    2.0,
}

# Symbol table for pretty-printing ─────────────────────────────────────────
_OP_SYMBOLS = {
    NodeType.ADD: "+", NodeType.SUB: "-",
    NodeType.MUL: "*", NodeType.DIV: "/", NodeType.POW: "^",
}
_FN_NAMES = {
    NodeType.SIN: "sin", NodeType.COS: "cos",
    NodeType.EXP: "exp", NodeType.LOG: "log",
}


def arity(node_type: NodeType) -> int:
    """Number of children an operator requires."""
    if node_type in BINARY_OPS:
        return 2
    if node_type in UNARY_OPS:
        return 1
    if node_type in TERMINALS:
        return 0
    if node_type == NodeType.OPEN_SLOT:
        return 0
    raise ValueError(f"Unknown node type: {node_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# AST node
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ASTNode:
    """Single node inside the tree."""
    node_type: NodeType
    children: List[int] = field(default_factory=list)   # indices into MathAST.nodes


# ═══════════════════════════════════════════════════════════════════════════════
# MathAST — the core data structure
# ═══════════════════════════════════════════════════════════════════════════════

class MathAST:
    """
    An equation represented as an incrementally-built abstract syntax tree.

    Construction protocol (used by MCTS):
        1. Start:   ast = MathAST(max_nodes=20, num_vars=2)
                     → single OPEN_SLOT at root
        2. Loop:    while not ast.is_complete:
                        actions = ast.legal_actions()
                        ast = ast.apply_action(choice)
        3. Done:    ast.to_string(["r", "v"])  →  "(1.0 / (r ^ 2.0))"
    """

    def __init__(self, max_nodes: int = 20, num_vars: int = 2):
        self.max_nodes = max_nodes
        self.num_vars = num_vars
        self.nodes: List[ASTNode] = [ASTNode(NodeType.OPEN_SLOT)]
        self._open_slots: List[int] = [0]          # BFS queue of unfilled positions

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def is_complete(self) -> bool:
        return len(self._open_slots) == 0

    @property
    def next_open_slot(self) -> Optional[int]:
        return self._open_slots[0] if self._open_slots else None

    @property
    def complexity(self) -> int:
        """Number of operator (non-terminal) nodes — for Occam's Razor."""
        return sum(1 for n in self.nodes
                   if n.node_type in BINARY_OPS or n.node_type in UNARY_OPS)

    # ── Action interface (for MCTS) ────────────────────────────────────────

    def legal_actions(self) -> List[int]:
        """Node types that can legally fill the next OPEN_SLOT."""
        if self.is_complete:
            return []

        actions: List[int] = []

        # How many more nodes we can add (replacing this slot counts as 0 net)
        remaining = self.max_nodes - self.num_nodes   # slots for NEW children
        open_after = len(self._open_slots) - 1        # other pending slots

        # Terminals are always legal (they don't create new slots) ──────────
        for c in CONSTANTS:
            actions.append(int(c))
        for i in range(self.num_vars):
            actions.append(int(NodeType.VAR_0) + i)

        # Unary ops: add 1 new open slot ────────────────────────────────────
        if remaining >= 1:
            for op in UNARY_OPS:
                actions.append(int(op))

        # Binary ops: add 2 new open slots ──────────────────────────────────
        if remaining >= 2:
            for op in BINARY_OPS:
                actions.append(int(op))

        return sorted(actions)

    def apply_action(self, action: int) -> MathAST:
        """
        Fill the next OPEN_SLOT with *action* and return a NEW MathAST.

        The original tree is never mutated (critical for MCTS rollbacks).
        """
        assert not self.is_complete, "Tree is already complete"
        assert action in self.legal_actions(), f"Illegal action {action}"

        node_type = NodeType(action)
        slot_idx = self._open_slots[0]

        # Deep copy ─────────────────────────────────────────────────────────
        new = MathAST.__new__(MathAST)
        new.max_nodes = self.max_nodes
        new.num_vars = self.num_vars
        new.nodes = [ASTNode(n.node_type, list(n.children)) for n in self.nodes]
        new._open_slots = list(self._open_slots)

        # Replace the slot ──────────────────────────────────────────────────
        new.nodes[slot_idx] = ASTNode(node_type)
        new._open_slots.pop(0)

        # Spawn children for operators ──────────────────────────────────────
        for _ in range(arity(node_type)):
            child_idx = len(new.nodes)
            new.nodes.append(ASTNode(NodeType.OPEN_SLOT))
            new.nodes[slot_idx].children.append(child_idx)
            new._open_slots.append(child_idx)

        return new

    # ── Neural-net conversion ──────────────────────────────────────────────

    def to_tensors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        node_features : ndarray, shape [max_nodes, vocab_size]
            One-hot encoded node types.  OPEN_SLOT rows are all-zero.
        adjacency     : ndarray, shape [max_nodes, max_nodes]
            adj[i][j] = 1  ⇔  node i is the parent of node j.
        """
        vocab_size = 13 + self.num_vars     # 5 binary + 5 unary + 3 const + vars

        features = np.zeros((self.max_nodes, vocab_size), dtype=np.float32)
        adj      = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)

        for i, node in enumerate(self.nodes):
            if i >= self.max_nodes:
                break
            nt = node.node_type
            if nt != NodeType.OPEN_SLOT and int(nt) < vocab_size:
                features[i, int(nt)] = 1.0
            for child_idx in node.children:
                if child_idx < self.max_nodes:
                    adj[i, child_idx] = 1.0

        return features, adj

    # ── Pretty-printing ────────────────────────────────────────────────────

    def to_string(self, var_names: Optional[List[str]] = None) -> str:
        """Human-readable algebraic string."""
        if not self.is_complete:
            return "<Incomplete Equation>"
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.num_vars)]

        def _fmt(idx: int) -> str:
            node = self.nodes[idx]
            t = node.node_type
            ch = node.children

            # Binary ────────────────────────────────────────────────────────
            if t in BINARY_OPS:
                return f"({_fmt(ch[0])} {_OP_SYMBOLS[t]} {_fmt(ch[1])})"

            # Unary ─────────────────────────────────────────────────────────
            if t in _FN_NAMES:
                return f"{_FN_NAMES[t]}({_fmt(ch[0])})"
            if t == NodeType.NEG:
                return f"(-{_fmt(ch[0])})"

            # Constants ─────────────────────────────────────────────────────
            if t in CONST_VALUES:
                v = CONST_VALUES[t]
                return str(int(v)) if v == int(v) else str(v)

            # Variables ─────────────────────────────────────────────────────
            if t in ALL_VARS:
                vi = int(t) - int(NodeType.VAR_0)
                return var_names[vi] if vi < len(var_names) else f"x{vi}"

            return "?"

        return _fmt(0)

    # ── Callable function (for fitness evaluation) ─────────────────────────

    def to_callable(self) -> Callable:
        """
        Return a pure function  f(state_vector) → scalar
        that evaluates this AST.  Safe against div-by-zero, overflow, etc.

        >>> eq = ...  # some complete MathAST
        >>> f = eq.to_callable()
        >>> f(jnp.array([2.0, 0.5]))   # evaluate at r=2.0, v=0.5
        """
        if not self.is_complete:
            raise ValueError("Cannot convert incomplete tree to callable")

        # Snapshot the node list so the closure is self-contained
        nodes = [(n.node_type, list(n.children)) for n in self.nodes]

        def _evaluate(state):
            """Recursive evaluator (pure Python, works with JAX tracers)."""
            import jax.numpy as jnp

            def _eval(idx: int):
                t, ch = nodes[idx]

                # Binary ops ────────────────────────────────────────────────
                if t == NodeType.ADD: return _eval(ch[0]) + _eval(ch[1])
                if t == NodeType.SUB: return _eval(ch[0]) - _eval(ch[1])
                if t == NodeType.MUL: return _eval(ch[0]) * _eval(ch[1])
                if t == NodeType.DIV:
                    d = _eval(ch[1])
                    return _eval(ch[0]) / jnp.where(jnp.abs(d) < 1e-10, 1e-10, d)
                if t == NodeType.POW:
                    b = _eval(ch[0])
                    e = _eval(ch[1])
                    return jnp.power(jnp.abs(b) + 1e-10, e)

                # Unary ops ─────────────────────────────────────────────────
                if t == NodeType.SIN: return jnp.sin(_eval(ch[0]))
                if t == NodeType.COS: return jnp.cos(_eval(ch[0]))
                if t == NodeType.EXP:
                    return jnp.exp(jnp.clip(_eval(ch[0]), -10.0, 10.0))
                if t == NodeType.LOG:
                    return jnp.log(jnp.abs(_eval(ch[0])) + 1e-10)
                if t == NodeType.NEG: return -_eval(ch[0])

                # Constants ─────────────────────────────────────────────────
                if t == NodeType.CONST_HALF: return jnp.float32(0.5)
                if t == NodeType.CONST_1:    return jnp.float32(1.0)
                if t == NodeType.CONST_2:    return jnp.float32(2.0)

                # Variables ─────────────────────────────────────────────────
                if t in ALL_VARS:
                    vi = int(t) - int(NodeType.VAR_0)
                    return state[vi]

                return jnp.float32(0.0)

            return _eval(0)

        return _evaluate

    # ── Repr ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "complete" if self.is_complete else f"{len(self._open_slots)} open"
        return f"MathAST({self.num_nodes} nodes, {status})"


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  AST GRAPH ENGINE — SELF TEST")
    print("=" * 60)

    # Build:  NEG(DIV(CONST_1, POW(VAR_0, CONST_2)))
    # Which is:  -(1 / r^2)
    ast = MathAST(max_nodes=20, num_vars=2)
    print(f"\n1. Initial state: {ast}")
    print(f"   Legal actions: {ast.legal_actions()}")

    # Step 1: NEG at root
    ast = ast.apply_action(int(NodeType.NEG))
    print(f"\n2. After NEG: {ast}")

    # Step 2: DIV (child of NEG)
    ast = ast.apply_action(int(NodeType.DIV))
    print(f"3. After DIV: {ast}")

    # Step 3: CONST_1 (left child of DIV = numerator)
    ast = ast.apply_action(int(NodeType.CONST_1))
    print(f"4. After CONST_1: {ast}")

    # Step 4: POW (right child of DIV = denominator)
    ast = ast.apply_action(int(NodeType.POW))
    print(f"5. After POW: {ast}")

    # Step 5: VAR_0 (base of POW = r)
    ast = ast.apply_action(int(NodeType.VAR_0))
    print(f"6. After VAR_0: {ast}")

    # Step 6: CONST_2 (exponent of POW = 2)
    ast = ast.apply_action(int(NodeType.CONST_2))
    print(f"7. After CONST_2: {ast}")

    assert ast.is_complete, "Tree should be complete!"

    # Pretty-print
    eq_str = ast.to_string(["r", "v"])
    print(f"\n   Equation: {eq_str}")
    print(f"   Complexity: {ast.complexity} operators")

    # Tensors
    feat, adj = ast.to_tensors()
    print(f"\n   Node features shape: {feat.shape}")
    print(f"   Adjacency matrix shape: {adj.shape}")

    # Callable
    import jax.numpy as jnp
    f = ast.to_callable()
    result = f(jnp.array([2.0, 0.0]))   # r=2.0, v=0.0
    expected = -1.0 / (2.0 ** 2)         # = -0.25
    print(f"\n   f(r=2.0, v=0.0) = {float(result):.6f}  (expected: {expected})")
    assert abs(float(result) - expected) < 1e-4, "Callable test FAILED!"

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
