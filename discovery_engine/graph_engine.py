#!/usr/bin/env python3
"""
================================================================================
          PROJECT BUDDHA: UNIVERSAL COMPUTATION GRAPH ENGINE (DAG)
================================================================================
Pure Mathematical Directed Acyclic Graph (DAG) Representation.
No Tokens, No Text Sequences, No Transformers.

A mathematical system is represented as a directed graph of operations
flowing from input state variables to time derivatives.
Supports arbitrary non-linear ODEs, transcendental operators, and coupled terms.
Directly serializes to Rust Omni-Compiler JSON for Pratyaksh C++ execution.
================================================================================
"""

import math
import copy
import json
import random
from typing import List, Dict, Tuple, Optional, Any, Set, Union


# Supported Primitive Operations
UNARY_OPS = ['neg', 'sin', 'cos', 'exp', 'abs', 'sqrt', 'tanh']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow']
ALL_OPS = UNARY_OPS + BINARY_OPS


class GraphNode:
    """A single computational node in the DAG."""
    __slots__ = ['id', 'node_type', 'op', 'var_idx', 'const_val', 'inputs']

    def __init__(self, node_id: int, node_type: str, 
                 op: Optional[str] = None, 
                 var_idx: Optional[int] = None, 
                 const_val: Optional[float] = None):
        self.id = node_id
        self.node_type = node_type  # 'var', 'const', 'op'
        self.op = op                # e.g. 'add', 'mul', 'sin'
        self.var_idx = var_idx      # For 'var': index of state y[idx]
        self.const_val = const_val  # For 'const': float parameter
        self.inputs: List[int] = [] # IDs of input nodes feeding into this node

    def copy(self) -> 'GraphNode':
        n = GraphNode(self.id, self.node_type, self.op, self.var_idx, self.const_val)
        n.inputs = list(self.inputs)
        return n


class ComputationGraph:
    """A complete multi-output mathematical Computation Graph (DAG).
       Computes dy/dt = f(t, y) for a system of dimension D.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.nodes: Dict[int, GraphNode] = {}
        self.output_nodes: List[Optional[int]] = [None] * dimension
        self.next_node_id = 0

        # Initialize input state variable nodes (y_0, y_1, ..., y_{D-1})
        self.var_nodes: List[int] = []
        for d in range(dimension):
            nid = self._alloc_id()
            node = GraphNode(nid, 'var', var_idx=d)
            self.nodes[nid] = node
            self.var_nodes.append(nid)

    def _alloc_id(self) -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def copy(self) -> 'ComputationGraph':
        g = ComputationGraph(self.dimension)
        g.next_node_id = self.next_node_id
        g.output_nodes = list(self.output_nodes)
        g.var_nodes = list(self.var_nodes)
        g.nodes = {nid: node.copy() for nid, node in self.nodes.items()}
        return g

    def add_constant(self, val: float) -> int:
        nid = self._alloc_id()
        self.nodes[nid] = GraphNode(nid, 'const', const_val=float(val))
        return nid

    def add_operator(self, op: str, input_ids: List[int]) -> int:
        if op in UNARY_OPS and len(input_ids) != 1:
            raise ValueError(f"Unary op {op} requires exactly 1 input, got {len(input_ids)}")
        if op in BINARY_OPS and len(input_ids) != 2:
            raise ValueError(f"Binary op {op} requires exactly 2 inputs, got {len(input_ids)}")

        nid = self._alloc_id()
        node = GraphNode(nid, 'op', op=op)
        node.inputs = list(input_ids)
        self.nodes[nid] = node
        return nid

    def set_output(self, dim_idx: int, node_id: int):
        if dim_idx >= self.dimension:
            raise IndexError(f"Dimension {dim_idx} out of range ({self.dimension})")
        self.output_nodes[dim_idx] = node_id

    def is_valid(self) -> bool:
        """A graph is valid if every output dimension has a wired root node
           and there are no dangling inputs or directed cycles.
        """
        if any(out is None or out not in self.nodes for out in self.output_nodes):
            return False

        # Cycle detection and dependency check via topological sort
        visited = set()
        visiting = set()

        def dfs(nid: int) -> bool:
            if nid in visiting:
                return False # Cycle detected
            if nid in visited:
                return True
            visiting.add(nid)
            node = self.nodes.get(nid)
            if node is None:
                return False
            for inp_id in node.inputs:
                if not dfs(inp_id):
                    return False
            visiting.remove(nid)
            visited.add(nid)
            return True

        for out_id in self.output_nodes:
            if not dfs(out_id):
                return False

        return True

    def evaluate(self, y: List[float], t: float = 0.0) -> Optional[List[float]]:
        """Numerically evaluate dy/dt = f(t, y). Returns None if undefined or NaN."""
        if not self.is_valid():
            return None

        cache: Dict[int, float] = {}

        def eval_node(nid: int) -> Optional[float]:
            if nid in cache:
                return cache[nid]
            node = self.nodes[nid]

            if node.node_type == 'var':
                val = y[node.var_idx]
            elif node.node_type == 'const':
                val = node.const_val
            elif node.node_type == 'op':
                arg_vals = [eval_node(inp) for inp in node.inputs]
                if any(v is None for v in arg_vals):
                    return None

                op = node.op
                try:
                    if op == 'neg': val = -arg_vals[0]
                    elif op == 'sin': val = math.sin(arg_vals[0])
                    elif op == 'cos': val = math.cos(arg_vals[0])
                    elif op == 'exp':
                        v = arg_vals[0]
                        val = math.exp(min(v, 80.0)) # Bounded to prevent inf
                    elif op == 'abs': val = abs(arg_vals[0])
                    elif op == 'sqrt':
                        val = math.sqrt(max(0.0, arg_vals[0]))
                    elif op == 'tanh': val = math.tanh(arg_vals[0])
                    elif op == 'add': val = arg_vals[0] + arg_vals[1]
                    elif op == 'sub': val = arg_vals[0] - arg_vals[1]
                    elif op == 'mul': val = arg_vals[0] * arg_vals[1]
                    elif op == 'div':
                        den = arg_vals[1]
                        if abs(den) < 1e-12:
                            sign = 1.0 if den >= 0 else -1.0
                            val = arg_vals[0] / (sign * 1e-12)
                        else:
                            val = arg_vals[0] / den
                    elif op == 'pow':
                        base, exp = arg_vals[0], arg_vals[1]
                        # Safe bounded power
                        if abs(base) < 1e-14 and exp < 0:
                            return None
                        val = math.pow(abs(base), min(max(exp, -4.0), 4.0))
                    else:
                        return None
                except Exception:
                    return None
            else:
                return None

            if not math.isfinite(val) or abs(val) > 1e12:
                return None

            cache[nid] = val
            return val

        results = []
        for out_id in self.output_nodes:
            res = eval_node(out_id)
            if res is None:
                return None
            results.append(res)
        return results

    def complexity(self) -> int:
        """Count only the active nodes connected to the outputs (Occam's razor)."""
        if not self.is_valid():
            return 999999

        active = set()
        def mark(nid: int):
            if nid in active: return
            active.add(nid)
            for inp in self.nodes[nid].inputs:
                mark(inp)

        for out_id in self.output_nodes:
            mark(out_id)
        return len(active)

    def to_rust_json(self, name: str, y0: List[float], t_span: List[float], dt: float) -> Optional[Dict[str, Any]]:
        """Converts DAG into tagged-enum JSON AST matching Rust buddha_omni_compiler."""
        if not self.is_valid():
            return None

        def node_to_ast(nid: int) -> Dict[str, Any]:
            node = self.nodes[nid]
            if node.node_type == 'var':
                return {"type": "State", "data": node.var_idx}
            elif node.node_type == 'const':
                return {"type": "Constant", "data": node.const_val}
            elif node.node_type == 'op':
                op = node.op
                if op == 'neg':
                    return {"type": "Neg", "data": node_to_ast(node.inputs[0])}
                elif op == 'sin':
                    return {"type": "Sin", "data": node_to_ast(node.inputs[0])}
                elif op == 'cos':
                    return {"type": "Cos", "data": node_to_ast(node.inputs[0])}
                elif op == 'exp':
                    return {"type": "Exp", "data": node_to_ast(node.inputs[0])}
                elif op == 'abs':
                    return {"type": "Abs", "data": node_to_ast(node.inputs[0])}
                elif op == 'sqrt':
                    return {"type": "Sqrt", "data": node_to_ast(node.inputs[0])}
                elif op == 'tanh':
                    return {"type": "Tanh", "data": node_to_ast(node.inputs[0])}
                elif op == 'add':
                    return {"type": "Add", "data": [node_to_ast(node.inputs[0]), node_to_ast(node.inputs[1])]}
                elif op == 'sub':
                    return {"type": "Sub", "data": [node_to_ast(node.inputs[0]), node_to_ast(node.inputs[1])]}
                elif op == 'mul':
                    return {"type": "Mul", "data": [node_to_ast(node.inputs[0]), node_to_ast(node.inputs[1])]}
                elif op == 'div':
                    return {"type": "Div", "data": [node_to_ast(node.inputs[0]), node_to_ast(node.inputs[1])]}
                elif op == 'pow':
                    return {"type": "Pow", "data": [node_to_ast(node.inputs[0]), node_to_ast(node.inputs[1])]}
            return {"type": "Constant", "data": 0.0}

        return {
            "name": name,
            "dimension": self.dimension,
            "is_pde": False,
            "equations": [node_to_ast(out_id) for out_id in self.output_nodes],
            "params": [],
            "default_y0": y0,
            "default_t_span": t_span,
            "default_dt": dt
        }

    def to_string(self, var_names: Optional[List[str]] = None) -> str:
        """Pretty-print equations."""
        if not self.is_valid():
            return "<Invalid or Incomplete Graph>"

        if var_names is None:
            var_names = [f"y{i}" for i in range(self.dimension)]

        def fmt(nid: int) -> str:
            n = self.nodes[nid]
            if n.node_type == 'var':
                return var_names[n.var_idx]
            elif n.node_type == 'const':
                v = n.const_val
                if abs(v - round(v)) < 1e-4: return str(int(round(v)))
                return f"{v:.4g}"
            elif n.node_type == 'op':
                op = n.op
                args = [fmt(inp) for inp in n.inputs]
                if op == 'neg': return f"(-{args[0]})"
                if op in UNARY_OPS: return f"{op}({args[0]})"
                if op == 'add': return f"({args[0]} + {args[1]})"
                if op == 'sub': return f"({args[0]} - {args[1]})"
                if op == 'mul': return f"({args[0]} * {args[1]})"
                if op == 'div': return f"({args[0]} / {args[1]})"
                if op == 'pow': return f"({args[0]}^{args[1]})"
            return "?"

        eq_strs = [f"d{var_names[d]}/dt = {fmt(self.output_nodes[d])}" for d in range(self.dimension)]
        return " | ".join(eq_strs)
