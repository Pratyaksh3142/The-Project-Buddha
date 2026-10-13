"""
Persistent Mathematical Atlas of Discovered Algebraic Universes.

Stores, deduplicates, and catalogs discovered consistent algebraic structures
by their coordinate-invariant signature (dimension, Killing eigenvalues, center dim, rank).
Zero human physics labels. Pure mathematical classification.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional


class AlgebraAtlas:
    """
    Thread-safe persistent catalog of discovered mathematical algebras.
    Uses coordinate-invariant fingerprints to identify distinct isomorphism classes.
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.catalog_file = os.path.join(storage_dir, "algebra_catalog.jsonl")
        self.fingerprints = set()
        self.total_discoveries = 0

        # Load existing discoveries if catalog exists
        if os.path.exists(self.catalog_file):
            with open(self.catalog_file, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.fingerprints.add(entry["invariant_fingerprint"])
                        self.total_discoveries += 1

    def compute_fingerprint(self, analysis: Dict[str, Any]) -> str:
        """
        Computes a coordinate-invariant fingerprint based on:
        - Dimension N
        - Axiom system
        - Center dimension
        - Semi-simplicity
        - Sorted rounded eigenvalues of Cartan-Killing metric
        """
        dim = analysis["dimension"]
        sys_type = analysis["axiom_system"]
        c_dim = analysis["center_dimension"]
        is_ss = analysis["is_semisimple"]
        evs = sorted([round(float(e), 2) for e in analysis["killing_eigenvalues"]])
        ev_str = "_".join(f"{e:.2f}" for e in evs)

        return f"dim{dim}_{sys_type}_cdim{c_dim}_ss{int(is_ss)}_ev[{ev_str}]"

    def register_discovery(self, analysis: Dict[str, Any], raw_tensor: np.ndarray, source_info: Dict[str, Any]) -> bool:
        """
        Registers a discovery. Returns True if this is a genuinely NEW invariant isomorphism class.
        """
        fp = self.compute_fingerprint(analysis)
        is_novel = fp not in self.fingerprints

        if is_novel:
            self.fingerprints.add(fp)
            self.total_discoveries += 1

            def sanitize(o):
                if isinstance(o, (np.integer, int)):
                    return int(o)
                elif isinstance(o, (np.floating, float)):
                    return float(o)
                elif isinstance(o, (np.ndarray, list)):
                    return [sanitize(x) for x in o]
                elif isinstance(o, dict):
                    return {k: sanitize(v) for k, v in o.items()}
                return o

            entry = sanitize({
                "discovery_id": self.total_discoveries,
                "invariant_fingerprint": fp,
                "dimension": analysis["dimension"],
                "axiom_system": analysis["axiom_system"],
                "nonzero_entries": analysis["nonzero_entries"],
                "center_dimension": analysis["center_dimension"],
                "is_non_abelian": not analysis["is_abelian"],
                "is_semisimple": analysis["is_semisimple"],
                "killing_determinant": analysis["killing_determinant"],
                "killing_eigenvalues": analysis["killing_eigenvalues"],
                "brackets": analysis["brackets"],
                "raw_tensor": raw_tensor.tolist(),
                "source_info": source_info
            })

            with open(self.catalog_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

        return is_novel
