"""
Project Buddha v2.0: Unified Relational Universe Engine.
Three-tier bilevel neuro-symbolic discovery across quantum matter, geometry, and algebra.
Zero hardcoding. Zero domain-specific silos.
"""

from unified_universe.substrate import UniversalSubstrate
from unified_universe.action_referee import UniversalActionReferee
from unified_universe.neural_pairformer import UnifiedIntuitionEngine
from unified_universe.mcts_engine import UnifiedMCTSEngine, compute_num_actions, decode_action
from unified_universe.differentiable_solver import DifferentiableActionSolver
from unified_universe.composable_action import ComposableAction, TensorContractionTerm, AVAILABLE_TERMS
from unified_universe.meta_referee import MetaReferee
from unified_universe.symbolic_distiller import SymbolicDistiller, DiscoveredSymbolicLaw

__all__ = [
    "UniversalSubstrate",
    "UniversalActionReferee",
    "UnifiedIntuitionEngine",
    "UnifiedMCTSEngine",
    "compute_num_actions",
    "decode_action",
    "DifferentiableActionSolver",
    "ComposableAction",
    "TensorContractionTerm",
    "AVAILABLE_TERMS",
    "MetaReferee",
    "SymbolicDistiller",
    "DiscoveredSymbolicLaw",
]
