"""
Quick verification runner for the Unified Relational Engine.
Runs a short discovery session and prints results.
"""

import os
import sys
import argparse
import time
import numpy as np

from unified_universe.substrate import UniversalSubstrate
from unified_universe.action_referee import UniversalActionReferee
from unified_universe.neural_pairformer import UnifiedIntuitionEngine
from unified_universe.mcts_engine import UnifiedMCTSEngine, compute_num_actions, decode_action


def run_quick_test(N: int = 5, K: int = 6, episodes: int = 5, sims: int = 32, steps: int = 6):
    print("=" * 80)
    print("  PROJECT BUDDHA: UNIFIED ENGINE QUICK VERIFICATION")
    print("=" * 80)
    print(f"  N={N} entities, K={K} channels")

    num_actions = compute_num_actions(N, K)
    node_dim = 2 * K
    print(f"  Actions: {num_actions}, Node dim: {node_dim}")

    referee = UniversalActionReferee(N=N, K=K)
    brain = UnifiedIntuitionEngine(N=N, node_dim=node_dim, pair_dim=K,
                                   num_actions=num_actions, dim=64, num_layers=3)
    mcts = UnifiedMCTSEngine(referee=referee, brain=brain,
                             num_simulations=sims, max_depth=steps)

    start = time.time()

    for ep in range(1, episodes + 1):
        sub = UniversalSubstrate(num_entities=N, num_channels=K, is_spatial=True)
        init_eval = referee.compute_action(sub)
        print(f"\n>>> Episode {ep}/{episodes} | Initial S: {init_eval['total_action']:.2f}")

        for step in range(1, steps + 1):
            action, probs, q = mcts.search(sub, temperature=max(0.1, 1.0 - ep / episodes))
            i, j, ch, delta = decode_action(action, N, K)
            sub.mutate_pair_tensor(i, j, ch, delta)

            if sub.is_spatial:
                _, _, dx = brain.predict(sub)
                sub.X += dx * 0.05

            ev = referee.compute_action(sub)
            qm = ev["quantum"]
            sp = ev["spatial"]
            al = ev["algebra"]

            print(
                f"  Step {step:02d} | ({i},{j}) ch={ch} Δ={delta:+.0f} | "
                f"S={ev['total_action']:7.1f} "
                f"E₀={qm['ground_energy']:6.2f} "
                f"Gap={qm['spectral_gap']:.3f} "
                f"Pair={qm['pairing_norm']:.3f} "
                f"Jac={al['jacobi_violation']:.1f} "
                f"Rg={sp['radius_of_gyration']:.2f}"
            )

    elapsed = time.time() - start
    print(f"\n  Completed in {elapsed:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=5)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    run_quick_test(N=args.entities, K=args.channels, episodes=args.episodes,
                   sims=args.sims, steps=args.steps)
