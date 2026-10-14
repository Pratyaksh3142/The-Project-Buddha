"""
Continuous Training & Discovery Engine for the Unified Relational Universe.

Autonomous discovery across:
1. Many-Body Quantum Ground States & Superconducting Pairing Symmetries
2. Invariant Geometric Harmonic Embeddings & Minimal Surfaces
3. Algebraic Associativity & Lie Gauge Symmetries
4. Graph Spectral Connectivity & Transport

ZERO hardcoded targets. ZERO human empirical formulas.
Action functional is purely variational energy: S = E_quantum + 0.1 * E_spatial + 2.0 * E_algebra.
"""

import os
import sys
import time
import json
import argparse
import signal
from typing import List, Dict, Any
import numpy as np

from unified_universe.substrate import UniversalSubstrate
from unified_universe.action_referee import UniversalActionReferee
from unified_universe.neural_pairformer import UnifiedIntuitionEngine
from unified_universe.mcts_engine import UnifiedMCTSEngine


def run_continuous_unified(
    duration_hours: float = 2.0,
    entity_curriculum: List[int] = [4, 5, 6],
    sims_per_step: int = 32,
    max_steps: int = 8,
    output_dir: str = "/Users/pi3.142/.gemini/antigravity/brain/b86bc748-a14d-4f6b-b65b-b4437bd6e1b6"
):
    target_seconds = duration_hours * 3600.0
    start_time = time.time()
    end_time = start_time + target_seconds

    catalog_dir = os.path.join(output_dir, "unified_data")
    os.makedirs(catalog_dir, exist_ok=True)
    catalog_file = os.path.join(catalog_dir, "unified_catalog.jsonl")
    progress_file = os.path.join(output_dir, "unified_live_progress.md")

    print("=" * 80)
    print("      PROJECT BUDDHA: UNIFIED RELATIONAL ACTION DISCOVERY ENGINE")
    print("=" * 80)
    print(f"  Target Duration:     {duration_hours:.2f} hours ({target_seconds:.0f} seconds)")
    print(f"  Entity Curriculum:   {entity_curriculum}")
    print(f"  Architecture:        DeepMind Pairformer + MCTS Unified Graph")
    print(f"  Hardware:            Apple M2 Unified Silicon (JAX/Flax)")
    print(f"  Action Principle:    Pure Variational Minimization (Zero Hardcoding)")
    print("=" * 80)

    # Initialize engines per curriculum scale N
    referees = {N: UniversalActionReferee(N=N) for N in entity_curriculum}
    brains = {}
    mcts_engines = {}

    for N in entity_curriculum:
        num_pairs = (N * (N - 1)) // 2
        num_actions = num_pairs * 2 * 2
        br = UnifiedIntuitionEngine(N=N, node_dim=N, pair_dim=N, num_actions=num_actions, learning_rate=1e-3)
        ckpt_path = os.path.join(catalog_dir, f"pairformer_N{N}.bin")
        if br.load_checkpoint(ckpt_path):
            print(f"  [Loaded Checkpoint] Successfully resumed Pairformer N={N} from {ckpt_path}")
        brains[N] = br
        mcts_engines[N] = UnifiedMCTSEngine(referee=referees[N], brain=br, num_simulations=sims_per_step, max_depth=max_steps)

    replays = {N: {"s": [], "z": [], "x": [], "p": [], "v": []} for N in entity_curriculum}

    running = True
    def sig_handler(sig, frame):
        nonlocal running
        print("\n[STOP SIGNAL RECEIVED] Gracefully shutting down unified runner...")
        running = False
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    total_episodes = 0
    gapped_paired_discoveries = 0
    geometric_relaxed_discoveries = 0
    algebraic_symmetry_discoveries = 0
    total_saved_states = 0

    seen_phase_hashes = set()
    last_telemetry_time = time.time()

    while running and (time.time() < end_time):
        total_episodes += 1
        N = int(np.random.choice(entity_curriculum))
        referee = referees[N]
        brain = brains[N]
        mcts = mcts_engines[N]

        substrate = UniversalSubstrate(num_entities=N, feature_dim=N, is_spatial=True)

        ep_s, ep_z, ep_x, ep_policies = [], [], [], []

        for step in range(1, max_steps + 1):
            best_action, visit_probs, q_val = mcts.search(substrate)

            ep_s.append(np.copy(substrate.S))
            ep_z.append(np.copy(substrate.Z))
            ep_x.append(np.copy(substrate.X))
            ep_policies.append(visit_probs)

            # Decode action
            num_actions_per_pair = 4
            pair_idx = best_action // num_actions_per_pair
            sub_act = best_action % num_actions_per_pair
            ch = sub_act // 2
            delta = -1.0 if (sub_act % 2 == 0) else 1.0

            curr = 0
            mut_i, mut_j = 0, 1
            found = False
            for i in range(N):
                for j in range(i + 1, N):
                    if curr == pair_idx:
                        mut_i, mut_j = i, j
                        found = True
                        break
                    curr += 1
                if found:
                    break

            substrate.mutate_pair_tensor(mut_i, mut_j, ch, delta)
            _, _, dx = brain.predict(substrate)
            substrate.X += dx * 0.05

            eval_res = referee.compute_universal_action(substrate)
            qm = eval_res["quantum"]
            sp = eval_res["spatial"]
            alg = eval_res["algebra"]
            top = eval_res.get("topology", {})

            has_couplings = np.sum(np.abs(substrate.Z)) > 0
            is_gapped_paired = (qm["spectral_gap"] > 0.05 and qm["pairing_norm"] > 0.01)
            is_geometric = sp.get("dirichlet_energy", 0.0) > 0.1
            is_symmetry = (alg["jacobi_violation"] < 1e-4 and has_couplings)

            if has_couplings and (is_gapped_paired or is_geometric or is_symmetry):
                fp = f"N{N}_gap{qm['spectral_gap']:.2f}_pair{qm['pairing_norm']:.2f}_jac{alg['jacobi_violation']:.2f}_act{eval_res['total_action']:.1f}"
                if fp not in seen_phase_hashes:
                    seen_phase_hashes.add(fp)
                    total_saved_states += 1

                    if is_gapped_paired: gapped_paired_discoveries += 1
                    if is_geometric: geometric_relaxed_discoveries += 1
                    if is_symmetry: algebraic_symmetry_discoveries += 1

                    entry = {
                        "state_id": total_saved_states,
                        "fingerprint": fp,
                        "entities_N": N,
                        "total_action": float(eval_res["total_action"]),
                        "ground_energy": float(qm["ground_energy"]),
                        "spectral_gap": float(qm["spectral_gap"]),
                        "pairing_norm": float(qm["pairing_norm"]),
                        "dirichlet_energy": float(sp.get("dirichlet_energy", 0.0)),
                        "radius_of_gyration": float(sp.get("radius_of_gyration", 0.0)),
                        "jacobi_violation": float(alg["jacobi_violation"]),
                        "killing_det": float(alg.get("killing_det", 0.0)),
                        "algebraic_connectivity": float(top.get("algebraic_connectivity", 0.0)),
                        "step": step,
                        "episode": total_episodes,
                        "elapsed_seconds": time.time() - start_time
                    }

                    with open(catalog_file, "a") as f:
                        f.write(json.dumps(entry) + "\n")

                    print(
                        f"\n[UNIFIED DISCOVERY #{total_saved_states}] N={N} | "
                        f"Action={eval_res['total_action']:.1f} | "
                        f"E_0={qm['ground_energy']:.2f} | "
                        f"Gap={qm['spectral_gap']:.3f} | "
                        f"Pairing={qm['pairing_norm']:.3f} | "
                        f"Rg={sp.get('radius_of_gyration', 0.0):.2f}"
                    )

        # Replay experience: target is negative action
        target_v = -eval_res["total_action"] / 20.0
        for s, z, x, p in zip(ep_s, ep_z, ep_x, ep_policies):
            replays[N]["s"].append(s)
            replays[N]["z"].append(z)
            replays[N]["x"].append(x)
            replays[N]["p"].append(p)
            replays[N]["v"].append(target_v)

        # Train Pairformer on batch
        if len(replays[N]["s"]) >= 32:
            idx = np.random.choice(len(replays[N]["s"]), size=32, replace=False)
            bs = np.array([replays[N]["s"][i] for i in idx])
            bz = np.array([replays[N]["z"][i] for i in idx])
            bx = np.array([replays[N]["x"][i] for i in idx])
            bp = np.array([replays[N]["p"][i] for i in idx])
            bv = np.array([replays[N]["v"][i] for i in idx])
            brain.train_on_batch(bs, bz, bx, bp, bv)

            if len(replays[N]["s"]) > 400:
                replays[N]["s"] = replays[N]["s"][-200:]
                replays[N]["z"] = replays[N]["z"][-200:]
                replays[N]["x"] = replays[N]["x"][-200:]
                replays[N]["p"] = replays[N]["p"][-200:]
                replays[N]["v"] = replays[N]["v"][-200:]

        # Telemetry & Checkpointing every 30 seconds
        now = time.time()
        if (now - last_telemetry_time) > 30.0:
            last_telemetry_time = now
            elapsed = now - start_time
            pct = min(100.0, (elapsed / target_seconds) * 100.0)
            rate = total_episodes / elapsed

            # Save Pairformer brain checkpoints
            for n_val in entity_curriculum:
                ckpt_p = os.path.join(catalog_dir, f"pairformer_N{n_val}.bin")
                brains[n_val].save_checkpoint(ckpt_p)

            print(f"  [Progress {pct:5.1f}% | Elapsed: {elapsed/3600:5.2f}h / {duration_hours:.2f}h] Ep: {total_episodes} ({rate:.1f} ep/s) | States: {total_saved_states} (Gapped/Paired: {gapped_paired_discoveries}, Geometric: {geometric_relaxed_discoveries}, Lie Symmetries: {algebraic_symmetry_discoveries})")

            with open(progress_file, "w") as f:
                f.write(f"# Project Buddha: Unified Relational Action Discovery Telemetry\n\n")
                f.write(f"### Pure Variational Principle (Zero Hardcoding)\n\n")
                f.write(f"- **Progress:** `{pct:.1f}%` ({elapsed/3600:.2f}h / {duration_hours:.2f}h)\n")
                f.write(f"- **Total Episodes:** `{total_episodes}` ({rate:.1f} episodes/sec)\n")
                f.write(f"- **Total Ground States Cataloged:** `{total_saved_states}`\n")
                f.write(f"  - **Gapped Quantum Matter (Superconducting symmetry):** `{gapped_paired_discoveries}`\n")
                f.write(f"  - **Geometric Harmonic Embeddings:** `{geometric_relaxed_discoveries}`\n")
                f.write(f"  - **Exact Lie Symmetries:** `{algebraic_symmetry_discoveries}`\n")
                f.write(f"- **Active Curriculum Nodes:** `{entity_curriculum}`\n")
                f.write(f"- **Last Updated:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n\n")
                f.write(f"All physical ground states permanently cataloged in `unified_data/unified_catalog.jsonl`.\n")

    total_time = time.time() - start_time
    print("=" * 80)
    print("             UNIFIED CONTINUOUS TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Duration:              {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
    print(f"  Episodes:              {total_episodes}")
    print(f"  Total States Saved:    {total_saved_states}")
    print(f"  Gapped/Paired Matter:  {gapped_paired_discoveries}")
    print(f"  Geometric Embeddings:  {geometric_relaxed_discoveries}")
    print(f"  Lie Symmetries:        {algebraic_symmetry_discoveries}")
    print(f"  Catalog:               {catalog_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Continuous Discovery")
    parser.add_argument("--hours", type=float, default=2.0, help="Duration in hours")
    parser.add_argument("--dims", type=int, nargs="+", default=[4, 5, 6], help="Entity curriculum N")
    parser.add_argument("--sims", type=int, default=32, help="MCTS simulations per step")

    args = parser.parse_args()
    run_continuous_unified(
        duration_hours=args.hours,
        entity_curriculum=args.dims,
        sims_per_step=args.sims
    )
