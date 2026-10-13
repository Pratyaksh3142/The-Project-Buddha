"""
Continuous Multi-Hour Training and Curriculum Discovery Engine for Project Buddha.

Runs self-play MCTS + JAX Neural Brain across multiple algebraic dimensions (N=3, 4, 5, 6),
catalogs all unique mathematical algebras into the persistent Atlas,
and continuously improves neural intuition over time.
Zero human targets. Zero physics constants.
"""

import os
import sys
import time
import argparse
import signal
from typing import List, Dict, Any
import numpy as np

from bootstrap_engine.axioms import evaluate_algebraic_axioms
from bootstrap_engine.generator import CandidateTensorSpace
from bootstrap_engine.neural_brain import NeuralIntuitionEngine
from bootstrap_engine.mcts import AxiomaticMCTS
from bootstrap_engine.extractor import DiscoveredAlgebraAnalyzer
from bootstrap_engine.atlas import AlgebraAtlas


def run_continuous_training(
    duration_hours: float = 5.0,
    dimensions: List[int] = [3, 4],
    axiom_system: str = "lie",
    target_mode: str = "any",
    sims_per_step: int = 48,
    max_steps: int = 12,
    output_dir: str = "/Users/pi3.142/.gemini/antigravity/brain/b86bc748-a14d-4f6b-b65b-b4437bd6e1b6"
):
    target_seconds = duration_hours * 3600.0
    start_time = time.time()
    end_time = start_time + target_seconds

    os.makedirs(output_dir, exist_ok=True)
    atlas = AlgebraAtlas(storage_dir=os.path.join(output_dir, "atlas_data"))

    print("=" * 80)
    print("      PROJECT BUDDHA: CONTINUOUS AXIOMATIC BOOTSTRAP DISCOVERY")
    print("=" * 80)
    print(f"  Target Duration:     {duration_hours:.2f} hours ({target_seconds:.0f} seconds)")
    print(f"  Target Dimensions:   {dimensions}")
    print(f"  Axiom System:        {axiom_system.upper()}")
    print(f"  Target Mode:         {target_mode.upper()}")
    print(f"  Pre-existing Atlas:  {atlas.total_discoveries} unique algebras loaded")
    print(f"  Hardware:            Apple M2 (JAX/Flax Neural Brain)")
    print("=" * 80)

    # Initialize brains and spaces for each dimension in curriculum
    spaces: Dict[int, CandidateTensorSpace] = {}
    brains: Dict[int, NeuralIntuitionEngine] = {}
    mcts_engines: Dict[int, AxiomaticMCTS] = {}

    is_skew = (axiom_system == "lie")
    for dim in dimensions:
        sp = CandidateTensorSpace(dimension=dim, skew_symmetric=is_skew)
        br = NeuralIntuitionEngine(tensor_shape=(dim, dim, dim), action_size=sp.action_size, learning_rate=1e-3)
        mc = AxiomaticMCTS(space=sp, brain=br, axiom_system=axiom_system, target_mode=target_mode, num_simulations=sims_per_step, max_depth=max_steps)
        spaces[dim] = sp
        brains[dim] = br
        mcts_engines[dim] = mc

    # Replay buffers per dimension
    replays: Dict[int, Dict[str, List]] = {
        dim: {"states": [], "policies": [], "values": []} for dim in dimensions
    }

    # Signal handler for graceful stop
    running = True
    def sig_handler(sig, frame):
        nonlocal running
        print("\n[STOP SIGNAL RECEIVED] Gracefully finishing current step and generating final summary...")
        running = False
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    total_episodes = 0
    novel_discoveries_this_run = 0
    last_telemetry_time = time.time()

    progress_file = os.path.join(output_dir, "atlas_live_progress.md")

    while running and (time.time() < end_time):
        total_episodes += 1
        # Alternate or weight dimensions (e.g. 60% N=3, 40% N=4)
        dim = np.random.choice(dimensions)
        space = spaces[dim]
        brain = brains[dim]
        mcts = mcts_engines[dim]

        # Init state
        if np.random.random() < 0.4:
            current_tensor = space.create_random_tensor(sparsity=0.7)
        else:
            current_tensor = space.create_null_tensor()

        ep_states = []
        ep_policies = []
        ep_solved = False
        last_action = None

        for step in range(1, max_steps + 1):
            eval_res = evaluate_algebraic_axioms(current_tensor, axiom_system)
            is_met = eval_res["is_consistent"]

            if is_met and target_mode == "semisimple":
                analyzer_temp = DiscoveredAlgebraAnalyzer(current_tensor, axiom_system)
                is_met = analyzer_temp.analyze()["is_semisimple"]

            if is_met:
                analyzer = DiscoveredAlgebraAnalyzer(current_tensor, axiom_system)
                analysis = analyzer.analyze()
                is_novel = atlas.register_discovery(
                    analysis=analysis,
                    raw_tensor=current_tensor,
                    source_info={"episode": total_episodes, "dimension": dim, "step": step, "elapsed": time.time() - start_time}
                )
                if is_novel:
                    novel_discoveries_this_run += 1
                    report = analyzer.generate_latex_report(analysis)
                    fn = f"discovered_N{dim}_id{atlas.total_discoveries}.md"
                    with open(os.path.join(output_dir, fn), "w") as f:
                        f.write(report)
                    print(f"\n[NOVEL DISCOVERY #{atlas.total_discoveries}] Dim={dim} | NonZero={analysis['nonzero_entries']} | Det(K)={analysis['killing_determinant']:.2f} | Non-Abelian={not analysis['is_abelian']}")
                ep_solved = True
                break

            best_action, visit_probs, q_val = mcts.search(current_tensor, last_action=last_action)
            last_action = best_action

            ep_states.append(np.copy(current_tensor))
            ep_policies.append(visit_probs)

            current_tensor = space.apply_action(current_tensor, best_action)

        # Buffer replay experience
        t_val = 1.0 if ep_solved else -0.5
        for s, p in zip(ep_states, ep_policies):
            replays[dim]["states"].append(s)
            replays[dim]["policies"].append(p)
            replays[dim]["values"].append(t_val)

        # Train neural brain if batch ready
        if len(replays[dim]["states"]) >= 32:
            idx = np.random.choice(len(replays[dim]["states"]), size=32, replace=False)
            bx = np.array([replays[dim]["states"][i] for i in idx])
            bp = np.array([replays[dim]["policies"][i] for i in idx])
            bv = np.array([replays[dim]["values"][i] for i in idx])
            brain.train_on_batch(bx, bp, bv)

            # Cap replay memory
            if len(replays[dim]["states"]) > 500:
                replays[dim]["states"] = replays[dim]["states"][-300:]
                replays[dim]["policies"] = replays[dim]["policies"][-300:]
                replays[dim]["values"] = replays[dim]["values"][-300:]

        # Telemetry update every 30 seconds
        now = time.time()
        if (now - last_telemetry_time) > 30.0:
            last_telemetry_time = now
            elapsed = now - start_time
            pct = min(100.0, (elapsed / target_seconds) * 100.0)
            rate = total_episodes / elapsed

            print(f"  [Progress {pct:5.1f}% | Elapsed: {elapsed/3600:5.2f}h / {duration_hours:.2f}h] Episodes: {total_episodes} ({rate:.1f} ep/s) | Atlas Total: {atlas.total_discoveries} (+{novel_discoveries_this_run} novel)")

            # Write markdown progress summary
            with open(progress_file, "w") as f:
                f.write(f"# Project Buddha: Continuous Discovery Telemetry\n\n")
                f.write(f"- **Progress:** `{pct:.1f}%` ({elapsed/3600:.2f}h / {duration_hours:.2f}h)\n")
                f.write(f"- **Episodes Evaluated:** `{total_episodes}` ({rate:.2f} episodes/sec)\n")
                f.write(f"- **Total Unique Isomorphism Classes in Atlas:** `{atlas.total_discoveries}`\n")
                f.write(f"- **New Discoveries This Session:** `{novel_discoveries_this_run}`\n")
                f.write(f"- **Active Curriculum Dimensions:** `{dimensions}`\n")
                f.write(f"- **Last Updated:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n\n")
                f.write(f"### Discovered Classes Catalog\n")
                f.write(f"The persistent catalog is indexed at `atlas_data/algebra_catalog.jsonl`.\n")

    total_time = time.time() - start_time
    print("=" * 80)
    print("                CONTINUOUS TRAINING RUN COMPLETE")
    print("=" * 80)
    print(f"  Total Duration:            {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
    print(f"  Total Episodes Run:        {total_episodes}")
    print(f"  Unique Algebras in Atlas:  {atlas.total_discoveries}")
    print(f"  Novel Found This Session:  {novel_discoveries_this_run}")
    print(f"  Catalog Path:              {atlas.catalog_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Buddha Continuous Training")
    parser.add_argument("--hours", type=float, default=5.0, help="Target training duration in hours")
    parser.add_argument("--dims", type=int, nargs="+", default=[3, 4], help="Curriculum dimensions")
    parser.add_argument("--axiom", type=str, default="lie", choices=["lie", "associative"])
    parser.add_argument("--mode", type=str, default="any", choices=["any", "semisimple"])
    parser.add_argument("--sims", type=int, default=48, help="MCTS simulations per step")

    args = parser.parse_args()
    run_continuous_training(
        duration_hours=args.hours,
        dimensions=args.dims,
        axiom_system=args.axiom,
        target_mode=args.mode,
        sims_per_step=args.sims
    )
