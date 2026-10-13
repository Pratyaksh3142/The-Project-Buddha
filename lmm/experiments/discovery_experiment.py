"""
End-to-end experiment: Train the LMM to discover Hooke's Law of Harmonic Oscillation (dv/dt = -x).
"""

from lmm.config import LMMConfig
from lmm.fitness import make_spring_dataset
from lmm.trainer import LMMTrainer
from lmm.inference import DiscoveryEngine


def main():
    print("=" * 80)
    print("      PROJECT BUDDHA: END-TO-END DISCOVERY EXPERIMENT")
    print("      TARGET LAW: HOOKE'S LAW OSCILLATOR (dv/dt = -x)")
    print("=" * 80)

    cfg = LMMConfig(
        max_nodes=12,
        max_vars=2,
        embedding_dim=128,
        mpnn_layers=3,
        mcts_simulations=40,
        batch_size=8,
        learning_rate=2e-3
    )

    dataset = make_spring_dataset()
    print(f"Dataset generated: {dataset.name} with {len(dataset.trajectories)} test trajectories.")

    # 1. Train the Brain through Self-Play RL
    trainer = LMMTrainer(cfg, dataset, seed=123)
    trainer.train_episodes(num_episodes=25, print_interval=5)

    # 2. Run Inference using the trained brain
    print("\nStarting Guided Discovery with Trained Brain...")
    engine = DiscoveryEngine(cfg, model=trainer.model, params=trainer.params)
    best_ast, best_fitness, best_eq = engine.discover(dataset, num_attempts=10, simulations_per_move=80)

    print("\n" + "=" * 80)
    print(f"FINAL RESULT:")
    print(f"  Target Law:        dv/dt = -x")
    print(f"  Discovered Formula: dv/dt = {best_eq}")
    print(f"  Fitness Score:     {best_fitness:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
